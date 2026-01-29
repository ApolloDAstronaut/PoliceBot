#!/usr/bin/python3
import os
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Duration

from sensor_msgs.msg import CompressedImage, Range
from std_msgs.msg import Bool, Header, ColorRGBA
from duckietown_msgs.msg import WheelsCmdStamped, LEDPattern

import numpy as np
import cv2


# ================= IMAGE DETECTOR =================

class ImageColorDetector(Node):
    """
    Robust black detection using grid + center weighting + debounce.
    Publishes Bool on /<vehicle>/black_detected
    """

    def __init__(self):
        super().__init__('image_color_detector')

        self.vehicle_name = os.getenv('VEHICLE_NAME', 'duckiebot').strip()
        img_topic = f'/{self.vehicle_name}/image/compressed'
        pub_topic = f'/{self.vehicle_name}/black_detected'

        self.create_subscription(CompressedImage, img_topic, self.image_callback, 10)
        self.pub = self.create_publisher(Bool, pub_topic, 10)

        # detection params
        self.frame_skip = 3
        self.counter = 0

        self.grid = 3
        self.v_thresh = 55           # black value threshold
        self.s_thresh = 60           # avoid dark colors
        self.avg_thresh = 0.004      # average fraction
        self.min_pixels = 10

        # debounce
        self.required_hits = 2
        self.hit_count = 0

        self.get_logger().info(f'ImageColorDetector listening on {img_topic}')

    def image_callback(self, msg: CompressedImage):
        if self.counter % self.frame_skip != 0:
            self.counter += 1
            return
        self.counter += 1

        try:
            arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return
        except Exception:
            return

        h, w = img.shape[:2]
        rw, rh = int(w * 0.4), int(h * 0.4)
        cx, cy = w // 2, h // 2
        roi = img[cy - rh//2:cy + rh//2, cx - rw//2:cx + rw//2]
        if roi.size == 0:
            return

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]

        tile_h = roi.shape[0] // self.grid
        tile_w = roi.shape[1] // self.grid

        weighted_sum = 0.0
        weight_total = 0.0

        for r in range(self.grid):
            for c in range(self.grid):
                y1 = r * tile_h
                x1 = c * tile_w
                tile_v = v[y1:y1+tile_h, x1:x1+tile_w]
                tile_s = s[y1:y1+tile_h, x1:x1+tile_w]
                if tile_v.size == 0:
                    continue

                mask = (tile_v < self.v_thresh) & (tile_s < self.s_thresh)
                count = np.count_nonzero(mask)
                if count < self.min_pixels:
                    continue

                frac = count / tile_v.size

                # center tiles matter more
                weight = 2.0 if (r == 1 and c == 1) else 1.0
                weighted_sum += frac * weight
                weight_total += weight

        avg = weighted_sum / max(weight_total, 1e-6)
        detected = avg >= self.avg_thresh

        if detected:
            self.hit_count += 1
        else:
            self.hit_count = 0

        out = Bool()
        out.data = self.hit_count >= self.required_hits
        self.pub.publish(out)


# ================= MOTION CONTROLLER =================

class MotionController(Node):
    """
    Turns → sees black → approaches → blinks → turns → repeats
    """

    def __init__(self):
        super().__init__('motion_controller')

        self.vehicle_name = os.getenv('VEHICLE_NAME', 'duckiebot').strip()

        self.create_subscription(Bool, f'/{self.vehicle_name}/black_detected', self.black_cb, 10)
        self.create_subscription(Range, f'/{self.vehicle_name}/range', self.range_cb, 10)

        self.wheels_pub = self.create_publisher(WheelsCmdStamped, f'/{self.vehicle_name}/wheels_cmd', 10)
        self.led_pub = self.create_publisher(LEDPattern, f'/{self.vehicle_name}/led_pattern', 10)

        # motion params (FASTER)
        self.scan_speed = (0.45, 0.0)
        self.forward_speed = (0.25, 0.25)
        self.turn_speed = 0.18

        self.distance_threshold = 0.07
        self.loss_timeout = 0.4

        self.state = 'SCANNING'
        self.latest_black_time = None
        self.latest_range = None
        self.turn_end = None
        self.blink_start = None
        self.led_on = False

        self.create_timer(0.08, self.control_loop)
        self.get_logger().info('MotionController ready')

    def black_cb(self, msg: Bool):
        if msg.data:
            self.latest_black_time = self.now()
            if self.state == 'SCANNING':
                self.get_logger().info('Black found → APPROACHING')
                self.state = 'APPROACHING'

    def range_cb(self, msg: Range):
        self.latest_range = msg.range

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def control_loop(self):
        t = self.now()

        if self.state == 'SCANNING':
            self.publish_wheels(*self.scan_speed)

        elif self.state == 'APPROACHING':
            if self.latest_black_time is None or t - self.latest_black_time > self.loss_timeout:
                self.get_logger().info('Lost black → SCANNING')
                self.state = 'SCANNING'
                return

            self.publish_wheels(*self.forward_speed)

            if self.latest_range is not None and self.latest_range < self.distance_threshold:
                self.get_logger().info('Arrived → BLINK')
                self.state = 'BLINK'
                self.blink_start = t
                self.publish_wheels(0.0, 0.0)

        elif self.state == 'BLINK':
            if int((t - self.blink_start) / 0.3) % 2 == 0:
                self.publish_led(True)
            else:
                self.publish_led(False)

            if t - self.blink_start > 1.2:
                self.publish_led(False)
                self.turn_end = self.get_clock().now() + Duration(seconds=1.6)
                self.state = 'TURNING'

        elif self.state == 'TURNING':
            if self.get_clock().now() < self.turn_end:
                self.publish_wheels(-self.turn_speed, self.turn_speed)
            else:
                self.state = 'SCANNING'
                self.latest_black_time = None

    def publish_wheels(self, l, r):
        msg = WheelsCmdStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'cmd'
        msg.vel_left = float(l)
        msg.vel_right = float(r)
        self.wheels_pub.publish(msg)

    def publish_led(self, on):
        p = LEDPattern()
        c = ColorRGBA()
        c.r = 1.0 if on else 0.0
        c.g = 1.0 if on else 0.0
        c.b = 0.0
        c.a = 1.0 if on else 0.0
        p.rgb_vals = [c for _ in range(5)]
        self.led_pub.publish(p)


# ================= MAIN =================

def main(args=None):
    rclpy.init(args=args)

    image_node = ImageColorDetector()
    motion_node = MotionController()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(image_node)
    executor.add_node(motion_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        motion_node.publish_wheels(0.0, 0.0)
        motion_node.publish_led(False)
        executor.shutdown()
        image_node.destroy_node()
        motion_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
