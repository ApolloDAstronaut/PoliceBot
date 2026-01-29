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
    Robust RED detection using HSV + grid + debounce
    Publishes Bool on /<vehicle>/red_detected
    """

    def __init__(self):
        super().__init__('image_color_detector')

        self.vehicle_name = os.getenv('VEHICLE_NAME', 'duckiebot').strip()
        img_topic = f'/{self.vehicle_name}/image/compressed'
        pub_topic = f'/{self.vehicle_name}/red_detected'

        self.create_subscription(CompressedImage, img_topic, self.image_callback, 10)
        self.pub = self.create_publisher(Bool, pub_topic, 10)

        self.frame_skip = 3
        self.counter = 0

        self.grid = 3
        self.avg_thresh = 0.003
        self.min_pixels = 8

        self.required_hits = 2
        self.hit_count = 0

        self.get_logger().info(f'Red detector listening on {img_topic}')

    def image_callback(self, msg: CompressedImage):
        if self.counter % self.frame_skip != 0:
            self.counter += 1
            return
        self.counter += 1

        try:
            img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
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

        lower_red1 = np.array([0, 90, 60])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 90, 60])
        upper_red2 = np.array([179, 255, 255])

        mask = cv2.inRange(hsv, lower_red1, upper_red1) | \
               cv2.inRange(hsv, lower_red2, upper_red2)

        tile_h = roi.shape[0] // self.grid
        tile_w = roi.shape[1] // self.grid

        weighted_sum = 0.0
        weight_total = 0.0

        for r in range(self.grid):
            for c in range(self.grid):
                tile = mask[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w]
                if tile.size == 0:
                    continue

                count = np.count_nonzero(tile)
                if count < self.min_pixels:
                    continue

                frac = count / tile.size
                weight = 2.0 if (r == 1 and c == 1) else 1.0
                weighted_sum += frac * weight
                weight_total += weight

        detected = (weighted_sum / max(weight_total, 1e-6)) >= self.avg_thresh
        self.hit_count = self.hit_count + 1 if detected else 0

        out = Bool()
        out.data = self.hit_count >= self.required_hits
        self.pub.publish(out)


# ================= MOTION CONTROLLER =================

class MotionController(Node):
    """
    Turns → sees red → approaches (POLICE LIGHTS) → arrived (RED FLASH) → turns → repeats
    """

    def __init__(self):
        super().__init__('motion_controller')

        self.vehicle_name = os.getenv('VEHICLE_NAME', 'duckiebot').strip()

        self.create_subscription(Bool, f'/{self.vehicle_name}/red_detected', self.red_cb, 10)
        self.create_subscription(Range, f'/{self.vehicle_name}/range', self.range_cb, 10)

        self.wheels_pub = self.create_publisher(WheelsCmdStamped, f'/{self.vehicle_name}/wheels_cmd', 10)
        self.led_pub = self.create_publisher(LEDPattern, f'/{self.vehicle_name}/led_pattern', 10)

        self.scan_speed = (0.45, 0.0)
        self.forward_speed = (0.28, 0.28)
        self.turn_speed = 0.18

        self.distance_threshold = 0.07
        self.loss_timeout = 0.45

        self.state = 'SCANNING'
        self.latest_red_time = None
        self.latest_range = None
        self.turn_end = None
        self.blink_start = None

        self.create_timer(0.08, self.control_loop)
        self.get_logger().info('MotionController ready')

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def red_cb(self, msg: Bool):
        if msg.data:
            self.latest_red_time = self.now()
            if self.state == 'SCANNING':
                self.state = 'APPROACHING'

    def range_cb(self, msg: Range):
        self.latest_range = msg.range

    def control_loop(self):
        t = self.now()

        if self.state == 'SCANNING':
            self.publish_wheels(*self.scan_speed)
            self.publish_led_off()

        elif self.state == 'APPROACHING':
            if self.latest_red_time is None or t - self.latest_red_time > self.loss_timeout:
                self.state = 'SCANNING'
                return

            self.publish_wheels(*self.forward_speed)

            # 🚓 POLICE LIGHTS (red / blue alternating)
            if int(t / 0.25) % 2 == 0:
                self.publish_led_color(1.0, 0.0, 0.0)  # red
            else:
                self.publish_led_color(0.0, 0.0, 1.0)  # blue

            if self.latest_range is not None and self.latest_range < self.distance_threshold:
                self.publish_wheels(0.0, 0.0)
                self.blink_start = t
                self.state = 'BLINK'

        elif self.state == 'BLINK':
            # 🔴 FAST RED FLASH
            on = int((t - self.blink_start) / 0.2) % 2 == 0
            self.publish_led_color(1.0, 0.0, 0.0) if on else self.publish_led_off()

            if t - self.blink_start > 1.2:
                self.publish_led_off()
                self.turn_end = self.get_clock().now() + Duration(seconds=1.6)
                self.state = 'TURNING'

        elif self.state == 'TURNING':
            if self.get_clock().now() < self.turn_end:
                self.publish_wheels(-self.turn_speed, self.turn_speed)
            else:
                self.latest_red_time = None
                self.state = 'SCANNING'

    def publish_wheels(self, l, r):
        msg = WheelsCmdStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'cmd'
        msg.vel_left = float(l)
        msg.vel_right = float(r)
        self.wheels_pub.publish(msg)

    def publish_led_color(self, r, g, b):
        p = LEDPattern()
        c = ColorRGBA(r=r, g=g, b=b, a=1.0)
        p.rgb_vals = [c for _ in range(5)]
        self.led_pub.publish(p)

    def publish_led_off(self):
        self.publish_led_color(0.0, 0.0, 0.0)


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
        motion_node.publish_led_off()
        executor.shutdown()
        image_node.destroy_node()
        motion_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()