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


# ============================================================
# IMAGE NODE — RED DETECTION
# ============================================================
class ImageColorDetector(Node):
    """
    Detect small RED region using grid-averaged detection inside a center ROI.
    Publishes Bool on /<vehicle>/black_detected (now meaning RED detected).
    """

    def __init__(self):
        super().__init__('image_color_detector')

        self.vehicle_name = os.getenv('VEHICLE_NAME', 'duckiebot').strip()
        self.img_topic = f'/{self.vehicle_name}/image/compressed'
        self.pub_topic = f'/{self.vehicle_name}/black_detected'

        self.create_subscription(
            CompressedImage,
            self.img_topic,
            self.image_callback,
            10
        )
        self.pub = self.create_publisher(Bool, self.pub_topic, 10)

        self.get_logger().info(
            f'RedDetector subscribing: {self.img_topic}, publishing: {self.pub_topic}'
        )

        self.frame_skip = 4
        self.counter = 0

        # Grid detection parameters
        self.grid_rows = 3
        self.grid_cols = 3
        self.avg_fraction_threshold = 0.005
        self.per_tile_min_pixels = 12

    def image_callback(self, msg: CompressedImage):
        if self.counter % self.frame_skip != 0:
            self.counter += 1
            return
        self.counter += 1

        if not msg.data:
            return

        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return
        except Exception:
            return

        h, w = img.shape[:2]

        # Center ROI (40% x 40%)
        rw, rh = int(w * 0.4), int(h * 0.4)
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - rw // 2), max(0, cy - rh // 2)
        x2, y2 = min(w, x1 + rw), min(h, y1 + rh)
        roi = img[y1:y2, x1:x2]

        if roi.size == 0:
            return

        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Red HSV masks (two ranges)
            mask1 = cv2.inRange(hsv, (0, 100, 50), (10, 255, 255))
            mask2 = cv2.inRange(hsv, (160, 100, 50), (179, 255, 255))
            red_mask = cv2.bitwise_or(mask1, mask2)

            rows, cols = self.grid_rows, self.grid_cols
            tile_h = max(1, roi.shape[0] // rows)
            tile_w = max(1, roi.shape[1] // cols)

            fractions = []

            for r in range(rows):
                for c in range(cols):
                    sy = r * tile_h
                    sx = c * tile_w
                    ey = sy + tile_h if r < rows - 1 else roi.shape[0]
                    ex = sx + tile_w if c < cols - 1 else roi.shape[1]

                    tile = red_mask[sy:ey, sx:ex]
                    if tile.size == 0:
                        fractions.append(0.0)
                        continue

                    red_pixels = int(np.count_nonzero(tile))
                    frac = red_pixels / float(tile.size)

                    if red_pixels < self.per_tile_min_pixels:
                        frac = 0.0

                    fractions.append(frac)

            avg_frac = float(np.mean(fractions))
            detected = avg_frac >= self.avg_fraction_threshold

        except Exception:
            return

        out = Bool()
        out.data = detected
        self.pub.publish(out)

        if detected:
            self.get_logger().info(f'Red detected: avg_frac={avg_frac:.4f}')


# ============================================================
# MOTION CONTROLLER
# ============================================================
class MotionController(Node):
    """
    SCANNING -> CONFIRM -> APPROACHING -> ARRIVED_BLINK -> TURNING -> SCANNING
    """

    def __init__(self):
        super().__init__('motion_controller')

        self.vehicle_name = os.getenv('VEHICLE_NAME', 'duckiebot').strip()

        self.black_topic = f'/{self.vehicle_name}/black_detected'
        self.range_topic = f'/{self.vehicle_name}/range'
        self.wheels_topic = f'/{self.vehicle_name}/wheels_cmd'
        self.led_topic = f'/{self.vehicle_name}/led_pattern'

        self.distance_threshold = 0.06
        self.loss_timeout = 0.35
        self.confirm_stop = 0.18
        self.scan_wheels = (1, 0.0)
        self.forward_wheels = (0.5, 0.5)
        self.turn_speed = 0.12
        self.turn_duration = 1.6
        self.control_rate_hz = 12.0
        self.blink_interval = 0.4

        self.state = 'SCANNING'
        self.latest_range = None
        self.last_black_time = None
        self.confirm_end_time = None
        self.arrived_start_time = None
        self.turn_end_time = None
        self.last_blink_time = 0.0
        self.led_on = False

        self.create_subscription(Bool, self.black_topic, self.black_callback, 10)
        self.create_subscription(Range, self.range_topic, self.range_callback, 10)

        self.wheels_pub = self.create_publisher(WheelsCmdStamped, self.wheels_topic, 10)
        self.led_pub = self.create_publisher(LEDPattern, self.led_topic, 10)

        self.create_timer(1.0 / self.control_rate_hz, self.control_loop)

        self.get_logger().info('MotionController ready')

    def black_callback(self, msg: Bool):
        now = self.get_clock().now().nanoseconds / 1e9
        if msg.data:
            self.last_black_time = now
            if self.state == 'SCANNING':
                self.state = 'CONFIRM'
                self.confirm_end_time = now + self.confirm_stop

    def range_callback(self, msg: Range):
        try:
            self.latest_range = float(msg.range)
        except Exception:
            self.latest_range = None

    def control_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9

        if self.state == 'SCANNING':
            self.publish_wheels(*self.scan_wheels, 'scanning')

        elif self.state == 'CONFIRM':
            self.publish_wheels(0.0, 0.0, 'confirm')
            if now >= self.confirm_end_time:
                if self.last_black_time and now - self.last_black_time <= self.loss_timeout:
                    self.state = 'APPROACHING'
                else:
                    self.state = 'SCANNING'

        elif self.state == 'APPROACHING':
            if not self.last_black_time or now - self.last_black_time > self.loss_timeout:
                self.state = 'SCANNING'
                return

            self.publish_wheels(*self.forward_wheels, 'approach')

            if self.latest_range and self.latest_range <= self.distance_threshold:
                self.publish_wheels(0.0, 0.0, 'arrived')
                self.state = 'ARRIVED_BLINK'
                self.arrived_start_time = now
                self.last_blink_time = now
                self.led_on = False

        elif self.state == 'ARRIVED_BLINK':
            if now - self.last_blink_time >= self.blink_interval:
                self.led_on = not self.led_on
                self.publish_led(self.led_on)
                self.last_blink_time = now

            if now - self.arrived_start_time >= 1.2:
                self.turn_end_time = self.get_clock().now() + Duration(seconds=self.turn_duration)
                self.state = 'TURNING'
                self.publish_led(False)

        elif self.state == 'TURNING':
            if self.get_clock().now() < self.turn_end_time:
                self.publish_wheels(-self.turn_speed, self.turn_speed, 'turning')
            else:
                self.publish_wheels(0.0, 0.0, 'done')
                self.state = 'SCANNING'
                self.latest_range = None
                self.last_black_time = None

    def publish_wheels(self, vl, vr, frame):
        msg = WheelsCmdStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame
        msg.vel_left = vl
        msg.vel_right = vr
        self.wheels_pub.publish(msg)

    def publish_led(self, on):
        p = LEDPattern()
        c = ColorRGBA()
        if on:
            c.r, c.g, c.b, c.a = 1.0, 1.0, 0.0, 1.0
        else:
            c.r, c.g, c.b, c.a = 0.0, 0.0, 0.0, 0.0
        p.rgb_vals = [c] * 5
        self.led_pub.publish(p)


# ============================================================
# MAIN
# ============================================================
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
        try:
            motion_node.publish_wheels(0.0, 0.0, 'shutdown')
            motion_node.publish_led(False)
        except Exception:
            pass

        executor.shutdown()
        image_node.destroy_node()
        motion_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()