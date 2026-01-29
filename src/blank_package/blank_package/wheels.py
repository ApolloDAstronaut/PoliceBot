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


class ImageColorDetector(Node):
    """Detect small red region in center ROI and publish Bool when seen."""

    def __init__(self):
        super().__init__('image_color_detector_red')
        self.vehicle_name = os.getenv('VEHICLE_NAME', 'duckiebot').strip()
        img_topic = f'/{self.vehicle_name}/image/compressed'
        self.pub_topic = f'/{self.vehicle_name}/red_detected'

        self.create_subscription(CompressedImage, img_topic, self.image_callback, 10)
        self.pub = self.create_publisher(Bool, self.pub_topic, 10)

        self.get_logger().info(f'ImageColorDetector subscribing: {img_topic}, publishing: {self.pub_topic}')

        self.frame_skip = 4  # check every 4th frame
        self.counter = 0
        self.red_pixel_threshold_absolute = 40
        self.red_pixel_threshold_fraction = 0.003

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
        rw, rh = int(w * 0.4), int(h * 0.4)
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - rw // 2), max(0, cy - rh // 2)
        x2, y2 = min(w, x1 + rw), min(h, y1 + rh)
        roi = img[y1:y2, x1:x2]

        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # red HSV mask (two ranges)
            lower_red1 = np.array([0, 100, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 50])
            upper_red2 = np.array([179, 255, 255])
            red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
            red_count = int(np.count_nonzero(red_mask))
            roi_area = max(1, roi.shape[0] * roi.shape[1])
            fractional_thresh = int(max(self.red_pixel_threshold_absolute, roi_area * self.red_pixel_threshold_fraction))
            detected = red_count >= fractional_thresh
        except Exception:
            return

        out = Bool()
        out.data = bool(detected)
        self.pub.publish(out)

        if detected:
            self.get_logger().info(f'Red detected (count={red_count}, thresh={fractional_thresh})')


class MotionController(Node):
    """
    SCANNING -> RED_DETECTED -> APPROACHING -> ARRIVED_BLINK -> TURNING -> SCANNING
    """

    def __init__(self):
        super().__init__('motion_controller_red')
        self.vehicle_name = os.getenv('VEHICLE_NAME', 'duckiebot').strip()

        self.red_topic = f'/{self.vehicle_name}/red_detected'
        self.range_topic = f'/{self.vehicle_name}/range'
        self.wheels_topic = f'/{self.vehicle_name}/wheels_cmd'
        self.led_topic = f'/{self.vehicle_name}/led_pattern'

        self.distance_threshold = 0.06
        self.hysteresis = 0.12
        self.scan_wheels = (0.4, 0.0)
        self.forward_wheels = (0.18, 0.18)
        self.turn_speed = 0.12
        self.turn_duration = 1.8
        self.control_rate_hz = 10.0
        self.blink_interval = 0.4

        self.state = 'SCANNING'
        self.latest_range = None
        self.last_blink_time = 0.0
        self.led_on = False
        self.turn_end_time = None

        self.create_subscription(Bool, self.red_topic, self.red_callback, 10)
        self.create_subscription(Range, self.range_topic, self.range_callback, 10)
        self.wheels_pub = self.create_publisher(WheelsCmdStamped, self.wheels_topic, 10)
        self.led_pub = self.create_publisher(LEDPattern, self.led_topic, 10)

        timer_period = 1.0 / float(self.control_rate_hz)
        self.create_timer(timer_period, self.control_loop)
        self.get_logger().info(f'MotionController ready: red={self.red_topic}, range={self.range_topic}')

    def red_callback(self, msg: Bool):
        if msg.data and self.state == 'SCANNING':
            self.get_logger().info('Red detected -> switching to APPROACHING')
            self.state = 'APPROACHING'

    def range_callback(self, msg: Range):
        try:
            self.latest_range = float(msg.range)
        except Exception:
            self.latest_range = None

    def control_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9

        if self.state == 'SCANNING':
            self.publish_wheels(self.scan_wheels[0], self.scan_wheels[1], frame_id='scanning')

        elif self.state == 'APPROACHING':
            self.publish_wheels(self.forward_wheels[0], self.forward_wheels[1], frame_id='approach')
            if self.latest_range is not None and self.latest_range <= self.distance_threshold:
                self.publish_wheels(0.0, 0.0, frame_id='arrived')
                self.get_logger().info(f'Arrived (range={self.latest_range:.3f}) -> blinking then turning slowly')
                self.state = 'ARRIVED_BLINK'
                self.last_blink_time = now
                self.led_on = False
                self.publish_led(False)

        elif self.state == 'ARRIVED_BLINK':
            if now - self.last_blink_time >= self.blink_interval:
                self.led_on = not self.led_on
                self.publish_led(self.led_on)
                self.last_blink_time = now
            if not hasattr(self, '_arrived_start_time'):
                self._arrived_start_time = now
            if now - self._arrived_start_time >= 1.2:
                self._arrived_start_time = None
                self.turn_end_time = self.get_clock().now() + Duration(seconds=self.turn_duration)
                self.state = 'TURNING'
                self.publish_led(False)

        elif self.state == 'TURNING':
            if self.turn_end_time is not None and self.get_clock().now() < self.turn_end_time:
                self.publish_wheels(-self.turn_speed, self.turn_speed, frame_id='turning')
            else:
                self.publish_wheels(0.0, 0.0, frame_id='stop_after_turn')
                self.get_logger().info('Turn complete -> resuming SCANNING')
                self.turn_end_time = None
                self.state = 'SCANNING'
                self.latest_range = None

        else:
            self.publish_wheels(0.0, 0.0, frame_id='stopped')

    def publish_wheels(self, vel_left: float, vel_right: float, frame_id: str = 'cmd'):
        msg = WheelsCmdStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.vel_left = float(vel_left)
        msg.vel_right = float(vel_right)
        self.wheels_pub.publish(msg)

    def publish_led(self, on: bool):
        p = LEDPattern()
        c = ColorRGBA()
        if on:
            c.r = 1.0
            c.g = 0.0
            c.b = 0.0
            c.a = 1.0
        else:
            c.r = 0.0
            c.g = 0.0
            c.b = 0.0
            c.a = 0.0
        p.rgb_vals = [c for _ in range(5)]
        self.led_pub.publish(p)


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
            motion_node.publish_wheels(0.0, 0.0, frame_id='shutdown')
            motion_node.publish_led(False)
        except Exception:
            pass
        executor.shutdown()
        image_node.destroy_node()
        motion_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
