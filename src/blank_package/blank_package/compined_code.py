#!/usr/bin/python3
import os
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import CompressedImage, Range
from std_msgs.msg import Bool, Header, ColorRGBA
from duckietown_msgs.msg import WheelsCmdStamped, LEDPattern

import numpy as np
import cv2
import time


class ImageColorDetector(Node):
    """Detect small black region in center ROI and publish Bool when seen."""

    def __init__(self):
        super().__init__('image_color_detector')
        self.vehicle_name = os.getenv('VEHICLE_NAME', 'duckiebot').strip()
        img_topic = f'/{self.vehicle_name}/image/compressed'
        self.pub_topic = f'/{self.vehicle_name}/black_detected'

        self.create_subscription(CompressedImage, img_topic, self.image_callback, 10)
        self.pub = self.create_publisher(Bool, self.pub_topic, 10)

        self.get_logger().info(f'ImageColorDetector subscribing: {img_topic}, publishing: {self.pub_topic}')

        self.frame_skip = 6
        self.counter = 0
        self.black_pixel_threshold_absolute = 200
        self.black_pixel_threshold_fraction = 0.01  # fraction of ROI

    def image_callback(self, msg: CompressedImage):
        # sample frames to reduce CPU load
        if self.counter % self.frame_skip != 0:
            self.counter += 1
            return
        self.counter += 1

        if not msg.data:
            self.get_logger().warn('Empty image data')
            return

        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                self.get_logger().warn('Failed to decode image')
                return
        except Exception as e:
            self.get_logger().error(f'Image decode error: {e}')
            return

        h, w = img.shape[:2]
        # Center ROI (40% x 40%)
        rw, rh = int(w * 0.4), int(h * 0.4)
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - rw // 2), max(0, cy - rh // 2)
        x2, y2 = min(w, x1 + rw), min(h, y1 + rh)
        roi = img[y1:y2, x1:x2]

        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2]
            black_mask = v < 50
            black_count = int(np.count_nonzero(black_mask))
            roi_area = roi.shape[0] * roi.shape[1]
            fractional_thresh = int(max(self.black_pixel_threshold_absolute, roi_area * self.black_pixel_threshold_fraction))
            detected = black_count >= fractional_thresh
        except Exception as e:
            self.get_logger().error(f'ROI processing error: {e}')
            return

        out = Bool()
        out.data = bool(detected)
        self.pub.publish(out)

        if detected:
            self.get_logger().info(f'Black detected (count={black_count}, thresh={fractional_thresh})')


class MotionController(Node):
    """
    SCANNING -> when black detected -> APPROACHING (all wheels forward)
    -> when range <= threshold -> ARRIVED_BLINK (stop + blink LEDs)
    -> when range increases above threshold + hysteresis -> go back to SCANNING
    """

    def __init__(self):
        super().__init__('motion_controller')
        self.vehicle_name = os.getenv('VEHICLE_NAME', 'duckiebot').strip()

        self.black_topic = f'/{self.vehicle_name}/black_detected'
        self.range_topic = f'/{self.vehicle_name}/range'
        self.wheels_topic = f'/{self.vehicle_name}/wheels_cmd'
        self.led_topic = f'/{self.vehicle_name}/led_pattern'

        # parameters
        self.distance_threshold = 0.05  # meters: considered "arrived"
        self.hysteresis = 0.15  # must increase by this to resume scanning
        self.scan_wheels = (0.5, 0.0)  # left wheel forward for scanning
        self.forward_wheels = (0.15, 0.15)  # slow forward to avoid overshoot
        self.control_rate_hz = 10.0
        self.blink_interval = 0.5

        # state
        self.state = 'SCANNING'
        self.latest_range = None
        self.last_blink_time = 0.0
        self.led_on = False

        # pubs/subs
        self.create_subscription(Bool, self.black_topic, self.black_callback, 10)
        self.create_subscription(Range, self.range_topic, self.range_callback, 10)
        self.wheels_pub = self.create_publisher(WheelsCmdStamped, self.wheels_topic, 10)
        self.led_pub = self.create_publisher(LEDPattern, self.led_topic, 10)

        # control timer
        self.create_timer(1.0 / self.control_rate_hz, self.control_loop)

        self.get_logger().info(f'MotionController ready: black={self.black_topic}, range={self.range_topic}, wheels={self.wheels_topic}, leds={self.led_topic}')

    def black_callback(self, msg: Bool):
        if msg.data and self.state == 'SCANNING':
            self.get_logger().info('Black detected -> switching to APPROACHING')
            self.state = 'APPROACHING'

    def range_callback(self, msg: Range):
        try:
            self.latest_range = float(msg.range)
        except Exception:
            self.latest_range = None

    def control_loop(self):
        now = time.time()

        if self.state == 'SCANNING':
            # turn (one wheel) to scan
            self.publish_wheels(*self.scan_wheels, frame_id='scanning')

        elif self.state == 'APPROACHING':
            # drive all wheels forward
            self.publish_wheels(*self.forward_wheels, frame_id='approach')

            # if close enough -> stop & blink
            if self.latest_range is not None and self.latest_range <= self.distance_threshold:
                self.publish_wheels(0.0, 0.0, frame_id='arrived')
                self.get_logger().info(f'Arrived (range={self.latest_range:.3f}) -> blinking LEDs')
                self.state = 'ARRIVED_BLINK'
                self.last_blink_time = now
                self.led_on = False
                # ensure LED off initially
                self.publish_led(False)

        elif self.state == 'ARRIVED_BLINK':
            # blink LEDs
            if now - self.last_blink_time >= self.blink_interval:
                self.led_on = not self.led_on
                self.publish_led(self.led_on)
                self.last_blink_time = now
            # check if object moved away enough to restart scanning
            if self.latest_range is not None and self.latest_range >= (self.distance_threshold + self.hysteresis):
                self.get_logger().info(f'Range increased to {self.latest_range:.3f} -> resume scanning')
                # turn off LEDs and go back to scanning
                self.publish_led(False)
                self.state = 'SCANNING'

        else:
            # fallback: ensure stopped
            self.publish_wheels(0.0, 0.0, frame_id='stopped')

    def publish_wheels(self, vel_left: float, vel_right: float, frame_id: str = 'cmd'):
        msg = WheelsCmdStamped()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        msg.header = header
        msg.vel_left = float(vel_left)
        msg.vel_right = float(vel_right)
        self.wheels_pub.publish(msg)

    def publish_led(self, on: bool):
        # build LEDPattern: 5 ColorRGBA entries (duckietown convention)
        p = LEDPattern()
        if on:
            color = ColorRGBA()
            color.r = 1.0
            color.g = 1.0
            color.b = 0.0
            color.a = 1.0
        else:
            color = ColorRGBA()
            color.r = 0.0
            color.g = 0.0
            color.b = 0.0
            color.a = 0.0
        p.rgb_vals = [color for _ in range(5)]
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
        # ensure stopped + leds off
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
