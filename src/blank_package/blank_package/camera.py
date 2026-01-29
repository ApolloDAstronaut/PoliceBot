#!/usr/bin/python3
import os
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import CompressedImage, Range
from std_msgs.msg import Bool, Header
from duckietown_msgs.msg import WheelsCmdStamped

import numpy as np
import cv2


class ImageColorDetector(Node):
    """Detects small black region in the center ROI and publishes Bool when seen."""

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
        self.black_pixel_threshold_fraction = 0.01

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
            v = hsv[:, :, 2]
            black_mask = v < 50
            black_count = int(np.count_nonzero(black_mask))
            roi_area = roi.shape[0] * roi.shape[1]
            fractional_thresh = int(max(self.black_pixel_threshold_absolute, roi_area * self.black_pixel_threshold_fraction))
            detected = black_count >= fractional_thresh
        except Exception:
            return

        out = Bool()
        out.data = bool(detected)
        self.pub.publish(out)


class MotionController(Node):
    """Scans, approaches black object, then resumes scanning continuously."""

    def __init__(self):
        super().__init__('motion_controller')
        self.vehicle_name = os.getenv('VEHICLE_NAME', 'duckiebot').strip()

        self.black_topic = f'/{self.vehicle_name}/black_detected'
        self.range_topic = f'/{self.vehicle_name}/range'
        self.wheels_topic = f'/{self.vehicle_name}/wheels_cmd'

        # parameters
        self.distance_threshold = 0.20   # meters
        self.scan_left = (0.5, 0.0)      # left wheel forward, right stopped
        self.forward_speed = (0.45, 0.45)
        self.control_rate_hz = 10.0

        self.state = 'SCANNING'  # SCANNING -> APPROACHING -> RESET
        self.latest_range = None

        self.create_subscription(Bool, self.black_topic, self.black_callback, 10)
        self.create_subscription(Range, self.range_topic, self.range_callback, 10)
        self.wheels_pub = self.create_publisher(WheelsCmdStamped, self.wheels_topic, 10)

        self.create_timer(1.0 / self.control_rate_hz, self.control_loop)
        self.get_logger().info(f'MotionController ready, scanning and approaching black objects.')

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
        if self.state == 'SCANNING':
            self.publish_wheels(*self.scan_left, frame_id='scanning')
        elif self.state == 'APPROACHING':
            if self.latest_range is not None and self.latest_range <= self.distance_threshold:
                self.publish_wheels(0.0, 0.0, frame_id='arrived')
                self.get_logger().info(f'Reached object (range={self.latest_range:.3f}). Resuming scan.')
                self.state = 'SCANNING'  # reset to scanning
            else:
                self.publish_wheels(*self.forward_speed, frame_id='approach')
        else:
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
        motion_node.publish_wheels(0.0, 0.0, frame_id='shutdown')
        executor.shutdown()
        image_node.destroy_node()
        motion_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
