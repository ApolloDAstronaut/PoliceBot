#!/usr/bin/python3
import os
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Duration

from sensor_msgs.msg import CompressedImage, Range
from std_msgs.msg import Bool, Header
from duckietown_msgs.msg import WheelsCmdStamped

import numpy as np
import cv2


class ImageColorDetector(Node):
    """Detects a small black region in the center ROI and publishes Bool when seen."""

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
        self.black_pixel_threshold_absolute = 200      # minimal pixels to consider "black region"
        self.black_pixel_threshold_fraction = 0.01    # or fraction of ROI area

    def image_callback(self, msg: CompressedImage):
        # sample frames to reduce CPU use
        if self.counter % self.frame_skip != 0:
            self.counter += 1
            return
        self.counter += 1

        if not msg.data:
            self.get_logger().warn('Empty image data')
            return

        # decode compressed image
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
        # center ROI (40% x 40% of frame)
        rw, rh = int(w * 0.4), int(h * 0.4)
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - rw // 2), max(0, cy - rh // 2)
        x2, y2 = min(w, x1 + rw), min(h, y1 + rh)
        roi = img[y1:y2, x1:x2]

        # detect dark pixels in ROI using HSV V channel
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2]
            black_mask = v < 50  # threshold; tweak if needed
            black_count = int(np.count_nonzero(black_mask))
            roi_area = roi.shape[0] * roi.shape[1]
            fractional_thresh = int(max(self.black_pixel_threshold_absolute, roi_area * self.black_pixel_threshold_fraction))
            detected = black_count >= fractional_thresh
        except Exception as e:
            self.get_logger().error(f'ROI processing error: {e}')
            return

        # publish boolean
        out = Bool()
        out.data = bool(detected)
        self.pub.publish(out)

        if detected:
            self.get_logger().info(f'Black detected in ROI (count={black_count}, thresh={fractional_thresh})')
        else:
            self.get_logger().debug(f'No black (count={black_count}, thresh={fractional_thresh})')


class MotionController(Node):
    """
    Scans by turning (one wheel active). When image node signals black_detected,
    transition to APPROACHING and drive forward until ToF range <= threshold, then stop and exit.
    """

    def __init__(self):
        super().__init__('motion_controller')
        self.vehicle_name = os.getenv('VEHICLE_NAME', 'duckiebot').strip()

        self.black_topic = f'/{self.vehicle_name}/black_detected'
        self.range_topic = f'/{self.vehicle_name}/range'
        self.wheels_topic = f'/{self.vehicle_name}/wheels_cmd'

        # parameters (tweak)
        self.distance_threshold = 0.20   # meters
        self.scan_left = (0.5, 0.0)      # left wheel, right wheel (turning in place using one wheel)
        self.forward_speed = (0.45, 0.45)
        self.control_rate_hz = 10.0

        # state
        self.state = 'SCANNING'  # SCANNING -> APPROACHING -> STOPPED
        self.latest_range = None

        # pubs/subs
        self.create_subscription(Bool, self.black_topic, self.black_callback, 10)
        self.create_subscription(Range, self.range_topic, self.range_callback, 10)
        self.wheels_pub = self.create_publisher(WheelsCmdStamped, self.wheels_topic, 10)

        # control timer
        timer_period = 1.0 / float(self.control_rate_hz)
        self.create_timer(timer_period, self.control_loop)

        self.get_logger().info(f'MotionController: black_topic={self.black_topic}, range_topic={self.range_topic}, wheels={self.wheels_topic}')

    def black_callback(self, msg: Bool):
        if msg.data and self.state == 'SCANNING':
            self.get_logger().info('Black signal received -> switching to APPROACHING')
            self.state = 'APPROACHING'

    def range_callback(self, msg: Range):
        try:
            self.latest_range = float(msg.range)
        except Exception:
            self.latest_range = None

    def control_loop(self):
        # called at control_rate_hz
        if self.state == 'SCANNING':
            # turn by running only one wheel (left wheel forward, right wheel zero)
            self.publish_wheels(self.scan_left[0], self.scan_left[1], frame_id='scanning')
        elif self.state == 'APPROACHING':
            # if range available and close enough, stop and shutdown
            if self.latest_range is not None and self.latest_range <= self.distance_threshold:
                self.publish_wheels(0.0, 0.0, frame_id='arrived')
                self.get_logger().info(f'Arrived: range={self.latest_range:.3f} <= {self.distance_threshold:.3f}. Stopping and shutting down.')
                # request shutdown; main will cleanup
                rclpy.shutdown()
                return
            # otherwise drive forward
            self.publish_wheels(self.forward_speed[0], self.forward_speed[1], frame_id='approach')
        else:
            # ensure stopped
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
        # ensure robot stopped
        try:
            motion_node.publish_wheels(0.0, 0.0, frame_id='shutdown')
        except Exception:
            pass
        executor.shutdown()
        image_node.destroy_node()
        motion_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
