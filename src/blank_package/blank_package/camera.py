#!/usr/bin/python3
import os
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Duration

from sensor_msgs.msg import CompressedImage, Range
from std_msgs.msg import Bool
from std_msgs.msg import Header
from duckietown_msgs.msg import WheelsCmdStamped

import numpy as np
import cv2


class ImageColorDetector(Node):
    """Detect if any (small) black region exists in the camera image and publish a Bool."""

    def __init__(self):
        super().__init__('image_color_detector')
        self.vehicle_name = os.getenv('VEHICLE_NAME', '').strip()
        topic = f'/{self.vehicle_name}/image/compressed' if self.vehicle_name else '/image/compressed'
        self.pub_topic = f'/{self.vehicle_name}/black_detected' if self.vehicle_name else '/black_detected'

        self.sub = self.create_subscription(CompressedImage, topic, self.image_callback, 10)
        self.pub = self.create_publisher(Bool, self.pub_topic, 10)

        self.get_logger().info(f'ImageColorDetector subscribing to: {topic}, publishing: {self.pub_topic}')

        self.frame_skip = 6          # check ~every 6th frame (tweak for performance)
        self.counter = 0
        self.black_pixel_thresh = 60  # number of black pixels to consider "detected" (tweak)
        self.roi = None               # optionally set (x,y,w,h) to look only in center

    def image_callback(self, msg: CompressedImage):
        # sample frames to reduce CPU usage
        if self.counter % self.frame_skip != 0:
            self.counter += 1
            return
        self.counter += 1

        if not msg.data:
            self.get_logger().warn('Received empty image data')
            return

        # decode
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                self.get_logger().warn('Failed to decode compressed image')
                return
        except Exception as e:
            self.get_logger().error(f'Exception decoding image: {e}')
            return

        # optional ROI (center) - comment out if not needed
        h, w = img.shape[:2]
        # small center ROI: 40% of width and height
        cx, cy = w // 2, h // 2
        rw, rh = int(w * 0.4), int(h * 0.4)
        x1 = max(0, cx - rw // 2)
        y1 = max(0, cy - rh // 2)
        x2 = min(w, x1 + rw)
        y2 = min(h, y1 + rh)
        roi_img = img[y1:y2, x1:x2]

        # convert to HSV and check V channel for dark pixels
        try:
            hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2]
            black_mask = v < 50  # threshold for "black"
            black_count = int(np.count_nonzero(black_mask))
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
            return

        # publish result if counts exceed threshold
        detected = Bool()
        detected.data = (black_count >= self.black_pixel_thresh)
        self.pub.publish(detected)

        if detected.data:
            self.get_logger().info(f'Black detected (count={black_count})')
        else:
            self.get_logger().debug(f'No black (count={black_count})')


class MotionController(Node):
    """
    Scanning (turning) until black_detected == True, then approach forward until ToF range < threshold, then stop and shutdown.
    """

    def __init__(self):
        super().__init__('motion_controller')
        self.vehicle_name = os.getenv('VEHICLE_NAME', '').strip()

        # topics
        self.black_topic = f'/{self.vehicle_name}/black_detected' if self.vehicle_name else '/black_detected'
        self.range_topic = f'/{self.vehicle_name}/range' if self.vehicle_name else '/range'
        self.wheels_topic = f'/{self.vehicle_name}/wheels_cmd' if self.vehicle_name else '/wheels_cmd'

        # params
        self.distance_threshold = 0.20  # meters, stop when closer than this
        self.scan_turn_speed = 0.35     # wheel units for turning (left wheel negative, right positive)
        self.forward_speed = 0.45       # forward wheel speeds
        self.control_rate_hz = 10.0

        # state
        self.latest_range = None
        self.black_seen = False
        self.state = 'SCANNING'  # SCANNING -> APPROACHING -> STOPPED

        # subscribers / publishers
        self.create_subscription(Bool, self.black_topic, self.black_callback, 10)
        self.create_subscription(Range, self.range_topic, self.range_callback, 10)
        self.wheels_pub = self.create_publisher(WheelsCmdStamped, self.wheels_topic, 10)

        # control timer
        timer_period = 1.0 / float(self.control_rate_hz)
        self.create_timer(timer_period, self.control_loop)

        self.get_logger().info(f'MotionController listening: {self.black_topic}, {self.range_topic}; publishing wheels: {self.wheels_topic}')

    def black_callback(self, msg: Bool):
        if msg.data:
            # set flag only when currently scanning
            if self.state == 'SCANNING':
                self.get_logger().info('Black signal received -> switch to APPROACHING')
                self.black_seen = True
                self.state = 'APPROACHING'
        else:
            # ignore False while approaching
            self.black_seen = False

    def range_callback(self, msg: Range):
        try:
            self.latest_range = float(msg.range)
        except Exception:
            self.latest_range = None

    def control_loop(self):
        # called at control_rate_hz; non-blocking
        if self.state == 'SCANNING':
            # rotate in place (left wheel negative, right positive) to scan environment
            self.publish_wheels(-self.scan_turn_speed, self.scan_turn_speed, frame_id='scanning')
        elif self.state == 'APPROACHING':
            # if we have range and are close enough -> stop and shutdown
            if self.latest_range is not None and self.latest_range <= self.distance_threshold:
                self.publish_wheels(0.0, 0.0, frame_id='stop')
                self.get_logger().info(f'Arrived within {self.latest_range:.3f} m <= {self.distance_threshold:.3f} m -> stopping and shutting down')
                # shutdown safely
                rclpy.shutdown()
                return
            # otherwise drive forward
            self.publish_wheels(self.forward_speed, self.forward_speed, frame_id='approach')
        else:
            # STOPPED or unknown -> ensure stopped
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
        # ensure robot stopped and cleanup
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