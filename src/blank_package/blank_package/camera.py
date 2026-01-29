#!/usr/bin/python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Range
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import Header

import numpy as np
import cv2
from rclpy.time import Duration


class ImageColorToFNode(Node):
    """
    Node that:
    - Subscribes to compressed camera image
    - Subscribes to ToF range
    - Saves every 30th image
    - Detects dominant color
    - Moves robot forward until ToF threshold, then stops
    """

    def __init__(self):
        super().__init__('image_color_tof_node')

        self.vehicle_name = os.getenv('VEHICLE_NAME', '')
        self.counter = 0
        self.distance_threshold = 0.2
        self.forward_speed = 0.5
        self.turn_speed = 0.5
        self.turn_duration = 1.2  # seconds

        # latest ToF range
        self.latest_range = None

        # Subscribers
        img_topic = f'/{self.vehicle_name}/image/compressed' if self.vehicle_name else '/image/compressed'
        self.create_subscription(CompressedImage, img_topic, self.image_callback, 10)

        range_topic = f'/{self.vehicle_name}/range' if self.vehicle_name else '/range'
        self.create_subscription(Range, range_topic, self.range_callback, 10)

        # Publisher
        wheels_topic = f'/{self.vehicle_name}/wheels_cmd' if self.vehicle_name else '/wheels_cmd'
        self.wheels_pub = self.create_publisher(WheelsCmdStamped, wheels_topic, 10)

        # Timer for control loop
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info(f"Node started: subscribing to {img_topic} and {range_topic}")

        # Internal state
        self.latest_image = None
        self.state = 'MOVING'  # other states: 'TURNING'
        self.turn_end_time = None

    # ---------------- Callbacks ----------------

    def image_callback(self, msg: CompressedImage):
        self.latest_image = msg
        if self.counter % 30 != 0:
            self.counter += 1
            return

        # Save image
        try:
            output_dir = "/workspace/images/"
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f'{self.counter}.jpg')
            with open(path, 'wb') as f:
                f.write(msg.data)
            self.get_logger().info(f'Saved image {self.counter}')
        except Exception as e:
            self.get_logger().warn(f'Failed to save image: {e}')

        # Detect dominant color
        dominant = self.get_dominant_color(msg)
        if dominant:
            self.get_logger().info(f"Dominant color: {dominant}")
        else:
            self.get_logger().info("No dominant color detected")

        self.counter += 1

    def range_callback(self, msg: Range):
        self.latest_range = msg.range

    # ---------------- Control loop ----------------

    def control_loop(self):
        """Non-blocking loop to drive robot based on ToF and turning state."""
        from rclpy.clock import Clock

        now = self.get_clock().now()

        if self.state == 'MOVING':
            if self.latest_range is not None and self.latest_range < self.distance_threshold:
                self.get_logger().info(f"Range {self.latest_range:.3f} < {self.distance_threshold}, stopping")
                self._publish_wheels(0.0, 0.0, 'stop')
                # start turn
                self.turn_end_time = now + Duration(seconds=self.turn_duration)
                self.state = 'TURNING'
            else:
                self._publish_wheels(self.forward_speed, self.forward_speed, 'forward')

        elif self.state == 'TURNING':
            if now < self.turn_end_time:
                # turn left in place
                self._publish_wheels(-self.turn_speed, self.turn_speed, 'turn')
            else:
                self.state = 'MOVING'
                self.turn_end_time = None
                self._publish_wheels(0.0, 0.0, 'stop_after_turn')
                self.get_logger().info("Turn finished, resuming forward")

    # ---------------- Utilities ----------------

    def _publish_wheels(self, vel_left, vel_right, frame_id='cmd'):
        msg = WheelsCmdStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.vel_left = float(vel_left)
        msg.vel_right = float(vel_right)
        self.wheels_pub.publish(msg)

    def get_dominant_color(self, msg: CompressedImage):
        """Return one of blue, yellow, red, green, black, or None"""
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            small = cv2.resize(img, (100, 100))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

            # HSV masks
            masks = {
                'red': cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255])) |
                       cv2.inRange(hsv, np.array([160, 100, 50]), np.array([179, 255, 255])),
                'green': cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255])),
                'blue': cv2.inRange(hsv, np.array([100, 50, 50]), np.array([140, 255, 255])),
                'yellow': cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255])),
                'black': (hsv[:, :, 2] < 50).astype(np.uint8) * 255
            }

            counts = {k: int(np.count_nonzero(v)) for k, v in masks.items()}
            if all(v == 0 for v in counts.values()):
                return None
            return max(counts, key=counts.get)
        except Exception as e:
            self.get_logger().error(f"Color detection failed: {e}")
            return None


def main():
    rclpy.init()
    node = ImageColorToFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # ensure robot stopped
        node._publish_wheels(0.0, 0.0, 'shutdown')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
