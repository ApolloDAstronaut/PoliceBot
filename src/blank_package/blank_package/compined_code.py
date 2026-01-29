#!/usr/bin/python3
import os
import rclpy
from rclpy.node import Node

import numpy as np
import cv2

from sensor_msgs.msg import Range
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped


class ColorBasedController(Node):
    def __init__(self):
        super().__init__('color_based_controller')

        self.vehicle_name = os.getenv('VEHICLE_NAME', 'duckiebot')

        image_topic = f'/{self.vehicle_name}/image/compressed'

        self.create_subscription(
            CompressedImage,
            image_topic,
            self.image_callback,
            10
        )

        self.wheels_pub = self.create_publisher(
            WheelsCmdStamped,
            f'/{self.vehicle_name}/wheels_cmd',
            10
        )

        self.get_logger().info(f"Subscribed to {image_topic}")

        self.frame_skip = 30
        self.counter = 0

    def image_callback(self, msg):
        # Skip frames for performance
        if self.counter % self.frame_skip != 0:
            self.counter += 1
            return

        if not msg.data:
            self.counter += 1
            return

        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            self.counter += 1
            return

        color = self.get_dominant_color(img)

        if color:
            self.get_logger().info(f"Detected color: {color}")
            self.react_to_color(color)
        else:
            self.stop()

        self.counter += 1

    def react_to_color(self, color, msg):
        distance = msg.range
        if color == "black":
            self.go_left()
        elif distance >= 0.2:
            self.move_forward()
        else:
            self.stop()

    def get_dominant_color(self, img):
        small = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        red_mask = (
            cv2.inRange(hsv, (0, 100, 50), (10, 255, 255)) |
            cv2.inRange(hsv, (160, 100, 50), (179, 255, 255))
        )
        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (140, 255, 255))
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))

        v = hsv[:, :, 2]
        black_mask = (v < 50).astype(np.uint8) * 255

        counts = {
            'red': np.count_nonzero(red_mask),
            'green': np.count_nonzero(green_mask),
            'blue': np.count_nonzero(blue_mask),
            'yellow': np.count_nonzero(yellow_mask),
            'black': np.count_nonzero(black_mask),
        }

        if all(count == 0 for count in counts.values()):
            return None

        return max(counts, key=counts.get)

    def move_forward(self):
        self.run_wheels(0.5, 0.5)

    def go_left(self):
        self.run_wheels(0.5, 0.0)

    def stop(self):
        self.run_wheels(0.0, 0.0)

    def run_wheels(self, vel_left, vel_right):
        msg = WheelsCmdStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.vel_left = vel_left
        msg.vel_right = vel_right
        self.wheels_pub.publish(msg)


def main():
    rclpy.init()
    node = ColorBasedController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
