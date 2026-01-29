#!/usr/bin/python3
import os
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

import numpy as np
import cv2


class ImageColorPublisher(Node):
    def __init__(self):
        super().__init__('image_color_publisher')

        self.vehicle_name = os.getenv('VEHICLE_NAME', '').strip()

        image_topic = (
            f'/{self.vehicle_name}/image/compressed'
            if self.vehicle_name else '/image/compressed'
        )

        self.create_subscription(
            CompressedImage,
            image_topic,
            self.image_callback,
            10
        )

        self.color_pub = self.create_publisher(
            String,
            f'/{self.vehicle_name}/detected_color',
            10
        )

        self.get_logger().info(f"Subscribed to {image_topic}")
        self.get_logger().info("Publishing dominant color on /detected_color")

        self.frame_skip = 30
        self.counter = 0

    def image_callback(self, msg):
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

        dominant = self.get_dominant_color(img)

        if dominant is not None:
            color_msg = String()
            color_msg.data = dominant
            self.color_pub.publish(color_msg)
            self.get_logger().info(f"Published color: {dominant}")

        self.counter += 1

    def get_dominant_color(self, img):
        small = cv2.resize(img, (100, 100))
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

        if all(v == 0 for v in counts.values()):
            return None

        return max(counts, key=counts.get)


def main(args=None):
    rclpy.init(args=args)
    node = ImageColorPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
