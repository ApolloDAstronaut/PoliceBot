#!/usr/bin/python3

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.output_dir = "/workspace/images/"
        os.makedirs(self.output_dir, exist_ok=True)
        self.vehicle_name = os.getenv('VEHICLE_NAME')
        self.counter = 0
        self.create_subscription(CompressedImage, f'/{self.vehicle_name}/image/compressed', self.save_image, 10)

    def save_image(self, msg):
        if self.counter % 10 != 0:
            self.counter += 1
            return
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        black_mask = gray < 30
        if np.any(black_mask):
            self.get_logger().info("Black detected")
        cv2.imshow("Image", img)
        self.counter += 1

def main():
    rclpy.init()
    image_saver = ImageSaver()
    rclpy.spin(image_saver)
    rclpy.shutdown()


if __name__ == '__main__':
    main()


""" 
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage

def save_image(self, msg):
    if self.counter % 10 != 0:
        self.counter += 1
        return
    img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return

    small = cv2.resize(img, (100, 100))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    colors = {
        "red": ((0, 100, 100), (10, 255, 255), (160, 100, 100), (179, 255, 255)),
        "green": ((40, 50, 50), (80, 255, 255)),
        "blue": ((100, 50, 50), (140, 255, 255)),
        "black": ((0, 0, 0), (180, 255, 30)),
        "yellow": ((20, 100, 100), (30, 255, 255)),
    }

    counts = {}
    for name, ranges in colors.items():
        if name == "red":
            mask = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1])) + \
                   cv2.inRange(hsv, np.array(ranges[2]), np.array(ranges[3]))
        elif name == "black":
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            mask = gray < 30
        else:
            mask = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
        counts[name] = np.sum(mask)

    dominant = max(counts, key=counts.get)
    self.get_logger().info(f"Most dominant color: {dominant}")
    self.counter += 1

"""