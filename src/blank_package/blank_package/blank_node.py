#!/usr/bin/python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

import numpy as np
import cv2


class ImageColorDetector(Node):
    def __init__(self):
        super().__init__('image_color_detector')
        self.output_dir = "/workspace/images/"
        os.makedirs(self.output_dir, exist_ok=True)
        self.vehicle_name = os.getenv('VEHICLE_NAME', '')
        self.counter = 0
        topic = f'/{self.vehicle_name}/image/compressed' if self.vehicle_name else '/image/compressed'
        self.create_subscription(CompressedImage, topic, self.save_and_detect, 10)
        self.get_logger().info(f"Subscribed to: {topic}")

    def save_and_detect(self, msg: CompressedImage):
        # process every 30th frame (same as original)
        if self.counter % 30 != 0:
            self.counter += 1
            return

        # save (optional - kept from original)
        try:
            path = os.path.join(self.output_dir, f'{self.counter}.jpg')
            with open(path, 'wb') as f:
                f.write(msg.data)
            self.get_logger().info(f'Saved image {self.counter}')
        except Exception as e:
            self.get_logger().warn(f'Failed to save image: {e}')

        # decode compressed image bytes to OpenCV image
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                self.get_logger().warn('Failed to decode image')
                self.counter += 1
                return
        except Exception as e:
            self.get_logger().error(f'Error decoding image: {e}')
            self.counter += 1
            return

        # compute dominant color among: blue, yellow, red, green, black
        dominant = self.get_dominant_color(img)
        if dominant is None:
            self.get_logger().info('No dominant color detected')
        else:
            self.get_logger().info(f'Most dominant color: {dominant}')

        self.counter += 1

    def get_dominant_color(self, img):
        """
        Return one of: 'blue', 'yellow', 'red', 'green', 'black'
        or None if nothing detected.
        """
        # work on small image for speed
        small = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # masks for colors
        # red needs two ranges on HSV hue circle
        lower_red1 = np.array([0, 100, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 50])
        upper_red2 = np.array([179, 255, 255])

        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])

        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])

        # black: low value in V channel
        v_channel = hsv[:, :, 2]
        black_mask = v_channel < 50

        # compute masks
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # count pixels (use nonzero counts)
        counts = {
            'red': int(np.count_nonzero(red_mask)),
            'green': int(np.count_nonzero(green_mask)),
            'blue': int(np.count_nonzero(blue_mask)),
            'yellow': int(np.count_nonzero(yellow_mask)),
            'black': int(np.count_nonzero(black_mask)),
        }

        # if all zero -> nothing detected
        if all(v == 0 for v in counts.values()):
            return None

        # return the color with the maximum count
        dominant = max(counts, key=counts.get)
        return dominant


def main():
    rclpy.init()
    node = ImageColorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
