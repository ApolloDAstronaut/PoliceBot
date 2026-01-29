#!/usr/bin/python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

import numpy as np
import cv2


class ImageColorDetector(Node):
    """
    ROS2 node that subscribes to a CompressedImage topic and logs the most dominant
    color among: red, green, blue, yellow, black. No movement, no timers, no blocking.
    """

    def __init__(self):
        super().__init__('image_color_detector')

        self.vehicle_name = os.getenv('VEHICLE_NAME', '').strip()
        topic = f'/{self.vehicle_name}/image/compressed' if self.vehicle_name else '/image/compressed'
        self.create_subscription(CompressedImage, topic, self.image_callback, 10)

        self.get_logger().info(f"ImageColorDetector subscribing to: {topic}")

        self.frame_skip = 30
        self.counter = 0

    def image_callback(self, msg: CompressedImage):
        # sample every Nth frame
        if self.counter % self.frame_skip != 0:
            self.counter += 1
            return

        # Defensive: ensure there is payload
        if not msg.data:
            self.get_logger().warn("Received empty image data")
            self.counter += 1
            return

        # Decode compressed image bytes (JPEG/PNG) into OpenCV image
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                self.get_logger().warn("Failed to decode compressed image")
                self.counter += 1
                return
        except Exception as e:
            self.get_logger().error(f"Exception during image decode: {e}")
            self.counter += 1
            return

        dominant = self.get_dominant_color(img)
        if dominant is None:
            self.get_logger().info("No dominant color detected")
        else:
            self.get_logger().info(f"Most dominant color: {dominant}")

        self.counter += 1

    def get_dominant_color(self, img):
        """
        Return one of: 'red', 'green', 'blue', 'yellow', 'black' or None.
        Works on a downscaled image for speed.
        """
        try:
            small = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

            # Masks (HSV ranges)
            red_mask = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255])) | \
                       cv2.inRange(hsv, np.array([160, 100, 50]), np.array([179, 255, 255]))
            green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
            blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([140, 255, 255]))
            yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))

            # Black: low value in V channel
            v = hsv[:, :, 2]
            black_mask = (v < 50).astype(np.uint8) * 255

            counts = {
                'red': int(np.count_nonzero(red_mask)),
                'green': int(np.count_nonzero(green_mask)),
                'blue': int(np.count_nonzero(blue_mask)),
                'yellow': int(np.count_nonzero(yellow_mask)),
                'black': int(np.count_nonzero(black_mask)),
            }

            # If all zero, nothing matched
            if all(c == 0 for c in counts.values()):
                return None

            # Return color with maximum count
            return max(counts, key=counts.get)

        except Exception as e:
            self.get_logger().error(f"Error in dominant color detection: {e}")
            return None


def main(args=None):
    rclpy.init(args=args)
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
