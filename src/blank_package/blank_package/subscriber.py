#!/usr/bin/python3
import os
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from duckietown_msgs.msg import WheelsCmdStamped


class ColorFollowerNode(Node):
    def __init__(self):
        super().__init__('color_follower_node')

        self.vehicle_name = os.getenv('VEHICLE_NAME', 'duckiebot')

        self.create_subscription(
            String,
            f'/{self.vehicle_name}/detected_color',
            self.color_callback,
            10
        )

        self.wheels_pub = self.create_publisher(
            WheelsCmdStamped,
            f'/{self.vehicle_name}/wheels_cmd',
            10
        )

        self.get_logger().info("Subscribed to /detected_color")

    def color_callback(self, msg):
        color = msg.data
        self.get_logger().info(f"Received color: {color}")

        if color == "red":
            self.go_left()
        elif color == "green":
            self.move_forward()
        else:
            self.stop()

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
    node = ColorFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
