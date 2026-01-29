#!/usr/bin/python3
import os
import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from sensor_msgs.msg import Range
from duckietown_msgs.msg import WheelsCmdStamped


class TofNode(Node):
    def __init__(self):
        super().__init__('tof_node')

        self.vehicle_name = os.getenv('VEHICLE_NAME', 'duckiebot')

        self.tof_sub = self.create_subscription(
            Range,
            f'/{self.vehicle_name}/range',
            self.check_range,
            10
        )

        self.wheels_pub = self.create_publisher(
            WheelsCmdStamped,
            f'/{self.vehicle_name}/wheels_cmd',
            10
        )

    def check_range(self, msg):
        distance = msg.range

        #if distance >= 1.5:
           #self.go_left()
        if distance > 1.5:
            self.move_forward()
        if distance <= 0.2:
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
    tof = TofNode()
    try:
        rclpy.spin(tof)
    finally:
        rclpy.stop()
        rclpy.shutdown()

if __name__ == '__main__':
    main()