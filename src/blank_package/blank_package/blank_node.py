#!/usr/bin/python3
import os
import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from sensor_msgs.msg import Range
from duckietown_msgs.msg import WheelsCmdStamped


class DistanceNode(Node):
    def __init__(self):
        super().__init__('distance')
        self.vehicle_name = os.getenv('VEHICLE_NAME')

        self.distance_sub = self.create_subscription(Range, f'/{self.vehicle_name}/distance_sensor', self.check_range, 10)

    def check_range(self, msg):
        distance = msg.range
        self.get_logger().info(f"Distance: {distance}")





def main():
    rclpy.init()
    distance_sensor = DistanceNode()
    rclpy.spin(distance_sensor)
    rclpy.shutdown()


if __name__ == '__main__':
    main()