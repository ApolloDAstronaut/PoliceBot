#!/usr/bin/python3
import os
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Duration

from sensor_msgs.msg import CompressedImage, Range
from std_msgs.msg import Bool, Header, ColorRGBA
from duckietown_msgs.msg import WheelsCmdStamped, LEDPattern

import numpy as np
import cv2
import time


class ImageColorDetector(Node):
    """
    Detect small black region using grid-averaged detection inside a center ROI.
    Publishes Bool on /<vehicle>/black_detected when average-black fraction exceeds threshold.
    """

    def __init__(self):
        super().__init__('image_color_detector')
        self.vehicle_name = os.getenv('VEHICLE_NAME', 'duckiebot').strip()
        img_topic = f'/{self.vehicle_name}/image/compressed'
        self.pub_topic = f'/{self.vehicle_name}/black_detected'

        self.create_subscription(CompressedImage, img_topic, self.image_callback, 10)
        self.pub = self.create_publisher(Bool, self.pub_topic, 10)

        self.get_logger().info(f'ImageColorDetector subscribing: {img_topic}, publishing: {self.pub_topic}')

        # parameters tuned for small-object detection
        self.frame_skip = 4
        self.counter = 0

        # grid detection params
        self.grid_rows = 3
        self.grid_cols = 3
        self.avg_fraction_threshold = 0.005    # average fraction across tiles to trigger (0.5%)
        self.per_tile_min_pixels = 12          # tile must have at least this many black pixels to count (avoids noise)

    def image_callback(self, msg: CompressedImage):
        # frame skip
        if self.counter % self.frame_skip != 0:
            self.counter += 1
            return
        self.counter += 1

        if not msg.data:
            return

        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return
        except Exception:
            return

        h, w = img.shape[:2]
        # center ROI (40% x 40%)
        rw, rh = int(w * 0.4), int(h * 0.4)
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - rw // 2), max(0, cy - rh // 2)
        x2, y2 = min(w, x1 + rw), min(h, y1 + rh)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return

        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2]
            # create grid and compute per-tile black fraction
            rows = self.grid_rows
            cols = self.grid_cols
            tile_h = max(1, roi.shape[0] // rows)
            tile_w = max(1, roi.shape[1] // cols)
            fractions = []
            for r in range(rows):
                for c in range(cols):
                    sy = r * tile_h
                    sx = c * tile_w
                    ey = sy + tile_h if (r < rows - 1) else roi.shape[0]
                    ex = sx + tile_w if (c < cols - 1) else roi.shape[1]
                    tile_v = v[sy:ey, sx:ex]
                    if tile_v.size == 0:
                        fractions.append(0.0)
                        continue
                    black_mask = tile_v < 50
                    black_count = int(np.count_nonzero(black_mask))
                    frac = black_count / float(tile_v.size)
                    # ignore tiny counts that are likely noise
                    if black_count < self.per_tile_min_pixels:
                        frac = 0.0
                    fractions.append(frac)
            avg_frac = float(np.mean(fractions))
            detected = avg_frac >= self.avg_fraction_threshold
        except Exception:
            return

        out = Bool()
        out.data = bool(detected)
        self.pub.publish(out)

        if detected:
            self.get_logger().info(f'Black detected: avg_frac={avg_frac:.4f}')



class MotionController(Node):
    """
    SCANNING -> CONFIRM (short stop) -> APPROACHING (all wheels forward, monitor image)
    -> ARRIVED_BLINK -> TURNING -> SCANNING
    If black disappears while APPROACHING (loss_timeout), go back to SCANNING.
    """

    def __init__(self):
        super().__init__('motion_controller')
        self.vehicle_name = os.getenv('VEHICLE_NAME', 'duckiebot').strip()

        self.black_topic = f'/{self.vehicle_name}/black_detected'
        self.range_topic = f'/{self.vehicle_name}/range'
        self.wheels_topic = f'/{self.vehicle_name}/wheels_cmd'
        self.led_topic = f'/{self.vehicle_name}/led_pattern'

        # params
        self.distance_threshold = 0.06     # meters to consider "arrived"
        self.loss_timeout = 0.35           # seconds: if no black seen for this long while approaching -> abort
        self.confirm_stop = 0.18          # seconds to stop and confirm before starting approach
        self.scan_wheels = (0.35, 0.0)    # single-wheel turn for scanning
        self.forward_wheels = (0.16, 0.16) # slow forward approach
        self.turn_speed = 0.12
        self.turn_duration = 1.6
        self.control_rate_hz = 12.0
        self.blink_interval = 0.4

        # state
        self.state = 'SCANNING'
        self.latest_range = None
        self.latest_black = False
        self.last_black_time = None
        self.confirm_end_time = None
        self.arrived_start_time = None
        self.turn_end_time = None
        self.last_blink_time = 0.0
        self.led_on = False

        # pubs/subs
        self.create_subscription(Bool, self.black_topic, self.black_callback, 10)
        self.create_subscription(Range, self.range_topic, self.range_callback, 10)
        self.wheels_pub = self.create_publisher(WheelsCmdStamped, self.wheels_topic, 10)
        self.led_pub = self.create_publisher(LEDPattern, self.led_topic, 10)

        self.create_timer(1.0 / float(self.control_rate_hz), self.control_loop)
        self.get_logger().info(f'MotionController ready: black={self.black_topic}, range={self.range_topic}')

    def black_callback(self, msg: Bool):
        now = self.get_clock().now().nanoseconds / 1e9
        if msg.data:
            self.latest_black = True
            self.last_black_time = now
            # trigger confirm if scanning
            if self.state == 'SCANNING':
                self.get_logger().info('Black seen -> entering CONFIRM state')
                self.state = 'CONFIRM'
                self.confirm_end_time = now + self.confirm_stop
        else:
            # do not immediately clear latest_black, keep timestamp for loss handling
            self.latest_black = False

    def range_callback(self, msg: Range):
        try:
            self.latest_range = float(msg.range)
        except Exception:
            self.latest_range = None

    def control_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9

        if self.state == 'SCANNING':
            # rotate to scan
            self.publish_wheels(*self.scan_wheels, frame_id='scanning')

        elif self.state == 'CONFIRM':
            # stop during confirmation window
            self.publish_wheels(0.0, 0.0, frame_id='confirm_stop')
            if self.confirm_end_time is not None and now >= self.confirm_end_time:
                # if recent black seen, go to APPROACHING, otherwise resume scanning
                if self.last_black_time is not None and (now - self.last_black_time) <= (self.loss_timeout):
                    self.get_logger().info('Confirm passed -> APPROACHING')
                    self.state = 'APPROACHING'
                else:
                    self.get_logger().info('Confirm failed -> resume SCANNING')
                    self.state = 'SCANNING'
                self.confirm_end_time = None

        elif self.state == 'APPROACHING':
            # If black has not been seen recently, abort and resume scanning
            if self.last_black_time is None or (now - self.last_black_time) > self.loss_timeout:
                self.get_logger().info('Lost black while approaching -> resume SCANNING')
                self.state = 'SCANNING'
                return

            # move forward
            self.publish_wheels(*self.forward_wheels, frame_id='approach')

            # check distance
            if self.latest_range is not None and self.latest_range <= self.distance_threshold:
                self.publish_wheels(0.0, 0.0, frame_id='arrived')
                self.get_logger().info(f'Arrived (range={self.latest_range:.3f}) -> blinking then turning slowly')
                self.state = 'ARRIVED_BLINK'
                self.arrived_start_time = now
                self.last_blink_time = now
                self.led_on = False
                self.publish_led(False)

        elif self.state == 'ARRIVED_BLINK':
            # blink LEDs
            if now - self.last_blink_time >= self.blink_interval:
                self.led_on = not self.led_on
                self.publish_led(self.led_on)
                self.last_blink_time = now

            # after short blink period, start turning
            if self.arrived_start_time is not None and (now - self.arrived_start_time) >= 1.2:
                self.arrived_start_time = None
                self.turn_end_time = self.get_clock().now() + Duration(seconds=self.turn_duration)
                self.state = 'TURNING'
                self.publish_led(False)

        elif self.state == 'TURNING':
            if self.turn_end_time is not None and self.get_clock().now() < self.turn_end_time:
                self.publish_wheels(-self.turn_speed, self.turn_speed, frame_id='turning')
            else:
                self.publish_wheels(0.0, 0.0, frame_id='stop_after_turn')
                self.get_logger().info('Turn complete -> resume SCANNING')
                self.turn_end_time = None
                self.state = 'SCANNING'
                # clear range and black history so we re-detect
                self.latest_range = None
                self.latest_black = False
                self.last_black_time = None

        else:
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

    def publish_led(self, on: bool):
        p = LEDPattern()
        if on:
            c = ColorRGBA()
            c.r = 1.0
            c.g = 1.0
            c.b = 0.0
            c.a = 1.0
        else:
            c = ColorRGBA()
            c.r = 0.0
            c.g = 0.0
            c.b = 0.0
            c.a = 0.0
        p.rgb_vals = [c for _ in range(5)]
        self.led_pub.publish(p)


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
        # ensure stopped + leds off
        try:
            motion_node.publish_wheels(0.0, 0.0, frame_id='shutdown')
            motion_node.publish_led(False)
        except Exception:
            pass
        executor.shutdown()
        image_node.destroy_node()
        motion_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
