import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from batmobile_perception import locate_func
import socket
import time
import struct


class JokerFinder(Node):
    def __init__(self):
        super().__init__('target_location')
        self.timer = self.create_timer(1/12.0, self.callback)

        #self.host = '192.168.0.24'  # IP of lenovo loq
        self.host = '192.168.0.182'  # IP of Raspberry Pi
        self.port = 8001
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect_socket()

    def connect_socket(self):
        while True:
            try:
                self.client_socket.connect((self.host, self.port))
                self.get_logger().info(f"Connected to server at {self.host}:{self.port}")
                break
            except socket.error as e:
                self.get_logger().warn(f"Connection failed: {e}, retrying in 2 seconds...")
                time.sleep(2)

    def callback(self):
        status, x, y, z = locate_func.get_location()
        if not status:
            x, y, z = 0.0, 0.0, 0.0

        data_str = f"({x:.2f},{y:.2f},{z:.2f})"
        data_bytes = data_str.encode('utf-8')
        header = struct.pack('!I', len(data_bytes))

        try:
            self.client_socket.sendall(header + data_bytes)
        except socket.error as e:
            self.get_logger().error(f"Send failed: {e}")
            self.connect_socket()


def main(args=None):
    rclpy.init(args=args)
    node = JokerFinder()
    try:
        rclpy.spin(node)
    finally:
        node.client_socket.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
