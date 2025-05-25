import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from batmobile_perception import locale2 as locate_func
from threading import Thread
import socket
import json

class JokerFinder(Node):
    def __init__(self):
        super().__init__('target_location')

        # Setup socket to RPi
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.rpi_ip = '192.168.1.101'  # <--- Replace with your RPi IP
        self.port = 9999

        try:
            self.client_socket.connect((self.rpi_ip, self.port))
            self.get_logger().info(f"Connected to RPi at {self.rpi_ip}:{self.port}")
        except Exception as e:
            self.get_logger().error(f"Socket connection failed: {e}")
            self.client_socket = None

        # Start live stream in background
        Thread(target=locate_func.start_live_stream, daemon=True).start()

        # Timer to get and send location
        self.timer = self.create_timer(0.5, self.callback)

    def callback(self):
        status, x, y, z = locate_func.get_location()

        if not status or (x, y, z) == (0.0, 0.0, 0.0):
            self.get_logger().info("Object not detected or sensor unavailable")
        else:
            self.get_logger().info(f"Object detected at {x:.2f}, {y:.2f}, {z:.2f}")
            if self.client_socket:
                try:
                    data = json.dumps({'x': x, 'y': y, 'z': z})
                    self.client_socket.sendall(data.encode())
                except Exception as e:
                    self.get_logger().error(f"Failed to send data: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = JokerFinder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

