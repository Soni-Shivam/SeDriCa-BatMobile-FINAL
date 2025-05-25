import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from batmobile_perception import locate_func
from threading import Thread
import socket

# Global socket variable
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '192.168.0.182'
port = 8001


def connect_socket():
    """Ensure socket is connected to the server."""
    global client_socket
  
    try:
        client_socket.connect((host, port))
        print(f"Connected to {host}:{port}")
    except socket.error as e:
        print(f"Socket error: {e}")
        client_socket.close()


class JokerFinder(Node):
    def __init__(self):
        super().__init__('target_location')
        self.pub = self.create_publisher(Point, 'get_point', 10)

        # Try to connect initially
        connect_socket()

        # Set up a timer for the callback
        timer_period = 1 / 24  # 24 Hz
        self.timer = self.create_timer(timer_period, self.callback)

    def callback(self):
        msg = Point()
        status, x, y, z = locate_func.get_location()

        if not status:
            msg.x = 0.0
            msg.y = 0.0
            msg.z = 0.0
            self.get_logger().error("Sensor Data Unavailable")
        else:
            msg.x = x
            msg.y = y
            msg.z = z
            if (x, y, z) == (0.0, 0.0, 0.0):
                self.get_logger().info("Object not detected")
            else:
                self.get_logger().info(f"Object detected at {x}, {y}, {z}")

        instruction = f"({msg.x},{msg.y},{msg.z})"

        # Attempt to send data over the socket
        try:
            client_socket.send(instruction.encode())
        except socket.error as e:
            self.get_logger().error(f"Socket error: {e}")
            self.get_logger().info("Reconnecting to socket...")
            connect_socket()  # Reconnect to the socket if it was closed

        # Publish the message
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    # Create and start the publisher node
    publisher = JokerFinder()

    rclpy.spin(publisher)

    # Close the socket when done
    client_socket.close()


if __name__ == '__main__':
    main()

