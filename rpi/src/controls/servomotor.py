import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from gpiozero import Servo
from time import sleep

class ServoController(Node):
    def __init__(self):
        super().__init__('servo_controller')
        

        self.subscription = self.create_subscription(
            Bool,
            'shoot',
            self.listener_callback,
            10)
        
 
        self.servo = Servo(20)
        self.servo.detach()
        self.get_logger().info('Servo initialized and detached.')

    def listener_callback(self, msg):
        if msg.data:
            self.get_logger().info('Received TRUE – Moving servo.')
            self.servo.max()
            sleep(1)
            self.servo.detach()
        else:
            self.get_logger().info('Received FALSE – No action.')

def main(args=None):
    rclpy.init(args=args)
    node = ServoController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

