import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist
from gpiozero import AngularServo, Motor
import time

class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control')

        # Motor Pins (forward, backward) TODO:check if the pin usage is appropriate 
        self.left_motor = Motor(forward=12, backward=13, pwm=True)
        self.right_motor = Motor(forward=22, backward=23, pwm=True)

        # Servos : switched to pins 32 and 33(gpio 12 and 13) as they have hardware pwm
        #self.p_servo = AngularServo(21, min_angle=-90, max_angle=90)
        #self.c_servo = AngularServo(13, min_angle=-90, max_angle=90)

        # Internal state
        self.orient = 90.0
        self.shoot = False

        # ROS 2 subscriptions
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.create_subscription(Float32, '/orient', self.orient_callback, 10)
        self.create_subscription(Bool, '/shoot', self.shoot_callback, 10)

        # Initial setup
        #self.p_servo.value = 0
        #self.c_servo.value = 1
        self.control_motors((0, True), (0, True))

        self.get_logger().info("Robot control node initialized")

    def control_motors(self, speed_L, speed_R):
    
        # Convert 0-100 to -1 to 1 range
        def to_motor_val(speed, direction):
            val = speed / 100.0
            return val if direction else -val
          # speed_L and speed_R are tuples of the form (speed : 0-100, direction:true/false)

        self.left_motor.value = to_motor_val(speed_L[0], speed_L[1])
        self.right_motor.value = to_motor_val(speed_R[0], speed_R[1])

    def cmd_vel_callback(self, msg):
        linear = msg.linear.x
        angular = msg.angular.z

        speed_L = 0
        speed_R = 0
        dir_L = True
        dir_R = True

        base_speed = min(abs(linear) * 100, 100)
        turn_factor = min(abs(angular) * 50, 100)

        if linear > 0:
            dir_L = True
            dir_R = True
            speed_L = base_speed
            speed_R = base_speed
        elif linear < 0:
            dir_L = False
            dir_R = False
            speed_L = base_speed
            speed_R = base_speed

        if angular > 0:
            speed_L = max(speed_L - turn_factor, 0)
            speed_R = min(speed_R + turn_factor, 100)
        elif angular < 0:
            speed_L = min(speed_L + turn_factor, 100)
            speed_R = max(speed_R - turn_factor, 0)

        if linear == 0 and angular != 0:
            dir_L = angular < 0
            dir_R = angular > 0
            speed_L = speed_R = turn_factor

        if linear == 0 and angular == 0:
            speed_L = 0
            speed_R = 0

        self.control_motors((speed_L, dir_L), (speed_R, dir_R))

    def orient_callback(self, msg):
        self.orient = msg.data
        #self.p_servo.value = (self.orient - 90) / 90

    def shoot_callback(self, msg):
        self.shoot = msg.data
        if self.shoot:
            #self.c_servo.value = -1
            time.sleep(1)
            #self.c_servo.value = 1

def main(args=None):
    rclpy.init(args=args)
    node = RobotControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down due to keyboard interrupt...")
    finally:
        # Stop the motors before exiting
        node.control_motors((0, True), (0, True))

        # Explicitly free GPIO resources
        node.left_motor.close()
        node.right_motor.close()
        
        # If using servos, also close them:
        # node.p_servo.close()
        # node.c_servo.close()

        node.get_logger().info("GPIO resources released. Node shutting down.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

