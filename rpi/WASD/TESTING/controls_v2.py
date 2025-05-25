import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist
from gpiozero import AngularServo, Motor
import time

class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control')

        # Motor Pins (forward, backward)
        self.left_motor = Motor(forward=12, backward=13, pwm=True)
        self.right_motor = Motor(forward=22, backward=23, pwm=True)

        # Using AngularServo instead of Servo for direct angle control
        # Pan servo - for orientation control
        self.p_servo = AngularServo(21, min_angle=-90, max_angle=90)
        # Catapult servo - for shooting mechanism
        self.c_servo = AngularServo(13, min_angle=-90, max_angle=90)

        # Internal state
        self.orient = 0.0  # 0 degrees is center position
        self.shoot = False

        # ROS 2 subscriptions
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.create_subscription(Float32, '/orient', self.orient_callback, 10)
        self.create_subscription(Bool, '/shoot', self.shoot_callback, 10)

        # Initial setup - set servos to neutral positions
        self.p_servo.angle = 0  # Center position
        self.p_servo.detach()  # Detach immediately after initialization
        
        self.c_servo.angle = 90  # Loading position
        time.sleep(0.2)
        self.c_servo.detach()  # Detach immediately after initialization
        
        self.control_motors((0, True), (0, True))

        self.get_logger().info("Robot control node initialized with AngularServo - servos detached to prevent jitter")

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
        # Convert incoming orient value (expected 0-180) to servo angle range (-90 to 90)
        self.orient = msg.data
        angle = self.orient - 90  # Convert from 0-180 range to -90 to 90 range
        
        # Ensure angle is within valid range
        angle = max(min(angle, 90), -90)
        
        self.get_logger().info(f"Setting pan servo to angle: {angle}°")
        self.p_servo.angle = angle
        
        # Detach the servo after setting the position to prevent jitter
        time.sleep(0.3)  # Brief delay to ensure servo reaches position
        self.p_servo.detach()
        self.get_logger().info("Pan servo detached to prevent jitter")

    def shoot_callback(self, msg):
        self.shoot = msg.data
        if self.shoot:
            # Launch sequence
            self.get_logger().info("Shooting mechanism activated")
            self.c_servo.angle = -90  # Launch position
            time.sleep(1)
            self.c_servo.angle = 90   # Reset to loading position
            time.sleep(0.5)  # Brief delay to ensure servo reaches position
            self.c_servo.detach()  # Detach after reset to prevent jitter
            self.get_logger().info("Shooting mechanism reset and servo detached")

    def __del__(self):
        # Clean shutdown - set servos to neutral positions
        if hasattr(self, 'p_servo'):
            self.p_servo.angle = 0
            time.sleep(0.3)
            self.p_servo.detach()
            self.get_logger().info("Pan servo set to neutral and detached during shutdown")
        if hasattr(self, 'c_servo'):
            self.c_servo.angle = 0
            time.sleep(0.3)
            self.c_servo.detach()
            self.get_logger().info("Catapult servo set to neutral and detached during shutdown")

def main(args=None):
    rclpy.init(args=args)
    node = RobotControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure servos are in neutral position and detached before shutdown
        if hasattr(node, 'p_servo'):
            node.p_servo.angle = 0
            time.sleep(0.3)
            node.p_servo.detach()
            print("Pan servo set to neutral and detached during shutdown")
        if hasattr(node, 'c_servo'):
            node.c_servo.angle = 0
            time.sleep(0.3)
            node.c_servo.detach()
            print("Catapult servo set to neutral and detached during shutdown")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()