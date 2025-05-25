import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist
from gpiozero import Servo, Motor
import time

class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control')

        self.left_motor = Motor(forward=17, backward=27, pwm=True)
        self.right_motor = Motor(forward=18, backward=12, pwm=True)

        # Servos : switched to pins 32 and 33(gpio 12 and 13) as they have hardware pwm
        #self.p_servo = AngularServo(21, min_angle=-90, max_angle=90)
        #self.p_servo = AngularServo(21, min_angle=-90, max_angle=90)
        self.servo=Servo(21)
        self.servo.detach()


        # Internal state
        self.orient = 90.0
        self.shoot = False

        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.create_subscription(Float32, '/orient', self.orient_callback, 10)
        self.create_subscription(Bool, '/shoot', self.shoot_callback, 10)

        #self.p_servo.value = 0
        #self.c_servo.value = 1
        self.control_motors((0, True), (0, True))
        self.control_motors((0, 0), (0, 0))

        self.get_logger().info("Robot control node initialized")

    def control_motors(self, speed_L, speed_R):
    
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

        base_speed = min(abs(linear), 100)
        turn_factor = min(abs(angular), 100)

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
            #time.sleep(1)
            #self.c_servo.value = 1
            i=0
            while i<1:
                self.servo.max()
                time.sleep(0.045) #old 0.07
                self.servo.detach()
                i+=1
                time.sleep(2)


def main(args=None):
    rclpy.init(args=args)
    node = RobotControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

