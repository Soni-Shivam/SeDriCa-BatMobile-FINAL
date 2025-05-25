#include <chrono>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "mp/msg/control_instructions.hpp"
#include <vector>
#include <cmath>
#include <thread>

using std::placeholders::_1;
float threshold = 0.6, theta = 0, epsilonA = 0.1, epsilonB = 0.1, linear_speed = 1, angular_speed = 0.5, default_cannon_angle = 0;
bool target_found = false, moving = false, turning = false, shooting = false, shot_fired = false;
const int CANNON_SETTLE_TIME_MS = 1000; // 1 second wait for cannon to settle

struct Point{
    float x;
    float y;
    float z;
};

Point target, endpnt, camera, Controls_msg; 

void calcEndpoint(){
    float bufferdist = threshold; //60 cm
    theta = atan2(target.x + camera.x, target.z + camera.z); // Using atan2 for better angle calculation
    endpnt.x = (target.x + camera.x) - bufferdist * sin(theta);
    endpnt.z = (target.z + camera.z) - bufferdist * cos(theta);
}

// Define ControlsPublisher first
class ControlsPublisher : public rclcpp::Node {
public:
    ControlsPublisher() : Node("control_publisher") {
        control_publisher = this->create_publisher<mp::msg::ControlInstructions>(
            "Control_instruction", 10);
    }

    static int setControl(float linear_speed, float angular_speed, float cannon_angle, bool shoot) {
        auto control_msg = mp::msg::ControlInstructions();
        control_msg.linear_speed = linear_speed;
        control_msg.angular_speed = angular_speed;
        control_msg.cannon_angle = cannon_angle;
        control_msg.shoot_cannon = shoot;
        
        RCLCPP_INFO(rclcpp::get_logger("control_publisher"), 
            "Publishing: linear=%.2f angular=%.2f cannon_angle=%.2f shoot=%d",
            linear_speed, angular_speed, cannon_angle, shoot);
            
        control_publisher->publish(control_msg);
        return 0;
    }

private:
    static rclcpp::Publisher<mp::msg::ControlInstructions>::SharedPtr control_publisher;
};

// Initialize static member
rclcpp::Publisher<mp::msg::ControlInstructions>::SharedPtr ControlsPublisher::control_publisher;

// Then define CoordinateSubscriber
class CoordinateSubscriber : public rclcpp::Node {
public:
    CoordinateSubscriber() : Node("coordinate_subscriber") {
        subscription_ = this->create_subscription<geometry_msgs::msg::Point>(
            "get_point", 10, std::bind(&CoordinateSubscriber::callback, this, _1));
    }

private:
    void callback(const geometry_msgs::msg::Point::SharedPtr msg) const {
        RCLCPP_INFO(this->get_logger(), "Received coordinates: x=%.2f, y=%.2f, z=%.2f",
            msg->x, msg->y, msg->z);
        target.x = msg->x;
        target.y = msg->y;
        target.z = msg->z;
        if (!target_found) {
            // Scanning phase - rotate right until target is found
            if (target.z != 0) {
                target_found = true;
                turning = true;
                shot_fired = false;  // <--- Add this line
                RCLCPP_INFO(this->get_logger(), "Target found! Starting turning phase.");
            } else {
                shooting = false;
                moving = false;
                turning = true;
                RCLCPP_INFO(this->get_logger(), "Scanning for target...");
                ControlsPublisher::setControl(0, angular_speed, default_cannon_angle, false);
            }
        }
        else if (turning) {
            // Calculate required angle to face target
            float thetha = atan2(target.x + camera.x, target.z + camera.z);
            // Check if we've reached the desired heading
            if (abs(thetha) < epsilonA) {
                turning = false;
                moving = true;
                RCLCPP_INFO(this->get_logger(), "Reached desired heading, starting approach.");
            } else {
                // Turn in the direction of the target
                float turn_speed = angular_speed * (thetha > 0 ? 1 : -1);
                RCLCPP_INFO(this->get_logger(), "Turning to face target (angle: %.2f)", thetha);
                ControlsPublisher::setControl(0, turn_speed, default_cannon_angle, false);
            }
        }
        else if (moving) {
            if (target.z == 0){
                moving = false;
                target_found = false;
                RCLCPP_INFO(this->get_logger(), "Target lost! Starting scanning phase.");
            }
            calcEndpoint();
            // Check if we've reached the endpoint
            float distance_to_endpoint = sqrt(pow(endpnt.x, 2) + pow(endpnt.z, 2));
            
            if (distance_to_endpoint <= epsilonB) {
                moving = false;
                shooting = true;
                RCLCPP_INFO(this->get_logger(), "Reached endpoint, preparing to shoot.");
                // First set the cannon angle and wait for it to settle
                ControlsPublisher::setControl(0, 0, M_PI/4, false);
                std::this_thread::sleep_for(std::chrono::milliseconds(CANNON_SETTLE_TIME_MS));
                RCLCPP_INFO(this->get_logger(), "Cannon angle set, waiting for settle time.");
            } else {
                // Move towards endpoint
                float move_speed = linear_speed * (distance_to_endpoint > 0 ? 1 : -1);
                RCLCPP_INFO(this->get_logger(), "Moving towards endpoint (distance: %.2f)", distance_to_endpoint);
                ControlsPublisher::setControl(move_speed, 0, default_cannon_angle, false);
            }
        } else if (shooting) {
            if (!shot_fired) {
                RCLCPP_INFO(this->get_logger(), "Shooting!");
                ControlsPublisher::setControl(0, 0, M_PI/4, true);
                shot_fired = true;
        
                // Optional: go back to scanning after shooting
                target_found = false;
                shooting = false;
                turning = false;
                moving = false;
                RCLCPP_INFO(this->get_logger(), "Shot fired. Returning to scanning phase.");
            }
        }
        else {
            // Default state
            ControlsPublisher::setControl(0, 0, default_cannon_angle, false);
        }        
    }
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr subscription_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    
    // Create both nodes
    auto controls_publisher = std::make_shared<ControlsPublisher>();
    auto coordinate_subscriber = std::make_shared<CoordinateSubscriber>();
    
    // Create the executor and add both nodes
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(controls_publisher);
    executor.add_node(coordinate_subscriber);
    
    // Spin
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}
