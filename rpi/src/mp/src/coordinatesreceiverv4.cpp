#include <chrono>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32.hpp"

#include <vector>
#include <cmath>


#include <thread>

using std::placeholders::_1;

// Globals
float default_cannon_angle = 0,theta=0,missing_threshold=10,missing_count=0;
bool target_found = false, moving = false, turning = false, shooting = false, shot_fired = false;
const int CANNON_SETTLE_TIME_MS = 1000;

struct Point {
    float x;
    float y;
    float z;
};

Point target, endpnt, camera = {0.02,0,0.175};

// Helper: calculate endpoint to stop before target
void calcEndpoint(int threshold) {
    float bufferdist = threshold;
    theta = atan2(target.x + camera.x, target.z + camera.z);
    endpnt.x = (target.x + camera.x) - bufferdist * sin(theta);
    endpnt.z = (target.z + camera.z) - bufferdist * cos(theta);
}

// Publisher node
class ControlsPublisher : public rclcpp::Node {
public:
    ControlsPublisher() : Node("controls_publisher") {
        motion_publisher = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        shoot_publisher = this->create_publisher<std_msgs::msg::Bool>("/shoot", 10);
        orient_publisher = this->create_publisher<std_msgs::msg::Float32>("/orient", 10);


        RCLCPP_INFO(this->get_logger(), "ControlsPublisher initialized");
    }

    int setControl(float linear_speed, float angular_speed, float cannon_angle, bool shoot) {
        geometry_msgs::msg::Twist twist_msg;
        std_msgs::msg::Bool shoot_msg;
        std_msgs::msg::Float32 orient_msg;

        

        twist_msg.linear.x = linear_speed;
        twist_msg.angular.z = angular_speed;
        orient_msg.data = cannon_angle;
        shoot_msg.data = shoot;

        motion_publisher->publish(twist_msg);
        shoot_publisher->publish(shoot_msg);
        orient_publisher->publish(orient_msg);

        RCLCPP_INFO(this->get_logger(),
                    "Published: linear=%.2f angular=%.2f cannon=%.2f shoot=%d",
                    linear_speed, angular_speed, cannon_angle, shoot);
        return 0;
    }

private:
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr motion_publisher;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr shoot_publisher;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr orient_publisher;
};

// Subscriber node
class CoordinateSubscriber : public rclcpp::Node {
public:
    CoordinateSubscriber(std::shared_ptr<ControlsPublisher> controls)
        : Node("coordinate_subscriber"), controls_publisher(controls) {
        subscription_ = this->create_subscription<geometry_msgs::msg::Point>(
            "get_point", 10, std::bind(&CoordinateSubscriber::callback, this, _1));
        RCLCPP_INFO(this->get_logger(), "CoordinateSubscriber initialized");


        this->declare_parameter<float>("linear", 45);
        this->declare_parameter<float>("angular", 69);
        this->declare_parameter<float>("threshold",1.5);
        this->declare_parameter<float>("epsilonA",0.08);
        this->declare_parameter<float>("epsilonB",0.1);


    }

private:
    std::shared_ptr<ControlsPublisher> controls_publisher;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr subscription_;

    void callback(const geometry_msgs::msg::Point::SharedPtr msg) {
        target.x = msg->x;
        target.y = msg->y;
        target.z = msg->z;

        float linear_speed = this->get_parameter("linear").get_parameter_value().get<float>();
        float angular_speed = this->get_parameter("angular").get_parameter_value().get<float>();
        float threshold = this->get_parameter("threshold").get_parameter_value().get<float>();
        float epsilonA = this->get_parameter("epsilonA").get_parameter_value().get<float>();
        float epsilonB = this->get_parameter("epsilonB").get_parameter_value().get<float>();
        
        
        

        RCLCPP_INFO(this->get_logger(), "Received: x=%.2f, y=%.2f, z=%.2f", msg->x, msg->y, msg->z);

        if (target.z != 0.0f) {
            missing_count = 0;  // valid detection → reset counter
        } else {
            missing_count++;    // no detection → increment
        }
    
        // 2) State machine
        if (!target_found) {
            // Declare “found” only when we see it
            if (missing_count == 0) {
                target_found = true;
                turning      = true;
                shot_fired   = false;
                RCLCPP_INFO(this->get_logger(), "Target found, start turning.");
            } else {
                // keep scanning
                shooting = false;
                moving   = false;
                turning  = true;
                RCLCPP_INFO(this->get_logger(), "Scanning...");
                controls_publisher->setControl(0.0f, angular_speed, default_cannon_angle, false);
            }
    
        } else if (turning) {
            // Compute angle to target
            float theta = std::atan2(target.x + camera.x, target.z + camera.z);
            if (std::fabs(theta) < epsilonA) {
                turning = false;
                moving  = true;
                RCLCPP_INFO(this->get_logger(), "Heading set, start moving.");
            } else {
                float turn_speed = angular_speed * (theta > 0 ? -1.0f : 1.0f);
                RCLCPP_INFO(this->get_logger(), "Turning (%.2f rad)", theta);
                controls_publisher->setControl(0.0f, turn_speed, default_cannon_angle, false);
            }
    
        } else if (moving) {
            // Declare “lost” only after threshold misses
            if (missing_count >= missing_threshold) {
                moving       = false;
                target_found = false;
                RCLCPP_INFO(this->get_logger(),
                            "Target lost for %f frames. Scanning again.", missing_count);
                // next loop will enter scanning branch
            } else {
                // Continue moving toward last known endpoint
                calcEndpoint(threshold);
                float distance = std::sqrt(endpnt.x*endpnt.x + endpnt.z*endpnt.z);
                if (distance <= epsilonB) {
                    moving   = false;
                    shooting = true;
                    RCLCPP_INFO(this->get_logger(), "Reached endpoint. Preparing to shoot.");
                    controls_publisher->setControl(0.0f, 0.0f, M_PI/4, false);
                    std::this_thread::sleep_for(std::chrono::milliseconds(CANNON_SETTLE_TIME_MS));
                } else {
                    RCLCPP_INFO(this->get_logger(), "Approaching target (%.2f m)", distance);
                    controls_publisher->setControl(linear_speed, 0.0f, default_cannon_angle, false);
                }
            }
    
        } else if (shooting && !shot_fired) {
            RCLCPP_INFO(this->get_logger(), "Shooting!");
            controls_publisher->setControl(0.0f, 0.0f, M_PI/4, true);
            shot_fired   = true;
            target_found = false;
            shooting     = false;
            turning      = false;
            moving       = false;
            RCLCPP_INFO(this->get_logger(), "Shot fired. Back to scanning.");
    
        } else {
            // idle / fallback
            controls_publisher->setControl(0.0f, 0.0f, default_cannon_angle, false);
        }
    }
        
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);

    auto controls_node = std::make_shared<ControlsPublisher>();
    auto coords_node = std::make_shared<CoordinateSubscriber>(controls_node);

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(controls_node);
    executor.add_node(coords_node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
