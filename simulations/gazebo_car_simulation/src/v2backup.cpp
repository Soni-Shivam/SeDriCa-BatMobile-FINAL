#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "tf2_msgs/msg/tf_message.hpp"
#include "pid.hpp"
#include "trajectory_generator.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include <cmath>
#include <vector>

class PIDSteerPublisher : public rclcpp::Node {
public:
    PIDSteerPublisher()
        : Node("pid_steer_publisher"),
        pid_(0.0, 0.0, 0),  // Set reasonable PID gains
        speed(3.0),
        angular_speed(4.0),
        dt(0.01),
        is_turning(false),
        current_waypoint_index_(1) {

        // Load path from CSV
        std::string path_file = ament_index_cpp::get_package_share_directory("mycarsim") + "/path.csv";
        path_ = trajectory_generator_.loadPath(path_file);

        if (path_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Path file is empty or not found!");
            return;
        }

        // Initialize car state
        car_position = path_[0];
        // Don't set heading here as it will be updated in the control loop

        // Create publisher for /cmd_vel
        velocity_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

        // Subscribe to TF messages from Gazebo
        tf_subscriber_ = this->create_subscription<tf2_msgs::msg::TFMessage>(
            "/model/vehicle_blue/tf", 10, std::bind(&PIDSteerPublisher::tfCallback, this, std::placeholders::_1));

        // Timer for control loop
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(dt * 1000)),
            std::bind(&PIDSteerPublisher::controlLoop, this));
    }

private:
    PID pid_;
    TrajectoryGenerator trajectory_generator_;
    std::vector<std::vector<double>> path_;
    std::vector<double> car_position;
    int current_waypoint_index_;
    double speed;
    double angular_speed;
    double dt;
    bool is_turning;  // Flag to track if we're in turning mode
    double heading, current_heading;

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr velocity_publisher_;
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_subscriber_;
    rclcpp::TimerBase::SharedPtr timer_;

    // Utility Functions
    std::vector<double> subtractVectors(const std::vector<double>& a, const std::vector<double>& b) {
        std::vector<double> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] - b[i];
        }
        return result;
    }

    double dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
        double result = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    double cross2D(const std::vector<double>& v1, const std::vector<double>& v2) {
        return v1[0] * v2[1] - v1[1] * v2[0];
    }

    std::vector<double> scaleVector(double scalar, const std::vector<double>& v) {
        std::vector<double> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = scalar * v[i];
        }
        return result;
    }

    double clamp(double value, double min, double max) {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

    // Cross Track Error Calculation
    double getCrossTrackError(const std::vector<double>& currentWaypoint,
                              const std::vector<double>& previousWaypoint,
                              const std::vector<double>& car) {
        std::vector<double> v1 = subtractVectors(currentWaypoint, previousWaypoint);
        std::vector<double> v2 = subtractVectors(car, previousWaypoint);
        std::vector<double> progress = scaleVector(dotProduct(v1, v2) / dotProduct(v1, v1), v1);
        std::vector<double> cteVector = subtractVectors(progress, v2);
        return sqrt(dotProduct(cteVector, cteVector));
    }

    // Get Target Heading
    /* double getTargetHeading(const std::vector<double>& carPos, const std::vector<double>& targetPos) {
        double dx = targetPos[0] - carPos[0];
        double dy = targetPos[1] - carPos[1];
        return atan2(dy, dx);
    } */
   double getTargetHeading(int nextIndex) {
        double dx = path_[nextIndex][0] - path_[nextIndex - 1][0];
        double dy = path_[nextIndex][1] - path_[nextIndex - 1][1];
        return atan2(dy, dx);
    }

    // Get Steering Direction
    int getSteerDirection(const std::vector<double>& currentWaypoint,
                          const std::vector<double>& previousWaypoint,
                          const std::vector<double>& car) {
        std::vector<double> v1 = subtractVectors(currentWaypoint, previousWaypoint);
        std::vector<double> v2 = subtractVectors(car, previousWaypoint);
        double crossProduct = cross2D(v1, v2);
        return (crossProduct > 0) ? -1 : 1;
    }

    // Check if Car has Passed a Waypoint
    bool hasPassedWaypoint(const std::vector<double>& carPos,
                           const std::vector<double>& fromPos,
                           const std::vector<double>& toPos) {
        std::vector<double> a = subtractVectors(carPos, fromPos);
        std::vector<double> b = subtractVectors(toPos, fromPos);
        double progress = dotProduct(a, b) / dotProduct(b, b);
        return progress >= 1.0;
    }

    // TF Callback to update position and heading
    void tfCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg) {
        // Find the transform from world to vehicle_blue
        for (const auto& transform : msg->transforms) {
            if (transform.child_frame_id == "vehicle_blue/chassis") {
                // Update position
                car_position[0] = transform.transform.translation.x;
                car_position[1] = transform.transform.translation.y;

                // Extract quaternion
                double qx = transform.transform.rotation.x;
                double qy = transform.transform.rotation.y;
                double qz = transform.transform.rotation.z;
                double qw = transform.transform.rotation.w;

                // Convert quaternion to yaw
                current_heading = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
                // Debug output
                /* RCLCPP_INFO(this->get_logger(), 
                    "TF Data - Position: (%.2f, %.2f), Heading: %.2f",
                    car_position[0], car_position[1], current_heading); */
                break;
            }
        }
    }

    void controlLoop() {
        if (current_waypoint_index_ >= path_.size()) return;
    
        // Check if we've reached the next waypoint
        if (hasPassedWaypoint(car_position, path_[current_waypoint_index_ - 1], path_[current_waypoint_index_])) {
            current_waypoint_index_++;
            if (current_waypoint_index_ >= path_.size()) return;
                        }
        
        // Compute desired heading to next waypoint
        double next_desired_heading = getTargetHeading(current_waypoint_index_+1);
        
        // Compute heading error (difference between next and current heading)
        double changeOfAngle = next_desired_heading - current_heading;
        // Normalize to [-π, π]
        changeOfAngle = atan2(sin(changeOfAngle), cos(changeOfAngle));
        
        // Compute Cross-Track Error
        double cte = getCrossTrackError(path_[current_waypoint_index_],
                                   path_[current_waypoint_index_ - 1],
                                   car_position);
        int steer_dir = getSteerDirection(path_[current_waypoint_index_],
                                       path_[current_waypoint_index_ - 1],
                                       car_position);
        cte *= steer_dir;  // Apply direction to CTE
        
        // Use PID to compute steering adjustment based on CTE
        double pid_control_command = pid_.compute(cte, dt);
        
        // Set command velocities
        geometry_msgs::msg::Twist cmd;
        cmd.linear.x = speed;
        
        std::vector<double> a = subtractVectors(car_position, path_[current_waypoint_index_ - 1]);
        std::vector<double> b = subtractVectors(path_[current_waypoint_index_], path_[current_waypoint_index_ - 1]);
        double progress = dotProduct(a, b) / dotProduct(b, b);
        double required_progress = 1 - (speed*sin(changeOfAngle)/angular_speed)/sqrt(dotProduct(b, b));
        
        // Start turning if we reach required calculated threshold
        if (progress > required_progress && !is_turning) {
            is_turning = true;
            RCLCPP_INFO(this->get_logger(), "Starting turn - Progress: %.2f", progress);
        }
        
        // Apply steering command to angular velocity with proper limiting
        if (is_turning) {
            // Keep turning until we reach the target heading (within a small threshold)
            if (std::abs(changeOfAngle) > 0.01) {
                cmd.angular.z = clamp(changeOfAngle, -angular_speed, angular_speed);
            } else {
                // We've reached the target heading, switch back to PID control
                is_turning = false;
                cmd.angular.z = -pid_control_command;
                RCLCPP_INFO(this->get_logger(), "Finished turn - Switching to PID");
            }
        } else {
            // Use PID control for path following
            cmd.angular.z = pid_control_command;
        }
        
        cmd.angular.z = clamp(cmd.angular.z, -angular_speed, angular_speed);
        
        velocity_publisher_->publish(cmd);
        
        // Debugging Output
        RCLCPP_INFO(this->get_logger(), 
                    "Target: (%.2f,%.2f) | Pos: (%.2f, %.2f) | Current Heading: %.2f | "
                    "Next Heading: %.2f | Change Required: %.2f | CTE: %.2f | "
                    "Progress: %.2f | Required Progress: %.2f | Turning: %d | Command: %.2f",
                    path_[current_waypoint_index_][0], path_[current_waypoint_index_][1],
                    car_position[0], car_position[1], 
                    current_heading, next_desired_heading, changeOfAngle, 
                    cte, progress, required_progress, is_turning, cmd.angular.z);
    }
};

// Main Function
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PIDSteerPublisher>());
    rclcpp::shutdown();
    return 0;
}
