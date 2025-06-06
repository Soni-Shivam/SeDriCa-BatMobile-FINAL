#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_array.hpp"  // Include PoseArray message
#include "pid.hpp"
#include "trajectory_generator.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include <cmath>
#include <vector>
using namespace std;
// goal to keep car position to 0,0 at all times 
// change the target points dfdynamically as the car moves forward.
class PIDSteerPublisher : public rclcpp::Node {
public:
    PIDSteerPublisher()
        : Node("pid_steer_publisher"),
        pid_(0.35,0,0.3),
        speed_(2.0),
        angular_speed_(2.0),
        dt_(0.01),
        current_waypoint_index_(1) {
        // Load path from CSV
        std::string path_file = ament_index_cpp::get_package_share_directory("gazebo_car_simulation") + "/path.csv";
        path_ = trajectory_generator_.loadPath(path_file);

        if (path_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Path file is empty or not found!");
            return;
        }
        // Initialize car state
        car_position_ = {0,0};
        for (auto waypoint: path_) {
            waypoint[0] -= path_[0][0];
            waypoint[1] -= path_[0][1];
        }
        double initialHeading  = 0;
        heading_ = getTargetHeading(car_position_, transformToCarFrame(path_[current_waypoint_index_], car_position_, initialHeading));

        // Create publisher for /cmd_vel
        velocity_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

        // Subscribe to PoseArray for position and orientation feedback
        pose_array_subscriber_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "/world/Moving_robot/pose/info", 10, std::bind(&PIDSteerPublisher::poseArrayCallback, this, std::placeholders::_1));

        // Timer for control loop
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(dt_ * 1000)),
            std::bind(&PIDSteerPublisher::controlLoop, this));
    }

private:
    PID pid_;
    TrajectoryGenerator trajectory_generator_;
    std::vector<std::vector<double>> path_;
    std::vector<double> car_position_;
    std::vector<double> global_car_pos = {0,0};

    int current_waypoint_index_;
    double speed_;
    double angular_speed_;
    double dt_;
    double heading_, global_current_heading_;
    const double K_cte = 0.5; 

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr velocity_publisher_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr pose_array_subscriber_;  // New subscriber
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

    vector<double> transformToCarFrame(const std::vector<double>& global_point, 
        const std::vector<double>& global_car_position, 
        double car_heading) {
        double dx = global_point[0] - global_car_position[0];
        double dy = global_point[1] - global_car_position[1];

        double cos_theta = cos(-car_heading);
        double sin_theta = sin(-car_heading);

        double local_x = cos_theta * dx - sin_theta * dy;
        double local_y = sin_theta * dx + cos_theta * dy;

        return {local_x, local_y};
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
    double getTargetHeading(const std::vector<double>& carPos, const std::vector<double>& targetPos) {
        double dx = targetPos[0] - carPos[0];
        double dy = targetPos[1] - carPos[1];
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
        return progress > 1.0;
    }

    // PoseArray Callback (updates position and heading)
    void poseArrayCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg) {
        if (msg->poses.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty PoseArray message!");
            return;
        }
        double virtualX = 0, virtualY =0;
        // Assuming the first pose in the array represents the robot's current position
        virtualX = msg->poses[1].position.x;
        virtualY = msg->poses[1].position.y;
        global_car_pos[0] = virtualX;
        global_car_pos[1] = virtualY;
        // Compute the current heading from the first pose's orientation
        global_current_heading_ = atan2(
            2.0 * (msg->poses[1].orientation.w * msg->poses[1].orientation.z +
                   msg->poses[1].orientation.x * msg->poses[1].orientation.y),
            1.0 - 2.0 * (msg->poses[1].orientation.y * msg->poses[1].orientation.y +
                         msg->poses[1].orientation.z * msg->poses[1].orientation.z));
    }

    void controlLoop() {
        // RCLCPP_INFO(this->get_logger(), "Control loop triggered!");
        /* RCLCPP_INFO(this->get_logger(), 
        "Pos: (%.2f, %.2f) | Current Heading: %.2f | Curr Index: %.2d",
        global_car_pos[0], global_car_pos[1], 
        global_current_heading_, current_waypoint_index_);
         */

        if (current_waypoint_index_ >= path_.size()) {
            RCLCPP_WARN(this->get_logger(), "All waypoints reached. Stopping control loop.");
            return;
        }
        

        vector<double> local_wp = transformToCarFrame(path_[current_waypoint_index_], global_car_pos, global_current_heading_);
        vector<double> local_last_wp = transformToCarFrame(path_[current_waypoint_index_-1], global_car_pos, global_current_heading_); 

        
        // Check if we've reached the next waypoint
        if (hasPassedWaypoint(car_position_, local_last_wp, local_wp)) {
            current_waypoint_index_++;
            if (current_waypoint_index_ >= path_.size()) return;
        }
        
        // Compute desired heading to next waypoint
        double desired_heading = getTargetHeading(car_position_, local_wp);
        
        // Compute heading error (difference between desired and current heading)
        double heading_error = desired_heading;
        // Normalize to [-π, π]
        heading_error = atan2(sin(heading_error), cos(heading_error));
        
        // Compute Cross-Track Error
        double cte = getCrossTrackError(local_wp,
                                   local_last_wp,
                                   car_position_);
        int steer_dir = getSteerDirection(local_wp,
                                       local_last_wp,
                                       car_position_);
        cte *= steer_dir;  // Apply direction to CTE
        
        // Use PID to compute steering adjustment based on CTE
        double steering_command = pid_.compute(cte, dt_);
        
        // Set command velocities
        geometry_msgs::msg::Twist cmd;
        cmd.linear.x = speed_;
        
        // Apply steering command to angular velocity with proper limiting
        // Combine with a proportional correction based on heading error
        cmd.angular.z = steering_command + 0.5 * heading_error;  // Add heading correction
        cmd.angular.z = clamp(cmd.angular.z, -angular_speed_, angular_speed_);
        
        velocity_publisher_->publish(cmd);
        
        // Debugging Output
        RCLCPP_INFO(this->get_logger(), 
                    "Target: (%.2f,%.2f) | Pos: (%.2f, %.2f) | Current Heading: %.2f | "
                    "Desired Heading: %.2f | Heading Error: %.2f | CTE: %.2f | Command: %.2f",
                    local_wp[0], local_wp[1],
                    car_position_[0], car_position_[1], 
                    global_current_heading_, desired_heading, heading_error, 
                    cte, cmd.angular.z);
    }
};

// Main Function
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PIDSteerPublisher>());
    rclcpp::shutdown();
    return 0;
}
