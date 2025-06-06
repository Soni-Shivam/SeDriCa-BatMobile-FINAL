#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "pid.hpp"
#include "trajectory_generator.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include <cmath>
#include <vector>

class PIDSteerPublisher : public rclcpp::Node {
public:
    PIDSteerPublisher()
        : Node("pid_steer_publisher"),
          pid_(0.8,0.5,1), // Initial PID values (Kp, Ki, Kd)
          speed_(5.0),
          angular_speed_(1.5),
          dt_(0.01),
          current_waypoint_index_(1) {

        // Load path from CSV
        std::string path_file = ament_index_cpp::get_package_share_directory("mycarsim") + "/path.csv";
        path_ = trajectory_generator_.loadPath(path_file);

        if (path_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Path file is empty or not found!");
            return;
        }

        // Initialize car state
        car_position_ = path_[0];
        heading_ = getTargetHeading(car_position_, path_[current_waypoint_index_]);

        // Create publisher for /cmd_vel
        velocity_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

        // Subscribe to odometry for real position feedback
        odom_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&PIDSteerPublisher::odomCallback, this, std::placeholders::_1));


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
    int current_waypoint_index_;
    double speed_;
    double angular_speed_;
    double dt_;
    double heading_, current_heading_;

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr velocity_publisher_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;
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


    // Odometry Callback (updates position and heading)
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        car_position_[0] = msg->pose.pose.position.x;
        car_position_[1] = msg->pose.pose.position.y;

        current_heading_ = atan2(
            2.0 * (msg->pose.pose.orientation.w * msg->pose.pose.orientation.z +
                   msg->pose.pose.orientation.x * msg->pose.pose.orientation.y),
            1.0 - 2.0 * (msg->pose.pose.orientation.y * msg->pose.pose.orientation.y +
                         msg->pose.pose.orientation.z * msg->pose.pose.orientation.z));
   
    }

    // Control Loop
    void controlLoop() {
        if (current_waypoint_index_ >= path_.size()) return;

        // Check if we've reached the next waypoint
        if (hasPassedWaypoint(car_position_, path_[current_waypoint_index_ - 1], path_[current_waypoint_index_])) {
            current_waypoint_index_++;
            if (current_waypoint_index_ >= path_.size()) return;
            heading_ = getTargetHeading(path_[current_waypoint_index_]);
        }

        // Compute Cross-Track Error
        double error = getCrossTrackError(path_[current_waypoint_index_],
                                          path_[current_waypoint_index_ - 1],
                                          car_position_);
        error *= getSteerDirection(path_[current_waypoint_index_],
                                   path_[current_waypoint_index_ - 1],
                                   car_position_);

        // Compute Steering Correction using PID
        double steer = pid_.compute(error, dt_);
        double delta_heading = clamp(steer, -angular_speed_ * dt_, angular_speed_ * dt_);
        heading_ += delta_heading;
        
       /*  // Update Position Based on New Heading
        car_position_[0] += speed_ * cos(heading_) * dt_;
        car_position_[1] += speed_ * sin(heading_) * dt_; */

        // Publish Velocity to /cmd_vel
        geometry_msgs::msg::Twist cmd;
        cmd.linear.x = speed_;
        double heading_error = heading_ - current_heading_;
        heading_error = atan2(sin(heading_error), cos(heading_error));
        if (abs(heading_error)>0.05) {
            cmd.angular.z = (getSteerDirection(path_[current_waypoint_index_],
                path_[current_waypoint_index_ - 1],
                car_position_) > 0) ? 1.0 : -1.0;
        }
        else {
            cmd.angular.z = 0.0;
        }
//        cmd.angular.z = delta_heading / dt_; // Angular velocity = heading change rate
        velocity_publisher_->publish(cmd);

        // Debugging Output
        RCLCPP_INFO(this->get_logger(), "| target:( (%.2f,%.2f) | Pos: (%.2f, %.2f) | currHeading: %.2f | Heading: %.2f | Steer: %.2f| CTE: %.2f",
                    path_[current_waypoint_index_][0], path_[current_waypoint_index_][1],
                    car_position_[0], car_position_[1], current_heading_, heading_, steer, error);
    }
};

// Main Function
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PIDSteerPublisher>());
    rclcpp::shutdown();
    return 0;
}