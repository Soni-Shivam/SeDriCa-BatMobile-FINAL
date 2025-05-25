#include <chrono>
#include <deque>
#include <cmath>
#include <vector>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;

// State machine states
enum class State { SCANNING, TURNING, MOVING, AIMING, SHOOTING };

struct ScanSample {
  rclcpp::Time time;
  float range;
};

// Publisher node
class ControlsPublisher : public rclcpp::Node {
public:
  ControlsPublisher() : Node("controls_publisher") {
    motion_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    shoot_pub_  = create_publisher<std_msgs::msg::Bool>("/shoot", 10);
    orient_pub_ = create_publisher<std_msgs::msg::Float32>("/orient", 10);
    RCLCPP_INFO(get_logger(), "ControlsPublisher initialized");
  }

  void setControl(float lin, float ang, float cannon, bool shoot) {
    geometry_msgs::msg::Twist t;
    std_msgs::msg::Bool s;
    std_msgs::msg::Float32 o;
    t.linear.x = lin;
    t.angular.z = ang;
    o.data = cannon;
    s.data = shoot;
    motion_pub_->publish(t);
    orient_pub_->publish(o);
    shoot_pub_->publish(s);
    RCLCPP_DEBUG(get_logger(), "Control cmd: lin=%.2f ang=%.2f cannon=%.2f shoot=%d",
                  lin, ang, cannon, shoot);
  }

private:
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr motion_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr shoot_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr orient_pub_;
};

// Subscriber node with full state machine logic
class CoordinateSubscriber : public rclcpp::Node {
public:
  CoordinateSubscriber(std::shared_ptr<ControlsPublisher> ctrl)
  : Node("coordinate_subscriber"), controls_(ctrl),
    state_(State::SCANNING), lost_debounce_(false), last_dir_(1)
  {
    point_sub_ = create_subscription<geometry_msgs::msg::Point>(
      "get_point", 10, std::bind(&CoordinateSubscriber::pointCallback, this, _1));

    declare_parameter<float>("linear", 60.0);
    declare_parameter<float>("angular", 85.0);
    declare_parameter<float>("threshold", 1.5);
    declare_parameter<float>("epsilonA", 0.08);
    declare_parameter<float>("epsilonB", 0.1);
    declare_parameter<float>("heading_deadband", 0.05);
    declare_parameter<float>("heading_hyst", 0.1);

    RCLCPP_INFO(get_logger(), "CoordinateSubscriber initialized in SCANNING");
  }

private:
  std::shared_ptr<ControlsPublisher> controls_;
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr point_sub_;
  State state_;
  rclcpp::Time state_start_;
  std::deque<ScanSample> history_;
  bool lost_debounce_;
  rclcpp::Time lost_start_;
  geometry_msgs::msg::Point target_;
  int last_dir_;
  rclcpp::TimerBase::SharedPtr timer_;

  float getParam(const std::string &n) {
    return get_parameter(n).get_parameter_value().get<float>();
  }

  void gotoState(State s) {
    state_ = s;
    state_start_ = now();
    history_.clear();
    lost_debounce_ = false;
    RCLCPP_INFO(get_logger(), "Entered state: %s", stateToString(s).c_str());
  }

  std::string stateToString(State s) {
    switch(s) {
      case State::SCANNING: return "SCANNING";
      case State::TURNING:  return "TURNING";
      case State::MOVING:   return "MOVING";
      case State::AIMING:   return "AIMING";
      case State::SHOOTING: return "SHOOTING";
    }
    return "UNKNOWN";
  }

  void pointCallback(const geometry_msgs::msg::Point::SharedPtr msg) {
    target_ = *msg;
    RCLCPP_DEBUG(get_logger(), "Received target z=%.2f", target_.z);

    if (state_ == State::TURNING) {
      history_.push_back({ now(), float(target_.z) });
      while (!history_.empty() && now() - history_.front().time > rclcpp::Duration(30s))
        history_.pop_front();
    }

    switch(state_) {
      case State::SCANNING: handleScanning(); break;
      case State::TURNING:  handleTurning();  break;
      case State::MOVING:   handleMoving();   break;
      case State::AIMING:   handleAiming();   break;
      case State::SHOOTING: handleShooting(); break;
    }
  }

  void handleScanning() {
    float ang = getParam("angular");
    controls_->setControl(0, ang, 0, false);
    if (target_.z != 0) gotoState(State::TURNING);
  }

  void handleTurning() {
    float epsA = getParam("epsilonA");
    float ang  = getParam("angular");
    // 30s timeout: reverse last dir
    if (now() - state_start_ > rclcpp::Duration(30s) && !history_.empty()) {
      int dir = -last_dir_;
      RCLCPP_INFO(get_logger(), "Timeout turning, reversing dir=%d", dir);
      controls_->setControl(0, ang*dir, 0, false);
      last_dir_ = dir;
      return;
    }
    if (target_.z == 0) return;
    // approximate angle via x/z ratio
    float theta = std::atan2(target_.x, target_.z);
    float dead = getParam("heading_deadband");
    float hyst = getParam("heading_hyst");
    if (std::fabs(theta) < dead) { gotoState(State::MOVING); return; }
    if (std::fabs(theta) < hyst) theta = last_dir_*dead;
    int dir = (theta>0?1:-1);
    controls_->setControl(0, ang*dir, 0, false);
    last_dir_ = dir;
    RCLCPP_DEBUG(get_logger(), "Turning dir=%d theta=%.3f", dir, theta);
  }

  void handleMoving() {
    float epsB = getParam("epsilonB");
    float lin  = getParam("linear");
    float thr  = getParam("threshold");
    if (target_.z == 0) {
      if (!lost_debounce_) { lost_debounce_=true; lost_start_=now(); RCLCPP_WARN(get_logger(), "Possible loss"); }
      else if (now()-lost_start_>rclcpp::Duration(2s)) gotoState(State::SCANNING);
      return;
    } else if (lost_debounce_) { lost_debounce_=false; RCLCPP_INFO(get_logger(),"Reacquired"); }
    float theta=std::atan2(target_.x,target_.z);
    float ex=target_.x-thr*std::sin(theta), ez=target_.z-thr*std::cos(theta);
    float d=std::hypot(ex,ez);
    if(d<=epsB) { gotoState(State::AIMING); }
    else { controls_->setControl(lin,0,0,false); RCLCPP_DEBUG(get_logger(),"Moving d=%.2f",d); }
  }

  void handleAiming() {
    RCLCPP_INFO(get_logger(), "Aiming");
    controls_->setControl(0,0,M_PI/4,false);
    timer_ = create_wall_timer(1000ms, [this](){ gotoState(State::SHOOTING); timer_->cancel(); });
  }

  void handleShooting() {
    RCLCPP_INFO(get_logger(), "Shooting");
    controls_->setControl(0,0,M_PI/4,true);
    gotoState(State::SCANNING);
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto controls   = std::make_shared<ControlsPublisher>();
  auto subscriber = std::make_shared<CoordinateSubscriber>(controls);
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(controls);
  exec.add_node(subscriber);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}
