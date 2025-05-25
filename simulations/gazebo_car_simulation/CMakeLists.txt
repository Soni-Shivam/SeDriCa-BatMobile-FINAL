cmake_minimum_required(VERSION 3.5)
project(gazebo_car_simulation)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(gazebo_msgs REQUIRED)

# Include the header files
include_directories(include)

# PID Steering Publisher Node
add_executable(pid_steer_publisher
  src/pid_steer_publisherv3.cpp
  src/pid.cpp
  src/trajectory_generator.cpp
)
ament_target_dependencies(pid_steer_publisher
  rclcpp
  geometry_msgs
  nav_msgs
  ament_index_cpp
)
install(TARGETS pid_steer_publisher DESTINATION lib/${PROJECT_NAME})

# Install path.csv
install(FILES path.csv DESTINATION share/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)

# Install models
install(DIRECTORY models DESTINATION share/${PROJECT_NAME}/)

ament_package()
