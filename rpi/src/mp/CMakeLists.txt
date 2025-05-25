cmake_minimum_required(VERSION 3.8)
project(mp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_typesupport_cpp REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}  "msg/ControlInstructions.msg")
ament_export_dependencies(rosidl_typesupport_cpp)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")


add_executable(coordinates_receiver src/coordinatesreceiverv4.cpp)
ament_target_dependencies(coordinates_receiver 
  rclcpp 
  geometry_msgs
  rosidl_typesupport_cpp
)

target_link_libraries(coordinates_receiver "${cpp_typesupport_target}")

install(TARGETS
  coordinates_receiver
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  msg
  DESTINATION share/${PROJECT_NAME}/msg
)



if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
