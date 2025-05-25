import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Bridge for cmd_vel (ROS 2 → Ignition)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist'],
            output='screen'
        ),

        # Bridge for odometry (Ignition → ROS 2)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/odom@nav_msgs/msg/Odometry@ignition.msgs.Odometry'],
            output='screen'
        ),

        # Bridge for pose info (Ignition → ROS 2)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/world/Moving_robot/pose/info@geometry_msgs/msg/PoseArray@gz.msgs.Pose_V'],
            output='screen'
        ),
    ])
