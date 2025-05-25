import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path to the SDF file
    model_file = os.path.join(
        get_package_share_directory('gazebo_car_simulation'),
        'models',
        'thecarandworld.sdf'
    )

    # Start Gazebo Ignition Fortress with the model file
    gazebo = ExecuteProcess(
        cmd=['ign', 'gazebo', model_file],
        output='screen'
    )

    # Start the velocity publisher node
    pid_pub = Node(
        package='gazebo_car_simulation',
        executable='pid_steer_publisher',
        name='pid_steer_publisher',
        output='screen'
    )

    # Path to the bridge launch file
    bridge_launch_file = os.path.join(
        get_package_share_directory('gazebo_car_simulation'),
        'launch',
        'bridge.launch.py'
    )

    # Include the bridge launch file
    bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(bridge_launch_file)
    )

    return LaunchDescription([
        gazebo,
        bridge,
        pid_pub,
    ])
