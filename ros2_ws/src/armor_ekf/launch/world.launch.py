from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share = Path(get_package_share_directory('armor_ekf'))
    return LaunchDescription([
        DeclareLaunchArgument('model_path', description='Absolute path to best.onnx'),
        DeclareLaunchArgument('video_path', description='Absolute path to test.mp4'),
        DeclareLaunchArgument('fps', default_value='30.0'),
        DeclareLaunchArgument('loop', default_value='false'),
        DeclareLaunchArgument('world_config', default_value=str(share / 'config' / 'world.yaml')),
        Node(package='armor_video_publisher', executable='video_publisher', output='screen',
             parameters=[{'video_path': LaunchConfiguration('video_path'),
                          'fps': ParameterValue(LaunchConfiguration('fps'), value_type=float),
                          'loop': ParameterValue(LaunchConfiguration('loop'), value_type=bool)}]),
        Node(package='armor_ekf', executable='armor_ekf_node', output='screen',
             parameters=[str(share / 'config' / 'ekf.yaml'), LaunchConfiguration('world_config'),
                         {'model_path': LaunchConfiguration('model_path')}]),
    ])
