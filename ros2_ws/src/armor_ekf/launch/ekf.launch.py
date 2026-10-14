from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    config = str(Path(get_package_share_directory('armor_ekf')) / 'config' / 'ekf.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('model_path', description='Absolute path to best.onnx'),
        DeclareLaunchArgument('video_path', description='Absolute path to test.mp4'),
        DeclareLaunchArgument('fps', default_value='30.0', description='Video playback FPS; 0 uses source FPS'),
        DeclareLaunchArgument('loop', default_value='false'),
        Node(package='armor_ekf', executable='armor_ekf_node', output='screen',
             parameters=[config, {'model_path': LaunchConfiguration('model_path'),
                                  'video_path': LaunchConfiguration('video_path'),
                                  'fps': ParameterValue(LaunchConfiguration('fps'), value_type=float),
                                  'loop': ParameterValue(LaunchConfiguration('loop'), value_type=bool)}]),
    ])
