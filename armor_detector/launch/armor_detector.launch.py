import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 获取包的 share 目录路径
    package_name = 'armor_detector'
    share_dir = get_package_share_directory(package_name)

    # 定义配置文件路径
    # 假设你的配置文件名为 armor_detector.yaml，放在 config 目录下
    config_file = os.path.join(share_dir, 'config', 'armor_detector.yaml')

    # 定义节点
    detector_node = Node(
        package=package_name,
        executable='armor_detector_node',
        name='armor_detector',
        output='both',
        emulate_tty=True,  # 允许在终端显示彩色日志
        parameters=[config_file], # 加载参数文件
        arguments=['--ros-args', '--log-level', 'INFO'], # 设置日志级别
    )

    return LaunchDescription([
        detector_node,
    ])