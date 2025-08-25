from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node, SetRemap
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get directories
    bringup_dir = FindPackageShare('navigation')
    nav2_dir = FindPackageShare('nav2_bringup')
    
    # Parameter files
    nav2_params_file = PathJoinSubstitution([
        bringup_dir, 'config', 'unity_params.yaml'
    ])
    
    map_yaml_file = LaunchConfiguration('map')

    declare_map_yaml_cmd = DeclareLaunchArgument(
        'map',
        default_value=PathJoinSubstitution([bringup_dir, 'maps', 'empty.yaml']),
                    description='Full path to map yaml file'
    )

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    # Declare arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    # Create Nav2 group with proper remappings
    nav2_group = GroupAction(
        actions=[
            # Apply remappings to all nodes in this group
            SetRemap(src='/cmd_vel', dst='/cmd_vel_nav2'),
            SetRemap(src='/map', dst='/rtabmap/map'),
            
            # Include Nav2 launch with parameters
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([nav2_dir, 'launch', 'bringup_launch.py'])
                ),
                launch_arguments={
                    'map': '',
                    'params_file': nav2_params_file,
                    'use_sim_time': use_sim_time,
                    'autostart': 'true',
                    'log_level': 'info'
                }.items()
            )
        ]
    )
    
    # Twist converter node
    twist_converter_node = Node(
        package='navigation',
        executable='twist_convert_node.py',
        name='twist_converter',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # boundary_filter_include = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([
    #         os.path.join(get_package_share_directory('rtabmap'), 
    #                     'launch', 'boundary_filter_launch.py')
    #     ]),
    #     launch_arguments={
    #         'boundary_topic': '/boundary_marker',
    #         'use_sim_time': 'true',
    #         'autostart': 'true'
    #     }.items()
    # )
    
    # Return launch description
    return LaunchDescription([
        declare_use_sim_time_cmd,
        # declare_map_yaml_cmd,
        nav2_group,
        twist_converter_node,
        # boundary_filter_include,
    ])
