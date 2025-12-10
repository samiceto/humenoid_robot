"""Launch file for the voice command pipeline."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for voice command pipeline."""
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    # Voice command pipeline node
    voice_command_node = Node(
        package='humanoid_robot',
        executable='voice_command_pipeline',
        name='voice_command_pipeline',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen',
        remappings=[
            ('/cmd_vel', '/diff_drive_controller/cmd_vel_unstamped'),
            ('/goal_pose', '/navigate_to_pose/goal'),
            ('/voice_command', '/microphone/audio_input'),
            ('/voice_response', '/speaker/audio_output')
        ]
    )

    # Microphone input simulator (for testing without actual mic)
    mic_simulator = Node(
        package='humanoid_robot',
        executable='microphone_simulator',
        name='microphone_simulator',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Audio output simulator (for testing without actual speaker)
    speaker_simulator = Node(
        package='humanoid_robot',
        executable='speaker_simulator',
        name='speaker_simulator',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time_arg,
        voice_command_node,
        mic_simulator,
        speaker_simulator
    ])