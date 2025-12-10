# ROS 2 Navigation with Nav2: Student Exercises

## Introduction

Navigation is a fundamental capability for mobile robots, enabling them to autonomously move from one location to another while avoiding obstacles. The Navigation2 (Nav2) stack in ROS 2 provides a comprehensive framework for implementing robot navigation, including path planning, obstacle avoidance, and localization. This tutorial will guide students through the implementation of navigation capabilities using Nav2, with practical exercises to reinforce learning.

The Nav2 stack represents the latest generation of navigation tools for ROS 2, building upon the lessons learned from the original ROS navigation stack. It provides improved performance, better maintainability, and enhanced capabilities for modern robotic applications. Understanding Nav2 is essential for students who wish to develop autonomous mobile robots capable of operating in complex environments.

## Learning Objectives

By completing this tutorial, students will be able to:
1. Understand the architecture and components of the Nav2 stack
2. Configure and launch a basic navigation system
3. Implement autonomous navigation to specified goals
4. Handle navigation recovery behaviors
5. Create and use custom costmaps for navigation
6. Debug common navigation issues

## Prerequisites

Before starting this tutorial, students should have:
- Basic understanding of ROS 2 concepts (nodes, topics, services, actions)
- Experience with ROS 2 launch files and parameters
- Basic knowledge of coordinate frames and transformations (tf2)
- Understanding of robot perception (LIDAR, cameras, etc.)
- Experience with basic robot control and simulation

## Nav2 Architecture Overview

The Nav2 stack is composed of several key components that work together to provide navigation capabilities:

### Core Components

1. **Navigation Server**: The main node that coordinates navigation tasks
2. **Planners Server**: Handles global and local path planning
3. **Controller Server**: Manages trajectory execution and control
4. **Recovery Server**: Handles recovery behaviors when navigation fails
5. **BT Navigator**: Behavior tree-based navigation executor
6. **Lifecycle Manager**: Manages the lifecycle of navigation components

### Navigation Process

The navigation process follows these steps:
1. A navigation goal is sent to the navigation server
2. The global planner computes a path from the current position to the goal
3. The local planner continuously updates the robot's trajectory to follow the global path while avoiding obstacles
4. The controller executes the trajectory by sending commands to the robot's base controller
5. If navigation fails, recovery behaviors are executed

## Exercise 1: Setting Up a Basic Navigation System

### Step 1: Create a Navigation Package

First, create a new ROS 2 package for navigation exercises:

```bash
cd ~/humenoid_robot_ws/src
ros2 pkg create --build-type ament_python nav2_exercises
cd nav2_exercises
```

### Step 2: Create Basic Navigation Configuration

Create a `config` directory and add basic navigation parameters:

```bash
mkdir config
```

Create `config/nav2_params.yaml`:

```yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.2
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
    scan_topic: scan

amcl_map_client:
  ros__parameters:
    use_sim_time: True

amcl_rclcpp_node:
  ros__parameters:
    use_sim_time: True

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    navigate_through_poses: False
    navigate_to_pose: True
    behavior_tree_xml_filename: "navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_compute_path_through_poses_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_assisted_teleop_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_drive_on_heading_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_globally_updated_goal_condition_bt_node
    - nav2_is_path_valid_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_truncate_path_local_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node
    - nav2_controller_cancel_bt_node
    - nav2_path_longer_on_approach_bt_node
    - nav2_wait_cancel_bt_node
    - nav2_spin_cancel_bt_node
    - nav2_back_up_cancel_bt_node
    - nav2_assisted_teleop_cancel_bt_node
    - nav2_drive_on_heading_cancel_bt_node

bt_navigator_rclcpp_node:
  ros__parameters:
    use_sim_time: True

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    failure_tolerance: 0.3
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Progress checker parameters
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    # Goal checker parameters
    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: True

    # Controller parameters
    FollowPath:
      plugin: "nav2_rotation_shim_controller::RotationShimController"
      progress_checker_plugin: "progress_checker"
      goal_checker_plugin: "goal_checker"
      primary_controller: "FollowPath"
      rotation_shim:
        plugin: "nav2_controller::SimpleProgressChecker"
        desired_linear_vel: 0.5
        wait_for_offset: 1.0
        offset_tolerance: 0.1

      FollowPath:
        plugin: "nav2_mppi_controller::MPPIController"
        time_steps: 50
        model_dt: 0.05
        batch_size: 1000
        vx_std: 0.2
        vy_std: 0.1
        wz_std: 0.2
        vx_max: 0.5
        vx_min: -0.3
        vy_max: 0.3
        wz_max: 0.3
        simulation_time: 2.0
        speed_limit_scale: 0.9
        model_type: "nav2_mppi_controller::OmnibaseModel"
        trajectory_visualization_plugin: "nav2_mppi_controller::PathTrajectoryVisualizer"
        reference_track_publisher_plugin: "nav2_mppi_controller::PathReferenceTrackPublisher"
        temperature: 0.3
        gamma: 0.015
        motion_model: "omni"
        aux_variance: 0.05

controller_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      robot_radius: 0.22
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: False
        origin_z: 0.0
        z_resolution: 0.05
        z_voxels: 16
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      always_send_full_costmap: True
  local_costmap_client:
    ros__parameters:
      use_sim_time: True
  local_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: True
      robot_radius: 0.22
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      always_send_full_costmap: True
  global_costmap_client:
    ros__parameters:
      use_sim_time: True
  global_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

map_server:
  ros__parameters:
    use_sim_time: True
    yaml_filename: "turtlebot3_world.yaml"

map_saver:
  ros__parameters:
    use_sim_time: True
    save_map_timeout: 5.0
    free_thresh_default: 0.25
    occupied_thresh_default: 0.65
    map_subscribe_transient_local: True

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true

planner_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

smoother_server:
  ros__parameters:
    use_sim_time: True
    smoother_plugins: ["simple_smoother"]
    simple_smoother:
      plugin: "nav2_smoother::SimpleSmoother"
      tolerance: 1.0e-10
      max_its: 1000
      do_refinement: True

behavior_server:
  ros__parameters:
    use_sim_time: True
    local_costmap_topic: local_costmap/costmap_raw
    global_costmap_topic: global_costmap/costmap_raw
    local_footprint_topic: local_costmap/published_footprint
    global_footprint_topic: global_costmap/published_footprint
    cycle_frequency: 10.0
    behavior_plugins: ["spin", "backup", "wait", "assisted_teleop", "drive_on_heading"]
    spin:
      plugin: "nav2_behaviors::Spin"
      spin_dist: 1.57
    backup:
      plugin: "nav2_behaviors::BackUp"
      backup_dist: 0.15
      backup_speed: 0.025
    wait:
      plugin: "nav2_behaviors::Wait"
      wait_duration: 1.0
    assisted_teleop:
      plugin: "nav2_behaviors::AssistedTeleop"
      enabled_rate: 20.0
      rotation_scaling_enabled: True
      min_rotation_rate: 0.0
      max_rotation_rate: 1.5
    drive_on_heading:
      plugin: "nav2_behaviors::DriveOnHeading"
      enabled_rate: 20.0
      plugin_rate: 20.0
      max_approach_linear_velocity: 0.5
      min_approach_linear_velocity: 0.1
      approach_collision_arc_length: 1.4
      approach_collision_threshold: 0.5

robot_state_publisher:
  ros__parameters:
    use_sim_time: True

waypoint_follower:
  ros__parameters:
    use_sim_time: True
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: True
      waypoint_pause_duration: 200
```

### Step 3: Create a Navigation Launch File

Create `launch/basic_navigation.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import ReplaceString


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    use_composition = LaunchConfiguration('use_composition')
    container_name = LaunchConfiguration('container_name')

    # Launch configuration variables specific to simulation
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')

    # Map fully qualified names to relative ones so the node's namespace can be prepended.
    # In case of the transforms (tf), currently, there doesn't seem to be a better alternative
    # https://github.com/ros/geometry2/issues/32
    # https://github.com/ros/robot_state_publisher/pull/30
    remappings = [('/tf', 'tf'),
                  ('/tf_static', 'tf_static')]

    # Create the node
    navigation_node = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        use_sim_time=use_sim_time,
        parameters=[params_file],
        remappings=remappings)

    controller_server_node = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        use_sim_time=use_sim_time,
        parameters=[params_file],
        remappings=remappings)

    planner_server_node = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        use_sim_time=use_sim_time,
        parameters=[params_file],
        remappings=remappings)

    recoveries_server_node = Node(
        package='nav2_recoveries',
        executable='recoveries_server',
        name='recoveries_server',
        output='screen',
        use_sim_time=use_sim_time,
        parameters=[params_file],
        remappings=remappings)

    bt_navigator_smoothing_node = Node(
        package='nav2_smoothing',
        executable='smoother_server',
        name='smoother_server',
        output='screen',
        use_sim_time=use_sim_time,
        parameters=[params_file],
        remappings=remappings)

    lifecycle_nodes = ['controller_server',
                       'planner_server',
                       'recoveries_server',
                       'bt_navigator',
                       'smoother_server']

    # Create lifecycle manager node
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {'autostart': autostart},
                    {'node_names': lifecycle_nodes}])

    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (Gazebo) clock if true'))

    ld.add_action(DeclareLaunchArgument(
        'autostart', default_value='true',
        description='Automatically startup the nav2 stack'))

    ld.add_action(DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            FindPackageShare('nav2_exercises').find('nav2_exercises'),
            'config', 'nav2_params.yaml'),
        description='Full path to the ROS2 parameters file to use'))

    ld.add_action(DeclareLaunchArgument(
        'use_composition', default_value='False',
        description='Use composed bringup if True'))

    ld.add_action(DeclareLaunchArgument(
        'container_name', default_value='nav2_container',
        description='the name of conatiner that nodes will load in if use composition'))

    # Add nodes to the launch description
    ld.add_action(navigation_node)
    ld.add_action(controller_server_node)
    ld.add_action(planner_server_node)
    ld.add_action(recoveries_server_node)
    ld.add_action(bt_navigator_smoothing_node)
    ld.add_action(lifecycle_manager)

    return ld
```

## Exercise 2: Creating a Simple Navigation Script

Create `scripts/simple_navigation.py`:

```python
#!/usr/bin/env python3

"""
Simple navigation script for Nav2
This script demonstrates how to send navigation goals to Nav2
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Header


class SimpleNavigator(Node):
    def __init__(self):
        super().__init__('simple_navigator')

        # Create action client for navigation
        self.nav_to_pose_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        # Publisher for initial pose (for AMCL)
        self.initial_pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            'initialpose',
            10
        )

        # Wait for action server
        self.nav_to_pose_client.wait_for_server()
        self.get_logger().info('Navigation server ready')

    def set_initial_pose(self, x, y, theta):
        """Set the initial pose of the robot"""
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header = Header()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.get_clock().now().to_msg()

        # Set position
        initial_pose.pose.pose.position.x = x
        initial_pose.pose.pose.position.y = y
        initial_pose.pose.pose.position.z = 0.0

        # Convert theta (in radians) to quaternion
        import math
        from tf_transformations import quaternion_from_euler
        quat = quaternion_from_euler(0, 0, theta)
        initial_pose.pose.pose.orientation.x = quat[0]
        initial_pose.pose.pose.orientation.y = quat[1]
        initial_pose.pose.pose.orientation.z = quat[2]
        initial_pose.pose.pose.orientation.w = quat[3]

        # Set covariance (low uncertainty)
        initial_pose.pose.covariance = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.1]

        self.initial_pose_publisher.publish(initial_pose)
        self.get_logger().info(f'Set initial pose to ({x}, {y}, {theta})')

    def navigate_to_pose(self, x, y, theta):
        """Send navigation goal to Nav2"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header = Header()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Set goal position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # Convert theta (in radians) to quaternion
        import math
        from tf_transformations import quaternion_from_euler
        quat = quaternion_from_euler(0, 0, theta)
        goal_msg.pose.pose.orientation.x = quat[0]
        goal_msg.pose.pose.orientation.y = quat[1]
        goal_msg.pose.pose.orientation.z = quat[2]
        goal_msg.pose.pose.orientation.w = quat[3]

        self.get_logger().info(f'Sending navigation goal to ({x}, {y}, {theta})')

        # Send goal
        future = self.nav_to_pose_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

        return future

    def goal_response_callback(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle result"""
        result = future.result().result
        self.get_logger().info(f'Navigation result: {result}')


def main(args=None):
    rclpy.init(args=args)

    navigator = SimpleNavigator()

    # Example: Set initial pose and navigate to goal
    # In practice, you would get these values from parameters or user input
    navigator.set_initial_pose(0.0, 0.0, 0.0)  # Start at origin

    # Wait a bit for the initial pose to be processed
    import time
    time.sleep(1)

    # Navigate to a goal
    navigator.navigate_to_pose(2.0, 2.0, 0.0)  # Go to (2, 2)

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercise 3: Creating Navigation with Recovery Behaviors

Create `scripts/navigation_with_recovery.py`:

```python
#!/usr/bin/env python3

"""
Navigation script with recovery behaviors
This script demonstrates how to handle navigation failures and recovery
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav2_msgs.msg import RecoveryInfo
from std_msgs.msg import Header


class RecoveryNavigator(Node):
    def __init__(self):
        super().__init__('recovery_navigator')

        # Create action client for navigation
        self.nav_to_pose_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        # Publisher for recovery commands
        self.recovery_pub = self.create_publisher(
            RecoveryInfo,
            'recovery_info',
            QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        )

        # Wait for action server
        self.nav_to_pose_client.wait_for_server()
        self.get_logger().info('Navigation server ready')

    def navigate_with_recovery(self, x, y, theta):
        """Navigate with built-in recovery behaviors"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header = Header()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Set goal position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # Convert theta to quaternion
        import math
        from tf_transformations import quaternion_from_euler
        quat = quaternion_from_euler(0, 0, theta)
        goal_msg.pose.pose.orientation.x = quat[0]
        goal_msg.pose.pose.orientation.y = quat[1]
        goal_msg.pose.pose.orientation.z = quat[2]
        goal_msg.pose.pose.orientation.w = quat[3]

        self.get_logger().info(f'Sending navigation goal to ({x}, {y}, {theta}) with recovery')

        # Send goal
        future = self.nav_to_pose_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

        return future

    def goal_response_callback(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle result"""
        result = future.result().result
        if result:
            self.get_logger().info(f'Navigation completed with result: {result}')
        else:
            self.get_logger().info('Navigation failed - recovery behaviors may have been triggered')


def main(args=None):
    rclpy.init(args=args)

    navigator = RecoveryNavigator()

    # Navigate to goal with recovery behaviors enabled
    navigator.navigate_with_recovery(3.0, 3.0, 1.57)  # Go to (3, 3) with 90 degree orientation

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercise 4: Creating a Navigation Monitor

Create `scripts/navigation_monitor.py`:

```python
#!/usr/bin/env python3

"""
Navigation monitor script
This script monitors navigation status and provides feedback
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class NavigationMonitor(Node):
    def __init__(self):
        super().__init__('navigation_monitor')

        # Subscriptions
        self.path_sub = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.current_goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        # Publisher for status updates
        self.status_pub = self.create_publisher(
            String,
            '/navigation_status',
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.get_logger().info('Navigation monitor started')

    def path_callback(self, msg):
        """Handle path updates"""
        if len(msg.poses) > 0:
            path_length = len(msg.poses)
            self.get_logger().info(f'Current path has {path_length} waypoints')

            # Publish status
            status_msg = String()
            status_msg.data = f'Following path with {path_length} waypoints'
            self.status_pub.publish(status_msg)

    def laser_callback(self, msg):
        """Handle laser scan updates - useful for obstacle detection"""
        # Check for obstacles in front of robot
        front_scan = msg.ranges[len(msg.ranges)//2]  # Front reading

        if front_scan < 0.5:  # Obstacle within 0.5m
            self.get_logger().warn(f'Obstacle detected at {front_scan:.2f}m ahead')

            status_msg = String()
            status_msg.data = f'Obstacle detected: {front_scan:.2f}m ahead'
            self.status_pub.publish(status_msg)

    def goal_callback(self, msg):
        """Handle goal updates"""
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.get_logger().info(f'New navigation goal set to ({x:.2f}, {y:.2f})')


def main(args=None):
    rclpy.init(args=args)

    monitor = NavigationMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercise 5: Complete Navigation System Launch File

Create `launch/navigation_system.launch.py` that combines all components:

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, SetParameter
from launch_ros.actions import PushRosNamespace


def generate_launch_description():
    # Launch arguments
    namespace = LaunchConfiguration('namespace')
    use_namespace = LaunchConfiguration('use_namespace')
    map_yaml_file = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart')
    use_composition = LaunchConfiguration('use_composition')
    use_respawn = LaunchConfiguration('use_respawn')

    # Launch files
    bringup_dir = get_package_share_directory('nav2_bringup')

    # Create launch description
    ld = LaunchDescription()

    # Declare launch arguments
    ld.add_action(DeclareLaunchArgument(
        'namespace', default_value='',
        description='Top-level namespace'))

    ld.add_action(DeclareLaunchArgument(
        'use_namespace', default_value='false',
        description='Whether to apply a namespace to the navigation stack'))

    ld.add_action(DeclareLaunchArgument(
        'map',
        default_value=os.path.join(bringup_dir, 'maps', 'turtlebot3_world.yaml'),
        description='Full path to map file to load'))

    ld.add_action(DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (Gazebo) clock if true'))

    ld.add_action(DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(bringup_dir, 'params', 'nav2_params.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes'))

    ld.add_action(DeclareLaunchArgument(
        'autostart', default_value='true',
        description='Automatically startup the nav2 stack'))

    ld.add_action(DeclareLaunchArgument(
        'use_composition', default_value='False',
        description='Whether to use composed bringup'))

    ld.add_action(DeclareLaunchArgument(
        'use_respawn', default_value='False',
        description='Whether to respawn if a node crashes. Applied when composition is disabled.'))

    # Set parameters
    ld.add_action(SetParameter('use_sim_time', use_sim_time))

    # Include navigation bringup
    nav2_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_dir, 'launch', 'navigation_launch.py')),
        launch_arguments={
            'use_namespace': use_namespace,
            'namespace': namespace,
            'use_sim_time': use_sim_time,
            'params_file': params_file,
            'autostart': autostart,
            'use_composition': use_composition,
            'use_respawn': use_respawn}.items())

    # Include localization bringup (AMCL)
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_dir, 'launch', 'localization_launch.py')),
        launch_arguments={
            'namespace': namespace,
            'use_sim_time': use_sim_time,
            'autostart': autostart,
            'params_file': params_file,
            'use_composition': use_composition,
            'use_respawn': use_respawn}.items())

    # Add our custom nodes
    simple_navigator = Node(
        package='nav2_exercises',
        executable='simple_navigation.py',
        name='simple_navigator',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    navigation_monitor = Node(
        package='nav2_exercises',
        executable='navigation_monitor.py',
        name='navigation_monitor',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Add actions to launch description
    ld.add_action(nav2_bringup_launch)
    ld.add_action(localization_launch)
    ld.add_action(simple_navigator)
    ld.add_action(navigation_monitor)

    return ld
```

## Exercise 6: Student Exercises

### Exercise 6.1: Basic Navigation
1. Launch the navigation system in simulation
2. Send a simple navigation goal using RViz
3. Observe the robot's behavior and path planning
4. Document your observations

### Exercise 6.2: Costmap Configuration
1. Modify the local and global costmap parameters
2. Experiment with different inflation radii
3. Test how different obstacle inflation affects navigation
4. Create a report on your findings

### Exercise 6.3: Recovery Behaviors
1. Create an obstacle in the robot's path
2. Observe how the recovery behaviors activate
3. Modify recovery behavior parameters
4. Test different recovery strategies

### Exercise 6.4: Custom Behavior Tree
1. Create a custom behavior tree for navigation
2. Modify the default navigation behavior
3. Test your custom navigation strategy
4. Compare with the default behavior

## Troubleshooting Common Issues

### Issue 1: Navigation Fails Immediately
- **Cause**: Incorrect TF frames or localization issues
- **Solution**: Verify TF tree and ensure proper localization

### Issue 2: Robot Oscillates Near Goal
- **Cause**: Goal tolerance settings or controller parameters
- **Solution**: Adjust goal tolerance and controller gains

### Issue 3: Path Planning Fails
- **Cause**: Map issues or costmap configuration
- **Solution**: Check map quality and costmap settings

### Issue 4: Recovery Behaviors Don't Trigger
- **Cause**: Recovery server not running or parameters incorrect
- **Solution**: Verify recovery server configuration

## Performance Optimization

### Memory Usage
- Reduce costmap resolution if not needed
- Limit the number of particles in AMCL if accuracy allows
- Use smaller behavior trees for simpler tasks

### Computation Time
- Optimize path planner parameters
- Use appropriate controller frequency
- Consider using simpler controllers for basic navigation

### Real-time Performance
- Ensure sufficient CPU resources
- Use appropriate QoS settings
- Monitor system performance during navigation

## Conclusion

This tutorial provided a comprehensive introduction to implementing navigation with the Nav2 stack in ROS 2. Students learned about the architecture of Nav2, how to configure navigation components, and how to create custom navigation applications. The exercises provided hands-on experience with core navigation concepts including path planning, obstacle avoidance, and recovery behaviors.

Understanding Nav2 is crucial for developing autonomous mobile robots capable of operating in real-world environments. The skills learned in this tutorial form the foundation for more advanced navigation applications and research in mobile robotics.

Students should continue to experiment with different Nav2 configurations and explore advanced features such as multi-robot navigation, dynamic obstacle avoidance, and integration with perception systems.