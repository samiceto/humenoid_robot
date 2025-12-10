# Nav2 Navigation Stack Configuration Guide

This document outlines the steps to configure the Nav2 navigation stack for development in the Physical AI & Humanoid Robotics course.

## Prerequisites

- Ubuntu 22.04 LTS
- ROS 2 Iron or Jazzy installed
- Isaac ROS 3.0+ packages installed (as per previous setup guide)
- Basic understanding of ROS 2 concepts

## Installation Steps

### 1. Install Nav2 Packages

Install the Nav2 packages and dependencies:

```bash
# Update package lists
sudo apt update

# Install Nav2 packages
sudo apt install -y ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-nav2-gui
sudo apt install -y ros-jazzy-nav2-simple-commander ros-jazzy-nav2-lifecycle-manager
sudo apt install -y ros-jazzy-nav2-behaviors ros-jazzy-nav2-rviz-plugins
sudo apt install -y ros-jazzy-nav2-msgs ros-jazzy-nav2-interfaces

# Install additional dependencies
sudo apt install -y ros-jazzy-robot-state-publisher ros-jazzy-joint-state-publisher
sudo apt install -y ros-jazzy-teleop-twist-keyboard ros-jazzy-xacro
sudo apt install -y ros-jazzy-depthimage-to-laserscan ros-jazzy-pointcloud-to-laserscan
```

### 2. Create Navigation Workspace

Create a workspace specifically for Nav2 customizations:

```bash
# Create workspace directory
mkdir -p ~/nav2_ws/src
cd ~/nav2_ws

# Source ROS 2
source /opt/ros/jazzy/setup.bash
```

### 3. Clone Nav2 Source Code (Optional for Development)

If you need to modify Nav2 source code for the course:

```bash
cd ~/nav2_ws/src

# Clone Nav2 source code
git clone -b jazzy https://github.com/ros-planning/navigation2.git

# Navigate to navigation2 directory
cd navigation2

# Checkout the specific version compatible with ROS 2 Jazzy
git checkout jazzy
```

### 4. Install Nav2 Dependencies

Install dependencies for Nav2 development:

```bash
cd ~/nav2_ws

# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Install dependencies using rosdep
rosdep install --from-paths src --ignore-src -r -y
```

### 5. Build Nav2 (if using source code)

If you cloned the source code, build Nav2:

```bash
cd ~/nav2_ws

# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Build Nav2 packages
colcon build --symlink-install --packages-select nav2_common nav2_util nav2_msgs nav2_map_server nav2_costmap_2d nav2_voxel_grid nav2_navfn_planner nav2_behavior_tree nav2_planner nav2_controller nav2_bt_navigator nav2_dwb_controller nav2_amcl nav2_lifecycle_manager nav2_world_model nav2_rviz_plugins

# Or build all Nav2 packages
colcon build --symlink-install --parallel-workers 4
```

### 6. Configure Nav2 for Isaac Sim Integration

Create configuration files for Nav2 that work with Isaac Sim:

```bash
# Create navigation configuration directory
mkdir -p ~/nav2_ws/src/nav2_config/config
cd ~/nav2_ws/src/nav2_config/config

# Create base local planner configuration
cat << 'EOF' > base_local_planner.yaml
controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugins: ["progress_checker"]
    goal_checker_plugins: ["goal_checker"]
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

    # FollowPath controller parameters
    FollowPath:
      plugin: "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
      desired_linear_vel: 0.5
      max_linear_accel: 2.5
      max_linear_decel: 5.0
      desired_angular_vel: 1.0
      max_angular_accel: 3.2
      min_angular_vel: 0.05
      max_rotational_vel: 1.0
      min_turn_radius: 0.0
      lookahead_dist: 0.6
      use_velocity_scaled_lookahead_dist: false
      lookahead_time: 1.5
      use_interpolation: true
      use_regulated_linear_velocity_scaling: true
      use_regulated_angular_velocity_scaling: true
      regulated_linear_scaling_min_radius: 0.9
      regulated_linear_scaling_min_speed: 0.25
      use_cost_regulated_linear_velocity_scaling: true
      cost_scaling_dist: 1.0
      cost_scaling_gain: 1.0
      inflation_cost_scaling_factor: 3.0
      replanning_wait_time: 0.5
EOF

# Create global costmap configuration
cat << 'EOF' > global_costmap.yaml
global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: True
      robot_radius: 0.3
      resolution: 0.05
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
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      always_send_full_costmap: True
EOF

# Create local costmap configuration
cat << 'EOF' > local_costmap.yaml
local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      robot_radius: 0.3
      resolution: 0.05
      plugins: ["obstacle_layer", "voxel_layer", "inflation_layer"]
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
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 10
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: pointcloud
        pointcloud:
          topic: /pointcloud
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "PointCloud2"
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.5
      always_send_full_costmap: False
EOF

# Create behavior tree configuration
cat << 'EOF' > nav2_bt.xml
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <RecoveryNode name="NavigateRecovery" number_of_retries="2">
      <PipelineSequence name="NavigateWithRecovery">
        <RateController name="RateController" hz="1.0">
          <RecoveryNode name="ClearingRecovery" number_of_retries="1">
            <Sequence name="ClearingActions">
              <ClearEntireCostmap name="ClearGlobalCostmap-Context" service_name="global_costmap/clear_entirely_global_costmap"/>
              <ClearEntireCostmap name="ClearLocalCostmap-Context" service_name="local_costmap/clear_entirely_local_costmap"/>
              <RecoveryNode name="PlanRecovery" number_of_retries="1">
                <Sequence name="PlanWithRecovery">
                  <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
                </Sequence>
                <ReactiveFallback name="PlanRecoveryFallback">
                  <GoalUpdated name="GoalUpdated"/>
                  <RoundRobin name="PlanRecoveryActions">
                    <BackUp backup_dist="0.15" backup_speed="0.025"/>
                    <Spin spin_dist="1.57"/>
                    <Wait wait_duration="5"/>
                  </RoundRobin>
                </ReactiveFallback>
              </RecoveryNode>
            </Sequence>
            <ReactiveFallback name="ClearingRecoveryFallback">
              <GoalUpdated name="GoalUpdated"/>
              <RoundRobin name="ClearingRecoveryActions">
                <BackUp backup_dist="0.15" backup_speed="0.025"/>
                <Spin spin_dist="1.57"/>
                <Wait wait_duration="5"/>
              </RoundRobin>
            </ReactiveFallback>
          </RecoveryNode>
        </RateController>
        <RecoveryNode name="ControlRecovery" number_of_retries="1">
          <Sequence name="ControlWithRecovery">
            <FollowPath path="{path}" controller_id="FollowPath"/>
          </Sequence>
          <ReactiveFallback name="ControlRecoveryFallback">
            <GoalUpdated name="GoalUpdated"/>
            <RoundRobin name="ControlRecoveryActions">
              <BackUp backup_dist="0.15" backup_speed="0.025"/>
              <Spin spin_dist="1.57"/>
              <Wait wait_duration="5"/>
            </RoundRobin>
          </ReactiveFallback>
        </RecoveryNode>
      </PipelineSequence>
      <ReactiveFallback name="NavigateRecoveryFallback">
        <GoalUpdated name="GoalUpdated"/>
        <RoundRobin name="NavigateRecoveryActions">
          <BackUp backup_dist="0.15" backup_speed="0.025"/>
          <Spin spin_dist="1.57"/>
          <Wait wait_duration="5"/>
        </RoundRobin>
      </ReactiveFallback>
    </RecoveryNode>
  </BehaviorTree>
</root>
EOF
```

### 7. Create Launch Files for Isaac Sim Integration

Create launch files that integrate Nav2 with Isaac Sim:

```bash
# Create launch directory
mkdir -p ~/nav2_ws/src/nav2_config/launch
cd ~/nav2_ws/src/nav2_config/launch

# Create navigation launch file
cat << 'EOF' > nav2_isaac_sim.launch.py
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='True')
    params_file = LaunchConfiguration('params_file')
    namespace = LaunchConfiguration('namespace', default='')
    autostart = LaunchConfiguration('autostart', default='True')
    use_composition = LaunchConfiguration('use_composition', default='False')
    container_name = LaunchConfiguration('container_name', default='nav2_container')

    # Launch configuration variables
    default_params_file_path = PathJoinSubstitution(
        [FindPackageShare('nav2_config'), 'config', 'nav2_params.yaml']
    )

    # Create a configuration file for navigation parameters
    nav2_params = os.path.join(
        os.path.dirname(__file__),
        '..',
        'config',
        'nav2_params.yaml'
    )

    # Launch the main Nav2 nodes
    nav2_bringup_launch_dir = PathJoinSubstitution(
        [FindPackageShare('nav2_bringup'), 'launch']
    )

    # Controller server
    controller_server_node = Node(
        package='nav2_controller',
        executable='controller_server',
        output='screen',
        parameters=[nav2_params, {'use_sim_time': use_sim_time}],
        remappings=[('cmd_vel', 'cmd_vel_nav')]
    )

    # Planner server
    planner_server_node = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[nav2_params, {'use_sim_time': use_sim_time}],
        remappings=[('goal_pose', 'goal_pose')]
    )

    # Recoveries server
    recoveries_server = Node(
        package='nav2_recoveries',
        executable='recoveries_server',
        name='recoveries_server',
        output='screen',
        parameters=[nav2_params, {'use_sim_time': use_sim_time}],
        remappings=[
            ('cmd_vel', 'cmd_vel_nav'),
            ('global_costmap', 'global_costmap'),
            ('local_costmap', 'local_costmap')
        ]
    )

    # BT navigator
    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[nav2_params, {'use_sim_time': use_sim_time}],
        remappings=[
            ('goal_pose', 'goal_pose'),
            ('feedback', 'follow_waypoints/feedback'),
            ('result', 'follow_waypoints/result'),
            ('behavior_tree_xml_filename', 'bt_navigator/xml_tree'),
            ('global_costmap', 'global_costmap'),
            ('local_costmap', 'local_costmap'),
            ('global_costmap/GlobalCostmapROS', 'global_costmap/costmap'),
            ('local_costmap/LocalCostmapROS', 'local_costmap/costmap')
        ]
    )

    # Lifecycle manager
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {'autostart': autostart},
                    {'node_names': ['controller_server',
                                    'planner_server',
                                    'recoveries_server',
                                    'bt_navigator']}]
    )

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file_path,
            description='Full path to the ROS2 parameters file to use for all launched nodes'
        ),
        DeclareLaunchArgument(
            'autostart',
            default_value='True',
            description='Automatically startup the nav2 stack'
        ),
        DeclareLaunchArgument(
            'use_composition',
            default_value='False',
            description='Whether to use composed bringup'
        ),
        DeclareLaunchArgument(
            'container_name',
            default_value='nav2_container',
            description='the name of conatiner that nodes will load in if use composition'
        ),
        # Launch nodes
        controller_server_node,
        planner_server_node,
        recoveries_server,
        bt_navigator,
        lifecycle_manager
    ])
EOF
```

### 8. Create Nav2 Parameters Configuration

Create the main Nav2 parameters file:

```bash
cd ~/nav2_ws/src/nav2_config/config

cat << 'EOF' > nav2_params.yaml
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
    save_pose_delay: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
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
    default_bt_xml_filename: "nav2_bt.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_compute_path_through_poses_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
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

    # FollowPath controller parameters
    FollowPath:
      plugin: "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
      desired_linear_vel: 0.5
      max_linear_accel: 2.5
      max_linear_decel: 5.0
      desired_angular_vel: 1.0
      max_angular_accel: 3.2
      min_angular_vel: 0.05
      max_rotational_vel: 1.0
      min_turn_radius: 0.0
      lookahead_dist: 0.6
      use_velocity_scaled_lookahead_dist: false
      lookahead_time: 1.5
      use_interpolation: true
      use_regulated_linear_velocity_scaling: true
      use_regulated_angular_velocity_scaling: true
      regulated_linear_scaling_min_radius: 0.9
      regulated_linear_scaling_min_speed: 0.25
      use_cost_regulated_linear_velocity_scaling: true
      cost_scaling_dist: 1.0
      cost_scaling_gain: 1.0
      inflation_cost_scaling_factor: 3.0
      replanning_wait_time: 0.5

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
      robot_radius: 0.3
      resolution: 0.05
      plugins: ["obstacle_layer", "voxel_layer", "inflation_layer"]
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
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 10
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: pointcloud
        pointcloud:
          topic: /pointcloud
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "PointCloud2"
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.5
      always_send_full_costmap: False
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
      robot_radius: 0.3
      resolution: 0.05
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

recoveries_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    recovery_plugins: ["spin", "backup", "wait"]
    spin:
      plugin: "nav2_recoveries::Spin"
    backup:
      plugin: "nav2_recoveries::BackUp"
    wait:
      plugin: "nav2_recoveries::Wait"
    use_sim_time: True

robot_state_publisher:
  ros__parameters:
    use_sim_time: True

waypoint_follower:
  ros__parameters:
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: True
      waypoint_pause_duration: 200
EOF
```

### 9. Create Package.xml for Nav2 Config Package

```bash
cd ~/nav2_ws/src/nav2_config

cat << 'EOF' > package.xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>nav2_config</name>
  <version>1.0.0</version>
  <description>Navigation 2 configuration package for Isaac Sim integration</description>
  <maintainer email="student@university.edu">Student</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <exec_depend>nav2_common</exec_depend>
  <exec_depend>nav2_bringup</exec_depend>
  <exec_depend>nav2_controller</exec_depend>
  <exec_depend>nav2_planner</exec_depend>
  <exec_depend>nav2_recoveries</exec_depend>
  <exec_depend>nav2_bt_navigator</exec_depend>
  <exec_depend>nav2_lifecycle_manager</exec_depend>
  <exec_depend>nav2_map_server</exec_depend>
  <exec_depend>nav2_amcl</exec_depend>
  <exec_depend>nav2_core</exec_depend>
  <exec_depend>nav2_util</exec_depend>
  <exec_depend>nav2_costmap_2d</exec_depend>
  <exec_depend>nav2_voxel_grid</exec_depend>
  <exec_depend>nav2_navfn_planner</exec_depend>
  <exec_depend>nav2_regulated_pursuit_controller</exec_depend>
  <exec_depend>launch</exec_depend>
  <exec_depend>launch_ros</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>
  <exec_depend>nav_msgs</exec_depend>
  <exec_depend>std_msgs</exec_depend>
  <exec_depend>tf2_ros</exec_depend>
  <exec_depend>tf2_geometry_msgs</exec_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
EOF

# Create CMakeLists.txt
cat << 'EOF' > CMakeLists.txt
cmake_minimum_required(VERSION 3.8)
project(nav2_config)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
EOF
```

### 10. Build the Nav2 Config Package

```bash
cd ~/nav2_ws

# Source ROS 2 and Isaac ROS workspace
source /opt/ros/jazzy/setup.bash
source ~/isaac_ros_ws/install/setup.bash

# Build the nav2_config package
colcon build --packages-select nav2_config --symlink-install
```

### 11. Source the Workspace

Add the workspace to your environment:

```bash
# Add to your .bashrc to make it permanent
echo "source ~/nav2_ws/install/setup.bash" >> ~/.bashrc
source ~/nav2_ws/install/setup.bash
```

### 12. Verify Installation

Test the Nav2 installation:

```bash
# Source all workspaces
source /opt/ros/jazzy/setup.bash
source ~/isaac_ros_ws/install/setup.bash
source ~/nav2_ws/install/setup.bash

# Check if Nav2 packages are available
ros2 pkg list | grep nav2

# Check launch files
find ~/nav2_ws/install -name "*launch*" -type f
```

## Isaac Sim Integration Testing

To test Nav2 with Isaac Sim, you would typically run:

```bash
# Terminal 1: Start Isaac Sim with a robot model
# (This would be done in Isaac Sim GUI or using Isaac Sim launch scripts)

# Terminal 2: Launch Nav2
source /opt/ros/jazzy/setup.bash
source ~/isaac_ros_ws/install/setup.bash
source ~/nav2_ws/install/setup.bash
ros2 launch nav2_config nav2_isaac_sim.launch.py

# Terminal 3: Send navigation goals
source /opt/ros/jazzy/setup.bash
source ~/isaac_ros_ws/install/setup.bash
source ~/nav2_ws/install/setup.bash
ros2 run nav2_test nav2_simple_commander
```

## Troubleshooting

### Common Issues:

1. **Parameter Configuration**: If navigation fails, check that `use_sim_time` is set to `True` in all configuration files
2. **TF Issues**: Ensure all coordinate frames are properly published (map, odom, base_link)
3. **Sensor Data**: Verify that laser scan and odometry data are being published correctly
4. **Costmap Issues**: Check that obstacle topics are connected properly

### Debugging Commands:

```bash
# Check TF tree
ros2 run tf2_tools view_frames

# Visualize in RViz
ros2 run rviz2 rviz2

# Check topics
ros2 topic list
ros2 topic echo /scan
```

## Course-Specific Configurations

For the Physical AI & Humanoid Robotics course, you may need to customize:

1. Robot radius in costmap configurations
2. Controller parameters for specific robot dynamics
3. Behavior tree for humanoid-specific navigation patterns
4. Sensor topics to match Isaac Sim robot model outputs

## Next Steps

After completing Nav2 configuration, proceed to:

1. Setting up Jetson Orin Nano development environment
2. Testing navigation with simulated humanoid robots
3. Creating course-specific navigation exercises

## References

- [Navigation2 Documentation](https://navigation.ros.org/)
- [ROS 2 Navigation Tutorials](https://navigation.ros.org/tutorials/)
- [Isaac Sim ROS Bridge Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/programming_tutorials/tutorial_3_ros_bridge.html)