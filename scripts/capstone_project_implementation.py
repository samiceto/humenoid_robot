#!/usr/bin/env python3
# capstone_project_implementation.py
# Complete end-to-end capstone project implementation
# Integrating all concepts from the Physical AI & Humanoid Robotics course

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.duration import Duration

from std_msgs.msg import String, Bool, Float32, Header
from sensor_msgs.msg import Image, PointCloud2, Imu, JointState, LaserScan
from geometry_msgs.msg import Twist, PoseStamped, Point, Vector3, Quaternion
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Time

from rclpy.action import ActionClient
from geometry_msgs.action import NavigateToPose
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import GetState

import threading
import time
import numpy as np
import math
import json
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import queue
import logging
from enum import Enum

# Import required libraries for perception and control
try:
    import cv2
    from cv_bridge import CvBridge
except ImportError:
    print("CV2 not available, using mock implementation for simulation")
    cv2 = None

try:
    import torch
    import torchvision
except ImportError:
    print("PyTorch not available, using mock implementation for simulation")
    torch = None


class RobotState(Enum):
    """Enumeration of robot states"""
    IDLE = "idle"
    NAVIGATING = "navigating"
    MANIPULATING = "manipulating"
    PERCEIVING = "perceiving"
    EMERGENCY_STOP = "emergency_stop"
    SAFETY_MODE = "safety_mode"


@dataclass
class RobotPose:
    """Data class for robot pose information"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass
class ObjectDetection:
    """Data class for object detection results"""
    id: str
    class_name: str
    confidence: float
    position: Point
    size: Vector3
    timestamp: float


class SafetyMonitor(Node):
    """Safety monitoring system for the capstone project"""

    def __init__(self):
        super().__init__('safety_monitor')

        # Publishers
        self.safety_status_pub = self.create_publisher(Bool, 'safety_status', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, 'emergency_stop', 10)
        self.safety_violation_pub = self.create_publisher(String, 'safety_violations', 10)

        # Subscribers
        qos_profile = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos_profile)

        # Timer for safety checks
        self.safety_timer = self.create_timer(0.05, self.safety_check)  # 20 Hz safety checks

        # Safety parameters
        self.max_linear_velocity = 0.5  # m/s
        self.max_angular_velocity = 0.5  # rad/s
        self.max_joint_velocity = 2.0  # rad/s
        self.min_battery_level = 0.1  # 10%
        self.min_distance_to_obstacle = 0.3  # meters

        # State tracking
        self.current_odom = None
        self.current_imu = None
        self.current_joints = None
        self.current_scan = None
        self.safety_violation = False
        self.emergency_stop_triggered = False

        # Performance tracking
        self.safety_check_count = 0
        self.violation_count = 0

        self.get_logger().info('Safety Monitor initialized')

    def odom_callback(self, msg):
        """Handle odometry updates"""
        self.current_odom = msg

    def imu_callback(self, msg):
        """Handle IMU updates"""
        self.current_imu = msg

    def joint_state_callback(self, msg):
        """Handle joint state updates"""
        self.current_joints = msg

    def scan_callback(self, msg):
        """Handle laser scan updates"""
        self.current_scan = msg

    def safety_check(self):
        """Perform periodic safety checks"""
        if self.emergency_stop_triggered:
            return  # Already in emergency stop state

        violations = []

        # Check velocity limits
        if self.current_odom:
            linear_vel = math.sqrt(
                self.current_odom.twist.twist.linear.x**2 +
                self.current_odom.twist.twist.linear.y**2 +
                self.current_odom.twist.twist.linear.z**2
            )
            angular_vel = math.sqrt(
                self.current_odom.twist.twist.angular.x**2 +
                self.current_odom.twist.twist.angular.y**2 +
                self.current_odom.twist.twist.angular.z**2
            )

            if linear_vel > self.max_linear_velocity:
                violations.append(f'Linear velocity exceeded: {linear_vel:.2f} > {self.max_linear_velocity}')

            if angular_vel > self.max_angular_velocity:
                violations.append(f'Angular velocity exceeded: {angular_vel:.2f} > {self.max_angular_velocity}')

        # Check joint velocity limits
        if self.current_joints and len(self.current_joints.velocity) > 0:
            for i, vel in enumerate(self.current_joints.velocity):
                if abs(vel) > self.max_joint_velocity:
                    violations.append(f'Joint {i} velocity exceeded: {abs(vel):.2f} > {self.max_joint_velocity}')

        # Check obstacle distance
        if self.current_scan:
            min_distance = min([r for r in self.current_scan.ranges if not math.isnan(r)], default=float('inf'))
            if min_distance < self.min_distance_to_obstacle:
                violations.append(f'Obstacle too close: {min_distance:.2f} < {self.min_distance_to_obstacle}')

        # Update counters
        self.safety_check_count += 1
        if violations:
            self.violation_count += 1

        # Publish safety status
        safety_msg = Bool()
        if violations:
            self.get_logger().warn(f'Safety violations: {", ".join(violations)}')
            safety_msg.data = False
            self.safety_violation = True
            self.trigger_emergency_stop(violations)
        else:
            safety_msg.data = True
            self.safety_violation = False

        self.safety_status_pub.publish(safety_msg)

    def trigger_emergency_stop(self, violations):
        """Trigger emergency stop procedure"""
        self.get_logger().error(f'EMERGENCY STOP TRIGGERED: {", ".join(violations)}')
        self.emergency_stop_triggered = True

        # Publish emergency stop command
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)

        # Publish violation details
        violation_msg = String()
        violation_msg.data = json.dumps({
            'timestamp': time.time(),
            'violations': violations,
            'safety_check_count': self.safety_check_count,
            'violation_count': self.violation_count
        })
        self.safety_violation_pub.publish(violation_msg)


class PerceptionSystem(Node):
    """Perception system for the capstone project"""

    def __init__(self):
        super().__init__('perception_system')

        # QoS profile for perception
        qos_profile = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        # Subscribers for sensor data
        self.rgb_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.rgb_callback, qos_profile)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, qos_profile)
        self.lidar_sub = self.create_subscription(
            PointCloud2, '/lidar/points', self.lidar_callback, qos_profile)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile)

        # Publishers for processed perception data
        self.objects_pub = self.create_publisher(MarkerArray, 'detected_objects', 10)
        self.map_pub = self.create_publisher(MarkerArray, 'environment_map', 10)
        self.perception_rate_pub = self.create_publisher(Float32, 'perception_rate', 10)

        # Internal state
        self.perception_queue = queue.Queue(maxsize=20)
        self.detected_objects = []
        self.environment_map = []

        # Performance tracking
        self.perception_times = deque(maxlen=100)
        self.perception_rate = 0.0

        # Threading for perception processing
        self.perception_thread = threading.Thread(target=self.perception_worker, daemon=True)
        self.perception_thread.start()

        # Performance timer
        self.performance_timer = self.create_timer(1.0, self.publish_performance)

        self.get_logger().info('Perception System initialized')

    def rgb_callback(self, msg):
        """Handle RGB camera data"""
        try:
            self.perception_queue.put(('rgb', msg), block=False)
        except queue.Full:
            self.get_logger().warn('Perception queue full, dropping RGB frame')

    def depth_callback(self, msg):
        """Handle depth camera data"""
        try:
            self.perception_queue.put(('depth', msg), block=False)
        except queue.Full:
            self.get_logger().warn('Perception queue full, dropping depth frame')

    def lidar_callback(self, msg):
        """Handle LIDAR data"""
        try:
            self.perception_queue.put(('lidar', msg), block=False)
        except queue.Full:
            self.get_logger().warn('Perception queue full, dropping LIDAR frame')

    def scan_callback(self, msg):
        """Handle laser scan data"""
        try:
            self.perception_queue.put(('scan', msg), block=False)
        except queue.Full:
            self.get_logger().warn('Perception queue full, dropping scan')

    def perception_worker(self):
        """Background thread for perception processing"""
        while rclpy.ok():
            try:
                sensor_type, sensor_data = self.perception_queue.get(timeout=1.0)
                start_time = time.time()

                # Process sensor data based on type
                if sensor_type == 'rgb':
                    objects = self.process_rgb_data(sensor_data)
                    if objects:
                        self.detected_objects = objects
                        self.publish_objects(objects)
                elif sensor_type == 'depth':
                    depth_map = self.process_depth_data(sensor_data)
                elif sensor_type == 'lidar':
                    obstacles = self.process_lidar_data(sensor_data)
                    if obstacles:
                        self.environment_map = obstacles
                        self.publish_map(obstacles)
                elif sensor_type == 'scan':
                    scan_obstacles = self.process_scan_data(sensor_data)
                    if scan_obstacles:
                        self.environment_map.extend(scan_obstacles)
                        self.publish_map(self.environment_map)

                # Track processing time and rate
                processing_time = time.time() - start_time
                self.perception_times.append(time.time())

                # Calculate perception rate
                if len(self.perception_times) >= 2:
                    time_diff = self.perception_times[-1] - self.perception_times[0]
                    if time_diff > 0:
                        self.perception_rate = len(self.perception_times) / time_diff
                    else:
                        self.perception_rate = 0.0

            except queue.Empty:
                continue  # Timeout, continue loop
            except Exception as e:
                self.get_logger().error(f'Perception processing error: {e}')

    def process_rgb_data(self, rgb_msg):
        """Process RGB image data for object detection"""
        # Simulate object detection (in real implementation, use Isaac ROS or similar)
        objects = []

        # Simulate detection of 3-5 objects
        for i in range(np.random.randint(3, 6)):
            obj = ObjectDetection(
                id=f'obj_{i}_{int(time.time())}',
                class_name=np.random.choice(['person', 'chair', 'table', 'box']),
                confidence=np.random.uniform(0.7, 0.95),
                position=Point(
                    x=np.random.uniform(-2.0, 2.0),
                    y=np.random.uniform(-2.0, 2.0),
                    z=np.random.uniform(0.0, 1.5)
                ),
                size=Vector3(
                    x=np.random.uniform(0.1, 0.5),
                    y=np.random.uniform(0.1, 0.5),
                    z=np.random.uniform(0.1, 1.0)
                ),
                timestamp=time.time()
            )
            objects.append(obj)

        return objects

    def process_depth_data(self, depth_msg):
        """Process depth image data"""
        # Process depth data for 3D reconstruction
        # This would typically involve converting to point cloud and processing
        pass

    def process_lidar_data(self, lidar_msg):
        """Process LIDAR point cloud data"""
        # Simulate obstacle detection from LIDAR data
        obstacles = []

        # Simulate detection of 5-10 obstacles
        for i in range(np.random.randint(5, 11)):
            obstacle = {
                'id': f'obs_{i}_{int(time.time())}',
                'position': {
                    'x': np.random.uniform(-5.0, 5.0),
                    'y': np.random.uniform(-5.0, 5.0),
                    'z': 0.0
                },
                'size': {
                    'x': np.random.uniform(0.2, 1.0),
                    'y': np.random.uniform(0.2, 1.0),
                    'z': np.random.uniform(0.5, 2.0)
                },
                'type': np.random.choice(['static', 'dynamic'])
            }
            obstacles.append(obstacle)

        return obstacles

    def process_scan_data(self, scan_msg):
        """Process laser scan data for obstacle detection"""
        # Simulate obstacle detection from laser scan
        obstacles = []

        # Process scan ranges to detect obstacles
        for i, range_val in enumerate(scan_msg.ranges):
            if not math.isnan(range_val) and range_val < 2.0:  # Obstacle within 2m
                angle = scan_msg.angle_min + i * scan_msg.angle_increment
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)

                obstacle = {
                    'id': f'scan_obs_{i}_{int(time.time())}',
                    'position': {'x': x, 'y': y, 'z': 0.0},
                    'size': {'x': 0.3, 'y': 0.3, 'z': 1.0},
                    'type': 'scan_detected'
                }
                obstacles.append(obstacle)

        return obstacles

    def publish_objects(self, objects):
        """Publish detected objects as markers"""
        marker_array = MarkerArray()

        for i, obj in enumerate(objects):
            marker = Marker()
            marker.header = Header()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = 'map'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position = obj.position
            marker.pose.orientation.w = 1.0

            marker.scale = obj.size
            marker.color.r = 1.0 if obj.class_name == 'person' else 0.0
            marker.color.g = 1.0 if obj.class_name == 'chair' else 0.0
            marker.color.b = 1.0 if obj.class_name == 'table' else 0.0
            marker.color.a = 0.7

            marker.ns = "detected_objects"
            marker.lifetime = Duration(seconds=2.0).to_msg()

            marker_array.markers.append(marker)

        self.objects_pub.publish(marker_array)

    def publish_map(self, obstacles):
        """Publish environment map as markers"""
        marker_array = MarkerArray()

        for i, obs in enumerate(obstacles):
            marker = Marker()
            marker.header = Header()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = 'map'
            marker.id = i + 1000  # Offset to avoid ID conflicts with objects
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = obs['position']['x']
            marker.pose.position.y = obs['position']['y']
            marker.pose.position.z = obs['position']['z'] / 2.0  # Center the box
            marker.pose.orientation.w = 1.0

            marker.scale.x = obs['size']['x']
            marker.scale.y = obs['size']['y']
            marker.scale.z = obs['size']['z']

            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
            marker.color.a = 0.3

            marker.ns = "environment_map"
            marker.lifetime = Duration(seconds=5.0).to_msg()

            marker_array.markers.append(marker)

        self.map_pub.publish(marker_array)

    def publish_performance(self):
        """Publish perception performance metrics"""
        rate_msg = Float32()
        rate_msg.data = self.perception_rate
        self.perception_rate_pub.publish(rate_msg)

        self.get_logger().debug(f'Perception rate: {self.perception_rate:.2f} Hz')


class PlanningSystem(Node):
    """Planning system for the capstone project"""

    def __init__(self):
        super().__init__('planning_system')

        # Subscribers
        self.map_sub = self.create_subscription(
            MarkerArray, 'environment_map', self.map_callback, 10)
        self.objects_sub = self.create_subscription(
            MarkerArray, 'detected_objects', self.objects_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)

        # Publishers
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)

        # Action server for navigation
        self.nav_to_pose_action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            self.execute_navigate_to_pose,
            goal_callback=self.goal_navigate_to_pose,
            cancel_callback=self.cancel_navigate_to_pose
        )

        # Internal state
        self.environment_map = []
        self.detected_objects = []
        self.current_pose = RobotPose()
        self.current_goal = None
        self.path_to_goal = []

        # Planning parameters
        self.planning_frequency = 2.0  # Hz
        self.planning_timer = self.create_timer(
            1.0 / self.planning_frequency, self.plan_callback)

        # Performance tracking
        self.planning_times = deque(maxlen=50)
        self.last_plan_time = 0.0

        self.get_logger().info('Planning System initialized')

    def map_callback(self, msg):
        """Handle environment map updates"""
        self.environment_map = msg.markers

    def objects_callback(self, msg):
        """Handle detected objects updates"""
        self.detected_objects = msg.markers

    def odom_callback(self, msg):
        """Handle odometry updates"""
        self.current_pose.x = msg.pose.pose.position.x
        self.current_pose.y = msg.pose.pose.position.y
        self.current_pose.z = msg.pose.pose.position.z

        # Convert quaternion to euler angles
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_pose.yaw = math.atan2(siny_cosp, cosy_cosp)

    def plan_callback(self):
        """Generate navigation plan if goal is set"""
        if self.current_goal:
            start_time = time.time()
            path = self.generate_path_to_goal()
            self.last_plan_time = time.time() - start_time

            if path:
                self.path_to_goal = path
                self.publish_path(path)

    def generate_path_to_goal(self):
        """Generate path to current goal considering obstacles"""
        if not self.current_goal:
            return []

        # Simulate path planning (in real implementation, use A*, RRT*, or similar)
        path = []

        # Get start and goal positions
        start = Point()
        start.x = self.current_pose.x
        start.y = self.current_pose.y
        start.z = self.current_pose.z

        goal = self.current_goal.pose.position

        # Calculate straight-line path with obstacle avoidance
        steps = 20
        for i in range(steps + 1):
            t = i / steps
            pos = Point()
            pos.x = start.x + t * (goal.x - start.x)
            pos.y = start.y + t * (goal.y - start.y)
            pos.z = start.z + t * (goal.z - start.z)

            # Add some random variation to simulate obstacle avoidance
            if i > 0 and i < steps:  # Don't modify start and end points
                pos.x += np.random.uniform(-0.1, 0.1)
                pos.y += np.random.uniform(-0.1, 0.1)

            path.append(pos)

        return path

    def publish_path(self, path):
        """Publish the generated path"""
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for point in path:
            pose = PoseStamped()
            pose.header = Header()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position = point
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def goal_navigate_to_pose(self, goal_request):
        """Handle navigation goal request"""
        self.get_logger().info(f'Received navigation goal: {goal_request.pose.pose.position}')
        return GoalResponse.ACCEPT

    def cancel_navigate_to_pose(self, goal_handle):
        """Handle navigation goal cancellation"""
        self.get_logger().info('Navigation goal canceled')
        return CancelResponse.ACCEPT

    def execute_navigate_to_pose(self, goal_handle):
        """Execute navigation to pose action"""
        self.get_logger().info('Executing navigation to pose')

        feedback_msg = NavigateToPose.Feedback()
        result_msg = NavigateToPose.Result()

        # Set the goal
        self.current_goal = goal_handle.request.pose

        # Simulate navigation execution
        for i in range(50):  # Simulate 50 steps of navigation
            if goal_handle.is_canceling:
                goal_handle.canceled()
                result_msg.result = -1
                return result_msg

            # Calculate distance to goal
            dist_to_goal = math.sqrt(
                (self.current_pose.x - self.current_goal.pose.position.x)**2 +
                (self.current_pose.y - self.current_goal.pose.position.y)**2
            )

            # Update feedback
            feedback_msg.current_pose.pose.position.x = self.current_pose.x
            feedback_msg.current_pose.pose.position.y = self.current_pose.y
            feedback_msg.distance_remaining = dist_to_goal

            goal_handle.publish_feedback(feedback_msg)

            # Check if close enough to goal
            if dist_to_goal < 0.2:  # 20cm tolerance
                self.get_logger().info('Reached goal position')
                result_msg.result = 1
                goal_handle.succeed()
                return result_msg

            time.sleep(0.1)  # Simulate time for movement

        # If we get here, we didn't reach the goal
        result_msg.result = 0
        goal_handle.abort()
        return result_msg


class ControlSystem(Node):
    """Control system for the capstone project"""

    def __init__(self):
        super().__init__('control_system')

        # Subscribers
        self.path_sub = self.create_subscription(
            Path, 'global_plan', self.path_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.control_rate_pub = self.create_publisher(Float32, 'control_rate', 10)

        # Internal state
        self.current_path = []
        self.current_odom = None
        self.current_imu = None
        self.robot_state = RobotState.IDLE
        self.control_enabled = True

        # Control parameters
        self.control_frequency = 100.0  # Hz (≥100 Hz requirement)
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency, self.control_callback)

        # PID controller parameters
        self.linear_kp = 1.0
        self.linear_ki = 0.0
        self.linear_kd = 0.0
        self.angular_kp = 2.0
        self.angular_ki = 0.0
        self.angular_kd = 0.0

        # PID accumulators
        self.linear_error_sum = 0.0
        self.angular_error_sum = 0.0
        self.last_linear_error = 0.0
        self.last_angular_error = 0.0

        # Performance tracking
        self.control_times = deque(maxlen=100)
        self.control_rate = 0.0

        self.get_logger().info('Control System initialized')

    def path_callback(self, msg):
        """Handle path updates"""
        self.current_path = msg.poses

    def odom_callback(self, msg):
        """Handle odometry updates"""
        self.current_odom = msg

    def imu_callback(self, msg):
        """Handle IMU updates"""
        self.current_imu = msg

    def control_callback(self):
        """Generate control commands at high frequency"""
        if not self.control_enabled:
            return

        start_time = time.time()

        cmd = Twist()
        if self.current_odom and self.current_path:
            cmd = self.compute_control_command()

        self.cmd_vel_pub.publish(cmd)

        # Track control rate
        self.control_times.append(time.time())
        if len(self.control_times) >= 2:
            time_diff = self.control_times[-1] - self.control_times[0]
            if time_diff > 0:
                self.control_rate = len(self.control_times) / time_diff
            else:
                self.control_rate = 0.0

    def compute_control_command(self):
        """Compute velocity commands based on current state and path"""
        cmd = Twist()

        if not self.current_path or not self.current_odom:
            return cmd

        # Get current position
        current_pos = self.current_odom.pose.pose.position
        current_yaw = self.get_robot_yaw()

        # Get next waypoint
        if len(self.current_path) > 0:
            next_waypoint = self.current_path[0].pose.position

            # Calculate error to next waypoint
            dx = next_waypoint.x - current_pos.x
            dy = next_waypoint.y - current_pos.y
            distance = math.sqrt(dx**2 + dy**2)

            # Calculate angle to waypoint
            angle_to_waypoint = math.atan2(dy, dx)
            angle_error = angle_to_waypoint - current_yaw

            # Normalize angle error to [-pi, pi]
            while angle_error > math.pi:
                angle_error -= 2 * math.pi
            while angle_error < -math.pi:
                angle_error += 2 * math.pi

            # PID control for linear velocity
            self.linear_error_sum += distance
            linear_error_rate = distance - self.last_linear_error

            linear_output = (
                self.linear_kp * distance +
                self.linear_ki * self.linear_error_sum * (1.0 / self.control_frequency) +
                self.linear_kd * linear_error_rate * self.control_frequency
            )

            # PID control for angular velocity
            self.angular_error_sum += angle_error
            angular_error_rate = angle_error - self.last_angular_error

            angular_output = (
                self.angular_kp * angle_error +
                self.angular_ki * self.angular_error_sum * (1.0 / self.control_frequency) +
                self.angular_kd * angular_error_rate * self.control_frequency
            )

            # Limit outputs
            linear_output = max(min(linear_output, 0.5), -0.5)  # ±0.5 m/s
            angular_output = max(min(angular_output, 1.0), -1.0)  # ±1.0 rad/s

            cmd.linear.x = linear_output
            cmd.angular.z = angular_output

            # Update last errors
            self.last_linear_error = distance
            self.last_angular_error = angle_error

        return cmd

    def get_robot_yaw(self):
        """Get robot yaw angle from orientation"""
        if not self.current_odom:
            return 0.0

        q = self.current_odom.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def enable_control(self):
        """Enable control system"""
        self.control_enabled = True
        self.get_logger().info('Control system enabled')

    def disable_control(self):
        """Disable control system"""
        self.control_enabled = False
        # Stop the robot
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info('Control system disabled')


class LearningSystem(Node):
    """Learning system for adaptive behavior"""

    def __init__(self):
        super().__init__('learning_system')

        # Subscribers
        self.perception_sub = self.create_subscription(
            MarkerArray, 'detected_objects', self.perception_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)

        # Publishers
        self.adaptation_pub = self.create_publisher(String, 'adaptation_commands', 10)

        # Internal state
        self.perception_history = deque(maxlen=100)
        self.motion_history = deque(maxlen=100)
        self.environment_model = {}

        # Learning parameters
        self.learning_frequency = 1.0  # Hz
        self.learning_timer = self.create_timer(
            1.0 / self.learning_frequency, self.learning_callback)

        # Performance tracking
        self.learning_iterations = 0

        self.get_logger().info('Learning System initialized')

    def perception_callback(self, msg):
        """Handle perception updates"""
        self.perception_history.append({
            'timestamp': time.time(),
            'objects': len(msg.markers),
            'data': msg.markers
        })

    def odom_callback(self, msg):
        """Handle odometry updates"""
        self.motion_history.append({
            'timestamp': time.time(),
            'position': msg.pose.pose.position,
            'velocity': msg.twist.twist.linear
        })

    def imu_callback(self, msg):
        """Handle IMU updates"""
        pass  # For learning system, we might use IMU for stability learning

    def learning_callback(self):
        """Main learning callback"""
        self.learning_iterations += 1

        # Analyze perception patterns
        if len(self.perception_history) > 10:
            self.analyze_environment_patterns()

        # Analyze motion patterns
        if len(self.motion_history) > 10:
            self.analyze_motion_patterns()

        # Update environment model
        self.update_environment_model()

        # Generate adaptation commands if needed
        adaptation_needed = self.check_adaptation_needed()
        if adaptation_needed:
            self.generate_adaptation_command()

    def analyze_environment_patterns(self):
        """Analyze patterns in environmental perception"""
        # Count object types and locations
        object_counts = {}
        object_locations = {}

        for record in list(self.perception_history)[-10:]:  # Last 10 records
            for marker in record['data']:
                obj_type = marker.ns
                if obj_type not in object_counts:
                    object_counts[obj_type] = 0
                    object_locations[obj_type] = []

                object_counts[obj_type] += 1
                object_locations[obj_type].append((marker.pose.position.x, marker.pose.position.y))

        self.environment_model['object_distribution'] = {
            'counts': object_counts,
            'locations': object_locations
        }

    def analyze_motion_patterns(self):
        """Analyze patterns in robot motion"""
        # Calculate average velocities and accelerations
        if len(self.motion_history) < 2:
            return

        velocities = []
        accelerations = []

        for i in range(1, len(self.motion_history)):
            dt = (self.motion_history[i]['timestamp'] -
                  self.motion_history[i-1]['timestamp'])
            if dt > 0:
                v = math.sqrt(
                    self.motion_history[i]['velocity'].x**2 +
                    self.motion_history[i]['velocity'].y**2
                )
                velocities.append(v)

        for i in range(1, len(velocities)):
            dt = (self.motion_history[i+1]['timestamp'] -
                  self.motion_history[i]['timestamp'])
            if dt > 0:
                a = (velocities[i] - velocities[i-1]) / dt
                accelerations.append(a)

        self.environment_model['motion_patterns'] = {
            'avg_velocity': np.mean(velocities) if velocities else 0.0,
            'avg_acceleration': np.mean(accelerations) if accelerations else 0.0,
            'velocity_std': np.std(velocities) if velocities else 0.0
        }

    def update_environment_model(self):
        """Update the internal environment model"""
        # This would update a more sophisticated model in a real implementation
        pass

    def check_adaptation_needed(self) -> bool:
        """Check if adaptation is needed"""
        # Adaptation needed if:
        # 1. Significant change in environment
        # 2. Performance degradation detected
        # 3. New patterns detected

        # For simulation, adapt every 10 iterations
        return self.learning_iterations % 10 == 0

    def generate_adaptation_command(self):
        """Generate adaptation commands based on learning"""
        adaptation_msg = String()

        # Simulate adaptation command
        adaptation_commands = [
            "increase_perception_sensitivity",
            "adjust_navigation_parameters",
            "modify_safety_thresholds",
            "update_environment_map"
        ]

        command = np.random.choice(adaptation_commands)
        adaptation_msg.data = json.dumps({
            'command': command,
            'timestamp': time.time(),
            'iteration': self.learning_iterations
        })

        self.adaptation_pub.publish(adaptation_msg)
        self.get_logger().info(f'Generated adaptation command: {command}')


class CapstoneSystem(Node):
    """Main capstone system that coordinates all components"""

    def __init__(self):
        super().__init__('capstone_system')

        # Initialize all subsystems
        self.safety_monitor = SafetyMonitor()
        self.perception_system = PerceptionSystem()
        self.planning_system = PlanningSystem()
        self.control_system = ControlSystem()
        self.learning_system = LearningSystem()

        # Publishers for system status
        self.status_pub = self.create_publisher(String, 'system_status', 10)
        self.system_performance_pub = self.create_publisher(String, 'system_performance', 10)

        # Subscribers for system monitoring
        self.safety_status_sub = self.create_subscription(
            Bool, 'safety_status', self.safety_status_callback, 10)
        self.perception_rate_sub = self.create_subscription(
            Float32, 'perception_rate', self.perception_rate_callback, 10)
        self.control_rate_sub = self.create_subscription(
            Float32, 'control_rate', self.control_rate_callback, 10)

        # Timer for system monitoring
        self.status_timer = self.create_timer(1.0, self.system_status_callback)

        # System state tracking
        self.safety_status = True
        self.perception_rate = 0.0
        self.control_rate = 0.0
        self.system_operational = True

        # Performance requirements
        self.min_perception_rate = 15.0  # Hz
        self.min_control_rate = 100.0   # Hz

        self.get_logger().info('Capstone System initialized with all subsystems')

    def safety_status_callback(self, msg):
        """Handle safety status updates"""
        self.safety_status = msg.data
        if not self.safety_status:
            self.system_operational = False
            self.get_logger().warn('System safety violation detected')

    def perception_rate_callback(self, msg):
        """Handle perception rate updates"""
        self.perception_rate = msg.data

    def control_rate_callback(self, msg):
        """Handle control rate updates"""
        self.control_rate = msg.data

    def system_status_callback(self):
        """Publish overall system status"""
        status_msg = String()

        # Check performance requirements
        perception_ok = self.perception_rate >= self.min_perception_rate
        control_ok = self.control_rate >= self.min_control_rate
        safety_ok = self.safety_status

        status_parts = [
            f"PERCEPTION: {self.perception_rate:.1f}Hz ({'✓' if perception_ok else '✗'})",
            f"CONTROL: {self.control_rate:.1f}Hz ({'✓' if control_ok else '✗'})",
            f"SAFETY: {'OK' if safety_ok else 'VIOLATION'}"
        ]

        status_msg.data = " | ".join(status_parts)
        self.status_pub.publish(status_msg)

        # Log system status
        self.get_logger().info(status_msg.data)

        # Check if all requirements are met
        if perception_ok and control_ok and safety_ok:
            self.get_logger().info('✓ All performance requirements met')
        else:
            self.get_logger().warn('✗ Some performance requirements not met')

    def run_performance_validation(self):
        """Run comprehensive performance validation"""
        validation_results = {
            'perception_rate_meets_requirement': self.perception_rate >= self.min_perception_rate,
            'control_rate_meets_requirement': self.control_rate >= self.min_control_rate,
            'safety_system_operational': self.safety_status,
            'perception_rate': self.perception_rate,
            'control_rate': self.control_rate,
            'requirements_met': (
                self.perception_rate >= self.min_perception_rate and
                self.control_rate >= self.min_control_rate and
                self.safety_status
            )
        }

        performance_msg = String()
        performance_msg.data = json.dumps(validation_results)
        self.system_performance_pub.publish(performance_msg)

        return validation_results


def main(args=None):
    rclpy.init(args=args)

    # Create the main capstone system
    capstone_system = CapstoneSystem()

    # Create a multi-threaded executor to handle all nodes
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(capstone_system)
    executor.add_node(capstone_system.safety_monitor)
    executor.add_node(capstone_system.perception_system)
    executor.add_node(capstone_system.planning_system)
    executor.add_node(capstone_system.control_system)
    executor.add_node(capstone_system.learning_system)

    try:
        # Run the system
        capstone_system.get_logger().info('Starting capstone system...')
        executor.spin()
    except KeyboardInterrupt:
        capstone_system.get_logger().info('Interrupted, shutting down...')
    finally:
        # Clean up
        capstone_system.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()