#!/usr/bin/env python3
"""
Navigation Action Sequences for Humanoid Robots

This script implements navigation action sequences for the Physical AI & Humanoid Robotics course.
It includes various navigation behaviors and action primitives for humanoid robots.
"""

import rclpy
from rclpy.action import ActionClient, GoalResponse, CancelResponse
from rclpy.node import Node
from rclpy.duration import Duration
import time

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Header
from action_msgs.msg import GoalStatus

# Navigation action interfaces (these would normally be defined in custom action files)
# For this example, we'll use standard navigation2 actions
from nav2_msgs.action import NavigateToPose, FollowPath, ComputePathToPose
from geometry_msgs.msg import Pose, Point, Quaternion


class NavigationActionSequences(Node):
    """
    Implements navigation action sequences for humanoid robots
    """

    def __init__(self):
        super().__init__('navigation_action_sequences')

        # Action clients for navigation
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.follow_path_client = ActionClient(self, FollowPath, 'follow_path')
        self.compute_path_client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Robot state
        self.current_pose = None
        self.navigation_active = False

        self.get_logger().info("Navigation Action Sequences initialized")

    def odom_callback(self, msg):
        """Update current pose from odometry"""
        self.current_pose = msg.pose.pose

    def create_pose_stamped(self, x, y, z, qx, qy, qz, qw, frame_id="map"):
        """Helper to create a PoseStamped message"""
        pose_stamped = PoseStamped()
        pose_stamped.header = Header()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = frame_id
        pose_stamped.pose.position = Point(x=x, y=y, z=z)
        pose_stamped.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        return pose_stamped

    def navigate_to_location(self, location_name, x, y, theta=0.0):
        """Navigate to a named location"""
        self.get_logger().info(f"Navigating to {location_name} at ({x}, {y}, {theta})")

        # Convert theta to quaternion
        import math
        qw = math.cos(theta / 2.0)
        qz = math.sin(theta / 2.0)

        # Create goal pose
        goal_pose = self.create_pose_stamped(x, y, 0.0, 0.0, 0.0, qz, qw)

        # Send navigation goal
        return self.send_navigate_to_pose_goal(goal_pose)

    def send_navigate_to_pose_goal(self, pose_stamped):
        """Send a NavigateToPose goal"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_stamped

        # Wait for action server
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Navigation action server not available")
            return False

        # Send goal
        goal_handle_future = self.nav_to_pose_client.send_goal_async(
            goal_msg,
            feedback_callback=self.navigation_feedback_callback
        )

        goal_handle_future.add_done_callback(self.navigation_goal_response_callback)
        return True

    def navigation_goal_response_callback(self, future):
        """Handle navigation goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Navigation goal rejected')
            return

        self.get_logger().info('Navigation goal accepted')
        self.navigation_active = True

        # Get result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.navigation_result_callback)

    def navigation_result_callback(self, future):
        """Handle navigation result"""
        result = future.result().result
        status = future.result().status
        self.navigation_active = False

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Navigation succeeded')
        else:
            self.get_logger().info(f'Navigation failed with status: {status}')

    def navigation_feedback_callback(self, feedback_msg):
        """Handle navigation feedback"""
        feedback = feedback_msg.feedback
        # Log progress
        self.get_logger().info(f'Navigating... Remaining distance: {feedback.distance_remaining:.2f}m')

    def execute_waypoint_sequence(self, waypoints):
        """Execute a sequence of waypoints"""
        self.get_logger().info(f"Executing waypoint sequence with {len(waypoints)} waypoints")

        for i, waypoint in enumerate(waypoints):
            x, y, theta = waypoint
            self.get_logger().info(f"Moving to waypoint {i+1}/{len(waypoints)}: ({x}, {y})")

            # Navigate to this waypoint
            success = self.navigate_to_location(f"waypoint_{i}", x, y, theta)

            if not success:
                self.get_logger().error(f"Failed to navigate to waypoint {i}")
                return False

            # Wait for navigation to complete
            while self.navigation_active:
                rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info("Waypoint sequence completed successfully")
        return True

    def create_navigation_sequence(self, sequence_type):
        """Create predefined navigation sequences"""
        sequences = {
            "patrol_route": [
                (0.0, 0.0, 0.0),      # Start
                (2.0, 0.0, 1.57),     # Turn right
                (2.0, 2.0, 3.14),     # Turn around
                (0.0, 2.0, -1.57),    # Turn left
                (0.0, 0.0, 0.0)       # Return to start
            ],
            "room_inspection": [
                (1.0, 0.5, 0.0),      # Entry point
                (1.5, 1.0, 0.78),     # Corner 1
                (1.0, 1.5, 1.57),     # Corner 2
                (0.5, 1.0, 2.35),     # Corner 3
                (1.0, 0.5, 3.14),     # Back to entry
            ],
            "delivery_route": [
                (0.0, 0.0, 0.0),      # Base station
                (3.0, 1.0, 0.0),      # Kitchen
                (1.0, 3.0, 1.57),     # Living room
                (0.0, 0.0, 0.0)       # Return to base
            ]
        }

        if sequence_type in sequences:
            return sequences[sequence_type]
        else:
            self.get_logger().error(f"Unknown sequence type: {sequence_type}")
            return []

    def follow_path_primitive(self, path_points):
        """Primitive for following a specific path"""
        self.get_logger().info(f"Following path with {len(path_points)} points")

        # In a real implementation, this would use the follow_path action
        # For now, we'll simulate by publishing velocity commands

        for i in range(len(path_points) - 1):
            start_point = path_points[i]
            end_point = path_points[i + 1]

            # Calculate direction and distance
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            distance = (dx**2 + dy**2)**0.5

            # Normalize direction
            if distance > 0:
                dx /= distance
                dy /= distance

            # Publish velocity command
            twist = Twist()
            twist.linear.x = 0.5 * dx  # Scale by desired speed
            twist.linear.y = 0.5 * dy
            twist.angular.z = 0.0  # No rotation in this simple example

            self.cmd_vel_pub.publish(twist)

            # Sleep for duration of movement
            duration = distance / 0.5  # Assuming 0.5 m/s speed
            self.get_logger().info(f"Moving to point {i+1}, distance: {distance:.2f}m")
            time.sleep(duration)

        # Stop robot
        stop_twist = Twist()
        self.cmd_vel_pub.publish(stop_twist)
        self.get_logger().info("Path following completed")

    def compute_path_to_pose_primitive(self, start_pose, goal_pose):
        """Primitive for computing path to a pose"""
        self.get_logger().info("Computing path to pose...")

        # In a real implementation, this would use the compute_path action
        # For now, we'll return a simple straight-line path

        # Calculate straight-line path
        dx = goal_pose.pose.position.x - start_pose.pose.position.x
        dy = goal_pose.pose.position.y - start_pose.pose.position.y
        distance = (dx**2 + dy**2)**0.5

        # Create simple path (straight line)
        path_points = []
        num_steps = max(10, int(distance * 10))  # 10 points per meter

        for i in range(num_steps + 1):
            fraction = i / num_steps if num_steps > 0 else 0
            x = start_pose.pose.position.x + fraction * dx
            y = start_pose.pose.position.y + fraction * dy

            path_points.append((x, y))

        self.get_logger().info(f"Computed path with {len(path_points)} points")
        return path_points


def main(args=None):
    """Main function to run navigation action sequences"""
    rclpy.init(args=args)

    try:
        nav_sequences = NavigationActionSequences()

        # Example: Execute a patrol route
        patrol_waypoints = nav_sequences.create_navigation_sequence("patrol_route")
        nav_sequences.execute_waypoint_sequence(patrol_waypoints)

        # Spin to handle callbacks
        rclpy.spin(nav_sequences)

    except KeyboardInterrupt:
        print("Shutting down navigation action sequences...")
    finally:
        nav_sequences.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()