#!/usr/bin/env python3
"""
Combined Navigation and Manipulation Action Sequences for Humanoid Robots

This script implements combined navigation and manipulation action sequences for the Physical AI & Humanoid Robotics course.
It demonstrates how to coordinate navigation and manipulation behaviors for complex tasks.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import math

from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

import time
import threading
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


class TaskStatus(Enum):
    """Enumeration for task status"""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CombinedActionSequences(Node):
    """
    Implements combined navigation and manipulation action sequences for humanoid robots
    """

    def __init__(self):
        super().__init__('combined_action_sequences')

        # Initialize navigation and manipulation components
        self.initialize_components()

        # Task management
        self.current_task_status = TaskStatus.IDLE
        self.task_cancel_requested = False

        # Publishers and subscribers
        self.status_pub = self.create_publisher(Header, '/task_status', 10)

        self.get_logger().info("Combined Action Sequences initialized")

    def initialize_components(self):
        """Initialize navigation and manipulation components"""
        # For this example, we'll create mock interfaces
        # In a real implementation, these would be proper ROS 2 interfaces
        self.navigation_component = MockNavigationComponent(self)
        self.manipulation_component = MockManipulationComponent(self)

    def execute_delivery_task(self, start_location: Tuple[float, float, float],
                             pickup_location: Tuple[float, float, float],
                             delivery_location: Tuple[float, float, float],
                             object_info: Dict[str, Any]) -> bool:
        """
        Execute a complete delivery task: navigate to pickup location,
        pick up object, navigate to delivery location, place object
        """
        self.get_logger().info("Starting delivery task...")

        try:
            # Step 1: Navigate to pickup location
            self.get_logger().info("Step 1: Navigating to pickup location")
            self.publish_task_status(TaskStatus.RUNNING)

            success = self.navigation_component.navigate_to_location(
                "pickup_area", pickup_location[0], pickup_location[1], pickup_location[2]
            )

            if not success:
                self.get_logger().error("Failed to navigate to pickup location")
                self.publish_task_status(TaskStatus.FAILED)
                return False

            # Wait for navigation to complete
            while self.navigation_component.is_active():
                if self.task_cancel_requested:
                    self.get_logger().info("Task cancelled during navigation to pickup")
                    self.publish_task_status(TaskStatus.CANCELLED)
                    return False
                time.sleep(0.1)

            # Step 2: Pick up object
            self.get_logger().info("Step 2: Picking up object")

            # Create target pose for object pickup
            pickup_pose = self.create_pose_from_location(pickup_location, object_info.get('orientation', (0, 0, 0, 1)))

            success = self.manipulation_component.pick_object(
                object_info.get('arm', 'right'), pickup_pose
            )

            if not success:
                self.get_logger().error("Failed to pick up object")
                self.publish_task_status(TaskStatus.FAILED)
                return False

            # Wait for manipulation to complete
            while self.manipulation_component.is_active():
                if self.task_cancel_requested:
                    self.get_logger().info("Task cancelled during pickup")
                    self.publish_task_status(TaskStatus.CANCELLED)
                    return False
                time.sleep(0.1)

            # Step 3: Navigate to delivery location
            self.get_logger().info("Step 3: Navigating to delivery location")

            success = self.navigation_component.navigate_to_location(
                "delivery_area", delivery_location[0], delivery_location[1], delivery_location[2]
            )

            if not success:
                self.get_logger().error("Failed to navigate to delivery location")
                self.publish_task_status(TaskStatus.FAILED)
                return False

            # Wait for navigation to complete
            while self.navigation_component.is_active():
                if self.task_cancel_requested:
                    self.get_logger().info("Task cancelled during navigation to delivery")
                    self.publish_task_status(TaskStatus.CANCELLED)
                    return False
                time.sleep(0.1)

            # Step 4: Place object
            self.get_logger().info("Step 4: Placing object")

            # Create target pose for object placement
            delivery_pose = self.create_pose_from_location(delivery_location, object_info.get('orientation', (0, 0, 0, 1)))

            success = self.manipulation_component.place_object(
                object_info.get('arm', 'right'), delivery_pose
            )

            if not success:
                self.get_logger().error("Failed to place object")
                self.publish_task_status(TaskStatus.FAILED)
                return False

            # Wait for manipulation to complete
            while self.manipulation_component.is_active():
                if self.task_cancel_requested:
                    self.get_logger().info("Task cancelled during placement")
                    self.publish_task_status(TaskStatus.CANCELLED)
                    return False
                time.sleep(0.1)

            self.get_logger().info("Delivery task completed successfully!")
            self.publish_task_status(TaskStatus.SUCCESS)
            return True

        except Exception as e:
            self.get_logger().error(f"Error during delivery task: {e}")
            self.publish_task_status(TaskStatus.FAILED)
            return False

    def execute_inspection_task(self, waypoints: List[Tuple[float, float, float]],
                               inspection_actions: List[Dict[str, Any]]) -> bool:
        """
        Execute an inspection task: visit waypoints and perform actions at each
        """
        self.get_logger().info(f"Starting inspection task with {len(waypoints)} waypoints")

        try:
            for i, waypoint in enumerate(waypoints):
                x, y, theta = waypoint
                self.get_logger().info(f"Visiting waypoint {i+1}/{len(waypoints)}: ({x}, {y}, {theta})")

                # Navigate to waypoint
                success = self.navigation_component.navigate_to_location(
                    f"waypoint_{i}", x, y, theta
                )

                if not success:
                    self.get_logger().error(f"Failed to navigate to waypoint {i+1}")
                    return False

                # Wait for navigation to complete
                while self.navigation_component.is_active():
                    if self.task_cancel_requested:
                        self.get_logger().info(f"Task cancelled during navigation to waypoint {i+1}")
                        return False
                    time.sleep(0.1)

                # Perform inspection actions at this waypoint
                for action in inspection_actions:
                    action_type = action.get('type', 'none')

                    if action_type == 'look_around':
                        # Perform look-around action (could involve head/pan-tilt)
                        self.look_around()
                    elif action_type == 'take_photo':
                        # Take photo at current location
                        self.take_photo()
                    elif action_type == 'scan_environment':
                        # Scan environment with sensors
                        self.scan_environment()
                    elif action_type == 'collect_sample':
                        # Collect sample if manipulator available
                        if 'arm' in action:
                            # Move arm to sample collection pose
                            sample_pose = action.get('pose', Pose())
                            success = self.manipulation_component.move_arm_to_pose(
                                action['arm'], sample_pose
                            )

                            if not success:
                                self.get_logger().error(f"Failed to collect sample at waypoint {i+1}")
                                return False

                            # Wait for action to complete
                            while self.manipulation_component.is_active():
                                if self.task_cancel_requested:
                                    return False
                                time.sleep(0.1)

            self.get_logger().info("Inspection task completed successfully!")
            return True

        except Exception as e:
            self.get_logger().error(f"Error during inspection task: {e}")
            return False

    def execute_cleaning_task(self, rooms_to_clean: List[str], cleaning_pattern: str = "grid") -> bool:
        """
        Execute a cleaning task: navigate through rooms and perform cleaning actions
        """
        self.get_logger().info(f"Starting cleaning task for rooms: {rooms_to_clean}")

        try:
            # Define cleaning paths for each room
            room_paths = self.generate_cleaning_paths(rooms_to_clean, cleaning_pattern)

            for room_idx, room in enumerate(rooms_to_clean):
                self.get_logger().info(f"Cleaning room {room_idx+1}/{len(rooms_to_clean)}: {room}")

                # Get waypoints for this room
                room_waypoints = room_paths.get(room, [])

                for waypoint_idx, waypoint in enumerate(room_waypoints):
                    x, y, theta = waypoint
                    self.get_logger().info(f"Moving to cleaning point {waypoint_idx+1}/{len(room_waypoints)} in {room}")

                    # Navigate to cleaning point
                    success = self.navigation_component.navigate_to_location(
                        f"{room}_cleaning_point_{waypoint_idx}", x, y, theta
                    )

                    if not success:
                        self.get_logger().error(f"Failed to navigate to cleaning point in {room}")
                        return False

                    # Wait for navigation to complete
                    while self.navigation_component.is_active():
                        if self.task_cancel_requested:
                            self.get_logger().info(f"Task cancelled during cleaning of {room}")
                            return False
                        time.sleep(0.1)

                    # Perform cleaning action
                    self.perform_cleaning_action()

                    # Wait briefly between cleaning actions
                    time.sleep(0.5)

            self.get_logger().info("Cleaning task completed successfully!")
            return True

        except Exception as e:
            self.get_logger().error(f"Error during cleaning task: {e}")
            return False

    def generate_cleaning_paths(self, rooms: List[str], pattern: str) -> Dict[str, List[Tuple[float, float, float]]]:
        """
        Generate cleaning paths for specified rooms based on cleaning pattern
        """
        room_paths = {}

        # Define example room layouts and cleaning paths
        room_layouts = {
            "kitchen": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.57), (0.0, 1.0, 3.14)],
            "living_room": [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.5, 1.57), (0.0, 1.5, 3.14)],
            "bedroom": [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (1.5, 1.0, 1.57), (0.0, 1.0, 3.14)]
        }

        for room in rooms:
            if room in room_layouts:
                if pattern == "grid":
                    # For grid pattern, return the room layout as waypoints
                    room_paths[room] = room_layouts[room]
                elif pattern == "spiral":
                    # For spiral pattern, generate spiral path
                    room_paths[room] = self.generate_spiral_path(room_layouts[room])
                else:
                    # Default to room layout
                    room_paths[room] = room_layouts[room]
            else:
                # If room not defined, create a simple rectangular path
                room_paths[room] = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.57), (0.0, 1.0, 3.14)]

        return room_paths

    def generate_spiral_path(self, room_layout: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Generate a spiral cleaning path for a room
        """
        # Simplified spiral generation
        # In practice, this would be more sophisticated
        spiral_points = []

        # Get bounding box of room
        xs = [point[0] for point in room_layout]
        ys = [point[1] for point in room_layout]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Generate spiral points within the room bounds
        step_size = 0.2  # 20cm steps
        x, y = (min_x + max_x) / 2, (min_y + max_y) / 2  # Start in center
        radius = 0.0
        angle = 0.0

        while radius < max(max_x - min_x, max_y - min_y) / 2:
            # Calculate next point in spiral
            new_x = x + radius * math.cos(angle)
            new_y = y + radius * math.sin(angle)

            # Keep within room bounds
            new_x = max(min_x + 0.1, min(max_x - 0.1, new_x))
            new_y = max(min_y + 0.1, min(max_y - 0.1, new_y))

            spiral_points.append((new_x, new_y, angle))

            # Increment for next point
            angle += 0.5  # 0.5 radian steps
            radius += step_size * 0.1  # Grow radius slowly

            if len(spiral_points) > 50:  # Prevent infinite loop
                break

        return spiral_points

    def create_pose_from_location(self, location: Tuple[float, float, float], orientation: Tuple[float, float, float, float]) -> Pose:
        """Create a Pose object from location and orientation tuples"""
        pose = Pose()
        pose.position = Point(x=location[0], y=location[1], z=location[2])
        pose.orientation = Quaternion(x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3])
        return pose

    def look_around(self):
        """Perform look-around action (simplified)"""
        self.get_logger().info("Looking around...")
        # In a real implementation, this would move head/pan-tilt cameras
        time.sleep(1.0)

    def take_photo(self):
        """Take photo action (simplified)"""
        self.get_logger().info("Taking photo...")
        # In a real implementation, this would trigger camera
        time.sleep(0.5)

    def scan_environment(self):
        """Scan environment action (simplified)"""
        self.get_logger().info("Scanning environment...")
        # In a real implementation, this would use LIDAR, cameras, etc.
        time.sleep(1.0)

    def perform_cleaning_action(self):
        """Perform cleaning action (simplified)"""
        self.get_logger().info("Performing cleaning action...")
        # In a real implementation, this would activate cleaning mechanisms
        time.sleep(0.5)

    def cancel_current_task(self):
        """Cancel the currently executing task"""
        self.task_cancel_requested = True
        self.get_logger().info("Task cancellation requested")

    def publish_task_status(self, status: TaskStatus):
        """Publish current task status"""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = f"task_status_{status.value}"
        self.status_pub.publish(header)

    def reset_task_state(self):
        """Reset task state for next execution"""
        self.task_cancel_requested = False
        self.current_task_status = TaskStatus.IDLE


class MockNavigationComponent:
    """Mock navigation component for demonstration purposes"""

    def __init__(self, node):
        self.node = node
        self.active = False

    def navigate_to_location(self, location_name: str, x: float, y: float, theta: float) -> bool:
        """Mock navigation to location"""
        self.node.get_logger().info(f"Mock navigating to {location_name} at ({x}, {y}, {theta})")
        self.active = True

        # Simulate navigation in a separate thread
        thread = threading.Thread(target=self._simulate_navigation, args=(x, y, theta))
        thread.daemon = True
        thread.start()

        return True

    def _simulate_navigation(self, x: float, y: float, theta: float):
        """Simulate navigation process"""
        time.sleep(2.0)  # Simulate 2 seconds of navigation
        self.active = False
        self.node.get_logger().info("Mock navigation completed")

    def is_active(self) -> bool:
        """Check if navigation is currently active"""
        return self.active


class MockManipulationComponent:
    """Mock manipulation component for demonstration purposes"""

    def __init__(self, node):
        self.node = node
        self.active = False

    def pick_object(self, arm_side: str, object_pose: Pose) -> bool:
        """Mock pick object"""
        self.node.get_logger().info(f"Mock picking object with {arm_side} arm")
        self.active = True

        # Simulate pick in a separate thread
        thread = threading.Thread(target=self._simulate_pick)
        thread.daemon = True
        thread.start()

        return True

    def place_object(self, arm_side: str, target_pose: Pose) -> bool:
        """Mock place object"""
        self.node.get_logger().info(f"Mock placing object with {arm_side} arm")
        self.active = True

        # Simulate place in a separate thread
        thread = threading.Thread(target=self._simulate_place)
        thread.daemon = True
        thread.start()

        return True

    def move_arm_to_pose(self, arm_side: str, target_pose: Pose) -> bool:
        """Mock move arm to pose"""
        self.node.get_logger().info(f"Mock moving {arm_side} arm to pose")
        self.active = True

        # Simulate movement in a separate thread
        thread = threading.Thread(target=self._simulate_move_arm)
        thread.daemon = True
        thread.start()

        return True

    def _simulate_pick(self):
        """Simulate pick process"""
        time.sleep(3.0)  # Simulate 3 seconds of pick operation
        self.active = False
        self.node.get_logger().info("Mock pick completed")

    def _simulate_place(self):
        """Simulate place process"""
        time.sleep(2.5)  # Simulate 2.5 seconds of place operation
        self.active = False
        self.node.get_logger().info("Mock place completed")

    def _simulate_move_arm(self):
        """Simulate arm movement"""
        time.sleep(2.0)  # Simulate 2 seconds of arm movement
        self.active = False
        self.node.get_logger().info("Mock arm movement completed")

    def is_active(self) -> bool:
        """Check if manipulation is currently active"""
        return self.active


def main(args=None):
    """Main function to run combined action sequences"""
    rclpy.init(args=args)

    try:
        combined_sequences = CombinedActionSequences()

        # Example: Execute a delivery task
        start_loc = (0.0, 0.0, 0.0)
        pickup_loc = (2.0, 1.0, 0.0)
        delivery_loc = (-1.0, 2.0, 0.0)

        object_info = {
            'arm': 'right',
            'name': 'item_box',
            'orientation': (0.0, 0.0, 0.0, 1.0)
        }

        success = combined_sequences.execute_delivery_task(start_loc, pickup_loc, delivery_loc, object_info)

        if success:
            combined_sequences.get_logger().info("Delivery task executed successfully!")
        else:
            combined_sequences.get_logger().error("Delivery task failed!")

        # Example: Execute an inspection task
        waypoints = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 1.57),
            (1.0, 1.0, 3.14),
            (0.0, 1.0, -1.57)
        ]

        inspection_actions = [
            {'type': 'look_around'},
            {'type': 'take_photo'},
            {'type': 'scan_environment'}
        ]

        success = combined_sequences.execute_inspection_task(waypoints, inspection_actions)

        if success:
            combined_sequences.get_logger().info("Inspection task executed successfully!")
        else:
            combined_sequences.get_logger().error("Inspection task failed!")

        # Spin to handle callbacks
        rclpy.spin(combined_sequences)

    except KeyboardInterrupt:
        print("Shutting down combined action sequences...")
    finally:
        combined_sequences.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()