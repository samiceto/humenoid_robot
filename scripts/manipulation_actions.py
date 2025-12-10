#!/usr/bin/env python3
"""
Manipulation Action Sequences for Humanoid Robots

This script implements manipulation action sequences for the Physical AI & Humanoid Robotics course.
It includes various manipulation behaviors and action primitives for humanoid robots.
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from std_msgs.msg import Header, Float64MultiArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from action_msgs.msg import GoalStatus

import math
import time
from typing import List, Tuple, Optional


class ManipulationActionSequences(Node):
    """
    Implements manipulation action sequences for humanoid robots
    """

    def __init__(self):
        super().__init__('manipulation_action_sequences')

        # Action clients for manipulation
        self.arm_trajectory_client = ActionClient(self, FollowJointTrajectory, 'arm_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(self, FollowJointTrajectory, 'gripper_controller/follow_joint_trajectory')

        # Publishers
        self.joint_command_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.desired_pose_pub = self.create_publisher(Pose, '/end_effector_pose', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # Robot state
        self.joint_states = {}
        self.manipulation_active = False

        # Robot configuration (example for a humanoid robot)
        self.right_arm_joints = [
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
            'right_elbow_pitch', 'right_forearm_yaw', 'right_wrist_pitch', 'right_wrist_yaw'
        ]

        self.left_arm_joints = [
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
            'left_elbow_pitch', 'left_forearm_yaw', 'left_wrist_pitch', 'left_wrist_yaw'
        ]

        self.right_gripper_joints = ['right_gripper_finger_left', 'right_gripper_finger_right']
        self.left_gripper_joints = ['left_gripper_finger_left', 'left_gripper_finger_right']

        self.get_logger().info("Manipulation Action Sequences initialized")

    def joint_state_callback(self, msg):
        """Update joint states"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_states[name] = msg.position[i]

    def create_joint_trajectory(self, joint_names: List[str], positions: List[float],
                              time_from_start: float = 5.0) -> JointTrajectory:
        """Create a joint trajectory message"""
        trajectory = JointTrajectory()
        trajectory.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = [0.0] * len(positions)  # Start with zero velocity
        point.accelerations = [0.0] * len(positions)  # Start with zero acceleration
        point.time_from_start = Duration(seconds=time_from_start).to_msg()

        trajectory.points.append(point)
        return trajectory

    def move_arm_to_pose(self, arm_side: str, target_pose: Pose) -> bool:
        """Move arm to a specific pose"""
        self.get_logger().info(f"Moving {arm_side} arm to target pose")

        # Determine which joints to use
        if arm_side.lower() == 'right':
            joint_names = self.right_arm_joints
        elif arm_side.lower() == 'left':
            joint_names = self.left_arm_joints
        else:
            self.get_logger().error(f"Invalid arm side: {arm_side}")
            return False

        # In a real implementation, this would involve inverse kinematics
        # For this example, we'll use predefined joint positions
        target_joints = self.calculate_inverse_kinematics(target_pose, arm_side)

        if target_joints is None:
            self.get_logger().error("Could not calculate inverse kinematics for target pose")
            return False

        # Create trajectory
        trajectory = self.create_joint_trajectory(joint_names, target_joints, 5.0)

        # Send trajectory goal
        return self.send_joint_trajectory_goal(trajectory, f"{arm_side}_arm_controller")

    def calculate_inverse_kinematics(self, target_pose: Pose, arm_side: str) -> Optional[List[float]]:
        """Calculate inverse kinematics for target pose (simplified)"""
        # This is a simplified IK calculation - in reality, you'd use a proper IK solver

        # Extract target position
        target_x = target_pose.position.x
        target_y = target_pose.position.y
        target_z = target_pose.position.z

        # For this example, return predefined joint angles for common poses
        # In practice, you'd use a proper IK solver like KDL, PyKDL, or MoveIt2

        # Example: Reach forward position
        if target_z > 0.5 and abs(target_x) < 0.3 and abs(target_y) < 0.3:
            # Reach forward position (approximate)
            if arm_side == 'right':
                return [0.0, 0.2, 0.0, -0.5, 0.0, 0.3, 0.0]  # Approximate joint angles
            else:  # left
                return [0.0, -0.2, 0.0, -0.5, 0.0, -0.3, 0.0]  # Approximate joint angles
        else:
            # Return home position
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def send_joint_trajectory_goal(self, trajectory: JointTrajectory, controller_name: str) -> bool:
        """Send a joint trajectory goal"""
        # Choose the appropriate action client
        if 'arm' in controller_name:
            client = self.arm_trajectory_client
        else:  # gripper
            client = self.gripper_client

        # Wait for action server
        if not client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"Action server {controller_name} not available")
            return False

        # Create goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory

        # Send goal
        goal_handle_future = client.send_goal_async(
            goal_msg,
            feedback_callback=self.trajectory_feedback_callback
        )

        goal_handle_future.add_done_callback(self.trajectory_goal_response_callback)
        return True

    def trajectory_goal_response_callback(self, future):
        """Handle trajectory goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Trajectory goal rejected')
            return

        self.get_logger().info('Trajectory goal accepted')
        self.manipulation_active = True

        # Get result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.trajectory_result_callback)

    def trajectory_result_callback(self, future):
        """Handle trajectory result"""
        result = future.result().result
        status = future.result().status
        self.manipulation_active = False

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Trajectory execution succeeded')
        else:
            self.get_logger().info(f'Trajectory execution failed with status: {status}')

    def trajectory_feedback_callback(self, feedback_msg):
        """Handle trajectory feedback"""
        # Log progress
        self.get_logger().info('Trajectory in progress...')

    def open_gripper(self, arm_side: str) -> bool:
        """Open the gripper on the specified arm"""
        self.get_logger().info(f"Opening {arm_side} gripper")

        if arm_side.lower() == 'right':
            joint_names = self.right_gripper_joints
            positions = [0.05, 0.05]  # Open position (adjust as needed)
        elif arm_side.lower() == 'left':
            joint_names = self.left_gripper_joints
            positions = [0.05, 0.05]  # Open position (adjust as needed)
        else:
            self.get_logger().error(f"Invalid arm side: {arm_side}")
            return False

        trajectory = self.create_joint_trajectory(joint_names, positions, 2.0)
        return self.send_joint_trajectory_goal(trajectory, f"{arm_side}_gripper_controller")

    def close_gripper(self, arm_side: str, force: float = 10.0) -> bool:
        """Close the gripper on the specified arm"""
        self.get_logger().info(f"Closing {arm_side} gripper with force {force}")

        if arm_side.lower() == 'right':
            joint_names = self.right_gripper_joints
            positions = [0.01, 0.01]  # Closed position (adjust as needed)
        elif arm_side.lower() == 'left':
            joint_names = self.left_gripper_joints
            positions = [0.01, 0.01]  # Closed position (adjust as needed)
        else:
            self.get_logger().error(f"Invalid arm side: {arm_side}")
            return False

        trajectory = self.create_joint_trajectory(joint_names, positions, 2.0)
        return self.send_joint_trajectory_goal(trajectory, f"{arm_side}_gripper_controller")

    def pick_object(self, arm_side: str, object_pose: Pose) -> bool:
        """Execute pick sequence for an object"""
        self.get_logger().info(f"Picking object with {arm_side} arm at position ({object_pose.position.x}, {object_pose.position.y}, {object_pose.position.z})")

        try:
            # Step 1: Move arm to approach position (slightly above object)
            approach_pose = Pose()
            approach_pose.position = Point(
                x=object_pose.position.x,
                y=object_pose.position.y,
                z=object_pose.position.z + 0.1  # 10cm above object
            )
            approach_pose.orientation = object_pose.orientation  # Preserve orientation

            self.get_logger().info("Moving to approach position...")
            success = self.move_arm_to_pose(arm_side, approach_pose)
            if not success:
                self.get_logger().error("Failed to move to approach position")
                return False

            # Wait for movement to complete
            while self.manipulation_active:
                rclpy.spin_once(self, timeout_sec=0.1)

            # Step 2: Move down to object
            self.get_logger().info("Moving down to object...")
            success = self.move_arm_to_pose(arm_side, object_pose)
            if not success:
                self.get_logger().error("Failed to move down to object")
                return False

            # Wait for movement to complete
            while self.manipulation_active:
                rclpy.spin_once(self, timeout_sec=0.1)

            # Step 3: Close gripper to grasp object
            self.get_logger().info("Closing gripper...")
            success = self.close_gripper(arm_side)
            if not success:
                self.get_logger().error("Failed to close gripper")
                return False

            # Wait for gripper to close
            time.sleep(1.0)

            # Step 4: Lift object slightly
            lift_pose = Pose()
            lift_pose.position = Point(
                x=object_pose.position.x,
                y=object_pose.position.y,
                z=object_pose.position.z + 0.1  # Lift 10cm
            )
            lift_pose.orientation = object_pose.orientation

            self.get_logger().info("Lifting object...")
            success = self.move_arm_to_pose(arm_side, lift_pose)
            if not success:
                self.get_logger().error("Failed to lift object")
                return False

            # Wait for movement to complete
            while self.manipulation_active:
                rclpy.spin_once(self, timeout_sec=0.1)

            self.get_logger().info("Pick operation completed successfully")
            return True

        except Exception as e:
            self.get_logger().error(f"Error during pick operation: {e}")
            return False

    def place_object(self, arm_side: str, target_pose: Pose) -> bool:
        """Execute place sequence for an object"""
        self.get_logger().info(f"Placing object with {arm_side} arm at position ({target_pose.position.x}, {target_pose.position.y}, {target_pose.position.z})")

        try:
            # Step 1: Move arm to above placement position
            approach_pose = Pose()
            approach_pose.position = Point(
                x=target_pose.position.x,
                y=target_pose.position.y,
                z=target_pose.position.z + 0.1  # 10cm above placement
            )
            approach_pose.orientation = target_pose.orientation  # Preserve orientation

            self.get_logger().info("Moving to placement approach position...")
            success = self.move_arm_to_pose(arm_side, approach_pose)
            if not success:
                self.get_logger().error("Failed to move to approach position")
                return False

            # Wait for movement to complete
            while self.manipulation_active:
                rclpy.spin_once(self, timeout_sec=0.1)

            # Step 2: Move down to placement position
            self.get_logger().info("Moving down to placement position...")
            success = self.move_arm_to_pose(arm_side, target_pose)
            if not success:
                self.get_logger().error("Failed to move down to placement position")
                return False

            # Wait for movement to complete
            while self.manipulation_active:
                rclpy.spin_once(self, timeout_sec=0.1)

            # Step 3: Open gripper to release object
            self.get_logger().info("Opening gripper to release object...")
            success = self.open_gripper(arm_side)
            if not success:
                self.get_logger().error("Failed to open gripper")
                return False

            # Wait for gripper to open
            time.sleep(1.0)

            # Step 4: Retract arm slightly
            retract_pose = Pose()
            retract_pose.position = Point(
                x=target_pose.position.x,
                y=target_pose.position.y,
                z=target_pose.position.z + 0.1  # Retract 10cm up
            )
            retract_pose.orientation = target_pose.orientation

            self.get_logger().info("Retracting arm...")
            success = self.move_arm_to_pose(arm_side, retract_pose)
            if not success:
                self.get_logger().error("Failed to retract arm")
                return False

            # Wait for movement to complete
            while self.manipulation_active:
                rclpy.spin_once(self, timeout_sec=0.1)

            self.get_logger().info("Place operation completed successfully")
            return True

        except Exception as e:
            self.get_logger().error(f"Error during place operation: {e}")
            return False

    def execute_manipulation_sequence(self, sequence: List[Tuple[str, any]]) -> bool:
        """Execute a sequence of manipulation actions"""
        self.get_logger().info(f"Executing manipulation sequence with {len(sequence)} actions")

        for i, (action, params) in enumerate(sequence):
            self.get_logger().info(f"Executing action {i+1}/{len(sequence)}: {action}")

            if action == "move_arm_to_pose":
                arm_side, pose = params
                success = self.move_arm_to_pose(arm_side, pose)
            elif action == "open_gripper":
                arm_side = params
                success = self.open_gripper(arm_side)
            elif action == "close_gripper":
                arm_side, force = params
                success = self.close_gripper(arm_side, force)
            elif action == "pick_object":
                arm_side, object_pose = params
                success = self.pick_object(arm_side, object_pose)
            elif action == "place_object":
                arm_side, target_pose = params
                success = self.place_object(arm_side, target_pose)
            else:
                self.get_logger().error(f"Unknown action: {action}")
                return False

            if not success:
                self.get_logger().error(f"Action {action} failed")
                return False

            # Wait for action to complete
            while self.manipulation_active:
                rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info("Manipulation sequence completed successfully")
        return True

    def create_manipulation_sequence(self, sequence_type: str) -> List[Tuple[str, any]]:
        """Create predefined manipulation sequences"""
        # Example target poses
        shelf_pose = Pose()
        shelf_pose.position = Point(x=0.5, y=0.0, z=0.8)
        shelf_pose.orientation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

        table_pose = Pose()
        table_pose.position = Point(x=0.3, y=0.2, z=0.2)
        table_pose.orientation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

        object_pose = Pose()
        object_pose.position = Point(x=0.4, y=0.1, z=0.2)
        object_pose.orientation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

        sequences = {
            "pick_and_place": [
                ("move_arm_to_pose", ("right", shelf_pose)),
                ("pick_object", ("right", object_pose)),
                ("place_object", ("right", table_pose)),
                ("move_arm_to_pose", ("right", shelf_pose))
            ],
            "gripper_test": [
                ("open_gripper", "right"),
                ("close_gripper", ("right", 10.0)),
                ("open_gripper", "right")
            ],
            "arm_sweep": [
                ("move_arm_to_pose", ("right", Pose(position=Point(x=0.3, y=-0.3, z=0.7)))),
                ("move_arm_to_pose", ("right", Pose(position=Point(x=0.3, y=0.3, z=0.7)))),
                ("move_arm_to_pose", ("right", Pose(position=Point(x=0.3, y=0.0, z=0.7))))
            ]
        }

        if sequence_type in sequences:
            return sequences[sequence_type]
        else:
            self.get_logger().error(f"Unknown sequence type: {sequence_type}")
            return []


def main(args=None):
    """Main function to run manipulation action sequences"""
    rclpy.init(args=args)

    try:
        manip_sequences = ManipulationActionSequences()

        # Example: Execute a pick and place sequence
        pick_place_sequence = manip_sequences.create_manipulation_sequence("pick_and_place")
        manip_sequences.execute_manipulation_sequence(pick_place_sequence)

        # Spin to handle callbacks
        rclpy.spin(manip_sequences)

    except KeyboardInterrupt:
        print("Shutting down manipulation action sequences...")
    finally:
        manip_sequences.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()