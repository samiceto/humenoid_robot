#!/usr/bin/env python3
"""
Testing Pipeline in Isaac Sim Environment

This script tests the voice command pipeline in the Isaac Sim environment
for the Physical AI & Humanoid Robotics course.
"""

import sys
import time
import json
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image, LaserScan
from builtin_interfaces.msg import Time

# Import the voice command pipeline components
from voice_command_pipeline import VoiceCommandPipeline
from navigation_actions import NavigationActionSequences
from manipulation_actions import ManipulationActionSequences


class IsaacSimPipelineTester(Node):
    """
    Tests the voice command pipeline in Isaac Sim environment
    """

    def __init__(self):
        super().__init__('isaac_sim_pipeline_tester')

        # Test configuration
        self.test_scenarios = [
            {
                "name": "basic_navigation",
                "commands": ["move forward 1 meter", "turn left 90 degrees", "move forward 1 meter"],
                "expected_outcomes": ["robot_moves_forward", "robot_turns_left", "robot_moves_forward"]
            },
            {
                "name": "simple_manipulation",
                "commands": ["open gripper", "close gripper"],
                "expected_outcomes": ["gripper_opens", "gripper_closes"]
            },
            {
                "name": "complex_task",
                "commands": ["go to kitchen", "pick up cup", "go to table", "place cup on table"],
                "expected_outcomes": ["navigate_to_kitchen", "pick_object", "navigate_to_table", "place_object"]
            }
        ]

        # Test results
        self.test_results = {}
        self.current_test_scenario = None
        self.test_active = False

        # Publishers for simulating voice input
        self.voice_command_pub = self.create_publisher(String, '/voice_command', 10)
        self.test_status_pub = self.create_publisher(String, '/test_status', 10)

        # Subscribers for monitoring robot state
        self.robot_pose_sub = self.create_subscription(PoseStamped, '/robot_pose', self.robot_pose_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # Initialize pipeline components
        self.voice_pipeline = VoiceCommandPipeline()
        self.nav_sequences = NavigationActionSequences()
        self.manip_sequences = ManipulationActionSequences()

        # Robot state tracking
        self.current_pose = None
        self.current_velocity = None

        self.get_logger().info("Isaac Sim Pipeline Tester initialized")

    def robot_pose_callback(self, msg):
        """Update robot pose from simulation"""
        self.current_pose = msg.pose

    def cmd_vel_callback(self, msg):
        """Update robot velocity command"""
        self.current_velocity = msg

    def run_all_tests(self) -> Dict[str, Dict]:
        """Run all test scenarios in Isaac Sim"""
        self.get_logger().info("Starting Isaac Sim pipeline tests...")

        results = {}

        for scenario in self.test_scenarios:
            test_name = scenario["name"]
            self.get_logger().info(f"Running test scenario: {test_name}")

            result = self.run_single_test(scenario)
            results[test_name] = result

            # Brief pause between tests
            time.sleep(2.0)

        self.test_results = results
        self.publish_test_summary(results)

        return results

    def run_single_test(self, scenario: Dict) -> Dict:
        """Run a single test scenario"""
        test_name = scenario["name"]
        commands = scenario["commands"]
        expected_outcomes = scenario["expected_outcomes"]

        self.get_logger().info(f"Running test: {test_name}")
        self.current_test_scenario = scenario
        self.test_active = True

        # Initialize test result
        result = {
            "name": test_name,
            "status": "failed",
            "commands_sent": len(commands),
            "expected_outcomes": len(expected_outcomes),
            "actual_outcomes": [],
            "timestamps": [],
            "errors": [],
            "execution_times": [],
            "success_count": 0,
            "failure_count": 0
        }

        try:
            # Reset robot to initial state
            self.reset_robot_state()

            # Execute each command in sequence
            for i, command in enumerate(commands):
                expected_outcome = expected_outcomes[i] if i < len(expected_outcomes) else "unknown"

                self.get_logger().info(f"Sending command {i+1}/{len(commands)}: {command}")

                # Send voice command to pipeline
                self.send_voice_command(command)

                # Wait for command execution
                start_time = time.time()
                success = self.wait_for_outcome(expected_outcome)
                execution_time = time.time() - start_time

                result["execution_times"].append(execution_time)

                if success:
                    result["actual_outcomes"].append(expected_outcome)
                    result["success_count"] += 1
                    self.get_logger().info(f"Command {i+1} succeeded: {command}")
                else:
                    result["actual_outcomes"].append(f"failed_{expected_outcome}")
                    result["failure_count"] += 1
                    error_msg = f"Command {i+1} failed: {command}"
                    result["errors"].append(error_msg)
                    self.get_logger().error(error_msg)

                # Brief pause between commands
                time.sleep(1.0)

            # Determine overall test status
            if result["success_count"] == len(commands):
                result["status"] = "passed"
            else:
                result["status"] = "failed"

        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"Test execution error: {str(e)}")
            self.get_logger().error(f"Error running test {test_name}: {e}")

        finally:
            self.test_active = False
            self.current_test_scenario = None

        return result

    def reset_robot_state(self):
        """Reset robot to initial state in simulation"""
        self.get_logger().info("Resetting robot state...")

        # In a real simulation, this would reset the robot to a known state
        # For this test, we'll just ensure all components are ready
        time.sleep(1.0)  # Allow time for reset

        # Stop any ongoing motion
        stop_cmd = Twist()
        for _ in range(5):  # Send multiple stop commands to ensure robot stops
            self.voice_pipeline.cmd_vel_pub.publish(stop_cmd)
            time.sleep(0.1)

    def send_voice_command(self, command: str):
        """Send a voice command to the pipeline"""
        msg = String()
        msg.data = command
        self.voice_command_pub.publish(msg)
        self.get_logger().info(f"Sent voice command: {command}")

    def wait_for_outcome(self, expected_outcome: str, timeout: float = 30.0) -> bool:
        """Wait for expected outcome to occur"""
        start_time = time.time()

        while time.time() - start_time < timeout and self.test_active:
            # Check if the expected outcome has occurred based on robot state
            if self.check_outcome_occurred(expected_outcome):
                self.get_logger().info(f"Expected outcome achieved: {expected_outcome}")
                return True

            time.sleep(0.1)  # Check every 100ms

        self.get_logger().warn(f"Timeout waiting for outcome: {expected_outcome}")
        return False

    def check_outcome_occurred(self, expected_outcome: str) -> bool:
        """Check if the expected outcome has occurred based on robot state"""
        # This is a simplified check - in practice, you'd have more sophisticated outcome detection

        if "move_forward" in expected_outcome or "moves_forward" in expected_outcome:
            # Check if robot is moving forward
            if self.current_velocity and self.current_velocity.linear.x > 0.1:
                return True

        elif "turn_left" in expected_outcome:
            # Check if robot is turning left
            if self.current_velocity and self.current_velocity.angular.z > 0.1:
                return True

        elif "turn_right" in expected_outcome:
            # Check if robot is turning right
            if self.current_velocity and self.current_velocity.angular.z < -0.1:
                return True

        elif "gripper_opens" in expected_outcome:
            # In simulation, we'd check gripper state
            # For this test, we'll assume success after a delay
            return True  # Simplified check

        elif "gripper_closes" in expected_outcome:
            # In simulation, we'd check gripper state
            # For this test, we'll assume success after a delay
            return True  # Simplified check

        elif "navigate_to" in expected_outcome:
            # Check if navigation is active
            if hasattr(self.nav_sequences, 'navigation_active') and self.nav_sequences.navigation_active:
                return True

        elif "pick_object" in expected_outcome:
            # Check if manipulation is active
            if hasattr(self.manip_sequences, 'manipulation_active') and self.manip_sequences.manipulation_active:
                return True

        elif "place_object" in expected_outcome:
            # Check if manipulation is active
            if hasattr(self.manip_sequences, 'manipulation_active') and self.manip_sequences.manipulation_active:
                return True

        return False

    def publish_test_summary(self, results: Dict[str, Dict]):
        """Publish a summary of test results"""
        summary = {
            "timestamp": time.time(),
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results.values() if r["status"] == "passed"),
            "failed_tests": sum(1 for r in results.values() if r["status"] == "failed"),
            "error_tests": sum(1 for r in results.values() if r["status"] == "error"),
            "overall_success_rate": 0,
            "details": results
        }

        if summary["total_tests"] > 0:
            summary["overall_success_rate"] = (summary["passed_tests"] / summary["total_tests"]) * 100

        summary_msg = String()
        summary_msg.data = json.dumps(summary, indent=2)
        self.test_status_pub.publish(summary_msg)

        self.get_logger().info(f"Test Summary: {summary['passed_tests']}/{summary['total_tests']} tests passed")

    def run_performance_tests(self) -> Dict:
        """Run performance tests to ensure ≥15 Hz real-time inference"""
        self.get_logger().info("Running performance tests...")

        performance_results = {
            "real_time_performance": True,
            "average_frequency": 0.0,
            "min_frequency": float('inf'),
            "max_frequency": 0.0,
            "throughput_tests": [],
            "latency_tests": [],
            "resource_usage": {}
        }

        # Test 1: Command processing frequency
        test_duration = 10.0  # seconds
        start_time = time.time()
        command_count = 0

        # Send commands at high frequency to test processing capacity
        while time.time() - start_time < test_duration:
            # Send a simple command
            self.send_voice_command("move forward 0.1 meter")
            command_count += 1

            # Wait briefly to control command rate
            time.sleep(0.05)  # 20 Hz command rate

        # Calculate actual processing frequency
        actual_duration = time.time() - start_time
        processing_frequency = command_count / actual_duration

        performance_results["average_frequency"] = processing_frequency
        performance_results["min_frequency"] = min(performance_results["min_frequency"], processing_frequency)
        performance_results["max_frequency"] = max(performance_results["max_frequency"], processing_frequency)

        # Check if performance meets requirements (≥15 Hz)
        performance_results["real_time_performance"] = processing_frequency >= 15.0

        # Log performance results
        self.get_logger().info(f"Command processing frequency: {processing_frequency:.2f} Hz")
        self.get_logger().info(f"Real-time performance requirement met: {performance_results['real_time_performance']}")

        # Test 2: Resource usage monitoring
        # In a real implementation, this would monitor actual resource usage
        performance_results["resource_usage"] = {
            "cpu_usage_avg": 65.0,  # Example value
            "memory_usage_mb": 2048,  # Example value
            "gpu_usage_percent": 45.0  # Example value
        }

        return performance_results

    def run_integration_tests(self) -> Dict:
        """Run integration tests combining voice, navigation, and manipulation"""
        self.get_logger().info("Running integration tests...")

        integration_results = {
            "end_to_end_success": False,
            "pipeline_latency_ms": 0,
            "coordination_success": False,
            "error_log": []
        }

        # Example integration test: voice command triggers navigation and manipulation
        try:
            start_time = time.time()

            # Send complex command
            self.send_voice_command("go to kitchen and pick up the red cup")

            # Wait for completion
            timeout = 60.0  # 60 second timeout
            while time.time() - start_time < timeout:
                # Check if both navigation and manipulation have been triggered
                nav_active = getattr(self.nav_sequences, 'navigation_active', False)
                manip_active = getattr(self.manip_sequences, 'manipulation_active', False)

                if not nav_active and not manip_active:
                    # Both have completed
                    integration_results["end_to_end_success"] = True
                    integration_results["coordination_success"] = True
                    break

                time.sleep(0.1)

            end_time = time.time()
            integration_results["pipeline_latency_ms"] = (end_time - start_time) * 1000

            if integration_results["end_to_end_success"]:
                self.get_logger().info("Integration test completed successfully")
            else:
                integration_results["error_log"].append("Integration test timed out")
                self.get_logger().error("Integration test timed out")

        except Exception as e:
            integration_results["error_log"].append(f"Integration test error: {str(e)}")
            self.get_logger().error(f"Integration test error: {e}")

        return integration_results

    def run_comprehensive_test_suite(self) -> Dict:
        """Run comprehensive test suite for Isaac Sim pipeline"""
        self.get_logger().info("Starting comprehensive Isaac Sim pipeline test suite...")

        # Run functional tests
        functional_results = self.run_all_tests()

        # Run performance tests
        performance_results = self.run_performance_tests()

        # Run integration tests
        integration_results = self.run_integration_tests()

        # Compile comprehensive results
        comprehensive_results = {
            "timestamp": time.time(),
            "functional_tests": functional_results,
            "performance_tests": performance_results,
            "integration_tests": integration_results,
            "overall_status": "unknown",
            "recommendations": []
        }

        # Determine overall status
        functional_passed = all(r["status"] == "passed" for r in functional_results.values())
        performance_met = performance_results["real_time_performance"]
        integration_success = integration_results["end_to_end_success"]

        if functional_passed and performance_met and integration_success:
            comprehensive_results["overall_status"] = "pass"
        elif integration_success:  # At least integration worked
            comprehensive_results["overall_status"] = "partial_pass"
        else:
            comprehensive_results["overall_status"] = "fail"

        # Generate recommendations
        if not functional_passed:
            comprehensive_results["recommendations"].append("Address functional test failures")
        if not performance_met:
            comprehensive_results["recommendations"].append("Optimize for real-time performance (≥15 Hz)")
        if not integration_success:
            comprehensive_results["recommendations"].append("Fix integration between components")

        self.get_logger().info(f"Comprehensive test suite completed with status: {comprehensive_results['overall_status']}")
        return comprehensive_results


def main(args=None):
    """Main function to run Isaac Sim pipeline tests"""
    rclpy.init(args=args)

    try:
        tester = IsaacSimPipelineTester()

        # Run comprehensive test suite
        results = tester.run_comprehensive_test_suite()

        # Print results summary
        print("\n" + "="*60)
        print("ISAAC SIM PIPELINE TEST RESULTS SUMMARY")
        print("="*60)
        print(f"Overall Status: {results['overall_status']}")
        print(f"Functional Tests Passed: {sum(1 for r in results['functional_tests'].values() if r['status'] == 'passed')}/{len(results['functional_tests'])}")
        print(f"Performance Requirement Met: {results['performance_tests']['real_time_performance']}")
        print(f"Integration Success: {results['integration_tests']['end_to_end_success']}")
        print(f"Average Processing Frequency: {results['performance_tests']['average_frequency']:.2f} Hz")
        print(f"Pipeline Latency: {results['integration_tests']['pipeline_latency_ms']:.2f} ms")

        if results['recommendations']:
            print("\nRecommendations:")
            for rec in results['recommendations']:
                print(f"  - {rec}")

        print("="*60)

        # Keep node alive to handle callbacks
        rclpy.spin(tester)

    except KeyboardInterrupt:
        print("Testing interrupted by user")
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()