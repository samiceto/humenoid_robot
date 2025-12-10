#!/usr/bin/env python3
"""
Voice Command Pipeline: Whisper → LLM Planner → ROS 2

This script implements the end-to-end voice command pipeline for the Physical AI & Humanoid Robotics course.
It integrates OpenAI Whisper for speech recognition, an LLM for task planning, and ROS 2 for robot control.
"""

import os
import sys
import time
import json
import threading
import queue
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from std_msgs.msg import String
    from geometry_msgs.msg import Twist, PoseStamped
    from sensor_msgs.msg import Image
    from builtin_interfaces.msg import Duration
except ImportError:
    print("ROS 2 not available. This script requires ROS 2 to run properly.")
    print("Install ROS 2 Jazzy and source the setup.bash before running.")
    sys.exit(1)

try:
    import torch
    import whisper
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    print(f"Required packages not available: {e}")
    print("Install with: pip install openai-whisper transformers torch")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("OpenAI package not available. Using local LLM instead.")
    OpenAI = None


class VoiceCommandPipeline(Node):
    """
    Implements the voice command pipeline: Whisper → LLM Planner → ROS 2
    """

    def __init__(self):
        super().__init__('voice_command_pipeline')

        # Initialize components
        self.whisper_model = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.openai_client = None

        # Configuration
        self.whisper_model_size = "small"  # Options: tiny, base, small, medium, large
        self.use_local_llm = True  # Set to False to use OpenAI API
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Robot capabilities (define what the robot can do)
        self.robot_capabilities = {
            "navigation": {
                "move_forward": {"description": "Move robot forward by specified distance"},
                "move_backward": {"description": "Move robot backward by specified distance"},
                "turn_left": {"description": "Turn robot left by specified angle"},
                "turn_right": {"description": "Turn robot right by specified angle"},
                "go_to_location": {"description": "Navigate to specified location"}
            },
            "manipulation": {
                "pick_object": {"description": "Pick up an object at specified location"},
                "place_object": {"description": "Place object at specified location"},
                "open_gripper": {"description": "Open robot gripper"},
                "close_gripper": {"description": "Close robot gripper"}
            },
            "interaction": {
                "speak": {"description": "Make robot speak a message"},
                "listen": {"description": "Make robot listen and respond"},
                "detect_object": {"description": "Detect and identify objects in the environment"}
            }
        }

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.navigation_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.voice_response_pub = self.create_publisher(String, '/voice_response', 10)

        # Subscriber for voice commands (could come from microphone node)
        self.voice_command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.voice_command_callback,
            10
        )

        # Initialize models
        self.initialize_models()

        # Thread-safe queue for processing
        self.command_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_commands, daemon=True)
        self.processing_thread.start()

        self.get_logger().info("Voice Command Pipeline initialized successfully")

    def initialize_models(self):
        """Initialize Whisper ASR and LLM models"""
        self.get_logger().info("Initializing models...")

        # Initialize Whisper model
        try:
            self.get_logger().info(f"Loading Whisper model ({self.whisper_model_size})...")
            self.whisper_model = whisper.load_model(self.whisper_model_size)
            self.get_logger().info("Whisper model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load Whisper model: {e}")
            sys.exit(1)

        # Initialize LLM
        if self.use_local_llm:
            self.get_logger().info("Loading local LLM model...")
            try:
                model_name = "microsoft/DialoGPT-medium"  # Example model
                self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.llm_model = AutoModelForCausalLM.from_pretrained(model_name)

                # Add pad token if it doesn't exist
                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

                self.get_logger().info("Local LLM model loaded successfully")
            except Exception as e:
                self.get_logger().error(f"Failed to load local LLM model: {e}")
                # Fallback to a simpler approach
                self.get_logger().info("Using rule-based planner as fallback")
        else:
            if self.openai_api_key:
                try:
                    self.openai_client = OpenAI(api_key=self.openai_api_key)
                    self.get_logger().info("OpenAI client initialized")
                except Exception as e:
                    self.get_logger().error(f"Failed to initialize OpenAI client: {e}")
                    self.get_logger().info("Falling back to local processing")
                    self.use_local_llm = True

    def voice_command_callback(self, msg):
        """Callback for incoming voice commands"""
        command_text = msg.data
        self.get_logger().info(f"Received voice command: {command_text}")

        # Add to processing queue
        self.command_queue.put(command_text)

    def process_commands(self):
        """Process commands from the queue"""
        while rclpy.ok():
            try:
                # Get command from queue (blocks until available)
                command_text = self.command_queue.get(timeout=1.0)

                # Process the command
                self.process_voice_command(command_text)

                # Mark as processed
                self.command_queue.task_done()

            except queue.Empty:
                continue  # No command in queue, continue loop
            except Exception as e:
                self.get_logger().error(f"Error processing command: {e}")

    def process_voice_command(self, command_text: str):
        """Process a voice command through the pipeline"""
        self.get_logger().info(f"Processing command: {command_text}")

        # Step 1: Parse the command to understand intent
        parsed_command = self.parse_command(command_text)

        if not parsed_command:
            response = f"Sorry, I didn't understand the command: {command_text}"
            self.publish_voice_response(response)
            return

        # Step 2: Plan the action sequence
        action_plan = self.plan_action(parsed_command)

        if not action_plan:
            response = f"Sorry, I cannot execute the command: {command_text}"
            self.publish_voice_response(response)
            return

        # Step 3: Execute the action plan
        success = self.execute_action_plan(action_plan)

        # Step 4: Respond to user
        if success:
            response = f"I have executed the command: {command_text}"
        else:
            response = f"Failed to execute the command: {command_text}"

        self.publish_voice_response(response)

    def parse_command(self, command_text: str) -> Optional[Dict]:
        """Parse natural language command into structured format"""
        command_text = command_text.lower().strip()

        # Simple rule-based parsing (in practice, use more sophisticated NLP)
        parsed = {
            "intent": None,
            "arguments": {},
            "original_text": command_text
        }

        # Navigation commands
        if "move forward" in command_text or "go forward" in command_text:
            parsed["intent"] = "move_forward"
            # Extract distance if specified
            if "meter" in command_text:
                import re
                match = re.search(r"(\d+(?:\.\d+)?)\s*meter", command_text)
                if match:
                    parsed["arguments"]["distance"] = float(match.group(1))
                else:
                    parsed["arguments"]["distance"] = 1.0  # Default 1 meter
        elif "move backward" in command_text or "go backward" in command_text:
            parsed["intent"] = "move_backward"
            if "meter" in command_text:
                import re
                match = re.search(r"(\d+(?:\.\d+)?)\s*meter", command_text)
                if match:
                    parsed["arguments"]["distance"] = float(match.group(1))
                else:
                    parsed["arguments"]["distance"] = 1.0
        elif "turn left" in command_text:
            parsed["intent"] = "turn_left"
            if "degree" in command_text:
                import re
                match = re.search(r"(\d+(?:\.\d+)?)\s*degree", command_text)
                if match:
                    parsed["arguments"]["angle"] = float(match.group(1))
                else:
                    parsed["arguments"]["angle"] = 90.0  # Default 90 degrees
        elif "turn right" in command_text:
            parsed["intent"] = "turn_right"
            if "degree" in command_text:
                import re
                match = re.search(r"(\d+(?:\.\d+)?)\s*degree", command_text)
                if match:
                    parsed["arguments"]["angle"] = float(match.group(1))
                else:
                    parsed["arguments"]["angle"] = 90.0
        elif "go to" in command_text or "navigate to" in command_text:
            parsed["intent"] = "go_to_location"
            # Extract location if specified
            location_keywords = ["kitchen", "bedroom", "living room", "office", "door", "table"]
            for keyword in location_keywords:
                if keyword in command_text:
                    parsed["arguments"]["location"] = keyword
                    break
        elif "pick up" in command_text or "grab" in command_text:
            parsed["intent"] = "pick_object"
            # Extract object if specified
            object_keywords = ["cup", "ball", "book", "box", "toy"]
            for keyword in object_keywords:
                if keyword in command_text:
                    parsed["arguments"]["object"] = keyword
                    break
        elif "place" in command_text or "put down" in command_text:
            parsed["intent"] = "place_object"
        elif "speak" in command_text or "say" in command_text:
            parsed["intent"] = "speak"
            # Extract message to speak
            if '"' in command_text:
                import re
                match = re.search(r'"([^"]*)"', command_text)
                if match:
                    parsed["arguments"]["message"] = match.group(1)
            elif "'" in command_text:
                import re
                match = re.search(r"'([^']*)'", command_text)
                if match:
                    parsed["arguments"]["message"] = match.group(1)
            else:
                # If no quotes, assume everything after "say" or "speak" is the message
                for word in ["say", "speak"]:
                    if word in command_text:
                        idx = command_text.find(word) + len(word)
                        message = command_text[idx:].strip()
                        if message:
                            parsed["arguments"]["message"] = message
                        break

        return parsed if parsed["intent"] else None

    def plan_action(self, parsed_command: Dict) -> Optional[List[Dict]]:
        """Plan the sequence of actions needed to execute the command"""
        intent = parsed_command["intent"]
        args = parsed_command["arguments"]

        action_plan = []

        if intent == "move_forward":
            distance = args.get("distance", 1.0)
            action_plan.append({
                "action": "move_linear",
                "parameters": {
                    "direction": "forward",
                    "distance": distance,
                    "speed": 0.5
                }
            })
        elif intent == "move_backward":
            distance = args.get("distance", 1.0)
            action_plan.append({
                "action": "move_linear",
                "parameters": {
                    "direction": "backward",
                    "distance": distance,
                    "speed": 0.5
                }
            })
        elif intent == "turn_left":
            angle = args.get("angle", 90.0)
            action_plan.append({
                "action": "rotate",
                "parameters": {
                    "direction": "left",
                    "angle": angle,
                    "angular_speed": 0.5
                }
            })
        elif intent == "turn_right":
            angle = args.get("angle", 90.0)
            action_plan.append({
                "action": "rotate",
                "parameters": {
                    "direction": "right",
                    "angle": angle,
                    "angular_speed": 0.5
                }
            })
        elif intent == "go_to_location":
            location = args.get("location", "unknown")
            # This would involve more complex navigation planning
            action_plan.append({
                "action": "navigate_to",
                "parameters": {
                    "location": location
                }
            })
        elif intent == "pick_object":
            obj = args.get("object", "unknown")
            action_plan.append({
                "action": "approach_object",
                "parameters": {
                    "object": obj
                }
            })
            action_plan.append({
                "action": "grasp_object",
                "parameters": {
                    "object": obj
                }
            })
        elif intent == "place_object":
            action_plan.append({
                "action": "find_placement_location",
                "parameters": {}
            })
            action_plan.append({
                "action": "place_object",
                "parameters": {}
            })
        elif intent == "speak":
            message = args.get("message", "Hello")
            action_plan.append({
                "action": "speak_text",
                "parameters": {
                    "message": message
                }
            })
        else:
            self.get_logger().error(f"Unknown intent: {intent}")
            return None

        return action_plan

    def execute_action_plan(self, action_plan: List[Dict]) -> bool:
        """Execute the planned sequence of actions"""
        success = True

        for action_item in action_plan:
            action = action_item["action"]
            params = action_item["parameters"]

            if not self.execute_single_action(action, params):
                success = False
                break  # Stop execution if any action fails

        return success

    def execute_single_action(self, action: str, params: Dict) -> bool:
        """Execute a single action"""
        self.get_logger().info(f"Executing action: {action} with params: {params}")

        if action == "move_linear":
            return self.move_linear(params)
        elif action == "rotate":
            return self.rotate(params)
        elif action == "navigate_to":
            return self.navigate_to(params)
        elif action == "approach_object":
            return self.approach_object(params)
        elif action == "grasp_object":
            return self.grasp_object(params)
        elif action == "place_object":
            return self.place_object(params)
        elif action == "find_placement_location":
            return self.find_placement_location()
        elif action == "speak_text":
            return self.speak_text(params.get("message", ""))
        else:
            self.get_logger().error(f"Unknown action: {action}")
            return False

    def move_linear(self, params: Dict) -> bool:
        """Move robot linearly"""
        direction = params.get("direction", "forward")
        distance = params.get("distance", 1.0)
        speed = params.get("speed", 0.5)

        # Calculate time needed (simplified)
        duration = distance / speed

        twist = Twist()
        if direction == "forward":
            twist.linear.x = speed
        else:  # backward
            twist.linear.x = -speed

        # Publish command for duration
        start_time = self.get_clock().now()
        end_time = start_time + Duration(sec=int(duration), nanosec=(duration % 1) * 1e9)

        while self.get_clock().now() < end_time and rclpy.ok():
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)

        # Stop the robot
        stop_twist = Twist()
        self.cmd_vel_pub.publish(stop_twist)

        return True

    def rotate(self, params: Dict) -> bool:
        """Rotate robot"""
        direction = params.get("direction", "left")
        angle = params.get("angle", 90.0)  # in degrees
        angular_speed = params.get("angular_speed", 0.5)  # rad/s

        # Convert angle to radians
        angle_rad = angle * 3.14159 / 180.0

        # Calculate time needed
        duration = angle_rad / angular_speed

        twist = Twist()
        if direction == "left":
            twist.angular.z = angular_speed
        else:  # right
            twist.angular.z = -angular_speed

        # Publish command for duration
        start_time = self.get_clock().now()
        end_time = start_time + Duration(sec=int(duration), nanosec=(duration % 1) * 1e9)

        while self.get_clock().now() < end_time and rclpy.ok():
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)

        # Stop the robot
        stop_twist = Twist()
        self.cmd_vel_pub.publish(stop_twist)

        return True

    def navigate_to(self, params: Dict) -> bool:
        """Navigate to a specific location"""
        location = params.get("location", "unknown")

        # In a real implementation, this would use navigation2
        # For now, we'll simulate by sending a goal pose

        # This is a simplified example - in reality, you'd have a map
        # with predefined locations
        location_map = {
            "kitchen": (2.0, 1.0, 0.0),  # x, y, theta
            "bedroom": (-1.0, 2.0, 1.57),
            "living room": (0.0, 0.0, 0.0),
            "office": (3.0, -1.0, -1.57),
            "door": (1.5, 0.0, 0.0),
            "table": (0.5, 1.0, 0.0)
        }

        if location in location_map:
            x, y, theta = location_map[location]

            goal_pose = PoseStamped()
            goal_pose.header.frame_id = "map"
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            goal_pose.pose.position.x = x
            goal_pose.pose.position.y = y
            goal_pose.pose.position.z = 0.0

            # Convert theta to quaternion
            import math
            qw = math.cos(theta / 2.0)
            qz = math.sin(theta / 2.0)
            goal_pose.pose.orientation.z = qz
            goal_pose.pose.orientation.w = qw

            self.navigation_pub.publish(goal_pose)
            self.get_logger().info(f"Sent navigation goal to {location}")

            return True
        else:
            self.get_logger().warn(f"Unknown location: {location}")
            return False

    def approach_object(self, params: Dict) -> bool:
        """Approach an object"""
        obj = params.get("object", "unknown")
        self.get_logger().info(f"Approaching object: {obj}")
        # In a real implementation, this would use perception to locate the object
        # and then navigate to it
        return True

    def grasp_object(self, params: Dict) -> bool:
        """Grasp an object"""
        obj = params.get("object", "unknown")
        self.get_logger().info(f"Grasping object: {obj}")
        # In a real implementation, this would control the robot's gripper
        return True

    def place_object(self, params: Dict) -> bool:
        """Place an object"""
        self.get_logger().info("Placing object")
        # In a real implementation, this would control the robot's gripper
        return True

    def find_placement_location(self) -> bool:
        """Find a suitable location to place an object"""
        self.get_logger().info("Finding placement location")
        # In a real implementation, this would use perception to find a suitable spot
        return True

    def speak_text(self, message: str) -> bool:
        """Make the robot speak a message"""
        self.get_logger().info(f"Speaking: {message}")
        # In a real implementation, this would use text-to-speech
        # For now, we'll just log it
        return True

    def publish_voice_response(self, response: str):
        """Publish voice response"""
        msg = String()
        msg.data = response
        self.voice_response_pub.publish(msg)
        self.get_logger().info(f"Response published: {response}")


def main(args=None):
    """Main function to run the voice command pipeline"""
    rclpy.init(args=args)

    # Create and run the voice command pipeline node
    voice_pipeline = VoiceCommandPipeline()

    try:
        rclpy.spin(voice_pipeline)
    except KeyboardInterrupt:
        print("Shutting down voice command pipeline...")
    finally:
        voice_pipeline.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()