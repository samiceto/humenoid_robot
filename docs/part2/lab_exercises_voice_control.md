---
sidebar_position: 6
title: "Lab Exercises: Voice-Controlled Systems"
description: "Hands-on lab exercises for implementing and testing voice-controlled humanoid robot systems"
---

# Lab Exercises: Voice-Controlled Systems

import ChapterIntro from '@site/src/components/ChapterIntro';
import RoboticsBlock from '@site/src/components/RoboticsBlock';
import HardwareSpec from '@site/src/components/HardwareSpec';
import ROSCommand from '@site/src/components/ROSCommand';
import SimulationEnv from '@site/src/components/SimulationEnv';

<ChapterIntro
  title="Lab Exercises: Voice-Controlled Systems"
  subtitle="Practical exercises for implementing voice-controlled humanoid robot systems"
  objectives={[
    "Implement voice command processing pipeline",
    "Test speech recognition in various acoustic conditions",
    "Validate natural language understanding accuracy",
    "Deploy voice-controlled behaviors on simulated and physical robots"
  ]}
/>

## Overview

This lab module provides hands-on exercises for implementing and testing voice-controlled systems for humanoid robots. Students will learn to integrate speech recognition, natural language understanding, and robot control to create responsive voice-controlled robot behaviors.

## Learning Objectives

After completing these exercises, students will be able to:
- Implement a complete voice command processing pipeline
- Test and evaluate speech recognition accuracy in different environments
- Design and validate natural language understanding systems
- Deploy voice-controlled behaviors on both simulated and physical robots
- Optimize system performance for real-time operation (≥15 Hz)

## Prerequisites

Before starting these exercises, students should have:
- Completed Chapters 4-6 (Isaac Sim, Advanced Simulation, Sim-to-Reality Transfer)
- Working knowledge of ROS 2 and basic Python programming
- Access to Isaac Sim environment or physical humanoid robot
- Understanding of basic robotics concepts (navigation, manipulation)

## Lab Exercise 1: Basic Voice Command Pipeline

### Exercise 1.1: Implement Speech Recognition Module

**Objective**: Create a basic speech recognition module using OpenAI Whisper

**Time Estimate**: 2-3 hours

**Tasks**:
1. Install Whisper and dependencies
2. Create a Python script that captures audio from microphone
3. Implement real-time speech recognition using Whisper
4. Test with various audio inputs

**Implementation Steps**:

```python
# speech_recognizer.py
import whisper
import pyaudio
import wave
import numpy as np
import threading
import queue

class SpeechRecognizer:
    def __init__(self, model_size="small"):
        """Initialize speech recognizer with Whisper model"""
        self.model = whisper.load_model(model_size)
        self.audio_queue = queue.Queue()

        # Audio configuration
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.record_seconds = 5

        self.audio = pyaudio.PyAudio()

    def record_audio(self):
        """Record audio from microphone"""
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        print("Recording...")
        frames = []

        for i in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)

        print("Finished recording")

        stream.stop_stream()
        stream.close()

        # Save to WAV file
        wf = wave.open("temp_audio.wav", 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        return "temp_audio.wav"

    def transcribe_audio(self, audio_file):
        """Transcribe recorded audio"""
        result = self.model.transcribe(audio_file)
        return result["text"]

    def continuous_recognition(self, callback_func):
        """Continuously listen and recognize speech"""
        def audio_thread():
            while True:
                audio_file = self.record_audio()
                text = self.transcribe_audio(audio_file)

                if text.strip():  # Only call back if text is recognized
                    callback_func(text)

                # Clean up temp file
                import os
                if os.path.exists(audio_file):
                    os.remove(audio_file)

        thread = threading.Thread(target=audio_thread, daemon=True)
        thread.start()
        return thread

# Example usage
def command_callback(transcribed_text):
    print(f"Recognized: {transcribed_text}")

recognizer = SpeechRecognizer()
thread = recognizer.continuous_recognition(command_callback)

# Keep main thread alive
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Stopping...")
```

**Testing**:
1. Test with simple commands: "move forward", "turn left", "stop"
2. Evaluate recognition accuracy in quiet environment
3. Test with different speaking voices and accents
4. Measure response time

### Exercise 1.2: Natural Language Understanding

**Objective**: Implement natural language understanding for robot commands

**Time Estimate**: 3-4 hours

**Tasks**:
1. Create a command parser that converts natural language to robot actions
2. Implement intent recognition for navigation and manipulation
3. Test with various command formulations

**Implementation Steps**:

```python
# command_parser.py
import re
from typing import Dict, List, Optional

class CommandParser:
    def __init__(self):
        self.intent_patterns = {
            'move_forward': [
                r'move forward(?: (\d+(?:\.\d+)?))? ?(meter|metre|m)?',
                r'go forward(?: (\d+(?:\.\d+)?))? ?(meter|metre|m)?',
                r'forward(?: (\d+(?:\.\d+)?))? ?(meter|metre|m)?'
            ],
            'move_backward': [
                r'move backward(?: (\d+(?:\.\d+)?))? ?(meter|metre|m)?',
                r'go backward(?: (\d+(?:\.\d+)?))? ?(meter|metre|m)?',
                r'back(?: (\d+(?:\.\d+)?))? ?(meter|metre|m)?'
            ],
            'turn_left': [
                r'turn left(?: (\d+(?:\.\d+)?))? ?(degree|degrees|deg)?',
                r'rotate left(?: (\d+(?:\.\d+)?))? ?(degree|degrees|deg)?'
            ],
            'turn_right': [
                r'turn right(?: (\d+(?:\.\d+)?))? ?(degree|degrees|deg)?',
                r'rotate right(?: (\d+(?:\.\d+)?))? ?(degree|degrees|deg)?'
            ],
            'go_to_location': [
                r'go to (kitchen|bedroom|living room|office|table|door)',
                r'navigate to (kitchen|bedroom|living room|office|table|door)',
                r'move to (kitchen|bedroom|living room|office|table|door)'
            ],
            'pick_object': [
                r'pick up (?:the )?(.+)',
                r'grasp (?:the )?(.+)',
                r'get (?:the )?(.+)',
                r'take (?:the )?(.+)'
            ],
            'place_object': [
                r'place (?:the )?(.+) (?:on|at) (.+)',
                r'put (?:the )?(.+) (?:on|at) (.+)',
                r'drop (?:the )?(.+) (?:on|at) (.+)'
            ],
            'speak': [
                r'say "([^"]*)"',
                r'speak "([^"]*)"',
                r'tell me (.+)'
            ]
        }

    def parse_command(self, text: str) -> Optional[Dict]:
        """Parse natural language command into structured format"""
        text = text.lower().strip()

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    # Extract parameters based on pattern groups
                    groups = match.groups()

                    if intent in ['move_forward', 'move_backward']:
                        distance = float(groups[0]) if groups[0] else 1.0
                        unit = groups[1] if len(groups) > 1 and groups[1] else 'meter'

                        # Convert to meters if needed
                        if unit in ['meter', 'metre', 'm']:
                            distance = distance
                        else:
                            distance = distance  # Assume meters if no unit specified

                        return {
                            'intent': intent,
                            'distance': distance,
                            'original_text': text
                        }

                    elif intent in ['turn_left', 'turn_right']:
                        angle = float(groups[0]) if groups[0] else 90.0
                        return {
                            'intent': intent,
                            'angle': angle,
                            'original_text': text
                        }

                    elif intent == 'go_to_location':
                        location = groups[0]
                        return {
                            'intent': intent,
                            'location': location,
                            'original_text': text
                        }

                    elif intent == 'pick_object':
                        obj = groups[0]
                        return {
                            'intent': intent,
                            'object': obj,
                            'original_text': text
                        }

                    elif intent == 'place_object':
                        obj = groups[0]
                        location = groups[1] if len(groups) > 1 else 'default'
                        return {
                            'intent': intent,
                            'object': obj,
                            'location': location,
                            'original_text': text
                        }

                    elif intent == 'speak':
                        message = groups[0] if groups else 'Hello'
                        return {
                            'intent': intent,
                            'message': message,
                            'original_text': text
                        }

        # If no pattern matched, return None
        return None

# Test the parser
parser = CommandParser()

test_commands = [
    "move forward 2 meters",
    "turn left 45 degrees",
    "go to kitchen",
    "pick up the red cup",
    "place the cup on the table",
    "say hello world"
]

print("Testing command parser:")
for cmd in test_commands:
    result = parser.parse_command(cmd)
    print(f"Input: '{cmd}' -> Output: {result}")
```

### Exercise 1.3: Robot Control Integration

**Objective**: Integrate command parsing with robot control

**Time Estimate**: 3-4 hours

**Tasks**:
1. Connect command parser to ROS 2 robot control
2. Implement action execution for parsed commands
3. Test complete pipeline

**Implementation Steps**:

```python
# robot_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String
import time

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

        # Robot state
        self.current_velocity = Twist()
        self.target_velocity = Twist()
        self.is_moving = False

        self.get_logger().info("Robot controller initialized")

    def execute_move_command(self, distance: float, direction: str = 'forward'):
        """Execute move command"""
        self.get_logger().info(f"Moving {direction} for {distance} meters")

        # Calculate time needed (assuming 0.5 m/s speed)
        duration = distance / 0.5  # seconds

        # Set target velocity
        if direction == 'forward':
            self.target_velocity.linear.x = 0.5
        elif direction == 'backward':
            self.target_velocity.linear.x = -0.5
        else:
            return False

        # Execute movement
        start_time = time.time()
        while time.time() - start_time < duration and rclpy.ok():
            self.cmd_vel_pub.publish(self.target_velocity)
            time.sleep(0.05)

        # Stop robot
        self.target_velocity.linear.x = 0.0
        self.cmd_vel_pub.publish(self.target_velocity)

        return True

    def execute_turn_command(self, angle: float, direction: str = 'left'):
        """Execute turn command"""
        self.get_logger().info(f"Turning {direction} by {angle} degrees")

        # Convert to radians
        angle_rad = angle * 3.14159 / 180.0

        # Calculate time needed (assuming 0.5 rad/s angular velocity)
        duration = angle_rad / 0.5  # seconds

        # Set target angular velocity
        if direction == 'left':
            self.target_velocity.angular.z = 0.5
        elif direction == 'right':
            self.target_velocity.angular.z = -0.5
        else:
            return False

        # Execute turn
        start_time = time.time()
        while time.time() - start_time < duration and rclpy.ok():
            self.cmd_vel_pub.publish(self.target_velocity)
            time.sleep(0.05)

        # Stop robot
        self.target_velocity.angular.z = 0.0
        self.cmd_vel_pub.publish(self.target_velocity)

        return True

    def control_loop(self):
        """Main control loop"""
        if self.is_moving:
            self.cmd_vel_pub.publish(self.target_velocity)

# Complete voice command handler
class VoiceCommandHandler:
    def __init__(self):
        rclpy.init()
        self.robot_controller = RobotController()
        self.command_parser = CommandParser()
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.robot_controller)

    def handle_voice_command(self, text: str):
        """Handle complete voice command from recognition to execution"""
        print(f"Processing command: {text}")

        # Parse the command
        parsed = self.command_parser.parse_command(text)
        if not parsed:
            print(f"Could not understand command: {text}")
            return False

        print(f"Parsed command: {parsed}")

        # Execute based on intent
        intent = parsed['intent']

        if intent == 'move_forward':
            return self.robot_controller.execute_move_command(
                parsed['distance'], 'forward'
            )
        elif intent == 'move_backward':
            return self.robot_controller.execute_move_command(
                parsed['distance'], 'backward'
            )
        elif intent == 'turn_left':
            return self.robot_controller.execute_turn_command(
                parsed['angle'], 'left'
            )
        elif intent == 'turn_right':
            return self.robot_controller.execute_turn_command(
                parsed['angle'], 'right'
            )
        else:
            print(f"Intent {intent} not implemented in this exercise")
            return False

# Example usage
def main():
    handler = VoiceCommandHandler()

    # Test commands
    test_commands = [
        "move forward 1 meter",
        "turn left 90 degrees",
        "move forward 0.5 meters"
    ]

    for cmd in test_commands:
        handler.handle_voice_command(cmd)
        time.sleep(2)  # Pause between commands

    # Spin to handle ROS callbacks
    try:
        handler.executor.spin()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        handler.robot_controller.destroy_node()
        rclpy.shutdown()
```

## Lab Exercise 2: Advanced Voice Control Features

### Exercise 2.1: Context-Aware Command Processing

**Objective**: Implement context-aware command processing that considers robot state and environment

**Time Estimate**: 4-5 hours

**Tasks**:
1. Track robot state and position
2. Implement context-aware command interpretation
3. Handle ambiguous commands based on context

### Exercise 2.2: Multi-turn Conversation System

**Objective**: Create a system that can handle multi-turn conversations

**Time Estimate**: 5-6 hours

**Tasks**:
1. Implement dialogue state tracking
2. Handle follow-up commands
3. Manage conversation context

### Exercise 2.3: Performance Optimization

**Objective**: Optimize the voice control system for real-time performance (≥15 Hz)

**Time Estimate**: 4-5 hours

**Tasks**:
1. Profile system performance
2. Implement optimizations (batching, caching, etc.)
3. Validate performance requirements

## Lab Exercise 3: Simulation Integration

### Exercise 3.1: Isaac Sim Integration

**Objective**: Integrate voice control system with Isaac Sim environment

**Time Estimate**: 4-6 hours

**Tasks**:
1. Set up Isaac Sim with humanoid robot model
2. Connect voice control pipeline to simulation
3. Test voice-controlled navigation and manipulation

**Implementation Steps**:

```python
# isaac_sim_integration.py
import carb
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.viewports import set_camera_view
import numpy as np

class IsaacSimVoiceController:
    def __init__(self):
        # Initialize Isaac Sim world
        self.world = World(stage_units_in_meters=1.0)

        # Get assets root path
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets path")
            return

        # Add robot to stage
        robot_path = assets_root_path + "/Isaac/Robots/Franka/franka_instanceable.usd"
        add_reference_to_stage(robot_path, "/World/Robot")

        # Create robot object
        self.robot = self.world.scene.add(
            Robot(
                prim_path="/World/Robot",
                name="franka_robot",
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0])
            )
        )

        # Set up camera view
        set_camera_view(eye=np.array([2.5, 2.5, 2.0]),
                       target=np.array([0.0, 0.0, 0.5]),
                       camera_prim=omni.usd.get_stage_update_events().get_current_camera())

        print("Isaac Sim world initialized with robot")

    def setup_ros_bridge(self):
        """Set up ROS 2 bridge for communication"""
        # Enable ROS bridge extension
        import omni.isaac.ros2_bridge
        omni.isaac.ros2_bridge._ros2_bridge.initialize_ros2_bridge()

        print("ROS 2 bridge initialized")

    def execute_navigation_command(self, target_position):
        """Execute navigation command in simulation"""
        # This would interface with navigation stack in simulation
        print(f"Navigating to position: {target_position}")

        # In a real implementation, this would send navigation goals
        # to the navigation stack running in simulation
        pass

    def execute_manipulation_command(self, object_name, action):
        """Execute manipulation command in simulation"""
        print(f"Manipulating object '{object_name}' with action '{action}'")

        # In a real implementation, this would control the robot's manipulator
        # in the simulation environment
        pass

# Example usage
def run_isaac_sim_demo():
    controller = IsaacSimVoiceController()
    controller.setup_ros_bridge()

    # Simulate voice commands
    commands = [
        {"intent": "navigate", "target": [1.0, 1.0, 0.0]},
        {"intent": "manipulate", "object": "cup", "action": "pick"}
    ]

    for cmd in commands:
        if cmd["intent"] == "navigate":
            controller.execute_navigation_command(cmd["target"])
        elif cmd["intent"] == "manipulate":
            controller.execute_manipulation_command(cmd["object"], cmd["action"])

        # Step the simulation
        controller.world.step(render=True)
```

## Lab Exercise 4: Physical Robot Deployment

### Exercise 4.1: Hardware Setup and Calibration

**Objective**: Set up and calibrate voice control system on physical humanoid robot

**Time Estimate**: 6-8 hours

**Tasks**:
1. Configure microphone array for optimal audio capture
2. Calibrate robot's physical dimensions and capabilities
3. Test basic voice commands on physical robot

### Exercise 4.2: Real-World Performance Testing

**Objective**: Test voice control system in real-world conditions

**Time Estimate**: 4-5 hours

**Tasks**:
1. Test in various acoustic environments
2. Validate performance requirements (≥15 Hz)
3. Evaluate robustness to environmental factors

## Assessment and Evaluation

### Performance Metrics

Students will be evaluated on:
- **Accuracy**: Correct interpretation of voice commands (≥90%)
- **Latency**: Response time from speech to action (≤500ms)
- **Throughput**: Processing frequency (≥15 Hz)
- **Robustness**: Performance in noisy environments
- **Completeness**: Successful execution of multi-step commands

### Assessment Rubric

| Criteria | Excellent (90-100%) | Good (80-89%) | Satisfactory (70-79%) | Needs Improvement (&lt;70%) |
|----------|-------------------|---------------|---------------------|------------------------|
| Command Recognition | ≥95% accuracy across all commands | 90-94% accuracy | 80-89% accuracy | &lt;80% accuracy |
| System Integration | Seamless integration, no errors | Minor integration issues | Some integration issues | Significant integration issues |
| Performance | ≥15 Hz, ≤500ms latency | 12-15 Hz, ≤600ms latency | 10-12 Hz, ≤700ms latency | &lt;10 Hz, >700ms latency |
| Documentation | Comprehensive, clear, well-commented | Good documentation with minor gaps | Adequate documentation | Poor or missing documentation |
| Problem Solving | Creative solutions, optimization | Good problem-solving approach | Basic problem-solving | Little evidence of problem-solving |

### Submission Requirements

Students must submit:
1. **Source Code**: Complete, well-documented implementation
2. **Test Results**: Performance metrics and evaluation data
3. **Documentation**: Setup guide, user manual, and technical report
4. **Video Demonstration**: 3-5 minute video showing system in action
5. **Reflection Report**: Analysis of challenges faced and lessons learned

## Troubleshooting and Support

<RoboticsBlock type="warning" title="Common Issues and Solutions">
- **Poor Recognition Accuracy**: Check microphone positioning, ambient noise, and audio preprocessing
- **High Latency**: Optimize model inference, reduce pipeline stages, use lighter models
- **Low Throughput**: Implement batching, optimize memory management, use asynchronous processing
- **Integration Problems**: Verify ROS 2 network configuration, topic names, and message types
</RoboticsBlock>

### Debugging Tools
- Audio quality analyzer
- Performance profiler
- ROS 2 topic monitoring tools
- Isaac Sim debugging utilities

## Extension Activities

For advanced students:
1. **Multilingual Support**: Extend system to support multiple languages
2. **Emotional Intelligence**: Add emotion recognition and response
3. **Proactive Interaction**: Implement proactive assistance behaviors
4. **Learning Capabilities**: Add ability to learn new commands through interaction

## Conclusion

These lab exercises provide hands-on experience with implementing voice-controlled systems for humanoid robots. Students will gain practical skills in speech recognition, natural language processing, robot control, and system integration while meeting the performance requirements for real-time operation.

The exercises progress from basic implementation to advanced features and real-world deployment, providing a comprehensive learning experience in voice-controlled robotics systems.