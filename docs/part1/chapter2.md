---
sidebar_position: 3
---

# Chapter 2: ROS 2 Fundamentals for Humanoid Systems

## Overview

Robot Operating System 2 (ROS 2) is the backbone of modern robotics development. This chapter introduces ROS 2 concepts essential for humanoid robotics, including nodes, topics, services, actions, parameters, and the distributed computing model.

## What is ROS 2?

ROS 2 is not an operating system but a collection of software frameworks that provide a rich set of libraries and tools to help create robot applications. Key features include:

- Distributed computing architecture
- Language independence (C++, Python, Rust, etc.)
- Real-time support
- Improved security and safety features
- Better support for commercial products

## Installing ROS 2 Iron/Iron

This section provides a comprehensive guide to installing ROS 2 Iron Irwini on Ubuntu 22.04 LTS.

### System Requirements

- Operating System: Ubuntu 22.04 (Jammy Jellyfish)
- RAM: At least 8GB (16GB recommended)
- Disk Space: At least 5GB
- Architecture: x86_64 or aarch64

### Setup Locale

Ensure your locale is set to support UTF-8:

```bash
locale  # check for UTF-8

sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

### Setup Sources

Enable required repositories:

```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
```

### Add ROS 2 GPG Key and Repository

```bash
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### Install ROS 2 Packages

```bash
sudo apt update
sudo apt install ros-iron-desktop
```

### Environment Setup

Source the ROS 2 environment:

```bash
source /opt/ros/iron/setup.bash
```

To automatically source the ROS 2 environment at every new terminal, add the following line to your `~/.bashrc` file:

```bash
echo "source /opt/ros/iron/setup.bash" >> ~/.bashrc
```

## ROS 2 Concepts for Humanoid Robots

### Nodes

A node is an executable that uses ROS 2 to communicate with other nodes. In humanoid robotics, you might have nodes for:

- Joint controllers
- Perception systems
- Navigation
- High-level decision making
- Sensor processing

### Topics and Messages

Topics are named buses over which nodes exchange messages. For humanoid robots:

- `/joint_states`: Current joint positions, velocities, and efforts
- `/cmd_vel`: Velocity commands for base movement
- `/tf`: Transformations between coordinate frames
- `/camera/image_raw`: Raw camera images
- `/imu/data`: Inertial measurement unit data

### Services

Services provide request/response communication. Common services in humanoid robots:

- `/spawn_entity`: Spawn objects in simulation
- `/delete_entity`: Remove objects from simulation
- `/set_parameters`: Change node parameters dynamically

### Actions

Actions are used for long-running tasks with feedback. In humanoid robotics:

- `/follow_joint_trajectory`: Execute complex joint movements
- `/move_base`: Navigate to a specific location
- `/pick_and_place`: Manipulation tasks

## ROS 2 Tools for Development

### ros2 run

Execute a node from a package:

```bash
ros2 run turtlesim turtlesim_node
```

### ros2 topic

Inspect and interact with topics:

```bash
ros2 topic list
ros2 topic echo /topic_name
ros2 topic pub /topic_name MessageType "data: value"
```

### ros2 service

Call services:

```bash
ros2 service list
ros2 service call /service_name ServiceType "{request_field: value}"
```

### rqt

GUI-based tool for visualizing ROS 2 data:

```bash
rqt
```

## Creating a ROS 2 Workspace

### Create the Workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

### Build the Workspace

```bash
colcon build
source install/setup.bash
```

## ROS 2 for Humanoid Robotics

ROS 2 is particularly well-suited for humanoid robotics due to:

- **Modularity**: Each subsystem can run as a separate node
- **Distributed Architecture**: Different computational units can run on different hardware
- **Standardized Interfaces**: Common message types for communication
- **Simulation Integration**: Seamless transition between simulation and real hardware
- **Large Ecosystem**: Extensive libraries for navigation, perception, and control

## Common ROS 2 Packages for Humanoid Robots

### Navigation Stack (Nav2)

Provides path planning, obstacle avoidance, and navigation capabilities:

```bash
sudo apt install ros-iron-navigation2 ros-iron-nav2-bringup
```

### Perception Packages (Isaac ROS)

Optimized for NVIDIA hardware for perception tasks:

```bash
# Install Isaac ROS packages (covered in later chapters)
```

### Control Packages

- `ros2_control`: Framework for robot control
- `joint_state_publisher`: Publish joint states
- `robot_state_publisher`: Publish robot state based on URDF

## Chapter Summary

This chapter introduced ROS 2 fundamentals essential for humanoid robotics development. We covered the installation process, core concepts (nodes, topics, services, actions), development tools, and how ROS 2 applies to humanoid robots. In the next chapter, we'll explore URDF (Unified Robot Description Format) for modeling humanoid robots.