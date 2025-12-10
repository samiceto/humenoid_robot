# Isaac ROS 3.0+ Installation Guide

This document outlines the steps to install Isaac ROS 3.0+ packages and dependencies for the Physical AI & Humanoid Robotics course.

## Prerequisites

- Ubuntu 22.04 LTS
- ROS 2 Iron or Jazzy installed (as specified in the course requirements)
- NVIDIA GPU with RTX or GTX 10xx/20xx/30xx/40xx series
- Isaac Sim 2024.2+ (installed as per previous setup guide)

## Installation Steps

### 1. Verify ROS 2 Installation

First, ensure ROS 2 Iron or Jazzy is properly installed:

```bash
# Source ROS 2 installation
source /opt/ros/jazzy/setup.bash  # or iron for ROS 2 Iron

# Verify installation
ros2 --version
```

### 2. Install Isaac ROS Dependencies

Install the required system dependencies:

```bash
# Update package lists
sudo apt update

# Install essential dependencies
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-vcstool
sudo apt install -y build-essential cmake pkg-config libusb-1.0-0-dev libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglx-dev libegl1-mesa-dev
sudo apt install -y libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libgstreamer-plugins-base1.0-dev
sudo apt install -y libtbb-dev libjpeg-dev libpng-dev libtiff-dev libopenexr-dev libwebp-dev
```

### 3. Set Up Workspace

Create a workspace for Isaac ROS packages:

```bash
# Create workspace directory
mkdir -p ~/isaac_ros_ws/src
cd ~/isaac_ros_ws

# Source ROS 2
source /opt/ros/jazzy/setup.bash
```

### 4. Clone Isaac ROS Packages

Clone the Isaac ROS repositories:

```bash
cd ~/isaac_ros_ws/src

# Clone Isaac ROS common repository
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git

# Clone Isaac ROS perception packages
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag.git
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline.git
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_depth_segmentation.git
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation.git

# Clone Isaac ROS manipulation packages
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_manipulation.git
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_freespace_segmentation.git

# Clone Isaac ROS navigation packages
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_navigation.git
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros.git

# Clone Isaac ROS bi3d (3D segmentation)
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_bi3d.git

# Clone Isaac ROS centerpose (pose estimation)
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_centerpose.git
```

### 5. Install Package Dependencies

Use rosdep to install all dependencies:

```bash
cd ~/isaac_ros_ws

# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Install dependencies using rosdep
rosdep install --from-paths src --ignore-src -r -y
```

### 6. Build Isaac ROS Packages

Build the Isaac ROS packages:

```bash
cd ~/isaac_ros_ws

# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Build all packages
colcon build --symlink-install --packages-select $(find src -name "package.xml" -exec dirname {} \; | xargs basename | tr '\n' ' ')

# Or build with specific packages if needed
colcon build --symlink-install --parallel-workers 4
```

### 7. Source the Workspace

After building, source the workspace:

```bash
# Add to your .bashrc to make it permanent
echo "source ~/isaac_ros_ws/install/setup.bash" >> ~/.bashrc
source ~/isaac_ros_ws/install/setup.bash
```

### 8. Install Additional Isaac ROS Extensions

Install Isaac ROS extensions for specific functionality:

```bash
cd ~/isaac_ros_ws/src

# Isaac ROS Apriltag Detection
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag_interfaces.git

# Isaac ROS Image Buffer
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_buffer.git

# Isaac ROS NITROS Data Converter
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros_type.git
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros_bridge.git

# Isaac ROS Essential Matrix Generator
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_essential_matrix_generator.git

# Isaac ROS Pose Covariance
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_covariance.git
```

### 9. Verify Installation

Test the installation by running a simple Isaac ROS node:

```bash
# Source both ROS 2 and Isaac ROS workspace
source /opt/ros/jazzy/setup.bash
source ~/isaac_ros_ws/install/setup.bash

# List available packages
ros2 pkg list | grep isaac

# Check if key packages are available
ros2 pkg executables | grep isaac
```

## Alternative: Pre-built Docker Images

For easier setup, you can also use pre-built Isaac ROS Docker images:

```bash
# Pull the Isaac ROS Docker image
docker pull nvcr.io/nvidia/isaac-ros:latest

# Run Isaac ROS container
docker run --gpus all -it --rm \
  --network=host \
  --env "DISPLAY" \
  --volume "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume "$(pwd):/workspace:rw" \
  --privileged \
  nvcr.io/nvidia/isaac-ros:latest
```

## Troubleshooting

### Common Issues:

1. **Build Failures**: If colcon build fails, try building packages individually:
   ```bash
   colcon build --packages-select isaac_ros_apriltag
   ```

2. **Dependency Issues**: If rosdep fails, manually install missing packages:
   ```bash
   sudo apt install ros-jazzy-vision-msgs ros-jazzy-sensor-msgs-py
   ```

3. **Memory Issues**: If building fails due to memory, limit parallel workers:
   ```bash
   colcon build --parallel-workers 2
   ```

## Package Overview

Key Isaac ROS packages for the course:

- `isaac_ros_apriltag`: AprilTag detection for pose estimation
- `isaac_ros_visual_slam`: Visual SLAM for localization and mapping
- `isaac_ros_image_pipeline`: Image processing pipelines
- `isaac_ros_depth_segmentation`: Depth-based segmentation
- `isaac_ros_pose_estimation`: 3D pose estimation
- `isaac_ros_manipulation`: Manipulation algorithms
- `isaac_ros_navigation`: Navigation stack enhancements

## Next Steps

After completing Isaac ROS setup, proceed to:

1. Nav2 navigation stack configuration
2. Integration testing with Isaac Sim
3. Setting up Python interfaces for Isaac ROS

## References

- [Isaac ROS GitHub Repository](https://github.com/NVIDIA-ISAAC-ROS)
- [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/index.html)
- [ROS 2 Installation Guide](https://docs.ros.org/en/jazzy/Installation.html)