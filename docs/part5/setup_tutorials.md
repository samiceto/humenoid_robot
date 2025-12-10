# Comprehensive Setup Tutorials for Physical AI & Humanoid Robotics Course

## Table of Contents
1. [System Requirements and Prerequisites](#system-requirements)
2. [Ubuntu 22.04 LTS Installation](#ubuntu-installation)
3. [ROS 2 Iron Installation and Configuration](#ros2-installation)
4. [Isaac Sim Installation and Setup](#isaac-sim-setup)
5. [Isaac ROS Installation and Configuration](#isaac-ros-setup)
6. [Jetson Orin Development Setup](#jetson-setup)
7. [Development Environment Configuration](#dev-environment)
8. [Hardware Integration Setup](#hardware-integration)
9. [Simulation Environment Setup](#simulation-setup)
10. [Testing and Validation](#testing-validation)

---

## System Requirements and Prerequisites {#system-requirements}

### Minimum Hardware Requirements
- **CPU**: Intel i7-10700K or AMD Ryzen 7 3700X (8+ cores, 3.0+ GHz per core)
- **GPU**: NVIDIA RTX 3060 8GB or equivalent (12GB+ recommended)
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 500GB NVMe SSD minimum (1TB recommended)
- **Network**: Stable internet connection (50+ Mbps for cloud features)

### Recommended Hardware Configuration
- **CPU**: Intel i9-12900K or AMD Ryzen 9 5900X
- **GPU**: NVIDIA RTX 4070 12GB or Jetson Orin Nano
- **RAM**: 32GB DDR4-3200
- **Storage**: 1TB+ NVMe SSD
- **OS**: Ubuntu 22.04 LTS

### Software Prerequisites
- Ubuntu 22.04 LTS (64-bit)
- NVIDIA GPU with CUDA support
- Internet access for package downloads
- Administrative privileges for system installation

---

## Ubuntu 22.04 LTS Installation {#ubuntu-installation}

### Pre-Installation Checklist
- [ ] Backup important data from current system
- [ ] Download Ubuntu 22.04 LTS ISO file
- [ ] Create bootable USB drive (8GB+ recommended)
- [ ] Verify system compatibility with Ubuntu

### Installation Steps

#### Step 1: Create Bootable USB Drive
```bash
# Download Ubuntu 22.04 LTS from https://ubuntu.com/download/desktop
# Create bootable USB using Rufus (Windows) or Startup Disk Creator (Ubuntu)

# On Ubuntu system:
sudo apt install usb-creator-gtk
# Use Startup Disk Creator to create bootable USB
```

#### Step 2: BIOS/UEFI Configuration
1. Restart computer and enter BIOS/UEFI setup (usually F2, F12, or Del during boot)
2. Disable Secure Boot (required for some drivers)
3. Set SATA mode to AHCI (not RAID)
4. Enable virtualization technology (Intel VT-x/AMD-V)
5. Save changes and exit

#### Step 3: Ubuntu Installation
1. Boot from USB drive
2. Select "Install Ubuntu"
3. Choose language and keyboard layout
4. Select "Normal installation" with updates
5. For disk partitioning:
   - Option A: "Erase disk and install Ubuntu" (for dedicated robotics machine)
   - Option B: "Install alongside existing OS" (for dual-boot setup)
   - Option C: "Something else" (for custom partitioning)

6. For custom partitioning (recommended):
   ```
   /boot/efi: 1GB, EFI System Partition
   /: 100GB, Ext4, primary
   /home: Remaining space, Ext4, primary
   swap: 16GB (or equal to RAM size), swap area
   ```

7. Set timezone and create user account
8. Complete installation and restart

#### Step 4: Post-Installation Configuration
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential development tools
sudo apt install build-essential cmake git vim curl wget htop

# Install Python development tools
sudo apt install python3-dev python3-pip python3-venv

# Install graphics drivers (for NVIDIA)
sudo apt install nvidia-driver-535 nvidia-settings

# Reboot to apply driver changes
sudo reboot
```

---

## ROS 2 Iron Installation and Configuration {#ros2-installation}

### Step 1: Set Up Sources and Keys
```bash
# Add ROS 2 GPG key
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add ROS 2 repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update package list
sudo apt update
```

### Step 2: Install ROS 2 Iron Desktop
```bash
# Install ROS 2 Iron with desktop packages
sudo apt install -y ros-iron-desktop

# Install additional ROS 2 tools
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### Step 3: Initialize rosdep
```bash
# Initialize rosdep
sudo rosdep init

# Update rosdep
rosdep update
```

### Step 4: Set Up ROS 2 Environment
```bash
# Add ROS 2 setup to bashrc
echo "source /opt/ros/iron/setup.bash" >> ~/.bashrc

# Reload bashrc
source ~/.bashrc

# Verify installation
ros2 --version
```

### Step 5: Create ROS 2 Workspace
```bash
# Create workspace directory
mkdir -p ~/robotics_ws/src
cd ~/robotics_ws

# Build workspace (initial empty build)
colcon build --packages-selectament-cmake-args -DCMAKE_BUILD_TYPE=Release

# Source the workspace
source install/setup.bash

# Add workspace to bashrc
echo "source ~/robotics_ws/install/setup.bash" >> ~/.bashrc
```

### Step 6: Install Additional ROS 2 Packages
```bash
# Install navigation stack
sudo apt install -y ros-iron-navigation2 ros-iron-nav2-bringup

# Install perception packages
sudo apt install -y ros-iron-vision-opencv ros-iron-image-transport ros-iron-camera-calibration

# Install control packages
sudo apt install -y ros-iron-ros2-control ros-iron-ros2-controllers

# Install simulation packages
sudo apt install -y ros-iron-gazebo-ros-pkgs ros-iron-joint-state-publisher
```

---

## Isaac Sim Installation and Setup {#isaac-sim-setup}

### Step 1: System Preparation
```bash
# Install NVIDIA drivers (if not already installed)
sudo apt install nvidia-driver-535

# Install Vulkan support
sudo apt install vulkan-tools vulkan-utils

# Install additional dependencies
sudo apt install mesa-utils libgl1-mesa-glx libgl1-mesa-dri
```

### Step 2: Download Isaac Sim
```bash
# Visit https://developer.nvidia.com/isaac-sim and download the latest version
# As of 2024, download Isaac Sim 2024.2+ for Ubuntu 22.04

# Create installation directory
mkdir -p ~/isaac_sim
cd ~/isaac_sim

# Extract downloaded file (replace with actual filename)
tar -xzf isaac-sim-2024.2.0.tar.gz
```

### Step 3: Install Isaac Sim Dependencies
```bash
# Navigate to Isaac Sim directory
cd ~/isaac_sim/isaac-sim-2024.2.0

# Install Python dependencies
python3 -m pip install -e .
```

### Step 4: Set Up Isaac Sim Environment
```bash
# Add Isaac Sim to bashrc
echo 'export ISAACSIM_PATH="$HOME/isaac_sim/isaac-sim-2024.2.0"' >> ~/.bashrc
echo 'export PYTHONPATH="$ISAACSIM_PATH/python:$PYTHONPATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$ISAACSIM_PATH/exts/omni.isaac.sim.python/bin:$LD_LIBRARY_PATH"' >> ~/.bashrc

# Reload bashrc
source ~/.bashrc
```

### Step 5: Test Isaac Sim Installation
```bash
# Navigate to Isaac Sim directory
cd ~/isaac_sim/isaac-sim-2024.2.0

# Launch Isaac Sim
./python.sh -m omni.isaac.kit --exec /isaac-sim/exts/omni.isaac.examples/scripts/example_1.py
```

### Step 6: Install Isaac Sim ROS Bridge
```bash
# Clone the ROS bridge repository
cd ~/robotics_ws/src
git clone -b release/stable https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_bridge.git

# Install dependencies
cd ~/robotics_ws
rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
colcon build --packages-select isaac_ros_common isaac_ros_launch
```

---

## Isaac ROS Installation and Configuration {#isaac-ros-setup}

### Step 1: Install Isaac ROS Dependencies
```bash
# Update package list
sudo apt update

# Install Isaac ROS meta-package
sudo apt install ros-iron-isaac-ros-dev

# Install specific Isaac ROS packages
sudo apt install \
  ros-iron-isaac-ros-apriltag \
  ros-iron-isaac-ros-visual-slam \
  ros-iron-isaac-ros-bi3d \
  ros-iron-isaac-ros-centerpose \
  ros-iron-isaac-ros-gxf-components \
  ros-iron-isaac-ros-gxf-extensions
```

### Step 2: Verify Isaac ROS Installation
```bash
# Check if Isaac ROS packages are available
apt list --installed | grep isaac-ros

# Test Isaac ROS launch files
ros2 launch isaac_ros_apriltag isaac_ros_apriltag.launch.py
```

### Step 3: Install Isaac ROS Tools
```bash
# Install Isaac ROS utilities
sudo apt install \
  ros-iron-isaac-ros-essentials \
  ros-iron-isaac-ros-test \
  ros-iron-isaac-ros-benchmark
```

### Step 4: Configure Isaac ROS Environment
```bash
# Add Isaac ROS paths to bashrc
echo 'export ISAAC_ROS_WS="$HOME/robotics_ws"' >> ~/.bashrc
echo 'export ISAAC_ROS_COMMON_DIR="$ISAAC_ROS_WS/src/isaac_ros_common"' >> ~/.bashrc

# Reload bashrc
source ~/.bashrc
```

---

## Jetson Orin Development Setup {#jetson-setup}

### Step 1: Jetson Orin Initial Setup
```bash
# This section applies to Jetson Orin hardware setup
# For development environment simulation on x86, skip to Step 2

# On Jetson Orin device:
# 1. Flash JetPack SDK (includes Ubuntu, ROS, and NVIDIA tools)
# 2. Connect to internet
# 3. Update system: sudo apt update && sudo apt upgrade
```

### Step 2: Cross-Compilation Setup for Jetson (x86 Host)
```bash
# Install cross-compilation tools
sudo apt install crossbuild-essential-arm64

# Set up Docker for Jetson containers
sudo apt install docker.io
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Step 3: Jetson Orin ROS 2 Setup (if deploying to actual Jetson)
```bash
# On Jetson Orin:
# Install ROS 2 Iron for Jetson
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-iron-desktop
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

sudo rosdep init
rosdep update

echo "source /opt/ros/iron/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Step 4: TensorRT Optimization Setup
```bash
# Install TensorRT Python bindings
pip3 install tensorrt

# Install optimization tools
sudo apt install nvidia-tensorrt-dev
```

---

## Development Environment Configuration {#dev-environment}

### Step 1: Install Development Tools
```bash
# Install version control
sudo apt install git git-gui gitk

# Install code editors/IDEs
sudo apt install code # Visual Studio Code
# Or install vim with plugins
sudo apt install vim-gtk3

# Install development utilities
sudo apt install gdb valgrind cmake-curses-gui
```

### Step 2: Configure VS Code for ROS 2 Development
```bash
# Install VS Code ROS extension
code --install-extension ms-iot.vscode-ros
code --install-extension ms-vscode.cpptools
code --install-extension ms-python.python

# Create VS Code workspace configuration
mkdir -p ~/robotics_ws/.vscode
cat > ~/robotics_ws/.vscode/settings.json << 'EOF'
{
    "cmake.configureOnOpen": true,
    "cmake.buildBeforeRun": true,
    "terminal.integrated.defaultProfile.linux": "bash",
    "terminal.integrated.profiles.linux": {
        "bash": {
            "path": "/bin/bash",
            "args": ["-c", "source /opt/ros/iron/setup.bash && source ~/robotics_ws/install/setup.bash && exec bash"]
        }
    },
    "python.defaultInterpreterPath": "/usr/bin/python3",
    "ros.distro": "iron"
}
EOF
```

### Step 3: Install Python Development Environment
```bash
# Install virtual environment tools
pip3 install virtualenv virtualenvwrapper

# Set up virtualenvwrapper
echo "source /usr/share/virtualenvwrapper/virtualenvwrapper.sh" >> ~/.bashrc
source ~/.bashrc

# Create robotics development environment
mkvirtualenv robotics_dev
workon robotics_dev

# Install Python packages for robotics
pip install numpy scipy matplotlib opencv-python torch torchvision
```

### Step 4: Install Additional Tools
```bash
# Install network tools
sudo apt install net-tools nmap

# Install monitoring tools
sudo apt install htop iotop nethogs

# Install compression tools
sudo apt install p7zip-full unrar-free unace

# Install documentation tools
sudo apt install doxygen graphviz
```

---

## Hardware Integration Setup {#hardware-integration}

### Step 1: Sensor Integration
```bash
# Install camera drivers
sudo apt install v4l-utils

# Install LIDAR support (example for common LIDARs)
sudo apt install ros-iron-scan-tools ros-iron-laser-filters

# Install IMU support
sudo apt install ros-iron-rtimulib-ros2 ros-iron-imu-tools
```

### Step 2: Communication Protocols
```bash
# Install serial communication tools
sudo apt install ros-iron-serial-driver ros-iron-rosbridge-suite

# Install CAN bus support (if needed)
sudo apt install can-utils
sudo apt install ros-iron-socketcan-interface

# Install Ethernet/IP support
sudo apt install ros-iron-ethernet-ip
```

### Step 3: Actuator Control
```bash
# Install motor control packages
sudo apt install ros-iron-ros2-control ros-iron-ros2-controllers

# Install specific actuator drivers (example)
sudo apt install ros-iron-dynamixel-sdk ros-iron-ros-canopen
```

---

## Simulation Environment Setup {#simulation-setup}

### Step 1: Gazebo Installation (Alternative to Isaac Sim)
```bash
# Install Gazebo Garden (most recent version)
sudo apt install gazebo libgazebo-dev

# Install ROS 2 Gazebo bridge
sudo apt install ros-iron-gazebo-ros-pkgs ros-iron-gazebo-plugins
```

### Step 2: Create Simulation Workspace
```bash
# Create simulation-specific workspace
mkdir -p ~/simulation_ws/src
cd ~/simulation_ws

# Clone common simulation packages
cd ~/simulation_ws/src
git clone https://github.com/ros-simulation/gazebo_ros_pkgs.git -b iron
git clone https://github.com/ros-simulation/simulators.git -b iron

# Build simulation workspace
cd ~/simulation_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select gazebo_ros_pkgs
source install/setup.bash
```

### Step 3: Configure Simulation Environment
```bash
# Add simulation environment to bashrc
echo 'export GAZEBO_MODEL_PATH="$GAZEBO_MODEL_PATH:$HOME/simulation_ws/src/gazebo_ros_pkgs/gazebo_ros/models"' >> ~/.bashrc
echo 'export GAZEBO_RESOURCE_PATH="$GAZEBO_RESOURCE_PATH:$HOME/simulation_ws/src/gazebo_ros_pkgs/gazebo_ros/worlds"' >> ~/.bashrc

# Reload bashrc
source ~/.bashrc
```

---

## Testing and Validation {#testing-validation}

### Step 1: Basic ROS 2 Test
```bash
# Test ROS 2 installation
ros2 run demo_nodes_cpp talker &
ros2 run demo_nodes_py listener

# Verify topics are communicating
# You should see messages passing between nodes
```

### Step 2: Isaac Sim Test
```bash
# Test Isaac Sim with a simple example
cd ~/isaac_sim/isaac-sim-2024.2.0
./python.sh -c "
import omni
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({'headless': False})
import omni.isaac.core.utils.prims as prim_utils
prim_utils.create_prim('/World/Cube', 'Cube', position=[0, 0, 1.0])
simulation_app.close()
"
```

### Step 3: Isaac ROS Test
```bash
# Test Isaac ROS components
ros2 launch isaac_ros_apriltag isaac_ros_apriltag.launch.py

# In another terminal, publish test image
ros2 topic pub /rgb_image sensor_msgs/msg/Image "{}" --field="height=480,width=640,encoding='rgb8',data=[0,0,0]"
```

### Step 4: System Integration Test
```bash
# Create a test launch file to verify all components work together
cat > ~/robotics_ws/src/test_launch.py << 'EOF'
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import find_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link']
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher'
        )
    ])
EOF

# Test the launch file
cd ~/robotics_ws
source install/setup.bash
ros2 launch src/test_launch.py
```

### Step 5: Performance Validation
```bash
# Test real-time performance capabilities
ros2 topic pub /performance_test std_msgs/msg/Float64 "data: 1.0" -r 100

# Monitor performance with:
# - htop for CPU usage
# - nvidia-smi for GPU usage
# - ros2 topic hz for message frequency
```

### Step 6: Complete System Validation
```bash
# Run comprehensive validation script
cat > ~/validate_setup.sh << 'EOF'
#!/bin/bash

echo "=== ROS 2 Validation ==="
if ros2 --version; then
    echo "✓ ROS 2 installation: OK"
else
    echo "✗ ROS 2 installation: FAILED"
fi

echo -e "\n=== Isaac Sim Validation ==="
if [ -d "$HOME/isaac_sim/isaac-sim-2024.2.0" ]; then
    echo "✓ Isaac Sim installation: OK"
else
    echo "✗ Isaac Sim installation: FAILED"
fi

echo -e "\n=== Isaac ROS Validation ==="
if dpkg -l | grep -q "isaac-ros"; then
    echo "✓ Isaac ROS installation: OK"
else
    echo "✗ Isaac ROS installation: FAILED"
fi

echo -e "\n=== Python Environment Validation ==="
if python3 -c "import cv2, numpy, torch" 2>/dev/null; then
    echo "✓ Python packages: OK"
else
    echo "✗ Python packages: FAILED"
fi

echo -e "\n=== System Resources Validation ==="
MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ "$MEM_GB" -ge 16 ]; then
    echo "✓ Memory (16GB+): OK ($MEM_GB GB)"
else
    echo "⚠ Memory (16GB+): LOW ($MEM_GB GB)"
fi

echo -e "\n=== GPU Validation ==="
if nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null; then
    echo "✓ NVIDIA GPU: OK"
else
    echo "✗ NVIDIA GPU: NOT FOUND"
fi

echo -e "\n=== Setup Validation Complete ==="
EOF

chmod +x ~/validate_setup.sh
~/validate_setup.sh
```

---

## Troubleshooting Common Issues

### ROS 2 Installation Issues
**Problem**: `rosdep init` fails with "Permission denied"
**Solution**:
```bash
sudo rosdep init
rosdep update
```

**Problem**: ROS 2 commands not found after installation
**Solution**: Check if ROS 2 is sourced in bashrc:
```bash
echo "source /opt/ros/iron/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Isaac Sim Issues
**Problem**: Isaac Sim fails to launch with graphics errors
**Solution**: Ensure NVIDIA drivers are properly installed:
```bash
sudo apt install nvidia-driver-535
sudo reboot
```

**Problem**: Isaac Sim Python environment conflicts
**Solution**: Use Isaac Sim's Python exclusively:
```bash
cd ~/isaac_sim/isaac-sim-2024.2.0
./python.sh -m pip install [package_name]
```

### Isaac ROS Issues
**Problem**: Isaac ROS packages not found
**Solution**: Verify installation and check ROS 2 environment:
```bash
source /opt/ros/iron/setup.bash
dpkg -l | grep isaac-ros
```

### Performance Issues
**Problem**: Simulation runs slowly
**Solution**:
1. Reduce simulation quality settings
2. Close unnecessary applications
3. Ensure sufficient RAM and GPU resources

---

## Final Configuration Checklist

Before starting the course, verify the following:

- [ ] Ubuntu 22.04 LTS installed and updated
- [ ] ROS 2 Iron installed and working
- [ ] Isaac Sim installed and launching
- [ ] Isaac ROS packages installed
- [ ] Development tools configured
- [ ] All hardware drivers installed (if applicable)
- [ ] System performance validated
- [ ] All tutorials tested successfully

---

## Support and Resources

### Official Documentation
- ROS 2 Iron: https://docs.ros.org/en/iron/
- Isaac Sim: https://docs.omniverse.nvidia.com/isaacsim/latest/
- Isaac ROS: https://nvidia-isaac-ros.github.io/

### Community Support
- ROS Answers: https://answers.ros.org/
- Isaac Sim Forum: https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/simulation/43
- GitHub Issues: Check repositories for specific packages

### Troubleshooting Resources
- System logs: `journalctl -xe`
- ROS 2 logs: `~/.ros/log/`
- Isaac Sim logs: `~/isaac_sim/isaac-sim-2024.2.0/logs/`