# Ubuntu 22.04 LTS Development Environment Configuration

This document outlines the steps to configure the Ubuntu 22.04 LTS development environment for the Physical AI & Humanoid Robotics course.

## Prerequisites

- Ubuntu 22.04 LTS installed (either native or WSL2)
- Internet connection
- Administrative privileges (sudo access)

## Initial System Setup

### 1. System Update and Basic Tools

Update the system and install essential development tools:

```bash
# Update package lists and upgrade system
sudo apt update && sudo apt upgrade -y

# Install essential development tools
sudo apt install -y build-essential cmake git vim htop curl wget unzip
sudo apt install -y python3-pip python3-dev python3-venv
sudo apt install -y terminator tmux
sudo apt install -y openssh-server openssh-client
```

### 2. Configure Git

Set up Git for development:

```bash
# Configure Git globally
git config --global user.name "Student Name"
git config --global user.email "student@university.edu"
git config --global core.editor vim
git config --global init.defaultBranch main

# Configure Git for efficient operations
git config --global core.preloadindex true
git config --global core.fscache true
git config --global gc.auto 256
```

## ROS 2 Installation (Iron or Jazzy)

### 1. Install ROS 2 Jazzy (or Iron as specified)

Install ROS 2 with all required packages:

```bash
# Add ROS 2 repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update package lists
sudo apt update

# Install ROS 2 Jazzy Desktop (includes GUI tools)
sudo apt install -y ros-jazzy-desktop ros-jazzy-ros-base
sudo apt install -y ros-jazzy-perception ros-jazzy-navigation2

# Install development tools
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### 2. Install Additional ROS 2 Packages

Install packages required for the robotics course:

```bash
# Navigation and mapping
sudo apt install -y ros-jazzy-navigation2 ros-jazzy-navigation2-msgs
sudo apt install -y ros-jazzy-slam-toolbox ros-jazzy-robot-localization

# Simulation
sudo apt install -y ros-jazzy-gazebo-ros ros-jazzy-gazebo-plugins ros-jazzy-gazebo-dev
sudo apt install -y ros-jazzy-ros-gz ros-jazzy-ros-gz-plugins

# Hardware interfaces
sudo apt install -y ros-jazzy-ros2-control ros-jazzy-ros2-controllers
sudo apt install -y ros-jazzy-hardware-interface ros-jazzy-controller-manager
sudo apt install -y ros-jazzy-joint-state-broadcaster ros-jazzy-velocity-controllers

# Visualization and tools
sudo apt install -y ros-jazzy-rviz2 ros-jazzy-rqt ros-jazzy-ros2bag ros-jazzy-rosbag2
sudo apt install -y ros-jazzy-teleop-tools ros-jazzy-joy ros-jazzy-xacro

# Robot modeling and simulation
sudo apt install -y ros-jazzy-urdf ros-jazzy-urdf-tutorial ros-jazzy-tf2-tools
sudo apt install -y ros-jazzy-robot-state-publisher ros-jazzy-joint-state-publisher
```

### 3. Initialize rosdep

Initialize rosdep for dependency management:

```bash
sudo rosdep init
rosdep update
```

## Development Environment Configuration

### 1. Set Up Workspace Structure

Create a standard workspace structure for the course:

```bash
# Create main robotics workspace
mkdir -p ~/robotics_ws/src
mkdir -p ~/robotics_ws/logs
mkdir -p ~/robotics_ws/build
mkdir -p ~/robotics_ws/install

# Create specific directories for different parts of the course
mkdir -p ~/robotics_ws/src/simulation
mkdir -p ~/robotics_ws/src/control
mkdir -p ~/robotics_ws/src/perception
mkdir -p ~/robotics_ws/src/navigation
mkdir -p ~/robotics_ws/src/hardware
```

### 2. Install and Configure IDE

Install VS Code with ROS 2 extensions:

```bash
# Download and install VS Code
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64,arm64 signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install -y code

# Install useful VS Code extensions
code --install-extension ms-vscode.cpptools
code --install-extension ms-python.python
code --install-extension redhat.vscode-yaml
code --install-extension twxs.cmake
code --install-extension ros-ide.extras
```

### 3. Configure Shell Environment

Enhance the shell environment for robotics development:

```bash
# Add to ~/.bashrc
cat << 'EOF' >> ~/.bashrc

# ROS 2 Environment
source /opt/ros/jazzy/setup.bash

# Robotics workspace
if [ -f ~/robotics_ws/install/setup.bash ]; then
    source ~/robotics_ws/install/setup.bash
fi

# ROS 2 Domain ID for network isolation
export ROS_DOMAIN_ID=1

# Colcon configuration
export COLCON_HOME=$HOME/.colcon
export COLCON_DEFAULTS_FILE=$HOME/.colcon/defaults.yaml

# Python path for robotics libraries
export PYTHONPATH=$HOME/robotics_ws/install/lib/python3.10/site-packages:$PYTHONPATH

# CUDA and GPU settings (if available)
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Performance settings
export RCUTILS_LOGGING_USE_STDOUT=1
export RCUTILS_LOGGING_BUFFERED_STREAM=1

# Aliases for common robotics commands
alias cw='cd ~/robotics_ws'
alias cs='cd ~/robotics_ws/src'
alias cb='cd ~/robotics_ws && colcon build --symlink-install'
alias sb='source ~/robotics_ws/install/setup.bash'
alias rb='cd ~/robotics_ws && colcon build --symlink-install && source install/setup.bash'

# Git aliases for robotics projects
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline'
EOF

# Create colcon defaults file
mkdir -p ~/.colcon
cat << 'EOF' > ~/.colcon/defaults.yaml
{
    "build": {
        "cmake-args": [
            "-DCMAKE_BUILD_TYPE=Release"
        ],
        "ament-cmake-args": [
            "-DCMAKE_BUILD_TYPE=Release"
        ],
        "parallel-workers": 4
    }
}
EOF
```

### 4. Install Additional Development Tools

Install tools that will be useful throughout the course:

```bash
# Install Docker for containerized development
sudo apt install -y docker.io
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install -y docker-compose-v2

# Install version control tools
sudo apt install -y gitk git-gui meld

# Install profiling and debugging tools
sudo apt install -y valgrind gdb strace
sudo apt install -y htop iotop nethogs

# Install documentation tools
sudo apt install -y doxygen graphviz
pip3 install sphinx sphinx-rtd-theme
```

## Python Environment Setup

### 1. Set Up Python Virtual Environment

Create a virtual environment for Python robotics development:

```bash
# Create a virtual environment
python3 -m venv ~/robotics_env
source ~/robotics_env/bin/activate

# Upgrade pip and install basic packages
pip install --upgrade pip setuptools wheel

# Install common Python libraries for robotics
pip install numpy scipy matplotlib
pip install opencv-python opencv-contrib-python
pip install pyyaml transforms3d
pip install pandas jupyter jupyterlab
pip install plotly dash
pip install requests

# Install ROS 2 Python tools
pip install ros-numpy
pip install rospkg catkin_pkg
```

### 2. Configure Jupyter for Robotics Development

Set up Jupyter for robotics experimentation:

```bash
# Create Jupyter config directory
mkdir -p ~/.jupyter

# Create Jupyter configuration
cat << 'EOF' > ~/.jupyter/jupyter_notebook_config.py
c = get_config()

# Allow remote access (for development environments)
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.allow_origin = '*'
c.NotebookApp.token = ''  # Set password in production
c.NotebookApp.disable_check_xsrf = True
EOF

# Install Jupyter extensions for robotics
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable --py --sys-prefix plotlywidget
```

## Simulation Environment Setup

### 1. Install Gazebo Garden (or Fortress as appropriate)

Install the Gazebo simulation environment:

```bash
# Add Gazebo repository
sudo curl -sSL http://get.gazebosim.org | sh

# Install Gazebo Garden
sudo apt install -y gz-garden

# Install additional Gazebo plugins and tools
sudo apt install -y libgazebo-dev
sudo apt install -y gazebo-plugin-base gazebo-plugin-libs
```

### 2. Configure Simulation Environment Variables

Add simulation-specific environment variables:

```bash
# Add to ~/.bashrc
cat << 'EOF' >> ~/.bashrc

# Gazebo Environment
source /usr/share/gz/setup.sh

# Gazebo models and worlds
export GZ_SIM_RESOURCE_PATH=$HOME/robotics_ws/src/simulation/models:$GZ_SIM_RESOURCE_PATH
export GZ_SIM_RESOURCE_PATH=/usr/share/gazebo/models:$GZ_SIM_RESOURCE_PATH

# Ignition Gazebo (older versions)
export IGN_GAZEBO_RESOURCE_PATH=$HOME/robotics_ws/src/simulation/worlds:$IGN_GAZEBO_RESOURCE_PATH
export IGN_GAZEBO_MODEL_PATH=$HOME/robotics_ws/src/simulation/models:$IGN_GAZEBO_MODEL_PATH
EOF
```

## Performance Optimization

### 1. System Tuning for Real-time Performance

Optimize the system for robotics applications:

```bash
# Install and configure CPU frequency scaling
sudo apt install -y cpufrequtils
sudo cpufreq-set -g performance

# Create a script to optimize system for robotics development
cat << 'EOF' > ~/setup_robotics_env.sh
#!/bin/bash
echo "Setting up Ubuntu 22.04 for robotics development..."

# Set CPU governor to performance mode
sudo cpufreq-set -g performance

# Increase file watchers (useful for development tools)
echo "fs.inotify.max_user_watches=524288" | sudo tee -a /etc/sysctl.conf
echo "fs.inotify.max_user_instances=256" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Increase shared memory size (for simulation)
echo "kernel.shmmax=134217728" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

echo "Robotics development environment setup complete."
EOF

chmod +x ~/setup_robotics_env.sh
```

### 2. Network Configuration for Robotics

Configure network settings appropriate for robotics:

```bash
# Create a script to set up ROS 2 networking
cat << 'EOF' > ~/setup_ros_network.sh
#!/bin/bash
echo "Setting up ROS 2 networking..."

# Set up ROS 2 domain ID
export ROS_DOMAIN_ID=1

# Set up multicast settings for ROS 2 discovery
echo "Setting up multicast for ROS 2..."
# Multicast settings are typically fine with defaults, but you can tune if needed

# Set up firewall for ROS 2
sudo ufw allow 11311  # Default ROS master port
sudo ufw allow 11345  # Default DDS port range start
sudo ufw allow 11346  # Default DDS port range end
sudo ufw allow 8080   # For web interfaces
sudo ufw allow 5672   # For AMQP if using

echo "ROS 2 networking setup complete."
EOF

chmod +x ~/setup_ros_network.sh
```

## Testing the Environment

### 1. Verify ROS 2 Installation

Test the ROS 2 installation:

```bash
# Source the ROS 2 environment
source /opt/ros/jazzy/setup.bash

# Check ROS 2 version
ros2 --version

# List available packages
ros2 pkg list | head -20

# Test basic ROS 2 functionality
# Terminal 1:
# ros2 run demo_nodes_cpp talker

# Terminal 2:
# ros2 run demo_nodes_py listener
```

### 2. Test Build System

Test the build system with a simple package:

```bash
cd ~/robotics_ws/src

# Create a test package
ros2 pkg create --build-type ament_cmake test_pkg

# Build the workspace
cd ~/robotics_ws
colcon build --packages-select test_pkg

# Source the workspace
source install/setup.bash

# Verify the package was built
ros2 pkg list | grep test_pkg
```

### 3. Test Python Environment

Test the Python environment:

```bash
# Test Python libraries
python3 -c "import numpy; print('NumPy version:', numpy.__version__)"
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"
python3 -c "import matplotlib; print('Matplotlib version:', matplotlib.__version__)"
```

## Course-Specific Configuration

### 1. Create Course Directory Structure

Set up directories specific to the course:

```bash
# Create course-specific directories
mkdir -p ~/robotics_course/{week1,week2,week3,week4,week5,week6,week7,week8,week9,week10,week11,week12,week13}
mkdir -p ~/robotics_course/projects/{hw1,hw2,hw3,hw4,midterm,final_project}
mkdir -p ~/robotics_course/simulations/{isaac_sim,gazebo,custom_worlds}
mkdir -p ~/robotics_course/datasets/{training,validation,testing}
```

### 2. Set Up Course-Specific Aliases

Add course-specific aliases to ~/.bashrc:

```bash
cat << 'EOF' >> ~/.bashrc

# Course-specific aliases
alias course='cd ~/robotics_course'
alias week1='cd ~/robotics_course/week1'
alias week2='cd ~/robotics_course/week2'
alias week3='cd ~/robotics_course/week3'
alias projects='cd ~/robotics_course/projects'
alias sims='cd ~/robotics_course/simulations'
alias datasets='cd ~/robotics_course/datasets'

# Course-specific functions
function start_week() {
    if [ -z "$1" ]; then
        echo "Usage: start_week <week_number>"
        return 1
    fi
    cd ~/robotics_course/week$1
    echo "Starting work on Week $1"
    mkdir -p exercises assignments notes
}

function submit_hw() {
    if [ -z "$1" ]; then
        echo "Usage: submit_hw <hw_number>"
        return 1
    fi
    echo "Preparing homework $1 for submission..."
    cd ~/robotics_course/projects/hw$1
    # Add submission logic here
}
EOF
```

## Troubleshooting

### Common Issues:

1. **Workspace Build Issues**: If colcon build fails, try:
   ```bash
   cd ~/robotics_ws
   rm -rf build install log
   colcon build --symlink-install --packages-select <specific_package>
   ```

2. **ROS 2 Network Issues**: If nodes can't communicate, check:
   ```bash
   # Ensure same ROS_DOMAIN_ID
   echo $ROS_DOMAIN_ID
   # Check network connectivity
   ros2 topic list
   ```

3. **Python Package Issues**: If Python packages aren't found:
   ```bash
   # Ensure virtual environment is activated
   source ~/robotics_env/bin/activate
   # Or check PYTHONPATH
   echo $PYTHONPATH
   ```

## Maintenance Scripts

Create useful maintenance scripts:

```bash
# Create a workspace cleanup script
cat << 'EOF' > ~/cleanup_workspace.sh
#!/bin/bash
echo "Cleaning up robotics workspace..."

cd ~/robotics_ws

# Remove build and log directories
rm -rf build install log

# Clean up any temporary files
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -delete

# Clean up system
sudo apt autoremove -y
sudo apt autoclean

echo "Workspace cleanup complete."
EOF

chmod +x ~/cleanup_workspace.sh
```

## Next Steps

After completing Ubuntu 22.04 LTS development environment configuration, proceed to:

1. Install Python 3.10+ and required robotics libraries
2. Set up GitHub repository with appropriate branching strategy
3. Test the complete development environment with a simple ROS 2 project

## References

- [ROS 2 Installation Guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debians.html)
- [Ubuntu 22.04 LTS Documentation](https://ubuntu.com/tutorials)
- [Gazebo Installation Guide](https://gazebosim.org/docs/garden/install)
- [Colcon Build Tool Documentation](https://colcon.readthedocs.io/en/released/)