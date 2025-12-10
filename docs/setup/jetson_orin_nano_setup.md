# Jetson Orin Nano Development and Testing Environment Setup

This document outlines the steps to set up the Jetson Orin Nano development and testing environment for the Physical AI & Humanoid Robotics course.

## Prerequisites

- Jetson Orin Nano Developer Kit (16GB or 8GB RAM)
- Power adapter (official NVIDIA 19V/6.32A for 16GB version or 19V/4.74A for 8GB version)
- MicroSD card (at least 32GB, Class 10 or higher recommended)
- Ethernet cable or Wi-Fi access
- Host computer for flashing the Jetson (Ubuntu 18.04/20.04/22.04 recommended)

## Hardware Setup

### 1. Unboxing and Physical Setup

1. Remove the Jetson Orin Nano Developer Kit from its packaging
2. Attach the heatsink and fan assembly to the module (if not pre-installed)
3. Connect the power adapter to the DC jack
4. Connect an Ethernet cable to the Gigabit Ethernet port
5. Connect a micro-USB cable to the micro-USB port (for serial console access if needed)
6. Connect an HDMI display to the HDMI port
7. Connect a USB keyboard and mouse to the USB ports

### 2. Power On and Initial Boot

1. Connect the power adapter to the DC jack
2. The Jetson should boot automatically
3. If using a display, you should see the boot process on the screen

## Software Setup

### 1. Flashing Jetson Orin Nano

The Jetson Orin Nano needs to be flashed with the appropriate OS and NVIDIA software stack:

#### Option A: Using SDK Manager (Recommended for beginners)

1. On your host computer, download and install NVIDIA SDK Manager:
   ```bash
   # Go to https://developer.nvidia.com/nvidia-sdk-manager
   # Download and install SDK Manager for your host OS
   ```

2. Connect the Jetson Orin Nano to your host computer via USB-C:
   - Use a USB-C to USB-C cable
   - Connect to the USB-C port on the Jetson carrier board
   - Ensure the Jetson is powered off
   - Hold the FORCE RECOVERY button while connecting power
   - Continue holding the button for 10 seconds, then release

3. In SDK Manager:
   - Log in with your NVIDIA Developer account
   - Select "Jetson Orin Nano" as the target device
   - Select "Ubuntu 22.04" as the target OS
   - Select "JetPack 5.1.3" or later (includes ROS 2 Jazzy support)
   - Check "Install on Target" for Isaac ROS packages
   - Start the flashing process

#### Option B: Using command line tools

1. Install Jetson SDK components on your host computer:
   ```bash
   # Install dependencies
   sudo apt update
   sudo apt install -y python3-pip python3-dev
   pip3 install jetson-stats
   ```

2. Download the appropriate JetPack SDK for Jetson Orin Nano

### 2. Initial Configuration

After flashing, configure the Jetson Orin Nano:

1. Complete the initial Ubuntu setup:
   - Set up user account
   - Configure timezone
   - Connect to Wi-Fi if needed

2. Update the system:
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo reboot
   ```

3. Install essential development tools:
   ```bash
   sudo apt install -y build-essential cmake git vim htop
   sudo apt install -y python3-pip python3-dev python3-venv
   sudo apt install -y curl wget unzip
   ```

### 3. Install NVIDIA Software Stack

1. Verify CUDA installation:
   ```bash
   nvidia-smi
   nvcc --version
   ```

2. Install additional NVIDIA tools:
   ```bash
   # Install Jetson Inference (for AI/ML tasks)
   sudo apt install -y libjetson-inference-dev

   # Install Jetson Multimedia API
   sudo apt install -y libargus-dev libv4l-dev

   # Install NVIDIA Container Runtime
   sudo apt install -y nvidia-container-toolkit
   ```

### 4. Set Up ROS 2 Environment

1. Install ROS 2 Jazzy (or Iron as specified in the course):
   ```bash
   # Add ROS 2 repository
   sudo apt update && sudo apt install -y curl gnupg lsb-release
   curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

   # Install ROS 2 packages
   sudo apt update
   sudo apt install -y ros-jazzy-ros-base ros-jazzy-ros-core
   sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
   ```

2. Install additional ROS 2 packages needed for robotics:
   ```bash
   sudo apt install -y ros-jazzy-navigation2 ros-jazzy-navigation2-msgs
   sudo apt install -y ros-jazzy-teleop-tools ros-jazzy-joy
   sudo apt install -y ros-jazzy-robot-state-publisher ros-jazzy-joint-state-publisher
   sudo apt install -y ros-jazzy-xacro ros-jazzy-urdf ros-jazzy-urdf-tutorial
   sudo apt install -y ros-jazzy-gazebo-ros ros-jazzy-gazebo-plugins
   ```

### 5. Install Isaac ROS Packages

1. Install Isaac ROS packages for Jetson:
   ```bash
   sudo apt install -y ros-jazzy-isaac-ros-common
   sudo apt install -y ros-jazzy-isaac-ros-apriltag
   sudo apt install -y ros-jazzy-isaac-ros-visual-slam
   sudo apt install -y ros-jazzy-isaac-ros-image-pipeline
   sudo apt install -y ros-jazzy-isaac-ros-bi3d
   sudo apt install -y ros-jazzy-isaac-ros-centerpose
   sudo apt install -y ros-jazzy-isaac-ros-pose-estimation
   ```

### 6. Set Up Development Environment

1. Create a workspace for development:
   ```bash
   mkdir -p ~/robotics_ws/src
   cd ~/robotics_ws
   source /opt/ros/jazzy/setup.bash
   colcon build --symlink-install
   ```

2. Set up environment variables:
   ```bash
   # Add to ~/.bashrc
   echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
   echo "source ~/robotics_ws/install/setup.bash" >> ~/.bashrc
   echo "export ROS_DOMAIN_ID=1" >> ~/.bashrc  # Set domain ID for network isolation
   echo "export CUDA_DEVICE_ORDER=PCI_BUS_ID" >> ~/.bashrc
   source ~/.bashrc
   ```

### 7. Configure Performance Mode

1. Set Jetson to maximum performance mode:
   ```bash
   sudo nvpmodel -m 0  # Maximum performance mode
   sudo jetson_clocks  # Lock all clocks to maximum frequency
   ```

2. Check current performance status:
   ```bash
   sudo nvpmodel -q
   jetson_clocks --show
   ```

### 8. Set Up Remote Development

1. Enable SSH for remote access:
   ```bash
   sudo systemctl enable ssh
   sudo systemctl start ssh
   ```

2. Set up passwordless SSH (optional, for development convenience):
   ```bash
   ssh-keygen -t rsa
   # Follow prompts to create SSH key pair
   ```

### 9. Install Course-Specific Libraries

1. Install Python libraries for robotics:
   ```bash
   pip3 install --upgrade pip
   pip3 install numpy scipy matplotlib
   pip3 install opencv-python
   pip3 install pyyaml
   pip3 install transforms3d
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. Install additional AI/ML libraries:
   ```bash
   pip3 install tensorflow
   pip3 install scikit-learn
   pip3 install pandas
   pip3 install jupyter
   ```

### 10. Configure Jetson for Real-time Performance

1. Set up CPU governor for performance:
   ```bash
   sudo apt install -y cpufrequtils
   sudo cpufreq-set -g performance
   ```

2. Configure real-time scheduling (optional):
   ```bash
   # Add current user to real-time group
   sudo usermod -a -G realtime $USER
   sudo sh -c "echo '$USER soft rtprio 99' >> /etc/security/limits.conf"
   sudo sh -c "echo '$USER hard rtprio 99' >> /etc/security/limits.conf"
   ```

## Testing Environment

### 1. Verify Hardware

1. Check GPU status:
   ```bash
   nvidia-smi
   ```

2. Check thermal status:
   ```bash
   sudo tegrastats  # Shows real-time stats including temperature
   ```

3. Check memory and CPU usage:
   ```bash
   htop
   ```

### 2. Verify ROS 2 Installation

1. Test ROS 2:
   ```bash
   source /opt/ros/jazzy/setup.bash
   ros2 topic list
   ros2 node list
   ```

2. Run a simple ROS 2 test:
   ```bash
   # Terminal 1
   ros2 run demo_nodes_cpp talker

   # Terminal 2
   ros2 run demo_nodes_py listener
   ```

### 3. Test Isaac ROS

1. Verify Isaac ROS installation:
   ```bash
   ros2 pkg list | grep isaac
   ```

2. Test Isaac ROS functionality:
   ```bash
   # Run a simple Isaac ROS node (example)
   ros2 run isaac_ros_apriltag apriltag_node
   ```

## Network Configuration

### 1. Set up Static IP (Optional)

For consistent network access in robotics applications:

```bash
# Edit netplan configuration
sudo nano /etc/netplan/01-network-manager-all.yaml

# Add static IP configuration (example):
network:
  version: 2
  renderer: networkd
  ethernets:
    eth0:
      dhcp4: false
      addresses: [192.168.1.100/24]
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
```

Apply the configuration:
```bash
sudo netplan apply
```

### 2. Configure Firewall

```bash
# Allow ROS 2 traffic
sudo ufw allow 11311  # Default ROS master port
sudo ufw allow 5672   # For AMQP if using
sudo ufw allow 8080   # For web interfaces if using
sudo ufw enable
```

## Optimization for Course Requirements

### 1. Performance Tuning

1. Optimize for real-time performance:
   ```bash
   # Create a script to set performance mode
   cat << 'EOF' > ~/setup_performance.sh
   #!/bin/bash
   echo "Setting up Jetson for optimal performance..."
   sudo nvpmodel -m 0
   sudo jetson_clocks
   sudo cpufreq-set -g performance
   echo "Performance setup complete."
   EOF

   chmod +x ~/setup_performance.sh
   ```

### 2. Power Management

1. For extended operation, consider power management:
   ```bash
   # Check power mode
   sudo tegrastats | grep -i power

   # To reduce power consumption when not needed:
   sudo nvpmodel -m 1  # For 10W mode (if available on your model)
   ```

## Troubleshooting

### Common Issues:

1. **Thermal Throttling**: If the Jetson overheats, ensure proper cooling and consider:
   ```bash
   # Monitor temperatures
   sudo tegrastats | grep -i temp
   ```

2. **Memory Issues**: Monitor memory usage:
   ```bash
   free -h
   # If needed, add swap space
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

3. **GPU Memory**: Monitor GPU memory:
   ```bash
   nvidia-smi -q -d MEMORY
   ```

## Maintenance Scripts

Create useful maintenance scripts:

```bash
# Create a system status script
cat << 'EOF' > ~/jetson_status.sh
#!/bin/bash
echo "=== Jetson Orin Nano Status ==="
echo "Time: $(date)"
echo "Uptime: $(uptime)"
echo ""
echo "=== System Info ==="
cat /etc/nv_tegra_release
echo ""
echo "=== Memory ==="
free -h
echo ""
echo "=== GPU ==="
nvidia-smi
echo ""
echo "=== Temperature ==="
sudo tegrastats | head -n 10
EOF

chmod +x ~/jetson_status.sh
```

## Next Steps

After completing Jetson Orin Nano setup, proceed to:

1. Configure Ubuntu 22.04 LTS development environment on development machines
2. Install Python 3.10+ and required robotics libraries
3. Set up GitHub repository with appropriate branching strategy

## References

- [NVIDIA Jetson Orin Nano Developer Guide](https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit)
- [JetPack SDK Documentation](https://docs.nvidia.com/jetson/archives/r35.4.1/index.html)
- [ROS 2 Installation Guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debians.html)
- [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/index.html)