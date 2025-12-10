# Isaac Sim 2024.2+ Development Environment Setup

This document outlines the steps to set up Isaac Sim 2024.2+ development environment for the Physical AI & Humanoid Robotics course.

## Prerequisites

- Ubuntu 22.04 LTS (or WSL2 with Ubuntu 22.04)
- NVIDIA GPU with RTX or GTX 10xx/20xx/30xx/40xx series
- NVIDIA Driver version 535 or higher
- CUDA 12.2
- Python 3.10 or 3.11 (Note: Python 3.12 may have compatibility issues with some Isaac Sim components)

## Installation Steps

### 1. Install NVIDIA Drivers and CUDA

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers (if not already installed)
sudo apt install nvidia-driver-550

# Install CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt update
sudo apt -y install cuda-12-2
```

### 2. Install Isaac Sim

The recommended approach is to download Isaac Sim from NVIDIA Developer website:

1. Go to [NVIDIA Isaac Sim Downloads](https://developer.nvidia.com/isaac-sim)
2. Create an NVIDIA Developer account if you don't have one
3. Download Isaac Sim 2024.2 or later
4. Follow the installation guide for your platform

Alternatively, you can use the Isaac Sim Omniverse launcher or container-based installation:

```bash
# Using Docker (recommended for development)
docker pull nvcr.io/nvidia/isaac-sim:4.2.0

# Create a docker run script
cat << 'EOF' > run_isaac_sim.sh
#!/bin/bash
xhost +local:docker
docker run --gpus all -it --rm \
  --network=host \
  --env "DISPLAY" \
  --env "QT_X11_NO_MITSHM=1" \
  --volume "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume "$(pwd):/workspace:rw" \
  --volume "~/.Xauthority:/root/.Xauthority:rw" \
  --volume "/tmp/.docker.xauth:/tmp/.docker.xauth:rw" \
  --privileged \
  --name isaac_sim \
  nvcr.io/nvidia/isaac-sim:4.2.0
EOF

chmod +x run_isaac_sim.sh
```

### 3. Configure Isaac Sim for Development

After installation, you need to set up the development environment:

```bash
# Create a virtual environment
python3 -m venv isaac_sim_env
source isaac_sim_env/bin/activate

# Install Isaac Sim Python dependencies
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install omni-isaac-gym-py
```

### 4. Verify Installation

```bash
# Test Isaac Sim installation
python3 -c "import omni; print('Isaac Sim Python API available')"
```

## Troubleshooting

### Common Issues:

1. **OpenGL Context Issues**: If you encounter OpenGL errors, ensure you have proper X11 forwarding set up
2. **CUDA Compatibility**: Make sure CUDA version matches Isaac Sim requirements
3. **GPU Memory**: Ensure sufficient GPU memory for simulation (minimum 8GB recommended)

### WSL2 Specific Setup:

For WSL2 users, ensure WSLg is enabled and GPU passthrough is configured:

```bash
# Check WSL version
wsl --list --verbose

# Update WSL kernel if needed
wsl --update

# Ensure WSLg is enabled (for Ubuntu 22.04)
grep -i wsl /proc/version
```

## Development Environment Configuration

### Environment Variables

Add these to your `.bashrc` or `.zshrc`:

```bash
export ISAACSIM_PATH="/path/to/isaac-sim"
export OMNI_USER="your_username"
export OMNI_PASS="your_password"
```

### IDE Configuration

For VS Code development with Isaac Sim:

1. Install the Python extension
2. Configure the interpreter to use the Isaac Sim virtual environment
3. Install the Isaac Sim extension if available

## Next Steps

After completing the Isaac Sim setup, proceed to:

1. Isaac ROS setup
2. Nav2 configuration
3. Testing with sample humanoid robot models

## References

- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
- [Isaac Sim GitHub Repository](https://github.com/isaac-sim/isaac-sim)
- [NVIDIA Developer Portal](https://developer.nvidia.com)