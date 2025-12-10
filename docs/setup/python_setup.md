# Python 3.10+ and Robotics Libraries Installation Guide

This document outlines the steps to install Python 3.10+ and required robotics libraries for the Physical AI & Humanoid Robotics course.

## Prerequisites

- Ubuntu 22.04 LTS with development tools installed
- Internet connection
- Administrative privileges (sudo access)

## Python Version Check

First, verify the current Python version:

```bash
python3 --version
```

Ubuntu 22.04 LTS comes with Python 3.10+ by default, which is suitable for this course. If you need to install or upgrade Python, follow the steps below.

## Installing Python 3.10+ (if needed)

### Option 1: Using deadsnakes PPA (for older Ubuntu versions or specific Python versions)

```bash
# Add the deadsnakes PPA
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.10 (or 3.11/3.12 as needed)
sudo apt install -y python3.10 python3.10-dev python3.10-venv
sudo apt install -y python3.11 python3.11-dev python3.11-venv  # Alternative
sudo apt install -y python3.12 python3.12-dev python3.12-venv  # Alternative

# Install pip for the specific Python version
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11  # If needed
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12  # If needed
```

### Option 2: Using pyenv (for managing multiple Python versions)

```bash
# Install dependencies
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
libffi-dev liblzma-dev

# Install pyenv
curl https://pyenv.run | bash

# Add pyenv to shell configuration
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Reload shell configuration
source ~/.bashrc

# Install Python 3.10+ using pyenv
pyenv install 3.10.14
pyenv install 3.11.8
pyenv install 3.12.1

# Set global Python version
pyenv global 3.10.14
```

## Creating Python Virtual Environment

Create a dedicated virtual environment for robotics development:

```bash
# Create a virtual environment
python3 -m venv ~/robotics_venv

# Activate the virtual environment
source ~/robotics_venv/bin/activate

# Upgrade pip to the latest version
pip install --upgrade pip setuptools wheel

# Verify Python version in the virtual environment
python --version
```

## Core Robotics Libraries

Install the core Python libraries required for robotics:

```bash
# Activate the virtual environment
source ~/robotics_venv/bin/activate

# Scientific computing libraries
pip install numpy==1.24.3 scipy==1.11.4 matplotlib==3.7.4

# Computer vision libraries
pip install opencv-python==4.8.1.78 opencv-contrib-python==4.8.1.78

# Data manipulation and analysis
pip install pandas==2.1.3

# Configuration and serialization
pip install pyyaml==6.0.1

# 3D transformations
pip install transforms3d==0.4.1

# HTTP requests
pip install requests==2.31.0

# Progress bars
pip install tqdm==4.66.1

# Type hints support
pip install typing-extensions==4.8.0
```

## AI and Machine Learning Libraries

Install AI/ML libraries for the Physical AI components:

```bash
# Activate the virtual environment
source ~/robotics_venv/bin/activate

# PyTorch for deep learning (with CUDA support if available)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# TensorFlow (alternative deep learning framework)
pip install tensorflow==2.14.0

# Scikit-learn for classical ML
pip install scikit-learn==1.3.2

# ONNX for model interoperability
pip install onnx==1.15.0

# ONNX Runtime for optimized inference
pip install onnxruntime==1.16.3 onnxruntime-gpu==1.16.3

# Hugging Face transformers (for LLM integration)
pip install transformers==4.35.2 accelerate==0.24.1

# Sentence transformers for embeddings
pip install sentence-transformers==2.2.2
```

## Robotics-Specific Libraries

Install Python libraries specifically for robotics applications:

```bash
# Activate the virtual environment
source ~/robotics_venv/bin/activate

# ROS 2 Python interfaces
pip install ros-numpy==1.0.0
pip install rospkg==1.5.0
pip install catkin-pkg==1.0.0

# Robot kinematics and dynamics
pip install pybullet==3.2.5  # Physics simulation
pip install roboticstoolbox-python==1.0.0  # Robotics toolbox
pip install spatialgeometry==0.1.0  # Spatial geometry tools
pip install swift==0.10.0  # Robotics simulation environment

# Control systems
pip install control==0.9.1  # Control systems library
pip install slycot==0.5.4  # Optional dependency for control systems

# Optimization
pip install casadi==3.6.4  # Optimization and optimal control
```

## Computer Vision and Perception Libraries

Install libraries for computer vision and perception:

```bash
# Activate the virtual environment
source ~/robotics_venv/bin/activate

# Image processing
pip install Pillow==10.1.0
pip install imageio==2.31.6

# Feature detection and matching
pip install scikit-image==0.22.0
pip install mahotas==1.4.11

# Point cloud processing
pip install open3d==0.18.0
pip install python-pcl==0.4.0  # Python wrapper for PCL

# 3D mesh processing
pip install trimesh==4.3.9
pip install meshio==5.3.4
```

## Audio Processing Libraries

Install libraries for audio processing (for voice control features):

```bash
# Activate the virtual environment
source ~/robotics_venv/bin/activate

# Audio processing
pip install pyaudio==0.2.14
pip install sounddevice==0.4.6
pip install soundfile==0.12.1

# Speech recognition
pip install SpeechRecognition==3.10.0
pip install pydub==0.25.1

# OpenAI Whisper for speech-to-text
pip install openai-whisper==20231117

# Additional audio libraries
pip install librosa==0.10.1
```

## Development and Visualization Tools

Install development and visualization tools:

```bash
# Activate the virtual environment
source ~/robotics_venv/bin/activate

# Jupyter notebooks
pip install jupyter==1.0.0 jupyterlab==4.0.8 notebook==7.0.6

# Interactive plotting
pip install plotly==5.17.0
pip install dash==2.14.1

# Visualization
pip install seaborn==0.13.0
pip install plotnine==0.12.1

# 3D visualization
pip install vtk==9.2.6
pip install mayavi==4.8.2

# Web frameworks (for web interfaces)
pip install flask==3.0.0
pip install fastapi==0.104.1 uvicorn==0.24.0
pip install python-multipart==0.0.6
```

## Testing Libraries

Install libraries for testing and validation:

```bash
# Activate the virtual environment
source ~/robotics_venv/bin/activate

# Testing frameworks
pip install pytest==7.4.3 pytest-cov==4.1.0
pip install pytest-asyncio==0.21.1

# Mocking and testing utilities
pip install mock==5.1.0
pip install responses==0.24.1

# Linting and formatting
pip install flake8==6.1.0
pip install black==23.10.1
pip install isort==5.12.0
```

## Creating Requirements File

Create a requirements file to track installed packages:

```bash
# Activate the virtual environment
source ~/robotics_venv/bin/activate

# Generate requirements file
pip freeze > ~/robotics_venv/requirements.txt

# Create a more manageable requirements file for the course
cat << 'EOF' > ~/robotics_venv/requirements-course.txt
# Core Scientific Libraries
numpy==1.24.3
scipy==1.11.4
matplotlib==3.7.4
pandas==2.1.3
pyyaml==6.0.1
transforms3d==0.4.1

# Computer Vision
opencv-python==4.8.1.78
opencv-contrib-python==4.8.1.78
Pillow==10.1.0
scikit-image==0.22.0

# AI/ML Libraries
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
tensorflow==2.14.0
scikit-learn==1.3.2
transformers==4.35.2

# Robotics Libraries
ros-numpy==1.0.0
pybullet==3.2.5
roboticstoolbox-python==1.0.0

# Audio Processing
pyaudio==0.2.14
SpeechRecognition==3.10.0
openai-whisper==20231117

# Development Tools
jupyter==1.0.0
plotly==5.17.0
seaborn==0.13.0
EOF
```

## Environment Configuration

Configure the environment to automatically activate the robotics virtual environment:

```bash
# Add to ~/.bashrc to automatically activate robotics environment
cat << 'EOF' >> ~/.bashrc

# Robotics Python Environment
if [ -d "$HOME/robotics_venv" ]; then
    source $HOME/robotics_venv/bin/activate
fi

# Python aliases for robotics development
alias python-robotics='source $HOME/robotics_venv/bin/activate && python'
alias pip-robotics='source $HOME/robotics_venv/bin/activate && pip'
alias jupyter-robotics='source $HOME/robotics_venv/bin/activate && jupyter lab'
EOF

# Reload bash configuration
source ~/.bashrc
```

## Verification and Testing

Test the Python environment installation:

```bash
# Activate the virtual environment
source ~/robotics_venv/bin/activate

# Test core libraries
python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import rospy; print('ROS Python available')" 2>/dev/null || echo "ROS Python not available (expected if not in ROS environment)"

# Test robotics libraries
python -c "import transforms3d; print('Transforms3D available')"
python -c "import pybullet; print('PyBullet available')"

# Test audio libraries
python -c "import pyaudio; print('PyAudio available')"
python -c "import speech_recognition; print('SpeechRecognition available')"

# Test visualization libraries
python -c "import matplotlib; print('Matplotlib available')"
python -c "import plotly; print('Plotly available')"
```

## Course-Specific Python Setup

Create a Python package structure for the course:

```bash
# Create course Python package
mkdir -p ~/robotics_course/python_packages/physical_ai_robotics
cd ~/robotics_course/python_packages/physical_ai_robotics

# Create __init__.py
touch __init__.py

# Create common utility modules
cat << 'EOF' > utils.py
"""
Utility functions for the Physical AI & Humanoid Robotics course.
"""
import numpy as np
import cv2
import torch
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Point, Pose, Quaternion
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2
import ros_numpy


def pose_to_transformation_matrix(pose):
    """
    Convert ROS Pose message to 4x4 transformation matrix.
    """
    import transforms3d
    pos = [pose.position.x, pose.position.y, pose.position.z]
    quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    rot_matrix = transforms3d.quaternions.quat2mat(quat)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rot_matrix
    transformation_matrix[:3, 3] = pos
    return transformation_matrix


def transformation_matrix_to_pose(matrix):
    """
    Convert 4x4 transformation matrix to ROS Pose message.
    """
    import transforms3d
    pos = matrix[:3, 3]
    rot_matrix = matrix[:3, :3]
    quat = transforms3d.quaternions.mat2quat(rot_matrix)

    pose = Pose()
    pose.position.x = pos[0]
    pose.position.y = pos[1]
    pose.position.z = pos[2]
    pose.orientation.x = quat[1]  # Note: transforms3d returns [w, x, y, z]
    pose.orientation.y = quat[2]
    pose.orientation.z = quat[3]
    pose.orientation.w = quat[0]

    return pose


def image_to_numpy(image_msg):
    """
    Convert ROS Image message to numpy array.
    """
    return ros_numpy.numpify(image_msg)


def numpy_to_image(numpy_img, encoding='rgb8'):
    """
    Convert numpy array to ROS Image message.
    """
    return ros_numpy.msgify(Image, numpy_img, encoding=encoding)


def pointcloud_to_numpy(pointcloud_msg):
    """
    Convert ROS PointCloud2 message to numpy array.
    """
    return ros_numpy.numpify(pointcloud_msg)


def get_rotation_matrix_from_vectors(v1, v2):
    """
    Get rotation matrix to rotate vector v1 to align with vector v2.
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Calculate cross product and dot product
    cross_product = np.cross(v1, v2)
    dot_product = np.dot(v1, v2)

    # Handle special cases
    if np.allclose(cross_product, 0) and dot_product > 0:
        # Vectors are parallel
        return np.eye(3)
    elif np.allclose(cross_product, 0) and dot_product < 0:
        # Vectors are anti-parallel
        # Find an arbitrary perpendicular vector
        if not np.allclose(v1[:2], 0):
            perp = np.array([-v1[1], v1[0], 0])
        else:
            perp = np.array([1, 0, 0])
        perp = perp / np.linalg.norm(perp)
        rotation_matrix = 2 * np.outer(perp, perp) - np.eye(3)
        return rotation_matrix

    # Calculate rotation matrix using Rodrigues' formula
    skew_symmetric = np.array([
        [0, -cross_product[2], cross_product[1]],
        [cross_product[2], 0, -cross_product[0]],
        [-cross_product[1], cross_product[0], 0]
    ])

    rotation_matrix = np.eye(3) + skew_symmetric + \
        np.dot(skew_symmetric, skew_symmetric) * (1 / (1 + dot_product))

    return rotation_matrix
EOF

# Create a setup.py for the package
cat << 'EOF' > setup.py
from setuptools import setup, find_packages

setup(
    name='physical_ai_robotics',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'opencv-python',
        'torch',
        'transforms3d',
        'pybullet',
        'rospkg',
        'catkin_pkg',
        'ros-numpy',
    ],
    author='Physical AI & Humanoid Robotics Course',
    description='Python utilities for the Physical AI & Humanoid Robotics course',
    python_requires='>=3.10',
)
EOF

# Create a requirements file for the course package
cat << 'EOF' > requirements.txt
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
opencv-python>=4.8.0
torch>=2.1.0
transforms3d>=0.4.1
pybullet>=3.2.5
rospkg>=1.5.0
catkin_pkg>=1.0.0
ros-numpy>=1.0.0
EOF
```

## Troubleshooting

### Common Issues:

1. **CUDA Compatibility**: If PyTorch CUDA operations fail, verify CUDA installation:
   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
   ```

2. **OpenCV Build Issues**: If OpenCV fails to import, try installing from conda-forge:
   ```bash
   # In the virtual environment
   pip uninstall opencv-python opencv-contrib-python
   conda install -c conda-forge opencv
   ```

3. **Memory Issues**: For large models, consider using:
   ```python
   # Enable memory-efficient attention mechanisms
   import torch
   torch.backends.cuda.matmul.allow_tf32 = True
   ```

4. **Package Conflicts**: If there are conflicts between packages:
   ```bash
   # Create a new clean environment
   deactivate
   rm -rf ~/robotics_venv
   python3 -m venv ~/robotics_venv
   source ~/robotics_venv/bin/activate
   # Reinstall packages in order of dependencies
   ```

## Performance Optimization

### 1. Configure PyTorch for Performance

```bash
# Create a PyTorch configuration in the virtual environment
cat << 'EOF' > ~/robotics_venv/torch_config.py
import torch

# Configure PyTorch for optimal performance
torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
torch.backends.cudnn.deterministic = False  # Faster but non-deterministic
torch.backends.cuda.matmul.allow_tf32 = True  # Use TensorFloat32 for matrix ops

# Memory optimization
torch.set_float32_matmul_precision('high')  # Use tensor cores when possible

print("PyTorch performance optimizations applied.")
EOF

# Add to Python startup
echo "source ~/robotics_venv/torch_config.py" >> ~/.bashrc
```

### 2. Set Up Jupyter for Robotics Development

```bash
# Create Jupyter configuration for robotics
mkdir -p ~/.jupyter
cat << 'EOF' > ~/.jupyter/jupyter_notebook_config.py
c = get_config()

# Allow remote access for development environments
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.allow_origin = '*'
c.NotebookApp.token = ''
c.NotebookApp.disable_check_xsrf = True

# Configure for robotics development
c.NotebookApp.contents_manager_class = 'notebook.services.contents.filemanager.FileContentsManager'
c.NotebookApp.kernel_spec_manager_class = 'jupyter_client.kernelspec.KernelSpecManager'
EOF

# Create a robotics-specific Jupyter kernel
source ~/robotics_venv/bin/activate
python -m ipykernel install --user --name=robotics --display-name="Python (Robotics)"
```

## Next Steps

After completing Python 3.10+ and robotics libraries installation, proceed to:

1. Set up GitHub repository with appropriate branching strategy
2. Create Python-based examples and tutorials for the course
3. Test the complete Python environment with robotics examples

## References

- [Python Package Index (PyPI)](https://pypi.org/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [OpenCV Python Installation](https://pypi.org/project/opencv-python/)
- [ROS Python Libraries](https://wiki.ros.org/rospy)
- [Transforms3D Documentation](https://matthew-brett.github.io/transforms3d/)