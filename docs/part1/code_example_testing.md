---
sidebar_position: 5
title: "Code Example Testing on Target Hardware"
description: "Process for testing all code examples on target hardware to meet performance requirements"
---

# Code Example Testing on Target Hardware

## Overview

This document outlines the process for testing all code examples on target hardware to ensure they meet the performance requirements (≥15 Hz on Jetson Orin Nano) for the Physical AI & Humanoid Robotics course. This testing ensures that all examples are not only theoretically correct but also practically executable on the target platform.

## Testing Requirements

### Performance Standards
- **Minimum Frequency**: ≥15 Hz for real-time operation
- **Maximum Memory**: &lt;2GB RAM consumption
- **Maximum CPU**: &lt;80% average CPU usage
- **Target Platform**: Jetson Orin Nano 8GB

### Hardware Specifications
- **Platform**: Jetson Orin Nano 8GB
- **CPU**: 6-core ARM Cortex-A78AE v8.2 64-bit
- **GPU**: NVIDIA Ampere architecture with 1024 CUDA cores
- **Memory**: 8GB 128-bit LPDDR5 @ 6000 MT/s
- **Storage**: MicroSD card or eMMC

## Testing Infrastructure

### Target Hardware Setup
1. **Jetson Orin Nano Development Kit**
   - Properly configured with Ubuntu 22.04 LTS
   - ROS 2 Jazzy installed and configured
   - Isaac ROS packages installed
   - All necessary drivers and dependencies

2. **Performance Monitoring Tools**
   - Jetson Stats (`jtop`) for real-time monitoring
   - System resource monitoring scripts
   - Frequency and timing measurement tools

### Testing Environment
- **Development Environment**: Ubuntu 22.04 LTS
- **ROS 2 Distribution**: Jazzy (or Iron as specified)
- **Isaac Sim**: 2024.2+ for simulation examples
- **Isaac ROS**: 3.0+ for perception and navigation

## Testing Process

### 1. Pre-Testing Setup
```bash
# Set Jetson to maximum performance mode
sudo nvpmodel -m 0  # Maximum performance mode
sudo jetson_clocks  # Lock all clocks to maximum frequency

# Verify performance mode
sudo nvpmodel -q
jetson_clocks --show
```

### 2. Code Example Classification

#### A. ROS 2 Node Examples
- Basic publisher/subscriber patterns
- Service and action implementations
- Parameter server usage
- TF transforms and coordinate frames

#### B. Perception Pipeline Examples
- Image processing and computer vision
- Sensor data processing
- Feature detection and matching
- Point cloud processing

#### C. Control System Examples
- Joint trajectory control
- PID controller implementations
- Inverse kinematics solutions
- Whole-body control algorithms

#### D. Navigation Examples
- Path planning algorithms
- Obstacle avoidance
- Localization and mapping
- Navigation stack integration

### 3. Performance Measurement Protocol

#### Frequency Measurement
```python
import time
import rclpy
from rclpy.node import Node

class PerformanceTester(Node):
    def __init__(self):
        super().__init__('performance_tester')
        self.start_time = time.time()
        self.iteration_count = 0
        self.frequency_samples = []

    def measure_frequency(self):
        current_time = time.time()
        elapsed = current_time - self.start_time
        self.iteration_count += 1

        if elapsed >= 1.0:  # Calculate frequency every second
            frequency = self.iteration_count / elapsed
            self.frequency_samples.append(frequency)
            self.get_logger().info(f'Current frequency: {frequency:.2f} Hz')

            # Reset for next measurement
            self.start_time = current_time
            self.iteration_count = 0

        return frequency
```

#### Resource Monitoring
```bash
# Monitor system resources during testing
# Memory usage
free -h

# CPU usage
top -b -n 1 | grep "Cpu(s)"

# GPU usage (if available)
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv

# Temperature monitoring
cat /sys/class/thermal/thermal_zone*/temp
```

### 4. Testing Procedures

#### Procedure A: ROS 2 Node Testing
1. **Build the example**
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select [package_name]
   source install/setup.bash
   ```

2. **Run with performance monitoring**
   ```bash
   # Terminal 1: Run the node
   ros2 run [package_name] [node_name]

   # Terminal 2: Monitor performance
   python3 scripts/monitor_performance.py --node [node_name]
   ```

3. **Record performance metrics**
   - Execution frequency
   - Memory consumption
   - CPU usage
   - Success rate

#### Procedure B: Simulation Integration Testing
1. **Launch Isaac Sim environment**
   ```bash
   # Launch simulation with example
   ros2 launch [simulation_package] [launch_file]
   ```

2. **Monitor integrated performance**
   - Combined simulation + ROS performance
   - Rendering performance
   - Control loop frequency

#### Procedure C: Hardware-in-Loop Testing
1. **Connect to physical hardware** (if available)
2. **Run control algorithms**
3. **Monitor real-time performance**
4. **Compare with simulation results**

## Test Scenarios

### Scenario 1: Basic ROS 2 Operations
- **Objective**: Verify basic ROS 2 functionality
- **Metrics**: Message frequency ≥ 15 Hz
- **Duration**: 5 minutes continuous operation
- **Acceptance**: 95% of samples ≥ 15 Hz

### Scenario 2: Perception Pipeline
- **Objective**: Test computer vision processing
- **Metrics**: Image processing rate ≥ 15 Hz
- **Duration**: 10 minutes with continuous input
- **Acceptance**: Average ≥ 15 Hz, peak memory < 1.5GB

### Scenario 3: Control Loop
- **Objective**: Validate real-time control
- **Metrics**: Control frequency ≥ 100 Hz (for critical systems)
- **Duration**: 30 seconds high-frequency operation
- **Acceptance**: ≥ 90 Hz sustained frequency

### Scenario 4: Navigation Stack
- **Objective**: Test navigation performance
- **Metrics**: Planning and execution frequency ≥ 10 Hz
- **Duration**: 5 minutes with continuous navigation
- **Acceptance**: ≥ 10 Hz with < 80% CPU usage

## Performance Validation Scripts

### Basic Performance Monitor
```bash
#!/bin/bash
# performance_monitor.sh - Monitor system during code execution

echo "Starting performance monitoring..."

# Monitor memory usage
echo "Memory usage at start:"
free -h

# Monitor CPU and GPU during execution
nvidia-smi dmon -s u -d 1 > gpu_monitor.log &
GPU_MON_PID=$!

# Run the target code example
echo "Running code example..."
$@

# Stop monitoring
kill $GPU_MON_PID 2>/dev/null

echo "Memory usage at end:"
free -h

echo "Performance monitoring complete."
```

### Frequency Validation Script
```python
#!/usr/bin/env python3
# frequency_validator.py - Validate execution frequency

import time
import sys
import subprocess

def validate_frequency(test_duration=60, min_frequency=15):
    """
    Validate that code executes at minimum required frequency
    """
    start_time = time.time()
    iteration_count = 0
    frequency_samples = []

    print(f"Validating frequency for {test_duration} seconds...")
    print(f"Minimum required frequency: {min_frequency} Hz")

    try:
        while time.time() - start_time < test_duration:
            # Record start time for this iteration
            iter_start = time.time()

            # Execute the test code
            # (This would be replaced with actual code execution)
            time.sleep(0.01)  # Simulate work

            # Calculate iteration time
            iter_time = time.time() - iter_start
            if iter_time > 0:
                iter_frequency = 1.0 / iter_time
                frequency_samples.append(iter_frequency)
                iteration_count += 1

            # Brief pause to allow other processes
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("Validation interrupted by user")

    # Calculate results
    if frequency_samples:
        avg_frequency = sum(frequency_samples) / len(frequency_samples)
        min_frequency_observed = min(frequency_samples)
        max_frequency_observed = max(frequency_samples)

        print(f"Results:")
        print(f"  Total iterations: {iteration_count}")
        print(f"  Average frequency: {avg_frequency:.2f} Hz")
        print(f"  Minimum frequency: {min_frequency_observed:.2f} Hz")
        print(f"  Maximum frequency: {max_frequency_observed:.2f} Hz")

        # Check if minimum requirement is met
        if min_frequency_observed >= min_frequency:
            print(f"  ✓ PASS: Minimum frequency requirement met")
            return True
        else:
            print(f"  ✗ FAIL: Minimum frequency requirement not met")
            return False
    else:
        print("  ✗ FAIL: No frequency samples collected")
        return False

if __name__ == "__main__":
    # Default values can be overridden by command line
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    min_freq = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0

    success = validate_frequency(duration, min_freq)
    sys.exit(0 if success else 1)
```

## Testing Report Template

### Individual Example Test Report
```
Code Example Test Report
========================

Example: [Example Name and Location]
Tester: [Name]
Date: [Date]
Hardware: Jetson Orin Nano 8GB

Test Configuration:
- ROS 2 Distribution: [Version]
- Isaac ROS: [Version]
- Performance Mode: [Mode]
- Environmental Conditions: [Temperature, etc.]

Performance Results:
- Average Frequency: XX.XX Hz
- Minimum Frequency: XX.XX Hz
- Maximum Frequency: XX.XX Hz
- Memory Usage: XX.XX MB
- CPU Usage: XX.XX%
- Success Rate: XX.XX%

Test Log:
[Detailed test log showing frequency measurements over time]

Pass/Fail: [PASS/FAIL]
Notes: [Any relevant observations or issues]

Tester Signature: _________________ Date: _______
```

### Aggregate Test Report
```
Course Code Example Testing Report
==================================

Test Period: [Date Range]
Tester: [Name]
Platform: Jetson Orin Nano 8GB

Summary:
- Total Examples Tested: XX
- Passed Examples: XX
- Failed Examples: XX
- Success Rate: XX.XX%

Performance Metrics:
- Average Frequency: XX.XX Hz
- Minimum Frequency Achieved: XX.XX Hz
- Memory Usage Range: XX-XX MB
- CPU Usage Range: XX-XX%

Examples by Category:
- ROS 2 Basics: XX/XX passed
- Perception: XX/XX passed
- Control: XX/XX passed
- Navigation: XX/XX passed

Critical Issues:
1. [Issue description and impact]
2. [Issue description and impact]

Recommendations:
1. [Specific recommendation]
2. [Specific recommendation]

Overall Assessment: [PASS/FAIL]
```

## Troubleshooting Common Issues

### Low Frequency Issues
- **Cause**: Inefficient algorithms or blocking operations
- **Solution**: Optimize algorithms, use threading, reduce computation

### High Memory Usage
- **Cause**: Memory leaks or inefficient data structures
- **Solution**: Profile memory usage, fix leaks, optimize storage

### CPU Overutilization
- **Cause**: CPU-intensive operations or poor threading
- **Solution**: Optimize algorithms, use GPU acceleration where possible

### Hardware Limitations
- **Cause**: Example too demanding for target hardware
- **Solution**: Simplify example or provide hardware upgrade path

## Continuous Integration

### Automated Testing Pipeline
- Integration with CI/CD pipeline
- Automated performance validation
- Regular regression testing
- Performance trend analysis

### Performance Monitoring
- Continuous performance tracking
- Trend analysis over time
- Early warning for performance degradation
- Hardware-specific optimization tracking

## Quality Assurance

### Testing Standards
- All examples must be tested on target hardware
- Performance metrics must be documented
- Results must be reproducible
- Testing environment must be standardized

### Validation Process
- Independent verification of results
- Cross-validation on different hardware units
- Student validation of test results
- Regular process review and improvement

This comprehensive testing process ensures that all code examples in the Physical AI & Humanoid Robotics course meet the performance requirements on the target Jetson hardware, providing students with reliable and performant examples that work in real-world conditions.