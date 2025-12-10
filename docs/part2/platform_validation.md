# Physical Platform Validation

## Overview

This document outlines the validation process for the voice command pipeline on physical humanoid platforms. The validation ensures that the pipeline works correctly on real hardware, meeting the performance requirements (≥15 Hz real-time inference) and functional specifications.

## Validation Requirements

### Performance Requirements
- **Minimum Frequency**: ≥15 Hz real-time inference
- **Response Time**: < 500ms for command processing
- **Throughput**: Handle 10+ simultaneous requests
- **Resource Usage**: < 80% CPU, < 2GB RAM

### Functional Requirements
- **Speech Recognition**: ≥90% accuracy in quiet environments
- **Command Interpretation**: ≥95% accuracy for common commands
- **Action Execution**: ≥98% success rate for basic actions
- **Safety**: Emergency stop functionality

## Hardware Platforms

### Primary Platform: Unitree G1
- **Specifications**:
  - Height: 1.3m
  - Weight: 74kg
  - DOF: 32 (16 per leg, 6 per arm, 4 in torso)
  - Computing: NVIDIA Jetson Orin AGX (64GB)
  - Sensors: IMU, cameras, force/torque sensors
  - Connectivity: WiFi, Bluetooth, Ethernet

### Secondary Platforms
- **Figure 02**: Advanced humanoid for comparison
- **Poppy Ergo Jr**: Educational platform for prototyping
- **Custom Platforms**: Research variants

## Validation Environment Setup

### Laboratory Configuration
```
Physical Platform ←→ ROS 2 Network ←→ Workstation ←→ Isaac Sim (for comparison)
```

### Equipment Required
- Physical humanoid robot platform
- ROS 2 compatible workstation (Ubuntu 22.04)
- Network infrastructure (low-latency WiFi/Ethernet)
- Audio equipment (microphones, speakers)
- Safety equipment (safety stop, barriers)
- Monitoring equipment (performance tracking)

### Network Configuration
```bash
# Robot network setup
export ROS_DOMAIN_ID=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Set up reliable communication
# Configure QoS profiles for different data types
```

## Validation Procedures

### 1. Pre-Deployment Validation

#### System Health Check
```python
def system_health_check():
    """Validate system readiness before testing"""
    checks = {
        "power_level": check_power_level(),
        "network_connectivity": check_network_connectivity(),
        "sensor_status": check_sensor_status(),
        "actuator_status": check_actuator_status(),
        "computing_resources": check_computing_resources()
    }

    return all(checks.values())
```

#### Component Validation
- **Microphone Array**: Verify audio input quality
- **Speakers**: Test audio output capability
- **Cameras**: Validate visual input systems
- **IMU**: Check orientation and acceleration sensors
- **Actuators**: Test joint position and torque control

### 2. Functional Validation

#### Speech Recognition Testing
```python
def test_speech_recognition():
    """Test speech recognition in various conditions"""
    test_scenarios = [
        {"environment": "quiet_lab", "distance": "1m", "noise_level": "low"},
        {"environment": "quiet_lab", "distance": "3m", "noise_level": "low"},
        {"environment": "noisy_lab", "distance": "1m", "noise_level": "medium"},
        {"environment": "controlled_noise", "distance": "2m", "noise_level": "high"}
    ]

    results = {}
    for scenario in test_scenarios:
        accuracy = run_speech_recognition_test(scenario)
        results[tuple(scenario.items())] = accuracy

    return results
```

#### Command Interpretation Testing
Test the NLU component with various command formulations:

```python
def test_command_interpretation():
    """Test command interpretation accuracy"""
    command_variations = {
        "move_forward": [
            "move forward 1 meter",
            "go forward by one meter",
            "move ahead 1 meter",
            "go straight 1 meter"
        ],
        "turn_left": [
            "turn left 90 degrees",
            "rotate left by 90 degrees",
            "pivot left 90 degrees"
        ],
        "pick_object": [
            "pick up the red cup",
            "grasp the red cup",
            "take the red cup"
        ]
    }

    interpretation_accuracy = {}
    for intent, variations in command_variations.items():
        correct_interpretations = 0
        for command in variations:
            parsed = parse_command(command)
            if parsed["intent"] == intent:
                correct_interpretations += 1

        accuracy = correct_interpretations / len(variations)
        interpretation_accuracy[intent] = accuracy

    return interpretation_accuracy
```

#### Action Execution Testing
Validate that interpreted commands result in correct physical actions:

```python
def test_action_execution():
    """Test physical action execution"""
    test_actions = [
        {"command": "move forward 1 meter", "expected": "move_1m_forward"},
        {"command": "turn left 90 degrees", "expected": "turn_90deg_left"},
        {"command": "wave arm", "expected": "arm_wave_motion"}
    ]

    execution_results = {}
    for action in test_actions:
        success = execute_and_verify_action(action["command"], action["expected"])
        execution_results[action["command"]] = success

    return execution_results
```

### 3. Performance Validation

#### Real-time Performance Testing
```python
def test_real_time_performance():
    """Test real-time performance requirements"""
    import time
    import threading

    # Test sustained performance over time
    start_time = time.time()
    command_count = 0
    interval_start = time.time()
    interval_commands = 0
    max_frequency = 0

    def send_commands_continuously():
        nonlocal command_count, interval_commands, max_frequency
        while time.time() - start_time < 60:  # Test for 1 minute
            if time.time() - interval_start >= 1.0:  # Measure frequency every second
                current_freq = interval_commands
                max_frequency = max(max_frequency, current_freq)

                # Reset for next interval
                interval_commands = 0
                interval_start = time.time()

            # Send command
            send_voice_command("move forward 0.1 meter")
            command_count += 1
            interval_commands += 1

            time.sleep(0.05)  # 20 commands per second

    # Run performance test
    command_thread = threading.Thread(target=send_commands_continuously)
    command_thread.start()
    command_thread.join()

    total_time = time.time() - start_time
    avg_frequency = command_count / total_time

    results = {
        "average_frequency": avg_frequency,
        "peak_frequency": max_frequency,
        "min_frequency_requirement_met": avg_frequency >= 15.0,
        "total_commands_processed": command_count
    }

    return results
```

#### Resource Utilization Monitoring
```python
def monitor_resource_utilization():
    """Monitor system resource usage during operation"""
    import psutil
    import threading

    resource_readings = []

    def collect_readings():
        while testing_active:
            reading = {
                "timestamp": time.time(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_mb": psutil.virtual_memory().used / (1024**2),
                "disk_io": psutil.disk_io_counters(),
                "network_io": psutil.net_io_counters()
            }
            resource_readings.append(reading)
            time.sleep(1)

    # Start monitoring in background
    monitor_thread = threading.Thread(target=collect_readings)
    monitor_thread.start()

    # Run tests
    run_functional_tests()

    # Stop monitoring
    testing_active = False
    monitor_thread.join()

    # Analyze results
    cpu_readings = [r["cpu_percent"] for r in resource_readings]
    memory_readings = [r["memory_percent"] for r in resource_readings]

    analysis = {
        "cpu_avg": sum(cpu_readings) / len(cpu_readings),
        "cpu_peak": max(cpu_readings),
        "cpu_within_limit": max(cpu_readings) < 80,
        "memory_avg": sum(memory_readings) / len(memory_readings),
        "memory_peak": max(memory_readings),
        "memory_within_limit": max(memory_readings) < 80  # Assuming 8GB RAM
    }

    return analysis
```

### 4. Safety and Reliability Testing

#### Emergency Stop Validation
```python
def test_emergency_stop():
    """Test emergency stop functionality"""
    # Start robot in motion
    send_voice_command("move forward indefinitely")
    time.sleep(2)  # Let robot start moving

    # Issue emergency stop
    trigger_emergency_stop()

    # Verify robot stops within safety time
    stopped = wait_for_stop(timeout=2.0)

    if not stopped:
        # Force stop if needed
        force_robot_stop()
        return False

    return True
```

#### Failure Recovery Testing
```python
def test_failure_recovery():
    """Test system recovery from various failures"""
    failure_scenarios = [
        {"type": "network_disconnect", "action": simulate_network_disconnect},
        {"type": "power_fluctuation", "action": simulate_power_fluctuation},
        {"type": "sensor_failure", "action": simulate_sensor_failure},
        {"type": "actuator_failure", "action": simulate_actuator_failure}
    ]

    recovery_results = {}
    for scenario in failure_scenarios:
        # Setup test
        initial_state = get_robot_state()

        # Apply failure
        scenario["action"]()

        # Wait for recovery
        recovered = wait_for_recovery(timeout=30.0)

        # Verify state after recovery
        final_state = get_robot_state()
        recovery_successful = verify_safe_state(initial_state, final_state)

        recovery_results[scenario["type"]] = {
            "recovered": recovered,
            "safe_state": recovery_successful,
            "time_to_recovery": calculate_recovery_time()
        }

    return recovery_results
```

## Validation Results Documentation

### Test Report Template
```python
def generate_validation_report(results):
    """Generate comprehensive validation report"""
    report = {
        "validation_date": datetime.now().isoformat(),
        "platform": "Unitree G1",
        "software_version": get_software_version(),
        "environment": "Laboratory Setting",
        "operator": get_operator_info(),
        "results": {
            "system_health": results["health_check"],
            "speech_recognition": results["speech_recognition"],
            "command_interpretation": results["command_interpretation"],
            "action_execution": results["action_execution"],
            "performance": results["performance"],
            "resource_utilization": results["resource_utilization"],
            "safety": results["safety"],
            "reliability": results["reliability"]
        },
        "compliance": {
            "real_time_performance": results["performance"]["min_frequency_requirement_met"],
            "resource_limits": results["resource_utilization"]["cpu_within_limit"],
            "functional_accuracy": calculate_overall_accuracy(results)
        },
        "recommendations": generate_recommendations(results),
        "signatures": {
            "engineer": None,
            "qa_specialist": None,
            "date": None
        }
    }

    return report
```

### Pass/Fail Criteria
- **Overall Pass**: All critical tests pass and ≥85% of total tests pass
- **Critical Tests**: Performance (≥15 Hz), safety systems, basic functionality
- **Major Failures**: Any safety violation, system crash, or major malfunction
- **Minor Issues**: Performance slightly below requirements, cosmetic issues

## Troubleshooting and Diagnostics

### Common Issues and Solutions

#### Audio Quality Issues
- **Problem**: Poor speech recognition accuracy
- **Diagnosis**: Check microphone positioning, ambient noise levels, audio preprocessing
- **Solution**: Adjust microphone gain, apply noise reduction, optimize positioning

#### Performance Degradation
- **Problem**: Processing frequency below 15 Hz
- **Diagnosis**: Monitor CPU/memory usage, identify bottlenecks
- **Solution**: Optimize algorithms, reduce model complexity, improve parallelization

#### Action Execution Failures
- **Problem**: Robot doesn't execute commands correctly
- **Diagnosis**: Check action server availability, joint limits, collision detection
- **Solution**: Calibrate sensors, adjust control parameters, verify robot state

### Diagnostic Tools
```bash
# Performance monitoring
ros2 run robot_performance_monitor performance_analyzer

# System health checker
ros2 run robot_health_checker system_diagnostics

# Audio quality analyzer
ros2 run audio_quality_analyzer real_time_analysis
```

## Continuous Validation

### Regression Testing
- **Daily**: Basic functionality tests
- **Weekly**: Performance and accuracy tests
- **Monthly**: Full validation suite

### Monitoring in Deployment
- Real-time performance metrics
- Usage analytics
- Failure detection and reporting
- Automatic recovery procedures

## Conclusion

Physical platform validation is critical for ensuring the voice command pipeline works reliably in real-world conditions. The validation process verifies both functional correctness and performance requirements, ensuring the system meets the ≥15 Hz real-time inference requirement while maintaining safety and reliability.

Successful validation on the physical platform demonstrates that the sim-to-real transfer was effective and that the system is ready for educational deployment.