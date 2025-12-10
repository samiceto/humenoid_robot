# Performance Testing Infrastructure for Jetson Deployment

This document outlines the performance testing infrastructure for validating that implementations meet the course requirements, specifically ≥15 Hz performance on Jetson Orin Nano.

## Performance Requirements

The Physical AI & Humanoid Robotics course has the following performance requirements:

- **Minimum Frequency**: ≥15 Hz for real-time operation
- **Maximum Memory**: &lt;2GB RAM consumption
- **Maximum CPU**: &lt;80% average CPU usage
- **Minimum Battery Life**: ≥120 minutes for mobile robot operations

## Testing Infrastructure

### 1. Performance Test Script

The `performance_test_jetson.py` script provides comprehensive performance validation:

```bash
# Run all performance tests
python3 scripts/performance_test_jetson.py

# Run a specific test
python3 scripts/performance_test_jetson.py --test basic_ros_node

# List available tests
python3 scripts/performance_test_jetson.py --list

# Use custom configuration
python3 scripts/performance_test_jetson.py --config path/to/config.json
```

### 2. Test Scenarios

The infrastructure includes several predefined test scenarios:

- **Basic ROS Node**: Tests fundamental ROS 2 node performance
- **Perception Pipeline**: Validates computer vision pipeline performance
- **Control Loop**: Tests robot control system responsiveness
- **Navigation Stack**: Validates full navigation stack performance

### 3. Resource Monitoring

The system continuously monitors:

- CPU usage percentage
- Memory consumption (MB)
- GPU utilization (for Jetson platforms)
- System temperature
- Message frequency on ROS topics

## Configuration

The performance testing can be configured through a JSON configuration file:

```json
{
  "requirements": {
    "min_frequency_hz": 15,
    "max_memory_mb": 2048,
    "max_cpu_percent": 80,
    "min_battery_life_minutes": 120
  },
  "test_scenarios": [
    {
      "name": "basic_ros_node",
      "description": "Basic ROS 2 node performance test",
      "command": "ros2 run demo_nodes_cpp talker",
      "duration": 30,
      "expected_frequency": 10
    }
  ],
  "monitoring": {
    "cpu_interval": 1.0,
    "memory_interval": 1.0,
    "frequency_interval": 0.1
  }
}
```

## Integration with CI/CD

The performance testing infrastructure integrates with the CI/CD pipeline:

1. **Pre-commit validation**: Basic performance checks
2. **Pull request checks**: Performance regression detection
3. **Deployment validation**: Final performance verification

## Report Generation

After running tests, the system generates:

1. **Markdown reports**: Human-readable test results
2. **JSON data**: Raw performance metrics for analysis
3. **Summary dashboards**: Visual performance trends

## Usage in Course

Students and instructors can use the performance testing infrastructure to:

- Validate their implementations meet course requirements
- Compare performance across different hardware configurations
- Identify performance bottlenecks in their code
- Document performance characteristics for assignments

## Troubleshooting

### Common Issues

1. **Insufficient permissions**: Run with appropriate permissions for system monitoring
2. **Missing dependencies**: Install required Python packages (psutil, etc.)
3. **Resource constraints**: Ensure adequate system resources for testing

### Performance Optimization Tips

1. Use efficient data structures and algorithms
2. Optimize ROS message serialization
3. Implement proper resource cleanup
4. Profile code to identify bottlenecks

## Validation Process

The performance validation process includes:

1. **Automated testing**: Scripted validation of all requirements
2. **Manual verification**: Expert review of critical components
3. **Hardware validation**: Testing on target Jetson hardware
4. **Simulation validation**: Performance in Isaac Sim environment

This infrastructure ensures that all course content and implementations meet the performance requirements for successful deployment on Jetson Orin Nano platforms.