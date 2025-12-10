#!/usr/bin/env python3
# test_capstone_performance.py
# Performance testing script for capstone project
# Validates ≥15 Hz performance requirement on Jetson Orin Nano

import time
import threading
import statistics
import subprocess
import json
import sys
from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import matplotlib.pyplot as plt


class PerformanceTester(Node):
    """Performance testing node for capstone project"""

    def __init__(self):
        super().__init__('performance_tester')

        # Performance tracking
        self.perception_rates = deque(maxlen=1000)
        self.control_rates = deque(maxlen=1000)
        self.system_cpu_usage = deque(maxlen=1000)
        self.system_memory_usage = deque(maxlen=1000)
        self.system_gpu_usage = deque(maxlen=1000)

        # Subscribers for performance metrics
        self.perception_rate_sub = self.create_subscription(
            Float32, 'perception_rate', self.perception_rate_callback, 10)
        self.control_rate_sub = self.create_subscription(
            Float32, 'control_rate', self.control_rate_callback, 10)

        # Publishers for test control
        self.test_status_pub = self.create_publisher(String, 'test_status', 10)

        # Test parameters
        self.test_duration = 60.0  # seconds
        self.min_perception_rate = 15.0  # Hz
        self.min_control_rate = 100.0    # Hz
        self.test_start_time = None
        self.test_active = False

        # Performance validation results
        self.validation_results = {}

        self.get_logger().info('Performance Tester initialized')

    def perception_rate_callback(self, msg):
        """Handle perception rate updates"""
        self.perception_rates.append(msg.data)

    def control_rate_callback(self, msg):
        """Handle control rate updates"""
        self.control_rates.append(msg.data)

    def get_system_metrics(self):
        """Get system performance metrics"""
        try:
            # Get CPU usage
            cpu_usage = float(subprocess.check_output(['top', '-bn1'],
                                                    text=True).split('\n')[2].split()[1].replace('%', ''))
            self.system_cpu_usage.append(cpu_usage)

            # Get memory usage
            mem_info = subprocess.check_output(['free', '-m'], text=True)
            mem_lines = mem_info.strip().split('\n')
            mem_total, mem_used = map(int, mem_lines[1].split()[1:3])
            mem_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 0
            self.system_memory_usage.append(mem_percent)

            # Get GPU usage (if available)
            try:
                gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu',
                                           '--format=csv,noheader,nounits'],
                                          capture_output=True, text=True, timeout=5)
                if gpu_result.returncode == 0:
                    gpu_usage = float(gpu_result.stdout.strip())
                    self.system_gpu_usage.append(gpu_usage)
                else:
                    self.system_gpu_usage.append(0.0)
            except:
                self.system_gpu_usage.append(0.0)

        except Exception as e:
            self.get_logger().warn(f'Error getting system metrics: {e}')

    def start_performance_test(self, duration=60.0):
        """Start performance test"""
        self.test_duration = duration
        self.test_start_time = time.time()
        self.test_active = True

        self.get_logger().info(f'Starting performance test for {duration} seconds...')

        # Test timer
        test_timer = self.create_timer(0.1, self.test_callback)

        # Wait for test to complete
        start_time = time.time()
        while time.time() - start_time < duration and self.test_active:
            time.sleep(0.1)

        test_timer.destroy()
        return self.analyze_results()

    def test_callback(self):
        """Test callback for periodic monitoring"""
        if not self.test_active:
            return

        # Get current time
        current_time = time.time()

        # Check if test duration has been reached
        if current_time - self.test_start_time >= self.test_duration:
            self.test_active = False
            return

        # Get system metrics
        self.get_system_metrics()

        # Publish test status
        status_msg = String()
        elapsed = current_time - self.test_start_time
        status_msg.data = f'Performance test running: {elapsed:.1f}/{self.test_duration:.1f}s'
        self.test_status_pub.publish(status_msg)

    def analyze_results(self):
        """Analyze performance test results"""
        results = {
            'test_duration': self.test_duration,
            'samples_collected': len(self.perception_rates),
            'perception_rate': {
                'mean': statistics.mean(self.perception_rates) if self.perception_rates else 0.0,
                'median': statistics.median(self.perception_rates) if self.perception_rates else 0.0,
                'min': min(self.perception_rates) if self.perception_rates else 0.0,
                'max': max(self.perception_rates) if self.perception_rates else 0.0,
                'std_dev': statistics.stdev(self.perception_rates) if len(self.perception_rates) > 1 else 0.0,
                'meets_requirement': (statistics.mean(self.perception_rates) if self.perception_rates else 0.0) >= self.min_perception_rate
            },
            'control_rate': {
                'mean': statistics.mean(self.control_rates) if self.control_rates else 0.0,
                'median': statistics.median(self.control_rates) if self.control_rates else 0.0,
                'min': min(self.control_rates) if self.control_rates else 0.0,
                'max': max(self.control_rates) if self.control_rates else 0.0,
                'std_dev': statistics.stdev(self.control_rates) if len(self.control_rates) > 1 else 0.0,
                'meets_requirement': (statistics.mean(self.control_rates) if self.control_rates else 0.0) >= self.min_control_rate
            },
            'system_metrics': {
                'cpu_usage': {
                    'mean': statistics.mean(self.system_cpu_usage) if self.system_cpu_usage else 0.0,
                    'max': max(self.system_cpu_usage) if self.system_cpu_usage else 0.0
                },
                'memory_usage': {
                    'mean': statistics.mean(self.system_memory_usage) if self.system_memory_usage else 0.0,
                    'max': max(self.system_memory_usage) if self.system_memory_usage else 0.0
                },
                'gpu_usage': {
                    'mean': statistics.mean(self.system_gpu_usage) if self.system_gpu_usage else 0.0,
                    'max': max(self.system_gpu_usage) if self.system_gpu_usage else 0.0
                }
            }
        }

        self.validation_results = results
        return results

    def print_results(self):
        """Print formatted performance results"""
        if not self.validation_results:
            print("No validation results available")
            return

        results = self.validation_results

        print("\n" + "="*60)
        print("CAPSTONE PROJECT PERFORMANCE VALIDATION RESULTS")
        print("="*60)

        print(f"Test Duration: {results['test_duration']:.1f} seconds")
        print(f"Samples Collected: {results['samples_collected']}")

        print("\nPERCEPTION SYSTEM PERFORMANCE:")
        print(f"  Mean Rate: {results['perception_rate']['mean']:.2f} Hz")
        print(f"  Median Rate: {results['perception_rate']['median']:.2f} Hz")
        print(f"  Min Rate: {results['perception_rate']['min']:.2f} Hz")
        print(f"  Max Rate: {results['perception_rate']['max']:.2f} Hz")
        print(f"  Std Dev: {results['perception_rate']['std_dev']:.2f} Hz")
        print(f"  Requirement (≥15 Hz): {'✓ PASS' if results['perception_rate']['meets_requirement'] else '✗ FAIL'}")

        print("\nCONTROL SYSTEM PERFORMANCE:")
        print(f"  Mean Rate: {results['control_rate']['mean']:.2f} Hz")
        print(f"  Median Rate: {results['control_rate']['median']:.2f} Hz")
        print(f"  Min Rate: {results['control_rate']['min']:.2f} Hz")
        print(f"  Max Rate: {results['control_rate']['max']:.2f} Hz")
        print(f"  Std Dev: {results['control_rate']['std_dev']:.2f} Hz")
        print(f"  Requirement (≥100 Hz): {'✓ PASS' if results['control_rate']['meets_requirement'] else '✗ FAIL'}")

        print("\nSYSTEM RESOURCE USAGE:")
        print(f"  CPU Usage: Mean={results['system_metrics']['cpu_usage']['mean']:.1f}%, Max={results['system_metrics']['cpu_usage']['max']:.1f}%")
        print(f"  Memory Usage: Mean={results['system_metrics']['memory_usage']['mean']:.1f}%, Max={results['system_metrics']['memory_usage']['max']:.1f}%")
        print(f"  GPU Usage: Mean={results['system_metrics']['gpu_usage']['mean']:.1f}%, Max={results['system_metrics']['gpu_usage']['max']:.1f}%")

        print("\nOVERALL VALIDATION:")
        perception_pass = results['perception_rate']['meets_requirement']
        control_pass = results['control_rate']['meets_requirement']

        if perception_pass and control_pass:
            print("  ✓ ALL PERFORMANCE REQUIREMENTS MET")
            print("  The capstone project meets the ≥15 Hz perception and ≥100 Hz control requirements")
        else:
            print("  ✗ PERFORMANCE REQUIREMENTS NOT MET")
            if not perception_pass:
                print("  - Perception rate requirement (≥15 Hz) not met")
            if not control_pass:
                print("  - Control rate requirement (≥100 Hz) not met")

        print("="*60)


def run_performance_validation():
    """Run complete performance validation"""
    print("Starting Capstone Project Performance Validation")
    print("Testing ≥15 Hz performance requirement on Jetson Orin Nano")
    print("-" * 50)

    # Initialize ROS 2
    rclpy.init()

    try:
        # Create performance tester node
        performance_tester = PerformanceTester()

        # Run the performance test
        print("Running 60-second performance test...")
        results = performance_tester.start_performance_test(duration=60.0)

        # Print results
        performance_tester.print_results()

        # Return validation status
        perception_pass = results['perception_rate']['meets_requirement']
        control_pass = results['control_rate']['meets_requirement']

        success = perception_pass and control_pass
        return success

    except KeyboardInterrupt:
        print("\nPerformance test interrupted by user")
        return False
    except Exception as e:
        print(f"\nError during performance test: {e}")
        return False
    finally:
        rclpy.shutdown()


def validate_jetson_orin_performance():
    """Validate performance specifically on Jetson Orin platform"""
    print("Validating performance on Jetson Orin platform...")

    # Check if running on Jetson platform
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip().replace('\x00', '')
        print(f"Running on: {model}")
    except:
        print("Could not determine hardware platform")
        print("Assuming test environment is properly configured")

    # Check Jetson-specific metrics
    try:
        # Check Jetson model and capabilities
        result = subprocess.run(['jetson_release', '-v'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Jetson platform: {result.stdout.strip()}")
        else:
            print("Could not determine Jetson platform details")
    except:
        print("jetson_release not available, continuing with validation")

    # Run the performance validation
    return run_performance_validation()


def create_performance_report(results, filename="performance_report.json"):
    """Create a detailed performance report"""
    report = {
        'timestamp': time.time(),
        'test_environment': 'Jetson Orin Nano',
        'test_duration': results['test_duration'],
        'requirements': {
            'perception_rate': '≥15 Hz',
            'control_rate': '≥100 Hz'
        },
        'results': results,
        'validation_passed': (
            results['perception_rate']['meets_requirement'] and
            results['control_rate']['meets_requirement']
        )
    }

    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Performance report saved to {filename}")
    return report


def plot_performance_metrics(results):
    """Create visualizations of performance metrics"""
    try:
        # Create time series plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Capstone Project Performance Metrics', fontsize=16)

        # Perception rate histogram
        axes[0, 0].hist(list(results['perception_rates']), bins=30, alpha=0.7)
        axes[0, 0].axvline(results['min_perception_rate'], color='red', linestyle='--', label='Required (15 Hz)')
        axes[0, 0].set_xlabel('Perception Rate (Hz)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Perception Rate Distribution')
        axes[0, 0].legend()

        # Control rate histogram
        axes[0, 1].hist(list(results['control_rates']), bins=30, alpha=0.7)
        axes[0, 1].axvline(results['min_control_rate'], color='red', linestyle='--', label='Required (100 Hz)')
        axes[0, 1].set_xlabel('Control Rate (Hz)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Control Rate Distribution')
        axes[0, 1].legend()

        # System resource usage
        axes[1, 0].bar(['CPU', 'Memory', 'GPU'], [
            results['system_metrics']['cpu_usage']['mean'],
            results['system_metrics']['memory_usage']['mean'],
            results['system_metrics']['gpu_usage']['mean']
        ])
        axes[1, 0].set_ylabel('Usage (%)')
        axes[1, 0].set_title('Average System Resource Usage')

        # Performance summary
        performance_summary = [
            results['perception_rate']['mean'],
            results['control_rate']['mean']
        ]
        requirement_lines = [results['min_perception_rate'], results['min_control_rate']]
        axes[1, 1].bar(['Perception', 'Control'], performance_summary, alpha=0.7, label='Actual')
        axes[1, 1].plot([-0.5, 1.5], [requirement_lines[0], requirement_lines[0]], 'r--', label='Perception Requirement')
        axes[1, 1].plot([-0.5, 1.5], [requirement_lines[1], requirement_lines[1]], 'r--', label='Control Requirement')
        axes[1, 1].set_ylabel('Rate (Hz)')
        axes[1, 1].set_title('Performance vs Requirements')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
        print("Performance metrics plot saved to performance_metrics.png")

    except ImportError:
        print("Matplotlib not available, skipping plot generation")
    except Exception as e:
        print(f"Error creating performance plots: {e}")


if __name__ == '__main__':
    # Validate performance on Jetson Orin
    success = validate_jetson_orin_performance()

    if success:
        print("\n🎉 PERFORMANCE VALIDATION SUCCESSFUL!")
        print("The capstone project meets all performance requirements on Jetson Orin Nano")
    else:
        print("\n❌ PERFORMANCE VALIDATION FAILED!")
        print("The capstone project does not meet the required performance on Jetson Orin Nano")
        sys.exit(1)