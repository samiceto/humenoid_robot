#!/usr/bin/env python3
"""
Hardware validation scripts for Isaac Sim compatibility.

This script validates that hardware configurations are compatible with Isaac Sim
and meet the requirements for the Physical AI & Humanoid Robotics course.
"""

import os
import sys
import json
import subprocess
import argparse
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class IsaacSimHardwareValidator:
    def __init__(self):
        self.validation_results = {}
        self.requirements = {
            "gpu": {
                "minimum_vram_gb": 8,
                "recommended_vram_gb": 12,
                "supported_manufacturers": ["NVIDIA"],
                "minimum_architecture": "Pascal"  # GTX 10xx series
            },
            "cpu": {
                "minimum_cores": 4,
                "recommended_cores": 8,
                "min_frequency_ghz": 2.5
            },
            "memory": {
                "minimum_gb": 16,
                "recommended_gb": 32
            },
            "os": {
                "supported": ["Linux", "Windows"],
                "recommended_linux": ["Ubuntu 20.04", "Ubuntu 22.04"]
            },
            "disk_space": {
                "minimum_gb": 20,  # For Isaac Sim installation
                "recommended_gb": 50  # Including models and scenes
            }
        }

    def check_gpu_compatibility(self) -> Dict:
        """Check GPU compatibility with Isaac Sim requirements."""
        result = {
            "name": "GPU Compatibility",
            "passed": False,
            "details": {},
            "issues": [],
            "recommendations": []
        }

        try:
            # Try nvidia-smi to get GPU information
            result_nvidia_smi = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                                            capture_output=True, text=True)

            if result_nvidia_smi.returncode == 0:
                gpu_info = result_nvidia_smi.stdout.strip().split(', ')
                if len(gpu_info) >= 2:
                    gpu_name = gpu_info[0].strip()
                    vram_gb = float(gpu_info[1].strip()) / 1024.0  # Convert MB to GB

                    result["details"] = {
                        "gpu_name": gpu_name,
                        "vram_gb": vram_gb
                    }

                    # Check VRAM requirements
                    if vram_gb >= self.requirements["gpu"]["minimum_vram_gb"]:
                        result["passed"] = True
                        result["issues"].append(f"GPU VRAM ({vram_gb:.1f}GB) meets minimum requirement ({self.requirements['gpu']['minimum_vram_gb']}GB)")
                    else:
                        result["issues"].append(f"GPU VRAM ({vram_gb:.1f}GB) below minimum requirement ({self.requirements['gpu']['minimum_vram_gb']}GB)")
                        result["recommendations"].append("Upgrade to GPU with at least 8GB VRAM for Isaac Sim")

                    # Check if GPU is NVIDIA
                    if "NVIDIA" in gpu_name.upper() or "GEFORCE" in gpu_name.upper() or "QUADRO" in gpu_name.upper() or "TESLA" in gpu_name.upper():
                        result["issues"].append("NVIDIA GPU detected - compatible with Isaac Sim")
                    else:
                        result["issues"].append("Non-NVIDIA GPU detected - Isaac Sim requires NVIDIA GPU")
                        result["passed"] = False
            else:
                result["issues"].append("Could not detect NVIDIA GPU. Isaac Sim requires NVIDIA GPU with CUDA support.")
                result["recommendations"].append("Install NVIDIA GPU and drivers")
        except FileNotFoundError:
            result["issues"].append("nvidia-smi not found. GPU information unavailable.")
            result["recommendations"].append("Install NVIDIA drivers and ensure nvidia-smi is available in PATH")
        except Exception as e:
            result["issues"].append(f"Error checking GPU: {str(e)}")

        return result

    def check_cpu_compatibility(self) -> Dict:
        """Check CPU compatibility with Isaac Sim requirements."""
        result = {
            "name": "CPU Compatibility",
            "passed": False,
            "details": {},
            "issues": [],
            "recommendations": []
        }

        try:
            # Get CPU information
            import psutil
            cpu_count = psutil.cpu_count(logical=False)  # Physical cores
            logical_count = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq()

            result["details"] = {
                "physical_cores": cpu_count,
                "logical_cores": logical_count,
                "max_frequency_ghz": cpu_freq.max / 1000.0 if cpu_freq else 0
            }

            # Check core requirements
            if cpu_count >= self.requirements["cpu"]["minimum_cores"]:
                result["passed"] = True
                result["issues"].append(f"CPU cores ({cpu_count}) meet minimum requirement ({self.requirements['cpu']['minimum_cores']})")
            else:
                result["issues"].append(f"CPU cores ({cpu_count}) below minimum requirement ({self.requirements['cpu']['minimum_cores']})")
                result["recommendations"].append(f"Use CPU with at least {self.requirements['cpu']['minimum_cores']} physical cores")

            # Check frequency requirements
            if cpu_freq and cpu_freq.max / 1000.0 >= self.requirements["cpu"]["min_frequency_ghz"]:
                result["issues"].append(f"CPU frequency ({cpu_freq.max/1000.0:.1f}GHz) meets minimum requirement ({self.requirements['cpu']['min_frequency_ghz']}GHz)")
            else:
                freq_str = f"{cpu_freq.max/1000.0:.1f}GHz" if cpu_freq else "unknown"
                result["issues"].append(f"CPU frequency ({freq_str}) below minimum requirement ({self.requirements['cpu']['min_frequency_ghz']}GHz)")
        except ImportError:
            result["issues"].append("psutil not available, cannot check CPU details")
            result["recommendations"].append("Install psutil: pip install psutil")
        except Exception as e:
            result["issues"].append(f"Error checking CPU: {str(e)}")

        return result

    def check_memory_compatibility(self) -> Dict:
        """Check system memory compatibility."""
        result = {
            "name": "Memory Compatibility",
            "passed": False,
            "details": {},
            "issues": [],
            "recommendations": []
        }

        try:
            import psutil
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)  # Convert to GB

            result["details"] = {
                "total_memory_gb": round(total_gb, 2),
                "available_memory_gb": round(memory.available / (1024**3), 2)
            }

            if total_gb >= self.requirements["memory"]["minimum_gb"]:
                result["passed"] = True
                result["issues"].append(f"System memory ({total_gb:.1f}GB) meets minimum requirement ({self.requirements['memory']['minimum_gb']}GB)")
            else:
                result["issues"].append(f"System memory ({total_gb:.1f}GB) below minimum requirement ({self.requirements['memory']['minimum_gb']}GB)")
                result["recommendations"].append(f"Upgrade to at least {self.requirements['memory']['minimum_gb']}GB RAM")

        except ImportError:
            result["issues"].append("psutil not available, cannot check memory details")
            result["recommendations"].append("Install psutil: pip install psutil")
        except Exception as e:
            result["issues"].append(f"Error checking memory: {str(e)}")

        return result

    def check_os_compatibility(self) -> Dict:
        """Check operating system compatibility."""
        result = {
            "name": "OS Compatibility",
            "passed": False,
            "details": {},
            "issues": [],
            "recommendations": []
        }

        system = platform.system()
        release = platform.release()
        version = platform.version()

        result["details"] = {
            "system": system,
            "release": release,
            "version": version
        }

        if system in self.requirements["os"]["supported"]:
            result["passed"] = True
            result["issues"].append(f"Operating system ({system}) is supported by Isaac Sim")

            if system == "Linux":
                # Try to get more specific Linux info
                try:
                    with open('/etc/os-release', 'r') as f:
                        os_info = {}
                        for line in f:
                            if '=' in line:
                                key, value = line.strip().split('=', 1)
                                os_info[key] = value.strip('"')

                        if 'NAME' in os_info and 'VERSION_ID' in os_info:
                            distro = f"{os_info['NAME']} {os_info['VERSION_ID']}"
                            result["details"]["distribution"] = distro

                            # Check if it's a recommended distribution
                            recommended = any(rec in distro for rec in self.requirements["os"]["recommended_linux"])
                            if recommended:
                                result["issues"].append(f"Linux distribution ({distro}) is recommended for Isaac Sim")
                            else:
                                result["recommendations"].append(f"Consider using a recommended distribution: {', '.join(self.requirements['os']['recommended_linux'])}")
                except:
                    pass
        else:
            result["issues"].append(f"Operating system ({system}) may not be fully supported by Isaac Sim")
            result["recommendations"].append(f"Use one of the supported OS: {', '.join(self.requirements['os']['supported'])}")

        return result

    def check_disk_space(self, path: str = "/") -> Dict:
        """Check disk space availability."""
        result = {
            "name": "Disk Space",
            "passed": False,
            "details": {},
            "issues": [],
            "recommendations": []
        }

        try:
            import shutil
            total, used, free = shutil.disk_usage(path)

            total_gb = total / (1024**3)
            free_gb = free / (1024**3)

            result["details"] = {
                "total_space_gb": round(total_gb, 2),
                "free_space_gb": round(free_gb, 2),
                "path": path
            }

            if free_gb >= self.requirements["disk_space"]["minimum_gb"]:
                result["passed"] = True
                result["issues"].append(f"Free disk space ({free_gb:.1f}GB) meets minimum requirement ({self.requirements['disk_space']['minimum_gb']}GB)")
            else:
                result["issues"].append(f"Free disk space ({free_gb:.1f}GB) below minimum requirement ({self.requirements['disk_space']['minimum_gb']}GB)")
                result["recommendations"].append(f"Free up at least {self.requirements['disk_space']['minimum_gb'] - free_gb:.1f}GB of disk space")
        except Exception as e:
            result["issues"].append(f"Error checking disk space: {str(e)}")

        return result

    def check_cuda_compatibility(self) -> Dict:
        """Check CUDA compatibility."""
        result = {
            "name": "CUDA Compatibility",
            "passed": False,
            "details": {},
            "issues": [],
            "recommendations": []
        }

        try:
            # Check nvcc version
            result_nvcc = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if result_nvcc.returncode == 0:
                version_line = [line for line in result_nvcc.stdout.split('\\n') if 'release' in line.lower()]
                if version_line:
                    result["details"]["nvcc_version"] = version_line[0].strip()
                    result["passed"] = True
                    result["issues"].append("CUDA compiler (nvcc) found")
                else:
                    result["issues"].append("Could not determine CUDA version from nvcc output")
            else:
                result["issues"].append("CUDA compiler (nvcc) not found")
                result["recommendations"].append("Install CUDA toolkit")
        except FileNotFoundError:
            result["issues"].append("nvcc command not found")
            result["recommendations"].append("Install CUDA toolkit and ensure nvcc is in PATH")
        except Exception as e:
            result["issues"].append(f"Error checking CUDA: {str(e)}")

        try:
            # Check nvidia-ml-py (if available)
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            result["details"]["cuda_devices"] = device_count
            result["issues"].append(f"Found {device_count} CUDA-capable device(s)")
        except ImportError:
            result["issues"].append("pynvml not available, skipping detailed CUDA device check")
            result["recommendations"].append("Install pynvml: pip install nvidia-ml-py")
        except Exception as e:
            result["issues"].append(f"Error accessing CUDA devices: {str(e)}")

        return result

    def run_all_validations(self) -> Dict:
        """Run all hardware compatibility validations."""
        print("Running Isaac Sim hardware compatibility validation...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "platform_info": {
                "system": platform.system(),
                "node": platform.node(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            },
            "tests": {},
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0
            }
        }

        # Run all validation tests
        validation_functions = [
            self.check_gpu_compatibility,
            self.check_cpu_compatibility,
            self.check_memory_compatibility,
            self.check_os_compatibility,
            self.check_disk_space,
            self.check_cuda_compatibility
        ]

        for validation_func in validation_functions:
            try:
                test_result = validation_func()
                test_name = validation_func.__name__.replace('check_', '').replace('_compatibility', '').replace('_space', '')
                results["tests"][test_name] = test_result

                if test_result["passed"]:
                    results["summary"]["passed_tests"] += 1
                    print(f"✅ {test_result['name']}: PASSED")
                else:
                    results["summary"]["failed_tests"] += 1
                    print(f"❌ {test_result['name']}: FAILED")
                    for issue in test_result["issues"]:
                        print(f"   - {issue}")
                    for rec in test_result["recommendations"]:
                        print(f"   💡 {rec}")
            except Exception as e:
                print(f"❌ {validation_func.__name__}: ERROR - {str(e)}")
                results["summary"]["failed_tests"] += 1

            results["summary"]["total_tests"] += 1

        return results

    def generate_report(self, results: Dict) -> str:
        """Generate a hardware compatibility report."""
        report = f"""
# Isaac Sim Hardware Compatibility Report

**Date**: {results['timestamp']}
**Platform**: {results['platform_info']['system']} {results['platform_info']['release']} ({results['platform_info']['machine']})

## Validation Summary
- Total Tests: {results['summary']['total_tests']}
- Passed: {results['summary']['passed_tests']}
- Failed: {results['summary']['failed_tests']}
- Success Rate: {(results['summary']['passed_tests']/results['summary']['total_tests']*100):.1f}% if results['summary']['total_tests'] > 0 else 0}%

## Detailed Results

"""

        for test_name, test_result in results['tests'].items():
            status = "✅ PASSED" if test_result['passed'] else "❌ FAILED"
            report += f"### {test_result['name']} - {status}\n\n"

            if test_result['details']:
                report += "**Details:**\n"
                for key, value in test_result['details'].items():
                    report += f"- {key.replace('_', ' ').title()}: {value}\n"
                report += "\n"

            if test_result['issues']:
                report += "**Issues:**\n"
                for issue in test_result['issues']:
                    report += f"- {issue}\n"
                report += "\n"

            if test_result['recommendations']:
                report += "**Recommendations:**\n"
                for rec in test_result['recommendations']:
                    report += f"- {rec}\n"
                report += "\n"

        # Add overall compatibility assessment
        passed_count = results['summary']['passed_tests']
        total_count = results['summary']['total_tests']

        if passed_count == total_count:
            report += "## Overall Assessment\n\n"
            report += "✅ **Your system is fully compatible with Isaac Sim!**\n\n"
            report += "All hardware requirements are met. You should be able to run Isaac Sim without issues.\n"
        elif passed_count >= total_count * 0.7:  # 70% threshold
            report += "## Overall Assessment\n\n"
            report += "⚠️ **Your system has partial compatibility with Isaac Sim.**\n\n"
            report += "Most requirements are met, but some components may limit performance. Review the failed tests above.\n"
        else:
            report += "## Overall Assessment\n\n"
            report += "❌ **Your system is not compatible with Isaac Sim.**\n\n"
            report += "Several requirements are not met. Please address the failed tests before attempting to run Isaac Sim.\n"

        report += f"""
## Minimum Requirements Summary

- **GPU**: NVIDIA GPU with ≥{self.requirements['gpu']['minimum_vram_gb']}GB VRAM
- **CPU**: ≥{self.requirements['cpu']['minimum_cores']} cores, ≥{self.requirements['cpu']['min_frequency_ghz']}GHz
- **Memory**: ≥{self.requirements['memory']['minimum_gb']}GB RAM
- **OS**: {', '.join(self.requirements['os']['supported'])}
- **Disk**: ≥{self.requirements['disk_space']['minimum_gb']}GB free space
- **CUDA**: Compatible NVIDIA GPU with CUDA support

For best performance in the Physical AI & Humanoid Robotics course,
consider meeting the recommended requirements where specified.
"""

        return report

    def save_report(self, results: Dict, filename: str = None):
        """Save the validation report."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"isaac_sim_compatibility_report_{timestamp}.md"

        report = self.generate_report(results)

        with open(filename, 'w') as f:
            f.write(report)

        print(f"Hardware compatibility report saved to: {filename}")

        # Also save raw JSON results
        json_filename = filename.replace('.md', '.json')
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Raw results saved to: {json_filename}")


def main():
    parser = argparse.ArgumentParser(description='Validate hardware compatibility with Isaac Sim')
    parser.add_argument('--output', type=str, help='Output report filename')
    parser.add_argument('--path', type=str, default='/', help='Path to check for disk space')

    args = parser.parse_args()

    validator = IsaacSimHardwareValidator()
    results = validator.run_all_validations()

    # Generate and save report
    validator.save_report(results, args.output)

    # Exit with appropriate code based on results
    if results["summary"]["failed_tests"] == 0:
        print(f"\n✅ All hardware compatibility checks passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {results['summary']['failed_tests']} hardware compatibility checks failed")
        sys.exit(1 if results["summary"]["failed_tests"] > 2 else 0)  # Only fail for major incompatibilities


if __name__ == "__main__":
    main()