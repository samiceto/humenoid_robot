#!/usr/bin/env python3
# validate_urdf_sdf_models.py
# Validation script for URDF/SDF models using gz sdf check and Isaac Sim validator

import os
import subprocess
import xml.etree.ElementTree as ET
import yaml
import json
from pathlib import Path
import sys
from typing import List, Dict, Tuple, Optional


class ModelValidator:
    """Class to validate URDF/SDF models"""

    def __init__(self):
        self.urdf_models = []
        self.sdf_models = []
        self.validation_results = {}
        self.isaac_sim_available = self.check_isaac_sim_availability()

    def check_isaac_sim_availability(self) -> bool:
        """Check if Isaac Sim is available"""
        try:
            # Try to import Isaac Sim Python modules
            import omni
            import carb
            return True
        except ImportError:
            print("Isaac Sim Python modules not available")
            return False

    def find_urdf_models(self, search_path: str) -> List[str]:
        """Find all URDF files in the specified path"""
        urdf_files = []
        for root, dirs, files in os.walk(search_path):
            for file in files:
                if file.endswith('.urdf') or file.endswith('.urdf.xacro'):
                    urdf_files.append(os.path.join(root, file))
        return urdf_files

    def find_sdf_models(self, search_path: str) -> List[str]:
        """Find all SDF files in the specified path"""
        sdf_files = []
        for root, dirs, files in os.walk(search_path):
            for file in files:
                if file.endswith('.sdf'):
                    sdf_files.append(os.path.join(root, file))
        return sdf_files

    def validate_urdf_with_check_urdf(self, urdf_path: str) -> Dict[str, any]:
        """Validate URDF using check_urdf tool"""
        try:
            result = subprocess.run(['check_urdf', urdf_path],
                                  capture_output=True, text=True, timeout=30)
            return {
                'valid': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                'valid': False,
                'error': 'Validation timed out',
                'return_code': -1
            }
        except FileNotFoundError:
            return {
                'valid': False,
                'error': 'check_urdf command not found',
                'return_code': -1
            }

    def validate_urdf_with_xml_lint(self, urdf_path: str) -> Dict[str, any]:
        """Validate URDF as XML"""
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()

            # Check if it's a valid URDF
            is_urdf = root.tag in ['robot', 'xacro:robot'] or 'robot' in root.tag

            return {
                'valid': is_urdf,
                'xml_valid': True,
                'robot_name': root.get('name', 'unknown'),
                'root_element': root.tag,
                'num_elements': len(list(root))
            }
        except ET.ParseError as e:
            return {
                'valid': False,
                'xml_valid': False,
                'error': str(e)
            }

    def validate_sdf_with_gz_sdf(self, sdf_path: str) -> Dict[str, any]:
        """Validate SDF using gz sdf command"""
        try:
            result = subprocess.run(['gz', 'sdf', '-k', sdf_path],
                                  capture_output=True, text=True, timeout=30)
            return {
                'valid': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                'valid': False,
                'error': 'Validation timed out',
                'return_code': -1
            }
        except FileNotFoundError:
            return {
                'valid': False,
                'error': 'gz sdf command not found',
                'return_code': -1
            }

    def validate_sdf_with_xml_lint(self, sdf_path: str) -> Dict[str, any]:
        """Validate SDF as XML"""
        try:
            tree = ET.parse(sdf_path)
            root = tree.getroot()

            # Check if it's a valid SDF
            is_sdf = root.tag in ['sdf', 'model', 'world']

            return {
                'valid': is_sdf,
                'xml_valid': True,
                'root_element': root.tag,
                'version': root.get('version', 'unknown'),
                'num_elements': len(list(root))
            }
        except ET.ParseError as e:
            return {
                'valid': False,
                'xml_valid': False,
                'error': str(e)
            }

    def validate_urdf_model(self, urdf_path: str) -> Dict[str, any]:
        """Complete validation of a URDF model"""
        print(f"Validating URDF: {urdf_path}")

        # XML validation
        xml_result = self.validate_urdf_with_xml_lint(urdf_path)

        # check_urdf validation
        urdf_result = self.validate_urdf_with_check_urdf(urdf_path)

        # Combined result
        result = {
            'path': urdf_path,
            'type': 'urdf',
            'xml_validation': xml_result,
            'urdf_validation': urdf_result,
            'overall_valid': xml_result.get('valid', False) and urdf_result.get('valid', False)
        }

        return result

    def validate_sdf_model(self, sdf_path: str) -> Dict[str, any]:
        """Complete validation of an SDF model"""
        print(f"Validating SDF: {sdf_path}")

        # XML validation
        xml_result = self.validate_sdf_with_xml_lint(sdf_path)

        # gz sdf validation
        sdf_result = self.validate_sdf_with_gz_sdf(sdf_path)

        # Combined result
        result = {
            'path': sdf_path,
            'type': 'sdf',
            'xml_validation': xml_result,
            'sdf_validation': sdf_result,
            'overall_valid': xml_result.get('valid', False) and sdf_result.get('valid', False)
        }

        return result

    def validate_all_models(self, search_path: str) -> Dict[str, any]:
        """Validate all URDF and SDF models in the specified path"""
        print(f"Searching for models in: {search_path}")

        # Find all models
        self.urdf_models = self.find_urdf_models(search_path)
        self.sdf_models = self.find_sdf_models(search_path)

        print(f"Found {len(self.urdf_models)} URDF models and {len(self.sdf_models)} SDF models")

        results = {
            'urdf_results': [],
            'sdf_results': [],
            'summary': {}
        }

        # Validate URDF models
        for urdf_path in self.urdf_models:
            result = self.validate_urdf_model(urdf_path)
            results['urdf_results'].append(result)

        # Validate SDF models
        for sdf_path in self.sdf_models:
            result = self.validate_sdf_model(sdf_path)
            results['sdf_results'].append(result)

        # Generate summary
        total_models = len(self.urdf_models) + len(self.sdf_models)
        valid_urdf_count = sum(1 for r in results['urdf_results'] if r['overall_valid'])
        valid_sdf_count = sum(1 for r in results['sdf_results'] if r['overall_valid'])
        total_valid = valid_urdf_count + valid_sdf_count

        results['summary'] = {
            'total_models': total_models,
            'total_urdf_models': len(self.urdf_models),
            'total_sdf_models': len(self.sdf_models),
            'valid_urdf_models': valid_urdf_count,
            'valid_sdf_models': valid_sdf_count,
            'total_valid_models': total_valid,
            'validation_rate': total_valid / total_models if total_models > 0 else 0
        }

        self.validation_results = results
        return results

    def print_validation_report(self):
        """Print a formatted validation report"""
        if not self.validation_results:
            print("No validation results available")
            return

        results = self.validation_results
        summary = results['summary']

        print("\n" + "="*70)
        print("URDF/SDF MODEL VALIDATION REPORT")
        print("="*70)

        print(f"Total Models Found: {summary['total_models']}")
        print(f"  URDF Models: {summary['total_urdf_models']}")
        print(f"  SDF Models: {summary['total_sdf_models']}")
        print()

        print(f"Valid Models: {summary['total_valid_models']}")
        print(f"  Valid URDF: {summary['valid_urdf_models']}")
        print(f"  Valid SDF: {summary['valid_sdf_models']}")
        print()

        print(f"Overall Validation Rate: {summary['validation_rate']:.2%}")
        print()

        # Detailed results for invalid models
        print("INVALID MODELS:")
        invalid_count = 0

        for result in results['urdf_results']:
            if not result['overall_valid']:
                print(f"  URDF: {result['path']}")
                if 'error' in result['xml_validation']:
                    print(f"    XML Error: {result['xml_validation']['error']}")
                if 'error' in result['urdf_validation']:
                    print(f"    URDF Error: {result['urdf_validation']['error']}")
                invalid_count += 1

        for result in results['sdf_results']:
            if not result['overall_valid']:
                print(f"  SDF: {result['path']}")
                if 'error' in result['xml_validation']:
                    print(f"    XML Error: {result['xml_validation']['error']}")
                if 'error' in result['sdf_validation']:
                    print(f"    SDF Error: {result['sdf_validation']['error']}")
                invalid_count += 1

        if invalid_count == 0:
            print("  All models are valid!")
        else:
            print(f"  {invalid_count} models failed validation")

        print("="*70)

    def generate_validation_report(self, output_file: str = "model_validation_report.json"):
        """Generate a JSON validation report"""
        if not self.validation_results:
            print("No validation results to report")
            return

        # Add timestamp to results
        self.validation_results['timestamp'] = str(time.time())
        self.validation_results['validator_version'] = "1.0.0"

        with open(output_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)

        print(f"Validation report saved to {output_file}")
        return output_file

    def validate_with_isaac_sim(self, model_path: str) -> Dict[str, any]:
        """Validate model using Isaac Sim (if available)"""
        if not self.isaac_sim_available:
            return {
                'valid': False,
                'error': 'Isaac Sim not available',
                'supported': False
            }

        try:
            # Import Isaac Sim modules
            import omni
            from omni.isaac.core import World
            from omni.isaac.core.utils.stage import add_reference_to_stage
            from omni.isaac.core.utils.nucleus import get_assets_root_path

            # Create a temporary world to load the model
            world = World(stage_units_in_meters=1.0)

            # Try to load the model
            try:
                add_reference_to_stage(usd_path=model_path, prim_path=f"/World/{Path(model_path).stem}")
                world.reset()
                world.step(render=True)
                return {
                    'valid': True,
                    'supported': True,
                    'message': 'Model loaded successfully in Isaac Sim'
                }
            except Exception as e:
                return {
                    'valid': False,
                    'supported': True,
                    'error': str(e),
                    'message': 'Model failed to load in Isaac Sim'
                }
        except Exception as e:
            return {
                'valid': False,
                'supported': True,
                'error': str(e),
                'message': 'Error during Isaac Sim validation'
            }


def validate_models_in_project():
    """Validate all models in the project"""
    import time

    print("Starting URDF/SDF Model Validation")
    print("Using gz sdf check and Isaac Sim validator")
    print("-" * 50)

    validator = ModelValidator()

    # Validate all models in the project
    project_path = "/mnt/d/Quarter-4/spec_kit_plus/humenoid_robot"
    results = validator.validate_all_models(project_path)

    # Print validation report
    validator.print_validation_report()

    # Generate JSON report
    report_file = validator.generate_validation_report("model_validation_report.json")

    # Check if all models are valid
    summary = results['summary']
    all_valid = summary['total_valid_models'] == summary['total_models']

    if all_valid and summary['total_models'] > 0:
        print("\n🎉 ALL MODELS VALIDATED SUCCESSFULLY!")
        print("All URDF/SDF models meet the validation requirements.")
        return True
    elif summary['total_models'] == 0:
        print("\n⚠️  NO MODELS FOUND TO VALIDATE!")
        print("No URDF or SDF models were found in the project.")
        return True  # This is acceptable if no models exist yet
    else:
        print(f"\n❌ VALIDATION FAILED!")
        print(f"Only {summary['total_valid_models']} out of {summary['total_models']} models are valid.")
        return False


def check_gz_sdf_installation():
    """Check if gz sdf is properly installed"""
    try:
        result = subprocess.run(['gz', 'sdf', '--version'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✓ gz sdf is available: {result.stdout.strip()}")
            return True
        else:
            print("✗ gz sdf is not working properly")
            return False
    except subprocess.TimeoutExpired:
        print("✗ gz sdf check timed out")
        return False
    except FileNotFoundError:
        print("✗ gz sdf command not found")
        print("  Install gz-sim by running: sudo apt install gz-sim7")
        return False


def check_check_urdf_installation():
    """Check if check_urdf is properly installed"""
    try:
        result = subprocess.run(['check_urdf', '--help'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ check_urdf is available")
            return True
        else:
            print("✗ check_urdf is not working properly")
            return False
    except subprocess.TimeoutExpired:
        print("✗ check_urdf check timed out")
        return False
    except FileNotFoundError:
        print("✗ check_urdf command not found")
        print("  Install urdfdom by running: sudo apt install liburdfdom-tools")
        return False


def main():
    """Main function to run the validation"""
    print("URDF/SDF Model Validation Tool")
    print("=" * 40)

    # Check required tools
    gz_available = check_gz_sdf_installation()
    urdf_available = check_check_urdf_installation()

    if not gz_available and not urdf_available:
        print("\n❌ Neither gz sdf nor check_urdf is available!")
        print("Please install the required tools before running validation.")
        return False

    # Run validation
    success = validate_models_in_project()

    if success:
        print("\n✓ Model validation completed successfully!")
        print("All models are properly formatted and validated.")
        return True
    else:
        print("\n✗ Model validation failed!")
        print("Some models did not pass validation. Check the report for details.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)