#!/usr/bin/env python3
# check_broken_links_deprecated_packages.py
# Script to check for broken links and deprecated packages in the course content

import os
import re
import requests
import subprocess
import yaml
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set
import time
from urllib.parse import urlparse, urljoin
import logging


class ContentValidator:
    """Class to validate course content for broken links and deprecated packages"""

    def __init__(self):
        self.broken_links = []
        self.deprecated_packages = []
        self.missing_files = []
        self.deprecated_content = []
        self.all_links = set()
        self.all_files = set()
        self.package_versions = {}

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Known deprecated packages and their replacements
        self.known_deprecated = {
            'ros:ros_comm': 'ros2:ros2-communication',
            'ros:geometry': 'ros2:tf2',
            'ros:navigation': 'ros2:nav2',
            'ros:vision_opencv': 'ros2:vision-opencv',
            'ros:common_msgs': 'ros2:common-interfaces',
            'rospack': 'ros2 pkg',
            'rosrun': 'ros2 run',
            'roslaunch': 'ros2 launch',
            'rosbag': 'ros2 bag',
        }

        # ROS 1 to ROS 2 command mappings
        self.ros1_to_ros2_commands = {
            'rostopic': 'ros2 topic',
            'rosservice': 'ros2 service',
            'rosnode': 'ros2 node',
            'rosparam': 'ros2 param',
            'roswtf': 'ros2 doctor',
        }

    def find_all_markdown_files(self, root_path: str) -> List[str]:
        """Find all markdown files in the project"""
        md_files = []
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.endswith('.md'):
                    md_files.append(os.path.join(root, file))
        return md_files

    def find_all_package_files(self, root_path: str) -> List[str]:
        """Find all package-related files (package.xml, CMakeLists.txt, etc.)"""
        package_files = []
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file in ['package.xml', 'CMakeLists.txt', 'requirements.txt', 'setup.py']:
                    package_files.append(os.path.join(root, file))
        return package_files

    def extract_links_from_file(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract all links from a markdown file"""
        links = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Regex patterns for different types of links
        patterns = [
            # Markdown links: [text](url)
            r'\[([^\]]+)\]\(([^)]+)\)',
            # HTML links: <a href="url">text</a>
            r'<a\s+href=["\']([^"\']+)["\'][^>]*>.*?</a>',
            # Direct URLs
            r'https?://[^\s\'"<>]+',
            # Image links
            r'!\[([^\]]*)\]\(([^)]+)\)',
        ]

        for line_num, line in enumerate(lines, 1):
            for pattern in patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    url = match.group(2) if len(match.groups()) > 1 else match.group(0)
                    if url.startswith(('http://', 'https://')):
                        links.append((url, line_num))

        return links

    def check_link_validity(self, url: str, timeout: int = 10) -> Dict[str, any]:
        """Check if a link is valid"""
        try:
            # Handle relative links by making them absolute
            if url.startswith('/'):
                # This is a relative link within the project
                project_path = Path(url[1:])  # Remove leading slash
                full_path = Path('/mnt/d/Quarter-4/spec_kit_plus/humenoid_robot') / project_path
                if full_path.exists():
                    return {'valid': True, 'type': 'local_file', 'path': str(full_path)}
                else:
                    return {'valid': False, 'type': 'local_file', 'error': 'File does not exist', 'path': str(full_path)}

            elif url.startswith(('.', '..')):
                # Relative link from current directory
                return {'valid': True, 'type': 'relative', 'url': url}  # Assume valid for now

            else:
                # HTTP/HTTPS link
                response = requests.head(url, timeout=timeout, allow_redirects=True)
                return {
                    'valid': response.status_code < 400,
                    'status_code': response.status_code,
                    'final_url': response.url,
                    'type': 'web'
                }

        except requests.exceptions.RequestException as e:
            return {'valid': False, 'error': str(e), 'type': 'web'}
        except Exception as e:
            return {'valid': False, 'error': str(e), 'type': 'unknown'}

    def check_all_links(self, root_path: str) -> Dict[str, any]:
        """Check all links in markdown files"""
        self.logger.info("Checking for broken links...")

        md_files = self.find_all_markdown_files(root_path)
        all_links = []

        # Extract all links from all markdown files
        for md_file in md_files:
            file_links = self.extract_links_from_file(md_file)
            for link, line_num in file_links:
                all_links.append((link, md_file, line_num))

        # Check validity of each link
        results = {
            'total_links': len(all_links),
            'valid_links': 0,
            'broken_links': [],
            'details': []
        }

        for link, file_path, line_num in all_links:
            self.logger.info(f"Checking link: {link} in {file_path}:{line_num}")
            validity = self.check_link_validity(link)

            link_result = {
                'url': link,
                'file': file_path,
                'line': line_num,
                'validity': validity
            }

            results['details'].append(link_result)

            if validity['valid']:
                results['valid_links'] += 1
            else:
                results['broken_links'].append(link_result)
                self.broken_links.append((link, file_path, line_num))

        return results

    def parse_package_xml(self, package_xml_path: str) -> Dict[str, any]:
        """Parse package.xml file to extract package information"""
        import xml.etree.ElementTree as ET

        try:
            tree = ET.parse(package_xml_path)
            root = tree.getroot()

            package_info = {
                'name': root.find('name').text if root.find('name') is not None else 'unknown',
                'version': root.find('version').text if root.find('version') is not None else 'unknown',
                'description': root.find('description').text if root.find('description') is not None else '',
                'maintainer': root.find('maintainer').text if root.find('maintainer') is not None else '',
                'license': root.find('license').text if root.find('license') is not None else '',
                'dependencies': [],
                'build_dependencies': [],
                'exec_dependencies': []
            }

            # Extract dependencies
            for dep in root.findall('depend'):
                package_info['dependencies'].append(dep.text)

            for dep in root.findall('build_depend'):
                package_info['build_dependencies'].append(dep.text)

            for dep in root.findall('exec_depend'):
                package_info['exec_dependencies'].append(dep.text)

            return package_info

        except ET.ParseError as e:
            self.logger.error(f"Error parsing package.xml {package_xml_path}: {e}")
            return None

    def check_ros_packages(self, root_path: str) -> Dict[str, any]:
        """Check for deprecated ROS packages"""
        self.logger.info("Checking for deprecated ROS packages...")

        package_files = self.find_all_package_files(root_path)
        results = {
            'total_packages': 0,
            'deprecated_packages': [],
            'deprecated_commands': [],
            'details': []
        }

        for pkg_file in package_files:
            if pkg_file.endswith('package.xml'):
                # Parse package.xml
                pkg_info = self.parse_package_xml(pkg_file)
                if pkg_info:
                    results['total_packages'] += 1

                    # Check dependencies for deprecated packages
                    all_deps = (pkg_info['dependencies'] +
                               pkg_info['build_dependencies'] +
                               pkg_info['exec_dependencies'])

                    for dep in all_deps:
                        if dep in self.known_deprecated:
                            deprecated_info = {
                                'package': dep,
                                'replacement': self.known_deprecated[dep],
                                'file': pkg_file,
                                'type': 'dependency'
                            }
                            results['deprecated_packages'].append(deprecated_info)
                            self.deprecated_packages.append(deprecated_info)

            elif pkg_file.endswith('requirements.txt'):
                # Check requirements.txt for deprecated packages
                with open(pkg_file, 'r') as f:
                    content = f.read()

                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package name (before version specifiers)
                        pkg_name = re.split(r'[<>=!]', line)[0].strip()
                        if pkg_name in self.known_deprecated:
                            deprecated_info = {
                                'package': pkg_name,
                                'replacement': self.known_deprecated[pkg_name],
                                'file': pkg_file,
                                'type': 'requirement'
                            }
                            results['deprecated_packages'].append(deprecated_info)
                            self.deprecated_packages.append(deprecated_info)

            # Check all files for deprecated commands
            self.check_deprecated_commands_in_file(pkg_file, results)

        return results

    def check_deprecated_commands_in_file(self, file_path: str, results: Dict):
        """Check a file for deprecated ROS commands"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Check for deprecated ROS 1 commands
            for ros1_cmd, ros2_cmd in self.ros1_to_ros2_commands.items():
                if ros1_cmd in content:
                    # Find line numbers where deprecated commands appear
                    lines = content.split('\n')
                    for line_num, line in enumerate(lines, 1):
                        if ros1_cmd in line:
                            deprecated_cmd = {
                                'command': ros1_cmd,
                                'replacement': ros2_cmd,
                                'file': file_path,
                                'line': line_num,
                                'context': line.strip()
                            }
                            results['deprecated_commands'].append(deprecated_cmd)
                            self.deprecated_content.append(deprecated_cmd)

        except Exception as e:
            self.logger.error(f"Error checking deprecated commands in {file_path}: {e}")

    def check_file_references(self, root_path: str) -> Dict[str, any]:
        """Check for missing file references"""
        self.logger.info("Checking for missing file references...")

        md_files = self.find_all_markdown_files(root_path)
        results = {
            'total_references': 0,
            'missing_files': [],
            'details': []
        }

        # Find all file references in markdown files
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Find relative file references
            # Look for [text](file.md), [text](image.png), etc.
            file_refs = re.findall(r'\[([^\]]+)\]\(([^)]*\.(md|png|jpg|jpeg|gif|pdf|svg))\)', content)

            for ref_text, ref_file, ext in file_refs:
                # Resolve relative path
                md_dir = Path(md_file).parent
                full_path = (md_dir / ref_file).resolve()

                # Check if file exists
                if not full_path.exists():
                    missing_ref = {
                        'reference': ref_file,
                        'full_path': str(full_path),
                        'from_file': md_file,
                        'reference_text': ref_text
                    }
                    results['missing_files'].append(missing_ref)
                    self.missing_files.append(missing_ref)

                results['total_references'] += 1

        return results

    def validate_all_content(self, root_path: str) -> Dict[str, any]:
        """Run all validation checks"""
        self.logger.info(f"Starting content validation in: {root_path}")

        results = {
            'links': self.check_all_links(root_path),
            'packages': self.check_ros_packages(root_path),
            'files': self.check_file_references(root_path),
            'summary': {}
        }

        # Generate summary
        total_broken = len(results['links']['broken_links'])
        total_deprecated_pkgs = len(results['packages']['deprecated_packages'])
        total_deprecated_cmds = len(results['packages']['deprecated_commands'])
        total_missing_files = len(results['files']['missing_files'])

        results['summary'] = {
            'total_broken_links': total_broken,
            'total_deprecated_packages': total_deprecated_pkgs,
            'total_deprecated_commands': total_deprecated_cmds,
            'total_missing_files': total_missing_files,
            'total_issues': total_broken + total_deprecated_pkgs + total_deprecated_cmds + total_missing_files,
            'all_clear': (total_broken == 0 and
                         total_deprecated_pkgs == 0 and
                         total_deprecated_cmds == 0 and
                         total_missing_files == 0)
        }

        return results

    def print_validation_report(self, results: Dict):
        """Print a formatted validation report"""
        summary = results['summary']

        print("\n" + "="*80)
        print("CONTENT VALIDATION REPORT")
        print("="*80)

        print(f"Broken Links: {summary['total_broken_links']}")
        if results['links']['broken_links']:
            for link_info in results['links']['broken_links']:
                print(f"  - {link_info['url']} in {link_info['file']}:{link_info['line_num']}")

        print(f"\nDeprecated Packages: {summary['total_deprecated_packages']}")
        if results['packages']['deprecated_packages']:
            for pkg_info in results['packages']['deprecated_packages']:
                print(f"  - {pkg_info['package']} -> {pkg_info['replacement']} in {pkg_info['file']}")

        print(f"\nDeprecated Commands: {summary['total_deprecated_commands']}")
        if results['packages']['deprecated_commands']:
            for cmd_info in results['packages']['deprecated_commands']:
                print(f"  - {cmd_info['command']} -> {cmd_info['replacement']} in {cmd_info['file']}:{cmd_info['line']}")
                print(f"    Context: {cmd_info['context']}")

        print(f"\nMissing Files: {summary['total_missing_files']}")
        if results['files']['missing_files']:
            for file_info in results['files']['missing_files']:
                print(f"  - {file_info['reference']} from {file_info['from_file']}")

        print(f"\nTotal Issues Found: {summary['total_issues']}")

        if summary['all_clear']:
            print("\n🎉 ALL CLEAR! No issues found in content validation.")
        else:
            print(f"\n❌ ISSUES FOUND! {summary['total_issues']} issues need to be addressed.")

        print("="*80)

    def generate_fix_suggestions(self) -> List[str]:
        """Generate suggestions for fixing identified issues"""
        suggestions = []

        # Broken links suggestions
        if self.broken_links:
            suggestions.append("BROKEN LINKS FIX SUGGESTIONS:")
            for url, file_path, line_num in self.broken_links:
                suggestions.append(f"  - In {file_path}:{line_num}, fix or remove broken link: {url}")
            suggestions.append("")

        # Deprecated packages suggestions
        if self.deprecated_packages:
            suggestions.append("DEPRECATED PACKAGES FIX SUGGESTIONS:")
            for pkg_info in self.deprecated_packages:
                old_pkg = pkg_info['package']
                new_pkg = pkg_info['replacement']
                suggestions.append(f"  - Replace '{old_pkg}' with '{new_pkg}' in {pkg_info['file']}")
            suggestions.append("")

        # Deprecated commands suggestions
        if self.deprecated_content:
            suggestions.append("DEPRECATED COMMANDS FIX SUGGESTIONS:")
            for cmd_info in self.deprecated_content:
                old_cmd = cmd_info['command']
                new_cmd = cmd_info['replacement']
                suggestions.append(f"  - Replace '{old_cmd}' with '{new_cmd}' in {cmd_info['file']}:{cmd_info['line']}")
            suggestions.append("")

        # Missing files suggestions
        if self.missing_files:
            suggestions.append("MISSING FILES FIX SUGGESTIONS:")
            for file_info in self.missing_files:
                suggestions.append(f"  - Create or fix reference to '{file_info['reference']}' in {file_info['from_file']}")
            suggestions.append("")

        return suggestions

    def create_validation_report_file(self, results: Dict, output_file: str = "content_validation_report.json"):
        """Create a detailed validation report file"""
        report = {
            'timestamp': time.time(),
            'validation_results': results,
            'fix_suggestions': self.generate_fix_suggestions()
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Validation report saved to {output_file}")
        return output_file


def validate_course_content():
    """Validate the entire course content"""
    print("Starting Course Content Validation")
    print("Checking for broken links and deprecated packages")
    print("-" * 50)

    validator = ContentValidator()
    project_path = "/mnt/d/Quarter-4/spec_kit_plus/humenoid_robot"

    # Run all validations
    results = validator.validate_all_content(project_path)

    # Print report
    validator.print_validation_report(results)

    # Create detailed report file
    report_file = validator.create_validation_report_file(results)

    # Determine if content passes validation
    summary = results['summary']
    content_valid = summary['all_clear']

    if content_valid:
        print("\n✓ Content validation passed!")
        print("No broken links or deprecated packages found.")
        return True
    else:
        print(f"\n✗ Content validation failed!")
        print(f"{summary['total_issues']} issues need to be addressed before content is ready.")

        # Print fix suggestions
        suggestions = validator.generate_fix_suggestions()
        print("\nFIX SUGGESTIONS:")
        for suggestion in suggestions:
            if suggestion == "":
                print("-" * 40)
            else:
                print(suggestion)

        return False


def check_package_versions():
    """Check if ROS packages are up to date"""
    print("Checking ROS package versions...")

    try:
        # Check ROS 2 Iron installation
        result = subprocess.run(['ros2', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ ROS 2 version: {result.stdout.strip()}")
        else:
            print("✗ ROS 2 not properly installed")
            return False

        # Check for common ROS packages
        common_packages = [
            'rclpy',
            'std_msgs',
            'sensor_msgs',
            'geometry_msgs',
            'nav_msgs',
            'visualization_msgs',
            'tf2_ros',
            'cv_bridge',
            'message_filters'
        ]

        missing_packages = []
        for pkg in common_packages:
            try:
                result = subprocess.run(['ros2', 'pkg', 'list'], capture_output=True, text=True)
                if pkg not in result.stdout:
                    missing_packages.append(pkg)
            except:
                missing_packages.append(pkg)

        if missing_packages:
            print(f"✗ Missing ROS packages: {', '.join(missing_packages)}")
            return False
        else:
            print("✓ All required ROS packages are available")

        return True

    except Exception as e:
        print(f"Error checking package versions: {e}")
        return False


def main():
    """Main function to run the validation"""
    print("Course Content Validation Tool")
    print("=" * 40)

    # Check package versions first
    packages_ok = check_package_versions()

    if not packages_ok:
        print("\n❌ Package validation failed!")
        print("Required ROS packages are missing or not properly installed.")
        return False

    # Validate course content
    content_ok = validate_course_content()

    if content_ok:
        print("\n✓ All content validation checks passed!")
        print("The course content is ready with no broken links or deprecated packages.")
        return True
    else:
        print("\n✗ Content validation failed!")
        print("Issues were found that need to be addressed before the course is ready.")
        return False


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)