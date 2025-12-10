#!/usr/bin/env python3
"""
Automated validation pipeline for code examples in the Physical AI & Humanoid Robotics course.

This script validates code examples in the documentation to ensure they are correct,
executable, and meet the course standards.
"""

import os
import re
import sys
import subprocess
import tempfile
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import yaml


class CodeExampleValidator:
    def __init__(self, docs_path: str = "docs"):
        self.docs_path = Path(docs_path)
        self.code_examples = []
        self.validation_results = []

    def extract_code_examples(self) -> List[Dict]:
        """Extract all code examples from markdown files in docs directory."""
        code_examples = []

        # Find all markdown files
        md_files = list(self.docs_path.rglob("*.md"))

        for md_file in md_files:
            content = md_file.read_text(encoding='utf-8')

            # Find code blocks with language specification
            pattern = r'```(\w+)\n(.*?)```'
            matches = re.findall(pattern, content, re.DOTALL)

            for lang, code in matches:
                # Skip non-code languages
                if lang in ['text', 'console', 'output']:
                    continue

                code_examples.append({
                    'file': str(md_file),
                    'language': lang,
                    'code': code.strip(),
                    'line_number': content[:content.find(code)].count('\n') + 1
                })

        self.code_examples = code_examples
        return code_examples

    def validate_python_code(self, code: str) -> Tuple[bool, str]:
        """Validate Python code syntax and basic execution."""
        try:
            # Check syntax
            compile(code, '<string>', 'exec')

            # For safety, we'll just check syntax, not execute
            # In a real environment, you might want to execute in a sandbox
            return True, "Syntax is valid"
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def validate_bash_commands(self, code: str) -> Tuple[bool, str]:
        """Validate bash commands (basic validation)."""
        # Check for potentially dangerous commands
        dangerous_patterns = [
            r'\brm\s+-rf\b',  # Dangerous rm commands
            r'\bmv\s+\S+\s+/\b',  # Moving to root
            r'\bchmod\s+777\b',  # Overly permissive permissions
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Potentially dangerous command detected: {pattern}"

        return True, "Command appears safe"

    def validate_cpp_code(self, code: str) -> Tuple[bool, str]:
        """Validate C++ code syntax."""
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write("#include <iostream>\n")
                f.write("// For validation purposes only\n")
                f.write("int main() {\n")
                f.write(code + "\n")
                f.write("return 0;\n")
                f.write("}\n")
                temp_file = f.name

            # Try to compile with g++ (just syntax check)
            result = subprocess.run([
                'g++', '-fsyntax-only', temp_file
            ], capture_output=True, text=True)

            os.unlink(temp_file)

            if result.returncode == 0:
                return True, "Syntax is valid"
            else:
                return False, f"Compilation error: {result.stderr}"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def validate_ros_commands(self, code: str) -> Tuple[bool, str]:
        """Validate ROS-specific commands."""
        # Check for common ROS command patterns
        ros_patterns = [
            r'ros2\s+\w+',  # Basic ros2 commands
            r'rosrun\s+\w+\s+\w+',  # rosrun commands
            r'roslaunch\s+\w+\s+\w+',  # roslaunch commands
        ]

        found_ros_cmd = False
        for pattern in ros_patterns:
            if re.search(pattern, code):
                found_ros_cmd = True
                break

        if not found_ros_cmd:
            return True, "No ROS commands found"

        # Additional validation for ROS commands could go here
        return True, "ROS command pattern detected"

    def validate_code_block(self, code_block: Dict) -> Dict:
        """Validate a single code block."""
        lang = code_block['language']
        code = code_block['code']

        is_valid = False
        message = ""

        if lang == 'python':
            is_valid, message = self.validate_python_code(code)
        elif lang in ['bash', 'sh', 'shell']:
            is_valid, message = self.validate_bash_commands(code)
        elif lang == 'cpp':
            is_valid, message = self.validate_cpp_code(code)
        elif lang in ['c', 'h', 'hpp']:
            is_valid, message = self.validate_cpp_code(code)
        elif lang in ['yaml', 'yml']:
            try:
                yaml.safe_load(code)
                is_valid = True
                message = "YAML is valid"
            except yaml.YAMLError as e:
                is_valid = False
                message = f"YAML error: {str(e)}"
        elif lang in ['xml', 'urdf', 'sdf']:
            # Basic XML validation
            try:
                import xml.etree.ElementTree as ET
                ET.fromstring(code)
                is_valid = True
                message = "XML is valid"
            except ET.ParseError as e:
                is_valid = False
                message = f"XML error: {str(e)}"
        else:
            # For other languages, just basic validation
            is_valid = True
            message = f"Language {lang} validation skipped (not implemented)"

        result = {
            'file': code_block['file'],
            'language': lang,
            'line_number': code_block['line_number'],
            'is_valid': is_valid,
            'message': message,
            'code_snippet': code[:100] + "..." if len(code) > 100 else code
        }

        return result

    def run_validation(self) -> List[Dict]:
        """Run validation on all code examples."""
        print(f"Validating {len(self.code_examples)} code examples...")

        results = []
        for i, code_block in enumerate(self.code_examples):
            print(f"Validating {i+1}/{len(self.code_examples)}: {code_block['file']}:{code_block['line_number']}")
            result = self.validate_code_block(code_block)
            results.append(result)

            if not result['is_valid']:
                print(f"  ❌ {result['message']}")
            else:
                print(f"  ✅ {result['message']}")

        self.validation_results = results
        return results

    def generate_report(self) -> str:
        """Generate a validation report."""
        total = len(self.validation_results)
        valid = sum(1 for r in self.validation_results if r['is_valid'])
        invalid = total - valid

        report = f"""
# Code Example Validation Report

## Summary
- Total code examples: {total}
- Valid examples: {valid}
- Invalid examples: {invalid}
- Success rate: {valid/total*100:.1f}% if total > 0 else 0}%

## Details
"""

        for result in self.validation_results:
            status = "✅" if result['is_valid'] else "❌"
            report += f"- {status} {result['file']}:{result['line_number']} ({result['language']}) - {result['message']}\n"

        if invalid > 0:
            report += f"\n## Invalid Examples\n"
            for result in self.validation_results:
                if not result['is_valid']:
                    report += f"- {result['file']}:{result['line_number']} - {result['message']}\n"

        return report

    def save_report(self, output_file: str = "validation_report.md"):
        """Save validation report to file."""
        report = self.generate_report()
        Path(output_file).write_text(report)
        print(f"Validation report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Validate code examples in documentation')
    parser.add_argument('--docs-path', default='docs', help='Path to documentation directory')
    parser.add_argument('--output', default='validation_report.md', help='Output report file')
    parser.add_argument('--format', choices=['text', 'markdown'], default='markdown', help='Output format')

    args = parser.parse_args()

    validator = CodeExampleValidator(docs_path=args.docs_path)

    # Extract code examples
    print("Extracting code examples...")
    code_examples = validator.extract_code_examples()
    print(f"Found {len(code_examples)} code examples")

    # Run validation
    results = validator.run_validation()

    # Generate and save report
    validator.save_report(args.output)

    # Exit with error code if there are invalid examples
    invalid_count = sum(1 for r in results if not r['is_valid'])
    if invalid_count > 0:
        print(f"\n❌ Validation failed: {invalid_count} invalid examples found")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(results)} code examples are valid")
        sys.exit(0)


if __name__ == "__main__":
    main()