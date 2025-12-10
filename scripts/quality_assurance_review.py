#!/usr/bin/env python3
# quality_assurance_review.py
# Script to conduct final content quality assurance review

import os
import re
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Tuple
import markdown
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, urljoin
import time

class QualityAssuranceReviewer:
    """Class to conduct comprehensive quality assurance review of the course content"""

    def __init__(self):
        self.review_results = {
            'structure': {},
            'content': {},
            'code': {},
            'links': {},
            'consistency': {},
            'readability': {},
            'accessibility': {}
        }
        self.total_issues = 0
        self.issues_found = []

        # Define quality standards
        self.quality_standards = {
            'min_chapter_length': 3000,  # words
            'max_chapter_length': 10000,  # words
            'min_heading_depth': 2,  # minimum heading level
            'max_heading_depth': 4,  # maximum heading level
            'min_code_examples_per_chapter': 3,
            'max_line_length': 120,
            'required_sections': ['Introduction', 'Learning Objectives', 'Summary', 'Exercises'],
            'forbidden_patterns': [
                r'\b(obviously|clearly|easily)\b',  # Condescending language
                r'(TODO|FIXME|XXX)',  # Incomplete content
                r'(?i)click here',  # Poor accessibility
            ]
        }

    def review_all_content(self) -> Dict:
        """Conduct comprehensive review of all course content"""
        print("Starting Quality Assurance Review...")
        print("=" * 60)

        # Review structure
        self.review_structure()

        # Review content
        self.review_content()

        # Review code examples
        self.review_code_examples()

        # Review links
        self.review_links()

        # Review consistency
        self.review_consistency()

        # Review readability
        self.review_readability()

        # Review accessibility
        self.review_accessibility()

        # Generate final report
        self.generate_report()

        return self.review_results

    def review_structure(self):
        """Review the structural integrity of the content"""
        print("\n1. Reviewing Content Structure...")

        structure_issues = []

        # Check if all required directories exist
        required_dirs = ['docs', 'src', 'static', 'scripts']
        for directory in required_dirs:
            if not os.path.exists(directory):
                structure_issues.append(f"Missing required directory: {directory}")

        # Check if all parts exist
        docs_path = Path('docs')
        expected_parts = ['part1', 'part2', 'part3', 'part4', 'part5', 'part6']
        for part in expected_parts:
            part_path = docs_path / part
            if not part_path.exists():
                structure_issues.append(f"Missing part directory: {part}")

        # Check if all chapters exist
        expected_chapters = [
            'chapter1', 'chapter2', 'chapter3',  # Part 1
            'chapter4', 'chapter5', 'chapter6',  # Part 2
            'chapter7', 'chapter8', 'chapter9',  # Part 3
            'chapter10', 'chapter11', 'chapter12',  # Part 4
            'chapter13', 'chapter14', 'chapter15',  # Part 5
            'chapter16', 'chapter17', 'chapter18',  # Part 6
        ]

        for chapter in expected_chapters:
            chapter_exists = False
            for part in expected_parts:
                chapter_path = docs_path / part / f"{chapter}.md"
                if chapter_path.exists():
                    chapter_exists = True
                    break
            if not chapter_exists:
                structure_issues.append(f"Missing chapter: {chapter}")

        # Check appendix structure
        appendix_files = [
            'hardware-specs.md',
            'code-references.md',
            'troubleshooting.md'
        ]
        for appendix_file in appendix_files:
            appendix_path = docs_path / 'appendix' / appendix_file
            if not appendix_path.exists():
                structure_issues.append(f"Missing appendix file: {appendix_file}")

        self.review_results['structure'] = {
            'issues_found': len(structure_issues),
            'issues': structure_issues,
            'status': 'PASS' if len(structure_issues) == 0 else 'FAIL'
        }

        self.total_issues += len(structure_issues)
        self.issues_found.extend(structure_issues)

        print(f"   Structure review: {len(structure_issues)} issues found")

    def review_content(self):
        """Review content quality and completeness"""
        print("\n2. Reviewing Content Quality...")

        content_issues = []

        # Find all markdown files
        md_files = []
        for root, dirs, files in os.walk('docs'):
            for file in files:
                if file.endswith('.md'):
                    md_files.append(os.path.join(root, file))

        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for minimum length
            word_count = len(content.split())
            if word_count < self.quality_standards['min_chapter_length']:
                content_issues.append(f"Content too short ({word_count} words) in {md_file}")

            # Check for forbidden patterns
            for pattern in self.quality_standards['forbidden_patterns']:
                if re.search(pattern, content, re.IGNORECASE):
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    content_issues.append(f"Forbidden pattern '{matches[0]}' found in {md_file}")

            # Check for proper heading structure
            lines = content.split('\n')
            headings = [line for line in lines if line.strip().startswith('#')]

            for heading in headings:
                level = len(heading) - len(heading.lstrip('#'))
                if level < self.quality_standards['min_heading_depth']:
                    content_issues.append(f"Heading too shallow (level {level}) in {md_file}")
                if level > self.quality_standards['max_heading_depth']:
                    content_issues.append(f"Heading too deep (level {level}) in {md_file}")

        self.review_results['content'] = {
            'issues_found': len(content_issues),
            'issues': content_issues,
            'status': 'PASS' if len(content_issues) == 0 else 'FAIL'
        }

        self.total_issues += len(content_issues)
        self.issues_found.extend(content_issues)

        print(f"   Content review: {len(content_issues)} issues found")

    def review_code_examples(self):
        """Review code examples for quality and correctness"""
        print("\n3. Reviewing Code Examples...")

        code_issues = []

        # Find all markdown files
        md_files = []
        for root, dirs, files in os.walk('docs'):
            for file in files:
                if file.endswith('.md'):
                    md_files.append(os.path.join(root, file))

        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find code blocks
            code_blocks = re.findall(r'```(\w+)?\n(.*?)```', content, re.DOTALL)

            # Count code examples
            if len(code_blocks) < self.quality_standards['min_code_examples_per_chapter']:
                code_issues.append(f"Insufficient code examples ({len(code_blocks)}) in {md_file}")

            # Check code quality
            for lang, code in code_blocks:
                # Check for line length
                lines = code.split('\n')
                for i, line in enumerate(lines, 1):
                    if len(line) > self.quality_standards['max_line_length']:
                        code_issues.append(f"Line too long ({len(line)} chars) in {md_file}:{i}")

                # Check for incomplete code
                if 'TODO' in code or 'FIXME' in code:
                    code_issues.append(f"Incomplete code example in {md_file}")

                # Check for proper ROS 2 syntax (if it's Python)
                if lang == 'python' or lang is None:
                    if re.search(r'import ros', code) and not re.search(r'import rclpy', code):
                        code_issues.append(f"Possible ROS 1 syntax in ROS 2 context in {md_file}")

        self.review_results['code'] = {
            'issues_found': len(code_issues),
            'issues': code_issues,
            'status': 'PASS' if len(code_issues) == 0 else 'FAIL'
        }

        self.total_issues += len(code_issues)
        self.issues_found.extend(code_issues)

        print(f"   Code review: {len(code_issues)} issues found")

    def review_links(self):
        """Review all links for validity"""
        print("\n4. Reviewing Links...")

        link_issues = []

        # Find all markdown files
        md_files = []
        for root, dirs, files in os.walk('docs'):
            for file in files:
                if file.endswith('.md'):
                    md_files.append(os.path.join(root, file))

        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find all links
            links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)

            for link_text, link_url in links:
                # Check if it's an external link
                if link_url.startswith(('http://', 'https://')):
                    try:
                        response = requests.head(link_url, timeout=10, allow_redirects=True)
                        if response.status_code >= 400:
                            link_issues.append(f"Broken external link: {link_url} in {md_file}")
                    except requests.RequestException:
                        link_issues.append(f"Unreachable external link: {link_url} in {md_file}")
                # Check if it's a relative link
                elif link_url.startswith(('.', '..')):
                    # Resolve relative path
                    file_dir = os.path.dirname(md_file)
                    full_path = os.path.join(file_dir, link_url)
                    if not os.path.exists(full_path):
                        link_issues.append(f"Broken relative link: {link_url} in {md_file}")

        self.review_results['links'] = {
            'issues_found': len(link_issues),
            'issues': link_issues,
            'status': 'PASS' if len(link_issues) == 0 else 'FAIL'
        }

        self.total_issues += len(link_issues)
        self.issues_found.extend(link_issues)

        print(f"   Link review: {len(link_issues)} issues found")

    def review_consistency(self):
        """Review content for consistency"""
        print("\n5. Reviewing Consistency...")

        consistency_issues = []

        # Find all markdown files
        md_files = []
        for root, dirs, files in os.walk('docs'):
            for file in files:
                if file.endswith('.md'):
                    md_files.append(os.path.join(root, file))

        # Check for consistent terminology
        terminology_issues = []
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read().lower()

            # Check for inconsistent terminology
            if 'ros2' in content and 'ros 2' in content:
                terminology_issues.append(f"Inconsistent ROS 2 terminology in {md_file}")
            if 'isaac sim' in content and 'isaac-sim' in content:
                terminology_issues.append(f"Inconsistent Isaac Sim terminology in {md_file}")

        consistency_issues.extend(terminology_issues)

        # Check for consistent formatting
        formatting_issues = []
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                # Check for trailing whitespace
                if line.rstrip() != line:
                    formatting_issues.append(f"Trailing whitespace in {md_file}:{i}")

        consistency_issues.extend(formatting_issues)

        self.review_results['consistency'] = {
            'issues_found': len(consistency_issues),
            'issues': consistency_issues,
            'status': 'PASS' if len(consistency_issues) == 0 else 'FAIL'
        }

        self.total_issues += len(consistency_issues)
        self.issues_found.extend(consistency_issues)

        print(f"   Consistency review: {len(consistency_issues)} issues found")

    def review_readability(self):
        """Review content for readability"""
        print("\n6. Reviewing Readability...")

        readability_issues = []

        # Find all markdown files
        md_files = []
        for root, dirs, files in os.walk('docs'):
            for file in files:
                if file.endswith('.md'):
                    md_files.append(os.path.join(root, file))

        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Convert markdown to plain text for readability analysis
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            plain_text = soup.get_text()

            # Check for complex sentences (simple heuristic)
            sentences = re.split(r'[.!?]+', plain_text)
            avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / len([s for s in sentences if s.strip()])

            if avg_sentence_length > 25:  # Too complex
                readability_issues.append(f"Complex sentences (avg {avg_sentence_length:.1f} words) in {md_file}")

        self.review_results['readability'] = {
            'issues_found': len(readability_issues),
            'issues': readability_issues,
            'status': 'PASS' if len(readability_issues) == 0 else 'FAIL'
        }

        self.total_issues += len(readability_issues)
        self.issues_found.extend(readability_issues)

        print(f"   Readability review: {len(readability_issues)} issues found")

    def review_accessibility(self):
        """Review content for accessibility"""
        print("\n7. Reviewing Accessibility...")

        accessibility_issues = []

        # Find all markdown files
        md_files = []
        for root, dirs, files in os.walk('docs'):
            for file in files:
                if file.endswith('.md'):
                    md_files.append(os.path.join(root, file))

        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for alt text in images
            images = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content)
            for alt_text, img_path in images:
                if not alt_text or alt_text.strip() == '':
                    accessibility_issues.append(f"Missing alt text for image {img_path} in {md_file}")

        self.review_results['accessibility'] = {
            'issues_found': len(accessibility_issues),
            'issues': accessibility_issues,
            'status': 'PASS' if len(accessibility_issues) == 0 else 'FAIL'
        }

        self.total_issues += len(accessibility_issues)
        self.issues_found.extend(accessibility_issues)

        print(f"   Accessibility review: {len(accessibility_issues)} issues found")

    def generate_report(self):
        """Generate comprehensive quality assurance report"""
        print("\n" + "=" * 60)
        print("QUALITY ASSURANCE REVIEW REPORT")
        print("=" * 60)

        print(f"\nTotal Issues Found: {self.total_issues}")

        for category, results in self.review_results.items():
            status_icon = "✅" if results['status'] == 'PASS' else "❌"
            print(f"\n{category.upper()}: {status_icon} {results['status']} ({results['issues_found']} issues)")
            if results['issues']:
                for issue in results['issues'][:5]:  # Show first 5 issues
                    print(f"  - {issue}")
                if len(results['issues']) > 5:
                    print(f"  ... and {len(results['issues']) - 5} more issues")

        if self.total_issues == 0:
            print("\n🎉 ALL CLEAR! No quality issues found.")
            print("The course content meets all quality assurance standards.")
        else:
            print(f"\n⚠️  QUALITY ISSUES FOUND! {self.total_issues} issues need to be addressed.")

            # Categorize issues by severity
            critical_issues = [issue for issue in self.issues_found if
                              any(keyword in issue.lower() for keyword in
                                  ['broken', 'missing', 'unreachable', 'forbidden'])]
            warning_issues = [issue for issue in self.issues_found if issue not in critical_issues]

            if critical_issues:
                print(f"\n🔴 CRITICAL ISSUES ({len(critical_issues)}):")
                for issue in critical_issues:
                    print(f"  - {issue}")

            if warning_issues:
                print(f"\n🟡 WARNING ISSUES ({len(warning_issues)}):")
                for issue in warning_issues:
                    print(f"  - {issue}")

        print("\n" + "=" * 60)

        # Save detailed report
        self.save_detailed_report()

    def save_detailed_report(self):
        """Save detailed QA report to file"""
        report_data = {
            'timestamp': time.time(),
            'total_issues': self.total_issues,
            'review_results': self.review_results,
            'all_issues': self.issues_found
        }

        with open('reports/quality_assurance_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)

        print("Detailed report saved to reports/quality_assurance_report.json")

def run_quality_assurance_review():
    """Run the complete quality assurance review"""
    print("Physical AI & Humanoid Robotics Course")
    print("Quality Assurance Review Tool")
    print("=" * 60)

    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)

    reviewer = QualityAssuranceReviewer()
    results = reviewer.review_all_content()

    # Return success status
    total_issues = sum(r['issues_found'] for r in results.values())
    return total_issues == 0

if __name__ == '__main__':
    success = run_quality_assurance_review()

    if success:
        print("\n✅ Quality assurance review completed successfully!")
        print("All content meets quality standards.")
    else:
        print("\n❌ Quality assurance review found issues!")
        print("Please address the issues before finalizing the course.")

    exit(0 if success else 1)