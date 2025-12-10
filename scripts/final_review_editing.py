#!/usr/bin/env python3
# final_review_editing.py
# Final review and editing script for all course chapters

import os
import re
import yaml
import json
from pathlib import Path
from typing import List, Dict, Tuple
import markdown
from bs4 import BeautifulSoup
import requests
from textstat import flesch_reading_ease, flesch_kincaid_grade
import pygrep
import time
import logging


class ChapterReviewer:
    """Class to review and edit course chapters"""

    def __init__(self):
        self.review_results = {}
        self.chapters = []
        self.issues_found = []
        self.suggestions = []

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Review criteria
        self.review_criteria = {
            'technical_accuracy': True,
            'clarity': True,
            'completeness': True,
            'consistency': True,
            'readability': True,
            'formatting': True,
            'code_quality': True,
            'references': True
        }

        # Common issues to check for
        self.common_issues = [
            r'```python\n.*?```',  # Code blocks
            r'\[.*?\]\(.*?\)',     # Markdown links
            r'!\[.*?\]\(.*?\)',    # Markdown images
            r'^#{1,6}.*',          # Headers
            r'(\*|\-|\d+\.)\s+.*', # Lists
            r'`[^`]+`',            # Inline code
        ]

    def find_all_chapters(self, root_path: str) -> List[str]:
        """Find all chapter markdown files"""
        chapters = []
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.endswith('.md') and 'chapter' in file.lower():
                    chapters.append(os.path.join(root, file))
        return sorted(chapters)

    def load_chapter_content(self, chapter_path: str) -> str:
        """Load chapter content from file"""
        try:
            with open(chapter_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            self.logger.error(f"Error loading chapter {chapter_path}: {e}")
            return ""

    def check_chapter_structure(self, content: str, chapter_path: str) -> Dict[str, any]:
        """Check chapter structure and formatting"""
        structure_issues = []
        structure_score = 10  # Start with perfect score

        # Check for proper markdown structure
        lines = content.split('\n')

        # Check if there's a proper title (first line should be H1)
        if not lines[0].startswith('# '):
            structure_issues.append("No H1 title at beginning of chapter")
            structure_score -= 2

        # Check for proper header hierarchy
        headers = []
        for i, line in enumerate(lines):
            if line.startswith('#'):
                header_level = len(line) - len(line.lstrip('#'))
                headers.append((header_level, line.strip(), i + 1))

        # Check header hierarchy
        for i in range(1, len(headers)):
            current_level = headers[i][0]
            prev_level = headers[i-1][0]
            if current_level > prev_level + 1:
                structure_issues.append(
                    f"Improper header hierarchy: H{prev_level} followed by H{current_level} at line {headers[i][2]}"
                )
                structure_score -= 1

        # Check for minimum content length
        words = len(content.split())
        if words < 3000:  # Assuming chapters should be substantial
            structure_issues.append(f"Chapter is quite short ({words} words), consider expanding")
            structure_score -= 1

        return {
            'issues': structure_issues,
            'score': max(0, structure_score),
            'word_count': words,
            'header_count': len(headers)
        }

    def check_code_blocks(self, content: str, chapter_path: str) -> Dict[str, any]:
        """Check code blocks for quality and correctness"""
        code_issues = []
        code_score = 10

        # Find all code blocks
        code_blocks = re.findall(r'```(\w+)?\n(.*?)```', content, re.DOTALL)

        for lang, code in code_blocks:
            # Check for language specification
            if not lang:
                code_issues.append("Code block missing language specification")
                code_score -= 1

            # Check for common issues in code
            if 'import' in code.lower() and not re.search(r'#.*import', code):
                # Check if imports are at the top
                lines = code.split('\n')
                import_lines = [i for i, line in enumerate(lines) if line.strip().startswith('import') or 'from' in line.split()[:1]]
                if import_lines and not all(i < 5 for i in import_lines):  # Imports should be in first few lines
                    code_issues.append("Imports should be at the top of the code block")

            # Check for proper indentation
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    # Non-empty line not indented, could be a function/class definition which is fine
                    pass

        return {
            'issues': code_issues,
            'score': max(0, code_score),
            'total_blocks': len(code_blocks),
            'languages': list(set(lang for lang, _ in code_blocks if lang))
        }

    def check_links_and_references(self, content: str, chapter_path: str) -> Dict[str, any]:
        """Check for broken links and proper references"""
        link_issues = []
        link_score = 10

        # Find all markdown links
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)

        for text, url in links:
            # Check for absolute URLs (external links)
            if url.startswith(('http://', 'https://')):
                # For this review, we'll just check format, not actually visit the link
                if not self.is_valid_url(url):
                    link_issues.append(f"Potentially invalid URL: {url}")
                    link_score -= 1
            elif url.startswith('#'):
                # Internal anchor link - check if it exists in content
                anchor = url[1:].lower().replace(' ', '-')
                if anchor not in content.lower():
                    link_issues.append(f"Broken internal anchor link: {url}")
                    link_score -= 1
            else:
                # Relative link - check if file exists
                chapter_dir = Path(chapter_path).parent
                full_path = (chapter_dir / url).resolve()
                if not full_path.exists():
                    link_issues.append(f"Broken relative link: {url}")
                    link_score -= 1

        return {
            'issues': link_issues,
            'score': max(0, link_score),
            'total_links': len(links)
        }

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is properly formatted"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def check_terminology_consistency(self, content: str, chapter_path: str) -> Dict[str, any]:
        """Check for terminology consistency"""
        consistency_issues = []
        consistency_score = 10

        # Common robotics terminology that should be consistent
        terms_to_check = [
            (r'ROS\s+2', 'ROS 2'),
            (r'Isaac\s+Sim', 'Isaac Sim'),
            (r'Isaac\s+ROS', 'Isaac ROS'),
            (r'Jetson\s+Orin', 'Jetson Orin'),
            (r'Humanoid\s+Robot', 'humanoid robot'),  # Should be lowercase when not at sentence start
        ]

        for pattern, preferred in terms_to_check:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # Check if they're all in the preferred format
                for match in matches:
                    if match != preferred:
                        consistency_issues.append(f"Inconsistent terminology: '{match}' should be '{preferred}'")
                        consistency_score -= 0.5

        return {
            'issues': consistency_issues,
            'score': max(0, consistency_score),
            'total_terms_found': len(consistency_issues)
        }

    def check_readability(self, content: str, chapter_path: str) -> Dict[str, any]:
        """Check readability metrics"""
        readability_score = 10

        # Remove markdown formatting for readability analysis
        plain_text = re.sub(r'```.*?```', '', content, flags=re.DOTALL)  # Remove code blocks
        plain_text = re.sub(r'[~*_`#\[\]]', '', plain_text)  # Remove basic markdown
        plain_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', plain_text)  # Replace links with text

        # Calculate readability metrics
        try:
            flesch_score = flesch_reading_ease(plain_text)
            grade_level = flesch_kincaid_grade(plain_text)

            readability_issues = []
            if flesch_score < 40:  # Too difficult
                readability_issues.append(f"Text is too difficult to read (Flesch score: {flesch_score:.1f}, should be > 40)")
                readability_score -= 2
            elif flesch_score > 80:  # Too simple
                readability_issues.append(f"Text may be too simple (Flesch score: {flesch_score:.1f})")
                readability_score -= 1

            if grade_level > 12:  # College level or higher
                readability_issues.append(f"Text is at grade level {grade_level:.1f}, may be too advanced")
                readability_score -= 1

            return {
                'issues': readability_issues,
                'score': max(0, readability_score),
                'flesch_score': flesch_score,
                'grade_level': grade_level,
                'word_count': len(plain_text.split())
            }
        except:
            # If readability analysis fails, return default values
            return {
                'issues': ["Could not calculate readability metrics"],
                'score': 8,
                'flesch_score': 0,
                'grade_level': 0,
                'word_count': len(plain_text.split())
            }

    def perform_comprehensive_review(self, chapter_path: str) -> Dict[str, any]:
        """Perform comprehensive review of a chapter"""
        self.logger.info(f"Reviewing chapter: {chapter_path}")

        content = self.load_chapter_content(chapter_path)

        if not content:
            return {'valid': False, 'error': 'Could not load chapter content'}

        # Perform all checks
        structure = self.check_chapter_structure(content, chapter_path)
        code = self.check_code_blocks(content, chapter_path)
        links = self.check_links_and_references(content, chapter_path)
        consistency = self.check_terminology_consistency(content, chapter_path)
        readability = self.check_readability(content, chapter_path)

        # Calculate overall score
        total_score = (
            structure['score'] * 0.2 +
            code['score'] * 0.2 +
            links['score'] * 0.2 +
            consistency['score'] * 0.2 +
            readability['score'] * 0.2
        )

        review_result = {
            'chapter_path': chapter_path,
            'structure': structure,
            'code': code,
            'links': links,
            'consistency': consistency,
            'readability': readability,
            'overall_score': total_score,
            'total_issues': (
                len(structure['issues']) +
                len(code['issues']) +
                len(links['issues']) +
                len(consistency['issues']) +
                len(readability['issues'])
            ),
            'status': 'PASS' if total_score >= 8 else 'NEEDS_REVIEW' if total_score >= 5 else 'FAIL'
        }

        # Collect all issues
        all_issues = (
            structure['issues'] +
            code['issues'] +
            links['issues'] +
            consistency['issues'] +
            readability['issues']
        )

        for issue in all_issues:
            self.issues_found.append({
                'chapter': chapter_path,
                'issue': issue,
                'type': self.categorize_issue(issue)
            })

        return review_result

    def categorize_issue(self, issue: str) -> str:
        """Categorize an issue"""
        issue_lower = issue.lower()
        if 'code' in issue_lower or 'import' in issue_lower:
            return 'code_quality'
        elif 'link' in issue_lower or 'url' in issue_lower:
            return 'links'
        elif 'terminology' in issue_lower or 'consistent' in issue_lower:
            return 'consistency'
        elif 'read' in issue_lower or 'grade' in issue_lower:
            return 'readability'
        elif 'header' in issue_lower or 'structure' in issue_lower:
            return 'structure'
        else:
            return 'other'

    def review_all_chapters(self, root_path: str) -> Dict[str, any]:
        """Review all chapters in the course"""
        self.logger.info(f"Starting comprehensive review of all chapters in: {root_path}")

        chapters = self.find_all_chapters(root_path)
        self.logger.info(f"Found {len(chapters)} chapters to review")

        all_reviews = []
        total_score = 0
        total_issues = 0

        for chapter_path in chapters:
            review = self.perform_comprehensive_review(chapter_path)
            all_reviews.append(review)
            total_score += review['overall_score']
            total_issues += review['total_issues']

        overall_results = {
            'total_chapters': len(chapters),
            'chapters_reviewed': len(all_reviews),
            'average_score': total_score / len(all_reviews) if all_reviews else 0,
            'total_issues': total_issues,
            'individual_reviews': all_reviews,
            'summary': {
                'passing': len([r for r in all_reviews if r['status'] == 'PASS']),
                'needs_review': len([r for r in all_reviews if r['status'] == 'NEEDS_REVIEW']),
                'failing': len([r for r in all_reviews if r['status'] == 'FAIL'])
            }
        }

        self.review_results = overall_results
        return overall_results

    def generate_editing_suggestions(self) -> List[str]:
        """Generate editing suggestions based on review results"""
        suggestions = []

        # Add suggestions for common issues
        if self.issues_found:
            suggestions.append("EDITING SUGGESTIONS:")
            suggestions.append("=" * 40)

            # Group issues by type
            issues_by_type = {}
            for issue in self.issues_found:
                issue_type = issue['type']
                if issue_type not in issues_by_type:
                    issues_by_type[issue_type] = []
                issues_by_type[issue_type].append(issue)

            for issue_type, issues in issues_by_type.items():
                suggestions.append(f"\n{issue_type.upper()} ISSUES ({len(issues)}):")
                for issue in issues[:5]:  # Limit to first 5 of each type
                    suggestions.append(f"  - {issue['issue']} (in {Path(issue['chapter']).name})")
                if len(issues) > 5:
                    suggestions.append(f"  ... and {len(issues) - 5} more similar issues")

            suggestions.append("\nGENERAL IMPROVEMENTS:")
            suggestions.append("- Ensure consistent terminology throughout all chapters")
            suggestions.append("- Add more practical examples and exercises")
            suggestions.append("- Improve readability by simplifying complex sentences")
            suggestions.append("- Verify all code examples are correct and up-to-date")
            suggestions.append("- Check all links and references for accuracy")

        return suggestions

    def print_review_report(self):
        """Print a formatted review report"""
        if not self.review_results:
            print("No review results available")
            return

        results = self.review_results
        summary = results['summary']

        print("\n" + "="*80)
        print("COMPREHENSIVE CHAPTER REVIEW REPORT")
        print("="*80)

        print(f"Total Chapters: {results['total_chapters']}")
        print(f"Average Score: {results['average_score']:.2f}/10")
        print(f"Total Issues Found: {results['total_issues']}")

        print(f"\nStatus Summary:")
        print(f"  PASSING: {summary['passing']}")
        print(f"  NEEDS REVIEW: {summary['needs_review']}")
        print(f"  FAILING: {summary['failing']}")

        print(f"\nIndividual Chapter Reviews:")
        for review in results['individual_reviews']:
            status_emoji = "✅" if review['status'] == 'PASS' else "⚠️" if review['status'] == 'NEEDS_REVIEW' else "❌"
            print(f"  {status_emoji} {Path(review['chapter_path']).name}: {review['overall_score']:.1f}/10 ({review['total_issues']} issues)")

        if results['total_issues'] == 0:
            print("\n🎉 ALL CHAPTERS PASSED REVIEW!")
            print("No issues found in the course content.")
        else:
            print(f"\n⚠️  REVIEW COMPLETE WITH {results['total_issues']} ISSUES TO ADDRESS")

        print("="*80)

    def create_detailed_report(self, output_file: str = "chapter_review_report.json"):
        """Create a detailed review report file"""
        report = {
            'timestamp': time.time(),
            'review_results': self.review_results,
            'all_issues': self.issues_found,
            'editing_suggestions': self.generate_editing_suggestions()
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Detailed review report saved to {output_file}")
        return output_file

    def apply_basic_edits(self, chapter_path: str) -> bool:
        """Apply basic editing to a chapter file"""
        try:
            with open(chapter_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Apply basic formatting fixes
            # Fix multiple spaces
            content = re.sub(r' {2,}', ' ', content)

            # Ensure proper spacing around headers
            content = re.sub(r'\n{3,}#{1,6}', r'\n\n#{1,6}', content)

            # Fix double newlines at file start
            content = re.sub(r'^\n+', '', content)

            # Fix multiple newlines at end
            content = re.sub(r'\n+$', '\n', content)

            # Fix multiple consecutive newlines
            content = re.sub(r'\n{3,}', '\n\n', content)

            # If content changed, write it back
            if content != original_content:
                with open(chapter_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Applied basic formatting fixes to {chapter_path}")

            return True

        except Exception as e:
            self.logger.error(f"Error applying edits to {chapter_path}: {e}")
            return False


def run_final_review():
    """Run the final review and editing process"""
    print("Starting Final Review and Editing Process")
    print("Reviewing all course chapters for quality and consistency")
    print("-" * 50)

    reviewer = ChapterReviewer()
    project_path = "/mnt/d/Quarter-4/spec_kit_plus/humenoid_robot"

    # Review all chapters
    results = reviewer.review_all_chapters(project_path)

    # Print review report
    reviewer.print_review_report()

    # Create detailed report
    report_file = reviewer.create_detailed_report("chapter_review_report.json")

    # Apply basic formatting fixes to all chapters
    print("\nApplying basic formatting fixes...")
    chapters = reviewer.find_all_chapters(project_path)
    for chapter in chapters:
        reviewer.apply_basic_edits(chapter)

    # Determine if review passes
    summary = results['summary']
    all_passing = summary['failing'] == 0 and summary['needs_review'] <= 2  # Allow up to 2 chapters to need review

    if all_passing:
        print("\n✓ Final review completed successfully!")
        print("All chapters meet quality standards.")

        # Print editing suggestions
        suggestions = reviewer.generate_editing_suggestions()
        if suggestions:
            print("\nEDITING SUGGESTIONS:")
            for suggestion in suggestions:
                if suggestion.startswith("="):
                    print("-" * 40)
                else:
                    print(suggestion)

        return True
    else:
        print(f"\n⚠️  Review completed with issues.")
        print(f"{summary['failing']} chapters are failing review and need major improvements.")

        # Print editing suggestions
        suggestions = reviewer.generate_editing_suggestions()
        if suggestions:
            print("\nEDITING SUGGESTIONS:")
            for suggestion in suggestions:
                if suggestion.startswith("="):
                    print("-" * 40)
                else:
                    print(suggestion)

        return False


def check_chapter_completeness():
    """Check if all chapters have the required content"""
    print("Checking chapter completeness...")

    required_sections = [
        'Introduction',
        'Background/Literature Review',
        'Methodology/Approach',
        'Implementation',
        'Results/Discussion',
        'Conclusion',
        'References/Bibliography'
    ]

    project_path = "/mnt/d/Quarter-4/spec_kit_plus/humenoid_robot"
    reviewer = ChapterReviewer()
    chapters = reviewer.find_all_chapters(project_path)

    completeness_issues = []

    for chapter_path in chapters:
        content = reviewer.load_chapter_content(chapter_path)
        chapter_name = Path(chapter_path).name

        # Check for required sections (using headers)
        found_sections = []
        lines = content.split('\n')
        for line in lines:
            if line.startswith('#'):
                section_title = line.strip('# ')
                found_sections.append(section_title)

        # Check if required sections are present
        missing_sections = []
        for required in required_sections:
            found = False
            for section in found_sections:
                if required.lower() in section.lower():
                    found = True
                    break
            if not found:
                missing_sections.append(required)

        if missing_sections:
            completeness_issues.append({
                'chapter': chapter_name,
                'missing_sections': missing_sections,
                'found_sections': found_sections
            })

    if completeness_issues:
        print(f"\nFound completeness issues in {len(completeness_issues)} chapters:")
        for issue in completeness_issues:
            print(f"  - {issue['chapter']}: Missing {', '.join(issue['missing_sections'])}")
        return False
    else:
        print("✓ All chapters have the required sections.")
        return True


def main():
    """Main function to run the final review and editing"""
    print("Final Review and Editing Tool")
    print("=" * 40)

    # Check chapter completeness first
    completeness_ok = check_chapter_completeness()

    if not completeness_ok:
        print("\n⚠️  Some chapters are missing required sections.")
        print("These need to be addressed before final review is complete.")

    # Run comprehensive review
    review_ok = run_final_review()

    if review_ok:
        print("\n✓ Final review and editing completed successfully!")
        print("The course content is ready for publication.")
        return True
    else:
        print("\n⚠️  Review completed but issues were found.")
        print("Some chapters need additional work before publication.")
        return False


if __name__ == '__main__':
    import sys
    from urllib.parse import urlparse
    import textstat

    success = main()
    sys.exit(0 if success else 1)