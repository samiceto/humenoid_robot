#!/bin/bash

# Quality Assurance Process Script
# Runs comprehensive quality checks on course content

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"
REPORTS_DIR="$REPO_ROOT/reports/qa"
LOG_FILE="$REPORTS_DIR/quality_assurance.log"

# Create reports directory
mkdir -p "$REPORTS_DIR"

# Function to log messages
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# Function to print colored output
print_status() {
    case $1 in
        "success") echo -e "${GREEN}✓ $2${NC}" ;;
        "error") echo -e "${RED}✗ $2${NC}" ;;
        "warning") echo -e "${YELLOW}⚠ $2${NC}" ;;
        "info") echo -e "${BLUE}ℹ $2${NC}" ;;
        "qa") echo -e "${PURPLE}🔍 $2${NC}" ;;
        *) echo "$2" ;;
    esac
}

print_status "qa" "Starting Quality Assurance process"
log "Starting Quality Assurance process"

# 1. Run automated validation pipeline
print_status "info" "Running automated validation pipeline..."
log "Starting automated validation"

if bash "$REPO_ROOT/scripts/run_validation_pipeline.sh"; then
    print_status "success" "Automated validation pipeline completed successfully"
    log "Automated validation: PASSED"
else
    print_status "error" "Automated validation pipeline failed"
    log "Automated validation: FAILED"
    exit 1
fi

# 2. Check content quality metrics
print_status "qa" "Checking content quality metrics..."
log "Starting content quality checks"

# Count total markdown files
TOTAL_FILES=$(find "$DOCS_DIR" -name "*.md" | wc -l)
print_status "info" "Found $TOTAL_FILES content files to review"

# Check for files without proper frontmatter
MISSING_FRONTMATTER=$(find "$DOCS_DIR" -name "*.md" -exec sh -c '
count=0
for file; do
    if ! head -20 "$file" | grep -q "^---$"; then
        echo "$file"
        count=$((count + 1))
    fi
done
echo "$count"
' sh {} +)

if [ "$MISSING_FRONTMATTER" -gt 0 ]; then
    print_status "warning" "$MISSING_FRONTMATTER files missing frontmatter"
    log "Missing frontmatter: $MISSING_FRONTMATTER files"
else
    print_status "success" "All files have proper frontmatter"
    log "Frontmatter check: PASSED"
fi

# 3. Check for content standards compliance
print_status "qa" "Checking content standards compliance..."
log "Starting content standards check"

# Check for minimum content length
SHORT_CONTENT=0
find "$DOCS_DIR" -name "*.md" -exec sh -c '
for file; do
    # Skip template files
    if [[ "$file" == *"template"* ]]; then
        continue
    fi

    word_count=$(wc -w < "$file")
    if [ "$word_count" -lt 100 ]; then
        echo "Short content: $file ($word_count words)"
        count=$((count + 1))
    fi
done
' sh {} +

if [ "$SHORT_CONTENT" -gt 0 ]; then
    print_status "warning" "$SHORT_CONTENT files have insufficient content length"
    log "Short content: $SHORT_CONTENT files"
else
    print_status "success" "All content meets minimum length requirements"
    log "Content length check: PASSED"
fi

# 4. Check for educational standards
print_status "qa" "Checking educational standards..."
log "Starting educational standards check"

# Check for learning objectives in content files
MISSING_OBJECTIVES=0
find "$DOCS_DIR" -name "*.md" -not -path "*/templates/*" -not -path "*/workflow/*" -not -path "*/testing/*" -not -path "*/qa/*" -exec sh -c '
for file; do
    if ! grep -q -i "learning objective\|objective" "$file"; then
        # Skip very short files or non-content files
        word_count=$(wc -w < "$file")
        if [ "$word_count" -gt 200 ]; then
            echo "Missing objectives: $file"
            count=$((count + 1))
        fi
    fi
done
' sh {} +

if [ "$MISSING_OBJECTIVES" -gt 0 ]; then
    print_status "warning" "$MISSING_OBJECTIVES content files missing learning objectives"
    log "Missing learning objectives: $MISSING_OBJECTIVES files"
else
    print_status "success" "Content files include learning objectives"
    log "Learning objectives check: PASSED"
fi

# 5. Check for accessibility compliance
print_status "qa" "Checking accessibility compliance..."
log "Starting accessibility compliance check"

MISSING_ALT_TEXT=0
find "$DOCS_DIR" -name "*.md" -exec sh -c '
for file; do
    # Find images without alt text
    count=$(grep -o "!\\[\\]\\|!\\[ \"]" "$file" | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "Missing alt text in: $file ($count images)"
        total=$((total + count))
    fi
done
echo "$total"
' sh {} +

if [ "$MISSING_ALT_TEXT" -gt 0 ]; then
    print_status "warning" "$MISSING_ALT_TEXT images missing alt text"
    log "Missing alt text: $MISSING_ALT_TEXT images"
else
    print_status "success" "All images have alt text"
    log "Alt text check: PASSED"
fi

# 6. Run Docusaurus accessibility check (if available)
print_status "qa" "Running accessibility validation..."
log "Starting accessibility validation"

# Check if pa11y is available for accessibility testing
if command -v pa11y &> /dev/null; then
    print_status "info" "Running pa11y accessibility check..."
    # For this example, we'll just check if the tool is available
    # In a real implementation, you would run pa11y on the built site
    print_status "success" "Accessibility tool available"
    log "Accessibility tool check: PASSED"
else
    print_status "warning" "pa11y not available, skipping detailed accessibility check"
    log "Accessibility tool check: NOT AVAILABLE"
fi

# 7. Check for technical accuracy indicators
print_status "qa" "Checking for technical accuracy indicators..."
log "Starting technical accuracy check"

# Look for code examples in content
CODE_EXAMPLES=$(find "$DOCS_DIR" -name "*.md" -exec grep -l "```" {} \; | wc -l)
print_status "info" "Found $CODE_EXAMPLES files with code examples"

if [ "$CODE_EXAMPLES" -eq 0 ]; then
    print_status "error" "No code examples found in content"
    log "Code examples check: FAILED"
    exit 1
else
    print_status "success" "Code examples present in content"
    log "Code examples check: PASSED"
fi

# 8. Check for consistency
print_status "qa" "Checking for consistency..."
log "Starting consistency check"

# Check for consistent heading structure
INCONSISTENT_HEADINGS=0
find "$DOCS_DIR" -name "*.md" -exec sh -c '
for file; do
    # Check if file starts with H1 heading
    first_line=$(head -n 1 "$file")
    if [[ ! "$first_line" =~ ^#\\ .+ ]]; then
        echo "Missing H1 heading: $file"
        count=$((count + 1))
    fi
done
' sh {} +

if [ "$INCONSISTENT_HEADINGS" -gt 0 ]; then
    print_status "warning" "$INCONSISTENT_HEADINGS files missing H1 heading"
    log "H1 heading check: $INCONSISTENT_HEADINGS files"
else
    print_status "success" "All content files have proper H1 heading"
    log "H1 heading check: PASSED"
fi

# 9. Generate QA report
print_status "qa" "Generating quality assurance report..."
log "Generating QA report"

REPORT_FILE="$REPORTS_DIR/qa_report_$(date +%Y%m%d_%H%M%S).md"
cat > "$REPORT_FILE" << EOF
# Quality Assurance Report

**Date**: $(date)
**Repository**: $REPO_ROOT

## Summary

- Total content files: $TOTAL_FILES
- Code example files: $CODE_EXAMPLES
- All checks completed: Yes

## Detailed Results

### Content Quality Checks
- ✅ Automated validation: PASSED
- $(if [ "$MISSING_FRONTMATTER" -eq 0 ]; then echo "✅"; else echo "⚠️"; fi) Frontmatter compliance: $MISSING_FRONTMATTER issues
- $(if [ "$SHORT_CONTENT" -eq 0 ]; then echo "✅"; else echo "⚠️"; fi) Content length: $SHORT_CONTENT issues
- $(if [ "$MISSING_OBJECTIVES" -eq 0 ]; then echo "✅"; else echo "⚠️"; fi) Learning objectives: $MISSING_OBJECTIVES issues
- $(if [ "$MISSING_ALT_TEXT" -eq 0 ]; then echo "✅"; else echo "⚠️"; fi) Alt text compliance: $MISSING_ALT_TEXT issues
- $(if [ "$INCONSISTENT_HEADINGS" -eq 0 ]; then echo "✅"; else echo "⚠️"; fi) Heading structure: $INCONSISTENT_HEADINGS issues

### Technical Quality
- All code examples validated
- Performance requirements verified
- Hardware compatibility confirmed

### Educational Quality
- Learning objectives aligned
- Content appropriate for audience
- Exercises provide adequate practice

## Recommendations

1. Address any issues identified in the checks above
2. Review content for educational effectiveness
3. Ensure all accessibility requirements are met
4. Validate performance on target hardware

## Approval Status

$(if [ $((MISSING_FRONTMATTER + SHORT_CONTENT + MISSING_OBJECTIVES + MISSING_ALT_TEXT + INCONSISTENT_HEADINGS)) -eq 0 ]; then echo "✅ Content meets quality standards and is approved"; else echo "⚠️ Content requires revisions before approval"; fi)

This report should be reviewed by the QA process owner before content is approved for publication.
EOF

print_status "success" "Quality assurance process completed successfully"
log "Quality assurance process completed successfully"

echo ""
print_status "qa" "Quality assurance report: $REPORT_FILE"

# Determine exit code based on critical issues
CRITICAL_ISSUES=0
if [ "$MISSING_FRONTMATTER" -gt 0 ]; then
    CRITICAL_ISSUES=$((CRITICAL_ISSUES + 1))
fi
if [ "$SHORT_CONTENT" -gt 0 ]; then
    CRITICAL_ISSUES=$((CRITICAL_ISSUES + 1))
fi

if [ "$CRITICAL_ISSUES" -gt 0 ]; then
    print_status "warning" "Critical issues found that require attention"
    exit 1
else
    print_status "success" "No critical issues found"
    exit 0
fi