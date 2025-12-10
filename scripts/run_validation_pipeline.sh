#!/bin/bash

# Automated validation pipeline for Physical AI & Humanoid Robotics course
# This script runs all validation checks on the course content

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"
REPORTS_DIR="$REPO_ROOT/reports"
LOG_FILE="$REPORTS_DIR/validation_pipeline.log"

# Create reports directory
mkdir -p "$REPORTS_DIR"

# Function to log messages
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Function to print colored output
print_status() {
    case $1 in
        "success") echo -e "${GREEN}✓ $2${NC}" ;;
        "error") echo -e "${RED}✗ $1$2${NC}" ;;
        "warning") echo -e "${YELLOW}⚠ $2${NC}" ;;
        "info") echo -e "${BLUE}ℹ $2${NC}" ;;
        *) echo "$2" ;;
    esac
}

# Start validation pipeline
print_status "info" "Starting Physical AI & Humanoid Robotics validation pipeline"
log "Starting validation pipeline"

# 1. Validate code examples
print_status "info" "Validating code examples..."
log "Starting code example validation"

if python3 "$REPO_ROOT/scripts/validate_code_examples.py" --docs-path "$DOCS_DIR" --output "$REPORTS_DIR/code_validation_report.md"; then
    print_status "success" "Code examples validation completed successfully"
    log "Code examples validation: PASSED"
else
    print_status "error" "Code examples validation failed"
    log "Code examples validation: FAILED"
    exit 1
fi

# 2. Check for broken links
print_status "info" "Checking for broken links..."
log "Starting broken link validation"

# Create a temporary file to store broken links
BROKEN_LINKS_FILE="$REPORTS_DIR/broken_links.txt"
> "$BROKEN_LINKS_FILE"  # Clear the file

# Find all markdown files and check for broken links
find "$DOCS_DIR" -name "*.md" -exec grep -H -o -E '\[([^\[\]]+)\]\(([^)]+)\)' {} \; | while read -r line; do
    file=$(echo "$line" | cut -d':' -f1)
    link=$(echo "$line" | grep -o -E '\(([^)]+)\)' | sed 's/[()]//g')

    # Skip external links and anchor links
    if [[ $link =~ ^https?:// ]] || [[ $link =~ ^# ]] || [[ $link =~ ^mailto: ]]; then
        continue
    fi

    # Resolve relative path from the markdown file's directory
    dir=$(dirname "$file")
    full_path="$dir/$link"

    # If it's a fragment (file#section), get just the file
    full_path=${full_path%%#*}

    # Handle relative paths like ../path/to/file
    full_path=$(realpath -m "$full_path" 2>/dev/null || echo "$full_path")

    # Check if file exists
    if [[ ! -f "$full_path" ]]; then
        echo "BROKEN: $file -> $link" >> "$BROKEN_LINKS_FILE"
    fi
done

if [[ -s "$BROKEN_LINKS_FILE" ]]; then
    print_status "error" "Broken links found:"
    cat "$BROKEN_LINKS_FILE"
    log "Broken links validation: FAILED"
    exit 1
else
    print_status "success" "No broken links found"
    log "Broken links validation: PASSED"
fi

# 3. Validate image references
print_status "info" "Validating image references..."
log "Starting image reference validation"

IMAGES_FILE="$REPORTS_DIR/image_validation.txt"
> "$IMAGES_FILE"

find "$DOCS_DIR" -name "*.md" -exec grep -H -o -E '!\[([^\[\]]*)\]\(([^)]+)\)' {} \; | while read -r line; do
    file=$(echo "$line" | cut -d':' -f1)
    image=$(echo "$line" | grep -o -E '\(([^)]+)\)' | sed 's/[()]//g')

    # Skip external images
    if [[ $image =~ ^https?:// ]]; then
        continue
    fi

    # Resolve relative path from the markdown file's directory
    dir=$(dirname "$file")
    full_path="$dir/$image"

    # Handle relative paths
    full_path=$(realpath -m "$full_path" 2>/dev/null || echo "$full_path")

    # Check if image exists
    if [[ ! -f "$full_path" ]]; then
        echo "MISSING IMAGE: $file -> $image" >> "$IMAGES_FILE"
    fi
done

if [[ -s "$IMAGES_FILE" ]]; then
    print_status "error" "Missing images found:"
    cat "$IMAGES_FILE"
    log "Image validation: FAILED"
    exit 1
else
    print_status "success" "All image references are valid"
    log "Image validation: PASSED"
fi

# 4. Validate frontmatter in markdown files
print_status "info" "Validating markdown frontmatter..."
log "Starting frontmatter validation"

FRONTMATTER_FILE="$REPORTS_DIR/frontmatter_validation.txt"
> "$FRONTMATTER_FILE"

find "$DOCS_DIR" -name "*.md" -exec bash -c '
for file; do
    # Check if file has frontmatter
    if grep -q "^---" "$file"; then
        # Extract frontmatter
        frontmatter=$(sed -n "/^---$/,/^---$/{/^---$/d; p}" "$file")

        # Validate YAML syntax
        if ! echo "$frontmatter" | python3 -c "import sys, yaml; yaml.safe_load(sys.stdin)" 2>/dev/null; then
            echo "INVALID YAML: $file" >> "'"$FRONTMATTER_FILE"'"
        fi
    fi
done
' _ {} +

if [[ -s "$FRONTMATTER_FILE" ]]; then
    print_status "error" "Invalid frontmatter found:"
    cat "$FRONTMATTER_FILE"
    log "Frontmatter validation: FAILED"
    exit 1
else
    print_status "success" "All frontmatter is valid"
    log "Frontmatter validation: PASSED"
fi

# 5. Check for common content issues
print_status "info" "Checking for common content issues..."
log "Starting content issue validation"

ISSUES_FILE="$REPORTS_DIR/content_issues.txt"
> "$ISSUES_FILE"

# Check for common issues
find "$DOCS_DIR" -name "*.md" -exec sh -c '
for file; do
    # Check for TODOs
    if grep -q "TODO\|FIXME" "$file"; then
        echo "TODO/FIXME in $file" >> "'"$ISSUES_FILE"'"
    fi

    # Check for placeholder content
    if grep -q "TODO\|FIXME\|placeholder\|PLACEHOLDER" "$file"; then
        echo "Placeholder content in $file" >> "'"$ISSUES_FILE"'"
    fi
done
' _ {} +

if [[ -s "$ISSUES_FILE" ]]; then
    print_status "warning" "Potential issues found:"
    cat "$ISSUES_FILE"
    log "Content issues validation: Some issues found (non-critical)"
else
    print_status "success" "No content issues found"
    log "Content issues validation: PASSED"
fi

# 6. Run Docusaurus build validation
print_status "info" "Running Docusaurus build validation..."
log "Starting Docusaurus build validation"

if cd "$REPO_ROOT" && npm run build 2>"$REPORTS_DIR/build_errors.log"; then
    print_status "success" "Docusaurus build completed successfully"
    log "Docusaurus build validation: PASSED"
else
    print_status "error" "Docusaurus build failed"
    log "Docusaurus build validation: FAILED"
    echo "Build errors:"
    cat "$REPORTS_DIR/build_errors.log"
    exit 1
fi

# Generate summary report
print_status "info" "Generating validation summary..."
log "Generating validation summary"

SUMMARY_FILE="$REPORTS_DIR/validation_summary.md"
cat > "$SUMMARY_FILE" << EOF
# Validation Summary Report

**Date**: $(date)

## Validation Results

- ✅ Code examples: PASSED
- ✅ Broken links: PASSED
- ✅ Image references: PASSED
- ✅ Frontmatter: PASSED
- ✅ Docusaurus build: PASSED

## Additional Checks
- ⚠ Content issues: See $ISSUES_FILE

## Validation Completed Successfully

All automated validation checks passed. The course content is ready for review.
EOF

print_status "success" "Validation pipeline completed successfully"
log "Validation pipeline completed successfully"

echo ""
print_status "info" "Reports generated in: $REPORTS_DIR"
print_status "info" "Summary report: $SUMMARY_FILE"