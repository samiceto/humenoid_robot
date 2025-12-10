#!/bin/bash

# Isaac Sim Hardware Validation Script
# Validates system compatibility for Isaac Sim and the Physical AI & Humanoid Robotics course

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORTS_DIR="$REPO_ROOT/reports/validation"
LOG_FILE="$REPORTS_DIR/isaac_sim_validation.log"

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
        *) echo "$2" ;;
    esac
}

print_status "info" "Starting Isaac Sim hardware compatibility validation"
log "Starting Isaac Sim hardware validation"

# 1. Check if required tools are available
print_status "info" "Checking required tools..."

REQUIRED_TOOLS=("python3" "nvidia-smi" "nvcc")
MISSING_TOOLS=()

for tool in "${REQUIRED_TOOLS[@]}"; do
    if ! command -v "$tool" &> /dev/null; then
        MISSING_TOOLS+=("$tool")
    fi
done

if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
    print_status "error" "Missing required tools: ${MISSING_TOOLS[*]}"
    for tool in "${MISSING_TOOLS[@]}"; do
        case $tool in
            "nvidia-smi")
                print_status "warning" "Install NVIDIA drivers for GPU detection"
                ;;
            "nvcc")
                print_status "warning" "Install CUDA toolkit for CUDA support"
                ;;
            "python3")
                print_status "warning" "Install Python 3 for validation scripts"
                ;;
        esac
    done
    log "Missing tools: ${MISSING_TOOLS[*]}"
else
    print_status "success" "All required tools are available"
    log "Required tools check: PASSED"
fi

# 2. Run Python-based hardware validation
print_status "info" "Running detailed hardware compatibility validation..."
log "Starting Python hardware validation"

if python3 "$REPO_ROOT/scripts/validate_isaac_sim_compatibility.py" --output "$REPORTS_DIR/isaac_sim_compatibility_report.md"; then
    print_status "success" "Hardware compatibility validation completed"
    log "Hardware validation: COMPLETED"
else
    print_status "error" "Hardware compatibility validation failed"
    log "Hardware validation: FAILED"
    exit 1
fi

# 3. Check Isaac Sim installation if present
print_status "info" "Checking Isaac Sim installation (if present)..."
log "Checking Isaac Sim installation"

ISAAC_SIM_PATHS=(
    "/opt/isaac-sim"
    "$HOME/.local/share/isaac-sim"
    "/usr/local/isaac-sim"
)

ISAAC_SIM_FOUND=false
for path in "${ISAAC_SIM_PATHS[@]}"; do
    if [ -d "$path" ]; then
        ISAAC_SIM_FOUND=true
        print_status "info" "Isaac Sim found at: $path"
        log "Isaac Sim found at: $path"

        # Check if Isaac Sim can be launched (basic check)
        if [ -f "$path/python.sh" ]; then
            print_status "info" "Isaac Sim Python interface available"
            log "Isaac Sim Python interface: AVAILABLE"
        else
            print_status "warning" "Isaac Sim Python interface not found at $path"
            log "Isaac Sim Python interface: NOT FOUND"
        fi
        break
    fi
done

if [ "$ISAAC_SIM_FOUND" = false ]; then
    print_status "info" "Isaac Sim not found (this is OK for validation)"
    log "Isaac Sim installation: NOT FOUND (expected for compatibility check)"
fi

# 4. Check system OpenGL capabilities
print_status "info" "Checking OpenGL capabilities..."
log "Checking OpenGL capabilities"

if command -v glxinfo &> /dev/null; then
    OPENGL_VERSION=$(glxinfo | grep "OpenGL version" | head -n1 | cut -d' ' -f3-)
    if [ -n "$OPENGL_VERSION" ]; then
        print_status "info" "OpenGL version: $OPENGL_VERSION"
        log "OpenGL version: $OPENGL_VERSION"

        # Check minimum OpenGL version (Isaac Sim requires 4.5+)
        if [[ $(echo "$OPENGL_VERSION >= 4.5" | bc -l 2>/dev/null || echo 0) -eq 1 ]]; then
            print_status "success" "OpenGL version meets Isaac Sim requirements"
            log "OpenGL version check: PASSED"
        else
            print_status "warning" "OpenGL version may be below Isaac Sim requirements"
            log "OpenGL version check: WARNING"
        fi
    else
        print_status "warning" "Could not determine OpenGL version"
        log "OpenGL version: UNKNOWN"
    fi
else
    print_status "warning" "glxinfo not available, skipping OpenGL check"
    log "glxinfo not available: OpenGL check skipped"
fi

# 5. Check system limits
print_status "info" "Checking system limits..."
log "Checking system limits"

# Check shared memory size
SHM_SIZE=$(df -h /dev/shm 2>/dev/null | tail -n1 | awk '{print $2}' | sed 's/G//')
if [ -n "$SHM_SIZE" ] && [ "$SHM_SIZE" -ge 2 ]; then
    print_status "success" "Shared memory size ($SHM_SIZE GB) is adequate"
    log "Shared memory check: PASSED ($SHM_SIZE GB)"
else
    print_status "warning" "Shared memory may be insufficient for Isaac Sim"
    log "Shared memory check: WARNING"
    if [ -n "$SHM_SIZE" ]; then
        print_status "info" "Current shared memory: ${SHM_SIZE}GB, Isaac Sim may need 2GB+"
    fi
fi

# 6. Generate summary
print_status "info" "Generating validation summary..."
log "Generating validation summary"

SUMMARY_FILE="$REPORTS_DIR/isaac_sim_validation_summary.md"
cat > "$SUMMARY_FILE" << EOF
# Isaac Sim Validation Summary

**Date**: $(date)

## System Compatibility Status

- ✅ System requirements check: COMPLETED
- ✅ Hardware compatibility: See detailed report
- $(if [ "$ISAAC_SIM_FOUND" = true ]; then echo "✅ Isaac Sim installation: FOUND"; else echo "ℹ Isaac Sim installation: NOT FOUND"; fi)
- $(if [ -n "$OPENGL_VERSION" ] && [[ $(echo "$OPENGL_VERSION >= 4.5" | bc -l 2>/dev/null || echo 0) -eq 1 ]]; then echo "✅ OpenGL compatibility: PASSED"; else echo "⚠ OpenGL compatibility: REVIEW NEEDED"; fi)
- $(if [ -n "$SHM_SIZE" ] && [ "$SHM_SIZE" -ge 2 ]; then echo "✅ System limits: ADEQUATE"; else echo "⚠ System limits: REVIEW NEEDED"; fi)

## Recommendations

Based on the validation results:

1. Ensure NVIDIA GPU with minimum 8GB VRAM is available
2. Verify CUDA toolkit is properly installed
3. Check that OpenGL 4.5+ is supported
4. Ensure adequate shared memory is configured
5. Install Isaac Sim from NVIDIA Developer website

## Next Steps

1. Review the detailed compatibility report: $(basename "$REPORTS_DIR/isaac_sim_compatibility_report.md")
2. Address any compatibility issues identified
3. Proceed with Isaac Sim installation if not already installed
4. Run Isaac Sim tutorials to validate functionality

EOF

print_status "success" "Isaac Sim hardware validation completed successfully"
log "Isaac Sim validation pipeline completed successfully"

echo ""
print_status "info" "Reports generated in: $REPORTS_DIR"
print_status "info" "Summary report: $SUMMARY_FILE"
print_status "info" "Detailed report: $REPORTS_DIR/isaac_sim_compatibility_report.md"