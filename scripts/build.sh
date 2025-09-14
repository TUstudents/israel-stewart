#!/bin/bash
# Build script for relativistic-turbulence-rg
# Comprehensive build and packaging with validation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
CLEAN=false
VERBOSE=false
QUIET=false
CHECK_ONLY=false
SKIP_TESTS=false
UPLOAD_PYPI=false
BUILD_DIR="dist"

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build and package the relativistic-turbulence-rg project.

OPTIONS:
    -h, --help          Show this help message
    -v, --verbose       Enable verbose output
    -q, --quiet         Suppress non-essential output
    -c, --clean         Clean build directory before building
    --check             Check build requirements without building
    --skip-tests        Skip running tests before building
    --upload            Upload to PyPI after successful build (requires auth)
    --build-dir DIR     Build output directory (default: $BUILD_DIR)

BUILD PROCESS:
    1. Check Python environment and dependencies
    2. Run tests (unless --skip-tests)
    3. Clean previous builds (if --clean)
    4. Build wheel and source distribution
    5. Validate built packages
    6. Optional: Upload to PyPI (if --upload)

EXAMPLES:
    $0                  # Standard build
    $0 --clean          # Clean build from scratch
    $0 --check          # Check build requirements only
    $0 --skip-tests     # Build without running tests
    $0 --upload         # Build and upload to PyPI

REQUIREMENTS:
    - uv (for package management)
    - build (for building packages)
    - twine (for PyPI upload, if --upload used)
EOF
}

# Logging functions
log_info() {
    if [[ "$QUIET" != "true" ]]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    fi
}

log_success() {
    if [[ "$QUIET" != "true" ]]; then
        echo -e "${GREEN}[SUCCESS]${NC} $1"
    fi
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${PURPLE}[VERBOSE]${NC} $1"
    fi
}

# Check if we're in the right directory
check_project_root() {
    if [[ ! -f "pyproject.toml" ]] || [[ ! -d "rtrg" ]]; then
        log_error "Must be run from project root (where pyproject.toml exists)"
        exit 1
    fi
}

# Check if required tools are available
check_dependencies() {
    local missing_tools=()

    if ! command -v uv >/dev/null 2>&1; then
        missing_tools+=("uv")
    fi

    if ! uv run python -c "import build" >/dev/null 2>&1; then
        log_warn "build package not available, trying to install..."
        if ! uv add --dev build; then
            missing_tools+=("build")
        fi
    fi

    if [[ "$UPLOAD_PYPI" == "true" ]]; then
        if ! uv run python -c "import twine" >/dev/null 2>&1; then
            log_warn "twine package not available for PyPI upload, trying to install..."
            if ! uv add --dev twine; then
                missing_tools+=("twine (for PyPI upload)")
            fi
        fi
    fi

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install missing tools and try again"
        exit 1
    fi
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -q|--quiet)
                QUIET=true
                shift
                ;;
            -c|--clean)
                CLEAN=true
                shift
                ;;
            --check)
                CHECK_ONLY=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --upload)
                UPLOAD_PYPI=true
                shift
                ;;
            --build-dir)
                if [[ -n "${2:-}" ]]; then
                    BUILD_DIR="$2"
                    shift 2
                else
                    log_error "Directory required for --build-dir"
                    exit 1
                fi
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Check Python environment
check_python_environment() {
    log_verbose "Checking Python environment..."

    # Check Python version
    local python_version
    python_version=$(uv run python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
    log_verbose "Python version: $python_version"

    if [[ "$python_version" == "unknown" ]]; then
        log_error "Unable to determine Python version"
        return 1
    fi

    # Check if minimum version requirement is met (3.12+)
    local major minor
    major=$(echo "$python_version" | cut -d. -f1)
    minor=$(echo "$python_version" | cut -d. -f2)

    if [[ $major -lt 3 ]] || [[ $major -eq 3 && $minor -lt 12 ]]; then
        log_error "Python 3.12+ required, found $python_version"
        return 1
    fi

    log_verbose "Python version check passed"

    # Check uv sync status
    if ! uv sync --dry-run >/dev/null 2>&1; then
        log_warn "Dependencies may be out of sync"
        log_info "Running uv sync to ensure dependencies are up to date..."
        uv sync
    fi

    return 0
}

# Run tests before building
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warn "Skipping tests as requested"
        return 0
    fi

    log_info "Running tests before building..."

    # Check if our test script exists
    if [[ -x "scripts/test.sh" ]]; then
        log_verbose "Using scripts/test.sh for testing"
        if [[ "$QUIET" == "true" ]]; then
            ./scripts/test.sh --quiet --fast
        elif [[ "$VERBOSE" == "true" ]]; then
            ./scripts/test.sh --verbose --fast
        else
            ./scripts/test.sh --fast
        fi
    else
        # Fallback to direct pytest
        log_verbose "Using direct pytest for testing"
        local pytest_args=("--tb=short" "-m" "not slow")
        [[ "$VERBOSE" == "true" ]] && pytest_args+=("-v")
        [[ "$QUIET" == "true" ]] && pytest_args+=("-q")

        uv run pytest "${pytest_args[@]}" tests/
    fi

    log_success "Tests passed"
}

# Clean build directory
clean_build() {
    if [[ "$CLEAN" == "true" ]]; then
        log_info "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
        rm -rf build/
        rm -rf *.egg-info/
        log_verbose "Removed $BUILD_DIR, build/, and *.egg-info/"
    fi

    # Ensure build directory exists
    mkdir -p "$BUILD_DIR"
}

# Build packages
build_packages() {
    log_info "Building packages..."

    # Use uv build if available, otherwise fall back to python -m build
    local build_cmd
    if uv --help | grep -q "build" 2>/dev/null; then
        build_cmd=("uv" "build")
        [[ "$VERBOSE" == "true" ]] && build_cmd+=("--verbose")
    else
        # Fallback to python -m build
        build_cmd=("uv" "run" "python" "-m" "build")
        [[ "$VERBOSE" == "true" ]] && build_cmd+=("--verbose")
    fi

    # Add output directory
    build_cmd+=("--outdir" "$BUILD_DIR")

    log_verbose "Build command: ${build_cmd[*]}"

    # Run the build
    if "${build_cmd[@]}"; then
        log_success "Build completed successfully"
    else
        log_error "Build failed"
        return 1
    fi

    # List built packages
    if [[ "$VERBOSE" == "true" ]] || [[ "$QUIET" != "true" ]]; then
        echo
        log_info "Built packages:"
        ls -la "$BUILD_DIR"/*.whl "$BUILD_DIR"/*.tar.gz 2>/dev/null | while read -r line; do
            echo "  $line"
        done
    fi
}

# Validate built packages
validate_packages() {
    log_info "Validating built packages..."

    local wheel_files
    wheel_files=($(find "$BUILD_DIR" -name "*.whl" 2>/dev/null || true))

    local sdist_files
    sdist_files=($(find "$BUILD_DIR" -name "*.tar.gz" 2>/dev/null || true))

    if [[ ${#wheel_files[@]} -eq 0 ]] && [[ ${#sdist_files[@]} -eq 0 ]]; then
        log_error "No built packages found in $BUILD_DIR"
        return 1
    fi

    # Basic validation
    for file in "${wheel_files[@]}" "${sdist_files[@]}"; do
        if [[ -f "$file" ]]; then
            local size
            size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo "unknown")
            log_verbose "Package: $(basename "$file") ($size bytes)"

            # Check if file is not empty
            if [[ "$size" == "0" ]] || [[ "$size" == "unknown" ]]; then
                log_error "Package file appears to be empty or invalid: $file"
                return 1
            fi
        fi
    done

    # Test installation in temporary environment if possible
    if [[ ${#wheel_files[@]} -gt 0 ]] && command -v python3 >/dev/null 2>&1; then
        log_verbose "Testing package installation..."
        local test_env=$(mktemp -d)

        if python3 -m venv "$test_env" >/dev/null 2>&1; then
            # shellcheck disable=SC1090
            source "$test_env/bin/activate"

            # Install the wheel
            if pip install "${wheel_files[0]}" >/dev/null 2>&1; then
                # Try importing the package
                if python -c "import rtrg" >/dev/null 2>&1; then
                    log_verbose "Package installation test passed"
                else
                    log_warn "Package installs but cannot be imported"
                fi
            else
                log_warn "Package installation test failed"
            fi

            deactivate
            rm -rf "$test_env"
        fi
    fi

    log_success "Package validation completed"
}

# Upload to PyPI
upload_to_pypi() {
    if [[ "$UPLOAD_PYPI" != "true" ]]; then
        return 0
    fi

    log_warn "Preparing to upload to PyPI..."

    # Check for credentials
    if [[ -z "${TWINE_USERNAME:-}" ]] && [[ -z "${TWINE_PASSWORD:-}" ]] && [[ ! -f ~/.pypirc ]]; then
        log_error "PyPI credentials not found"
        log_error "Set TWINE_USERNAME/TWINE_PASSWORD or configure ~/.pypirc"
        return 1
    fi

    # Confirm upload
    echo -e "${YELLOW}WARNING: This will upload packages to PyPI!${NC}"
    echo "Packages to upload:"
    ls -la "$BUILD_DIR"/*.whl "$BUILD_DIR"/*.tar.gz 2>/dev/null || true
    echo
    read -p "Continue with upload? [y/N] " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Upload cancelled by user"
        return 0
    fi

    log_info "Uploading to PyPI..."

    if uv run twine upload "$BUILD_DIR"/*; then
        log_success "Successfully uploaded to PyPI"
    else
        log_error "Upload to PyPI failed"
        return 1
    fi
}

# Generate build report
generate_report() {
    if [[ "$QUIET" == "true" ]]; then
        return
    fi

    echo
    log_info "Build Summary:"

    # Configuration
    local config_items=()
    [[ "$CLEAN" == "true" ]] && config_items+=("clean build")
    [[ "$SKIP_TESTS" == "true" ]] && config_items+=("skipped tests")
    [[ "$UPLOAD_PYPI" == "true" ]] && config_items+=("uploaded to PyPI")

    if [[ ${#config_items[@]} -gt 0 ]]; then
        echo "  Configuration: ${config_items[*]}"
    fi

    # Built packages
    if [[ -d "$BUILD_DIR" ]]; then
        echo "  Output directory: $BUILD_DIR/"
        local wheel_count
        wheel_count=$(find "$BUILD_DIR" -name "*.whl" 2>/dev/null | wc -l)
        local sdist_count
        sdist_count=$(find "$BUILD_DIR" -name "*.tar.gz" 2>/dev/null | wc -l)

        echo "  Built packages: $wheel_count wheel(s), $sdist_count source distribution(s)"
    fi
}

# Main execution
main() {
    parse_args "$@"
    check_project_root

    # Show what we're about to do
    if [[ "$QUIET" != "true" ]]; then
        echo -e "${BLUE}Relativistic Turbulence RG - Build Script${NC}"
        echo
    fi

    # Check-only mode
    if [[ "$CHECK_ONLY" == "true" ]]; then
        log_info "Checking build requirements..."
        check_dependencies
        check_python_environment
        log_success "Build requirements check completed"
        exit 0
    fi

    # Full build process
    check_dependencies
    check_python_environment
    run_tests
    clean_build
    build_packages
    validate_packages
    upload_to_pypi

    generate_report
    log_success "Build completed successfully!"
}

# Run main function with all arguments
main "$@"
