#!/bin/bash
# Test script for relativistic-turbulence-rg
# Comprehensive test runner with coverage, parallel execution, and reporting

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
COVERAGE=false
PARALLEL=false
BENCHMARK=false
VERBOSE=false
QUIET=false
FILTER=""
TEST_CATEGORY=""
REPORT_DIR="test-reports"
JUNIT_XML=false

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS] [TEST_PATH]

Run comprehensive test suite with various options and reporting.

OPTIONS:
    -h, --help          Show this help message
    -v, --verbose       Enable verbose test output
    -q, --quiet         Suppress non-essential output
    -c, --coverage      Run tests with coverage reporting
    -p, --parallel      Run tests in parallel (using pytest-xdist)
    -b, --benchmark     Include benchmark tests
    -f, --filter EXPR   Filter tests by expression (pytest -k)
    -j, --junit         Generate JUnit XML report
    --unit              Run only unit tests
    --integration       Run only integration tests
    --physics           Run only physics validation tests
    --slow              Include slow tests
    --fast              Exclude slow tests
    --report-dir DIR    Output directory for reports (default: $REPORT_DIR)

EXAMPLES:
    $0                      # Run all tests
    $0 --coverage           # Run with coverage report
    $0 --unit --parallel    # Run unit tests in parallel
    $0 -f "tensor"          # Run tests matching "tensor"
    $0 --benchmark --slow   # Run benchmarks including slow tests
    $0 tests/unit/          # Run specific test directory

TEST CATEGORIES:
    unit         Fast unit tests for individual components
    integration  Integration tests for component interactions
    benchmark    Performance benchmark tests
    physics      Tests validating physics correctness
    slow         Tests that take more than a few seconds
    numerical    Tests for numerical accuracy and stability
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
    if [[ ! -f "pyproject.toml" ]] || [[ ! -d "tests" ]]; then
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

    if ! uv run pytest --version >/dev/null 2>&1; then
        missing_tools+=("pytest (via uv)")
    fi

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install missing tools and try again"
        exit 1
    fi
}

# Parse command line arguments
parse_args() {
    local test_path=""

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
            -c|--coverage)
                COVERAGE=true
                shift
                ;;
            -p|--parallel)
                PARALLEL=true
                shift
                ;;
            -b|--benchmark)
                BENCHMARK=true
                shift
                ;;
            -f|--filter)
                if [[ -n "${2:-}" ]]; then
                    FILTER="$2"
                    shift 2
                else
                    log_error "Filter expression required for --filter"
                    exit 1
                fi
                ;;
            -j|--junit)
                JUNIT_XML=true
                shift
                ;;
            --unit)
                TEST_CATEGORY="unit"
                shift
                ;;
            --integration)
                TEST_CATEGORY="integration"
                shift
                ;;
            --physics)
                TEST_CATEGORY="physics"
                shift
                ;;
            --slow)
                export INCLUDE_SLOW="true"
                shift
                ;;
            --fast)
                export EXCLUDE_SLOW="true"
                shift
                ;;
            --report-dir)
                if [[ -n "${2:-}" ]]; then
                    REPORT_DIR="$2"
                    shift 2
                else
                    log_error "Directory required for --report-dir"
                    exit 1
                fi
                ;;
            --)
                shift
                test_path="$*"
                break
                ;;
            -*)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
            *)
                test_path="$1"
                shift
                ;;
        esac
    done

    export TEST_PATH="$test_path"
}

# Prepare test environment
prepare_environment() {
    # Create report directory
    mkdir -p "$REPORT_DIR"
    log_verbose "Created report directory: $REPORT_DIR"

    # Set up Python path
    export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
    log_verbose "Set PYTHONPATH to include current directory"
}

# Build pytest command with all options
build_pytest_command() {
    local pytest_args=()

    # Basic configuration
    pytest_args+=("--tb=short")
    pytest_args+=("--strict-markers")
    pytest_args+=("--strict-config")

    # Verbose/quiet modes
    if [[ "$VERBOSE" == "true" ]]; then
        pytest_args+=("-v")
    elif [[ "$QUIET" == "true" ]]; then
        pytest_args+=("-q")
    fi

    # Coverage
    if [[ "$COVERAGE" == "true" ]]; then
        pytest_args+=("--cov=rtrg")
        pytest_args+=("--cov-report=html:$REPORT_DIR/htmlcov")
        pytest_args+=("--cov-report=xml:$REPORT_DIR/coverage.xml")
        pytest_args+=("--cov-report=term-missing")
    fi

    # Parallel execution
    if [[ "$PARALLEL" == "true" ]]; then
        # Use number of CPU cores
        local num_cores
        num_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "4")
        pytest_args+=("-n" "$num_cores")
        log_verbose "Running tests in parallel with $num_cores workers"
    fi

    # JUnit XML reporting
    if [[ "$JUNIT_XML" == "true" ]]; then
        pytest_args+=("--junit-xml=$REPORT_DIR/junit.xml")
    fi

    # Test category markers
    local marker_expr=""
    if [[ -n "$TEST_CATEGORY" ]]; then
        case "$TEST_CATEGORY" in
            "unit")
                marker_expr="unit"
                ;;
            "integration")
                marker_expr="integration"
                ;;
            "physics")
                marker_expr="physics"
                ;;
        esac
    fi

    # Benchmark tests
    if [[ "$BENCHMARK" == "true" ]]; then
        if [[ -n "$marker_expr" ]]; then
            marker_expr="$marker_expr or benchmark"
        else
            marker_expr="benchmark"
        fi
    fi

    # Slow test handling
    if [[ "${EXCLUDE_SLOW:-}" == "true" ]]; then
        if [[ -n "$marker_expr" ]]; then
            marker_expr="($marker_expr) and not slow"
        else
            marker_expr="not slow"
        fi
    elif [[ "${INCLUDE_SLOW:-}" != "true" ]] && [[ "$BENCHMARK" != "true" ]]; then
        # By default, exclude slow tests unless explicitly requested
        if [[ -n "$marker_expr" ]]; then
            marker_expr="($marker_expr) and not slow"
        else
            marker_expr="not slow"
        fi
    fi

    if [[ -n "$marker_expr" ]]; then
        pytest_args+=("-m" "$marker_expr")
    fi

    # Filter expression
    if [[ -n "$FILTER" ]]; then
        pytest_args+=("-k" "$FILTER")
    fi

    # Test path
    if [[ -n "$TEST_PATH" ]]; then
        pytest_args+=("$TEST_PATH")
    else
        pytest_args+=("tests/")
    fi

    echo "${pytest_args[@]}"
}

# Run the actual tests
run_tests() {
    local pytest_cmd
    read -ra pytest_cmd <<< "$(build_pytest_command)"

    log_info "Running test suite..."
    if [[ "$VERBOSE" == "true" ]]; then
        log_verbose "Command: uv run pytest ${pytest_cmd[*]}"
    fi

    local start_time
    start_time=$(date +%s)

    # Run pytest
    local exit_code=0
    uv run pytest "${pytest_cmd[@]}" || exit_code=$?

    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Report results
    if [[ $exit_code -eq 0 ]]; then
        log_success "All tests passed in ${duration}s"
    else
        log_error "Some tests failed (exit code: $exit_code)"
    fi

    return $exit_code
}

# Generate summary report
generate_summary() {
    if [[ "$QUIET" == "true" ]]; then
        return
    fi

    echo
    log_info "Test Summary:"

    # Test configuration
    local config_items=()
    [[ "$COVERAGE" == "true" ]] && config_items+=("coverage")
    [[ "$PARALLEL" == "true" ]] && config_items+=("parallel")
    [[ "$BENCHMARK" == "true" ]] && config_items+=("benchmarks")
    [[ -n "$TEST_CATEGORY" ]] && config_items+=("$TEST_CATEGORY tests")
    [[ -n "$FILTER" ]] && config_items+=("filtered: '$FILTER'")
    [[ "${EXCLUDE_SLOW:-}" == "true" ]] && config_items+=("excluding slow")
    [[ "${INCLUDE_SLOW:-}" == "true" ]] && config_items+=("including slow")

    if [[ ${#config_items[@]} -gt 0 ]]; then
        echo "  Configuration: ${config_items[*]}"
    fi

    # Report locations
    if [[ -d "$REPORT_DIR" ]]; then
        echo "  Reports saved to: $REPORT_DIR/"
        if [[ "$COVERAGE" == "true" ]]; then
            echo "    - HTML coverage: $REPORT_DIR/htmlcov/index.html"
            echo "    - XML coverage: $REPORT_DIR/coverage.xml"
        fi
        if [[ "$JUNIT_XML" == "true" ]]; then
            echo "    - JUnit XML: $REPORT_DIR/junit.xml"
        fi
    fi
}

# Main execution
main() {
    parse_args "$@"
    check_project_root
    check_dependencies
    prepare_environment

    # Show what we're about to do
    if [[ "$QUIET" != "true" ]]; then
        echo -e "${BLUE}Relativistic Turbulence RG - Test Runner${NC}"
        echo
    fi

    if run_tests; then
        generate_summary
        log_success "Testing completed successfully!"
        exit 0
    else
        generate_summary
        log_error "Testing failed - see output above for details"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"
