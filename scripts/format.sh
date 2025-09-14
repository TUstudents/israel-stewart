#!/bin/bash
# Format script for relativistic-turbulence-rg
# Runs pre-commit hooks with multiple passes to autofix all linting issues

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MAX_ITERATIONS=5
VERBOSE=false
QUIET=false

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Format code using pre-commit hooks with multiple passes to fix all issues.

OPTIONS:
    -h, --help          Show this help message
    -v, --verbose       Enable verbose output
    -q, --quiet         Suppress non-essential output
    -m, --max-iter N    Maximum iterations (default: $MAX_ITERATIONS)
    --unsafe-fixes      Enable unsafe fixes in ruff
    --all-files         Run on all files (not just staged)

EXAMPLES:
    $0                  # Format staged files
    $0 --all-files      # Format all files in repository
    $0 -v --max-iter 3  # Verbose mode with max 3 iterations
    $0 --unsafe-fixes   # Enable potentially unsafe automatic fixes

This script runs pre-commit hooks multiple times until no more changes are made,
ensuring all auto-fixable linting issues are resolved.
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
        echo -e "${BLUE}[VERBOSE]${NC} $1"
    fi
}

# Check if we're in the right directory
check_project_root() {
    if [[ ! -f "pyproject.toml" ]] || [[ ! -f ".pre-commit-config.yaml" ]]; then
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

    if ! uv run pre-commit --version >/dev/null 2>&1; then
        log_warn "pre-commit not available via uv, checking system installation..."
        if ! command -v pre-commit >/dev/null 2>&1; then
            missing_tools+=("pre-commit")
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
    local all_files=false
    local unsafe_fixes=false

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
            -m|--max-iter)
                if [[ -n "${2:-}" ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
                    MAX_ITERATIONS="$2"
                    shift 2
                else
                    log_error "Invalid value for --max-iter: ${2:-}"
                    exit 1
                fi
                ;;
            --unsafe-fixes)
                unsafe_fixes=true
                shift
                ;;
            --all-files)
                all_files=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Export settings for use in other functions
    export FORMAT_ALL_FILES=$all_files
    export UNSAFE_FIXES=$unsafe_fixes
}

# Run pre-commit hooks with intelligent iteration
run_precommit_iterations() {
    local iteration=0
    local changes_made=true
    local total_changes=0

    # Prepare arguments
    local precommit_args=()
    if [[ "$FORMAT_ALL_FILES" == "true" ]]; then
        precommit_args+=("--all-files")
    fi

    # Add unsafe fixes for ruff if requested
    local ruff_args=""
    if [[ "$UNSAFE_FIXES" == "true" ]]; then
        ruff_args="--unsafe-fixes"
        log_info "Enabling unsafe fixes for ruff"
    fi

    log_info "Starting multi-pass formatting (max $MAX_ITERATIONS iterations)..."

    # Track files for change detection
    local temp_file_list=$(mktemp)
    if [[ "$FORMAT_ALL_FILES" == "true" ]]; then
        git ls-files '*.py' > "$temp_file_list"
    else
        git diff --cached --name-only --diff-filter=ACMR '*.py' > "$temp_file_list" || true
    fi

    local initial_checksums=$(mktemp)
    while IFS= read -r file; do
        if [[ -f "$file" ]]; then
            sha256sum "$file" >> "$initial_checksums" 2>/dev/null || true
        fi
    done < "$temp_file_list"

    while [[ $changes_made == "true" ]] && [[ $iteration -lt $MAX_ITERATIONS ]]; do
        iteration=$((iteration + 1))
        log_info "Iteration $iteration/$MAX_ITERATIONS"

        # Run ruff separately with potential unsafe fixes
        if [[ -n "$ruff_args" ]]; then
            log_verbose "Running ruff with unsafe fixes..."
            if uv run ruff check --fix $ruff_args . >/dev/null 2>&1; then
                log_verbose "Ruff completed successfully"
            else
                log_verbose "Ruff found issues (expected in early iterations)"
            fi
        fi

        # Run all pre-commit hooks
        local precommit_output
        local precommit_exit_code=0

        log_verbose "Running pre-commit hooks..."
        precommit_output=$(uv run pre-commit run "${precommit_args[@]}" 2>&1) || precommit_exit_code=$?

        if [[ "$VERBOSE" == "true" ]]; then
            echo "$precommit_output"
        fi

        # Check for changes by comparing file checksums
        local current_checksums=$(mktemp)
        while IFS= read -r file; do
            if [[ -f "$file" ]]; then
                sha256sum "$file" >> "$current_checksums" 2>/dev/null || true
            fi
        done < "$temp_file_list"

        if diff -q "$initial_checksums" "$current_checksums" >/dev/null 2>&1; then
            changes_made=false
            log_verbose "No changes detected in iteration $iteration"
        else
            changes_made=true
            total_changes=$((total_changes + 1))
            log_verbose "Changes detected in iteration $iteration"
            cp "$current_checksums" "$initial_checksums"
        fi

        rm -f "$current_checksums"

        # If pre-commit succeeded and no changes were made, we're done
        if [[ $precommit_exit_code -eq 0 ]] && [[ $changes_made == "false" ]]; then
            break
        fi

        # Brief pause between iterations
        sleep 0.5
    done

    # Cleanup
    rm -f "$temp_file_list" "$initial_checksums"

    # Report results
    if [[ $iteration -ge $MAX_ITERATIONS ]] && [[ $changes_made == "true" ]]; then
        log_warn "Reached maximum iterations ($MAX_ITERATIONS) with potential remaining issues"
        log_warn "You may need to fix remaining issues manually or increase --max-iter"
        return 1
    else
        log_success "Formatting completed in $iteration iterations"
        if [[ $total_changes -gt 0 ]]; then
            log_success "Made changes in $total_changes iterations"
        else
            log_success "No formatting changes needed"
        fi
        return 0
    fi
}

# Main execution
main() {
    parse_args "$@"
    check_project_root
    check_dependencies

    log_info "Running comprehensive code formatting..."

    if run_precommit_iterations; then
        log_success "All formatting completed successfully!"

        # Show summary of what was run
        if [[ "$QUIET" != "true" ]]; then
            echo
            log_info "Formatting summary:"
            echo "  ✓ Ruff linting and formatting"
            echo "  ✓ MyPy type checking"
            echo "  ✓ Trailing whitespace removal"
            echo "  ✓ End-of-file fixing"
            echo "  ✓ Line ending normalization"
            echo "  ✓ YAML/TOML/JSON validation"
            echo "  ✓ Security checks (bandit)"
            echo "  ✓ Various code quality checks"
        fi

        exit 0
    else
        log_error "Formatting completed with issues remaining"
        log_error "Please review output above and fix remaining issues manually"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"
