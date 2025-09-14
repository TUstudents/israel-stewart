# Development Scripts

This directory contains development scripts to streamline common tasks for the relativistic-turbulence-rg project.

## Scripts Overview

### 🎨 `format.sh` - Code Formatting
Comprehensive code formatting with multi-pass pre-commit hook execution.

```bash
# Format staged files with multiple passes
./scripts/format.sh

# Format all files in repository
./scripts/format.sh --all-files

# Enable unsafe fixes (more aggressive auto-fixing)
./scripts/format.sh --unsafe-fixes

# Verbose mode to see what's being fixed
./scripts/format.sh --verbose --max-iter 3
```

**Key Features:**
- **Multi-pass execution**: Runs pre-commit hooks multiple times until no changes are made
- **Auto-fixes all issues**: Handles ruff linting, formatting, type checking, and more
- **Intelligent iteration**: Stops when no more changes are detected
- **Configurable**: Control maximum iterations, verbosity, and fix aggressiveness

### 🧪 `test.sh` - Test Runner
Comprehensive test runner with coverage, parallel execution, and reporting.

```bash
# Run all tests (excluding slow ones)
./scripts/test.sh

# Run with coverage reporting
./scripts/test.sh --coverage

# Run specific test categories
./scripts/test.sh --unit --parallel
./scripts/test.sh --integration
./scripts/test.sh --physics

# Run with filtering
./scripts/test.sh --filter "tensor"
./scripts/test.sh --benchmark --slow

# Generate reports
./scripts/test.sh --coverage --junit --report-dir reports/
```

**Key Features:**
- **Comprehensive coverage**: Unit, integration, physics, and benchmark tests
- **Parallel execution**: Speed up tests with pytest-xdist
- **Detailed reporting**: HTML coverage, JUnit XML, custom reports
- **Flexible filtering**: Run specific tests or categories
- **Performance-aware**: Exclude slow tests by default

### 🏗️ `build.sh` - Build and Package
Build and package the project with validation and optional PyPI upload.

```bash
# Standard build
./scripts/build.sh

# Clean build from scratch
./scripts/build.sh --clean

# Build without running tests (fast)
./scripts/build.sh --skip-tests

# Check requirements only
./scripts/build.sh --check

# Build and upload to PyPI (interactive)
./scripts/build.sh --upload
```

**Key Features:**
- **Complete build pipeline**: Tests → Clean → Build → Validate → Upload
- **Multi-format packages**: Both wheel (.whl) and source distribution (.tar.gz)
- **Validation**: Tests package installation and importability
- **PyPI integration**: Secure upload with credential validation
- **Environment checks**: Validates Python version and dependencies

## Quick Start

1. **Format your code** before committing:
   ```bash
   ./scripts/format.sh --all-files
   ```

2. **Run tests** to ensure everything works:
   ```bash
   ./scripts/test.sh --coverage
   ```

3. **Build packages** for distribution:
   ```bash
   ./scripts/build.sh --clean
   ```

## Integration with Development Workflow

### Pre-commit Hook Integration
The format script is designed to work seamlessly with the existing pre-commit configuration:

```bash
# Install pre-commit hooks (run once)
uv run pre-commit install

# Format code (more thorough than git hooks)
./scripts/format.sh

# Commit with confidence
git commit -m "Your changes"
```

### CI/CD Integration
All scripts are designed for CI/CD environments:

```yaml
# Example GitHub Actions workflow
- name: Format check
  run: ./scripts/format.sh --quiet --all-files

- name: Run tests
  run: ./scripts/test.sh --coverage --junit --quiet

- name: Build packages
  run: ./scripts/build.sh --skip-tests
```

### Development Cycle
Typical development workflow:

```bash
# 1. Start development
git checkout -b feature/new-feature

# 2. Make changes
# ... edit files ...

# 3. Format and test
./scripts/format.sh
./scripts/test.sh --unit

# 4. Commit changes
git add .
git commit -m "Add new feature"

# 5. Full validation before merge
./scripts/test.sh --all
./scripts/build.sh --clean

# 6. Push and create PR
git push origin feature/new-feature
```

## Script Options and Configuration

### Common Options
All scripts support these common options:
- `-h, --help`: Show detailed help message
- `-v, --verbose`: Enable verbose output  
- `-q, --quiet`: Suppress non-essential output

### Environment Variables
- `PYTHONPATH`: Automatically set to include project root
- `UV_*`: Respect uv configuration variables
- `TWINE_*`: PyPI upload credentials (build.sh only)

### Exit Codes
All scripts use standard exit codes:
- `0`: Success
- `1`: General error
- `2`: Configuration error
- `130`: Interrupted by user (Ctrl+C)

## Dependencies

### Required Tools
- `uv`: Modern Python package manager
- `python`: 3.12+ (as specified in pyproject.toml)

### Python Packages (installed via uv)
- `pytest`: Testing framework
- `pytest-cov`: Coverage reporting
- `pytest-xdist`: Parallel test execution
- `ruff`: Linting and formatting
- `mypy`: Type checking
- `pre-commit`: Git hooks
- `build`: Package building (auto-installed)
- `twine`: PyPI upload (auto-installed if needed)

## Troubleshooting

### Format Script Issues
```bash
# If formatting seems stuck
./scripts/format.sh --verbose --max-iter 2

# If pre-commit hooks fail
uv run pre-commit clean
uv run pre-commit install --install-hooks

# For urgent fixes (skip hooks)
git commit --no-verify -m "emergency fix"
```

### Test Script Issues
```bash
# If tests hang or are too slow
./scripts/test.sh --fast --unit

# If parallel tests fail
./scripts/test.sh --no-parallel

# For debugging test failures
./scripts/test.sh --verbose --filter "failing_test"
```

### Build Script Issues
```bash
# If build fails due to tests
./scripts/build.sh --skip-tests

# Check dependencies
./scripts/build.sh --check

# Clean slate rebuild
rm -rf dist/ build/ *.egg-info/
./scripts/build.sh --clean
```

## Customization

### Adding Custom Checks
To add custom validation to any script, edit the respective file:

```bash
# Example: Add custom physics validation to test.sh
# Add to run_tests() function:
if [[ "$TEST_CATEGORY" == "physics" ]]; then
    log_info "Running custom physics validation..."
    # Your custom checks here
fi
```

### Project-Specific Configuration
All scripts respect existing project configuration:
- `pyproject.toml`: Main project configuration
- `.pre-commit-config.yaml`: Pre-commit hooks
- `pytest.ini`: Test configuration  
- `.gitignore`: Files to ignore

## Contributing

When modifying these scripts:
1. Follow the existing pattern of logging functions
2. Maintain backward compatibility with existing options
3. Add appropriate help text for new options
4. Test scripts on both development and CI environments
5. Update this README with any new features

## License

These scripts are part of the relativistic-turbulence-rg project and are subject to the same CC BY-NC-SA 4.0 license.
