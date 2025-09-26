# Logging Configuration Guide

## Overview

The Israel-Stewart hydrodynamics package provides comprehensive structured logging to help with debugging, performance analysis, and scientific validation of relativistic fluid dynamics simulations.

## Quick Start

### Basic Configuration

```python
from israel_stewart.utils.logging_config import configure_logging

# Console logging at INFO level
configure_logging(level="INFO")

# With file output
from pathlib import Path
configure_logging(
    level="DEBUG",
    log_file=Path("simulation.log"),
    enable_performance=True
)
```

### Environment Variables

Set logging configuration through environment variables:

```bash
# Basic configuration
export ISRAEL_STEWART_LOG_LEVEL=DEBUG
export ISRAEL_STEWART_LOG_FORMAT=console

# Enable specific logging features
export ISRAEL_STEWART_LOG_PERFORMANCE=true
export ISRAEL_STEWART_LOG_MEMORY=true

# File logging
export ISRAEL_STEWART_LOG_FILE=/path/to/simulation.log

# Your Python script will automatically use these settings
python your_simulation.py
```

### Runtime Control

Adjust logging behavior at runtime:

```python
from israel_stewart.utils.logging_config import (
    set_log_level,
    enable_performance_logging,
    enable_memory_debugging,
    get_logging_status
)

# Change log level during simulation
set_log_level("DEBUG")  # More verbose
set_log_level("WARNING")  # Less verbose

# Toggle performance monitoring
enable_performance_logging(True)   # Enable detailed timing
enable_performance_logging(False)  # Disable for production

# Toggle memory debugging
enable_memory_debugging(True)   # Track memory usage
enable_memory_debugging(False)  # Disable memory tracking

# Check current status
status = get_logging_status()
print(f"Main level: {status['main_level']}")
print(f"Performance enabled: {status['performance_enabled']}")
```

## Log Levels and When to Use

### DEBUG
- Detailed solver step information
- Memory allocation tracking
- Boundary condition details
- Physics computation internals

```python
configure_logging(level="DEBUG", enable_performance=True)
# Produces detailed output for development
```

### INFO
- Solver convergence status
- Physics conservation checks
- Performance summaries
- Configuration changes

```python
configure_logging(level="INFO")
# Good for production monitoring
```

### WARNING
- Failed physics validations
- Numerical instabilities
- Fallback method usage
- Resource constraints

### ERROR/CRITICAL
- Solver failures
- Memory exhaustion
- Physics violations
- System critical issues

## Specialized Loggers

### Performance Logger

```python
from israel_stewart.utils.logging_config import performance_logger

# Log operation timing
performance_logger.log_operation(
    operation="implicit_solve",
    duration=2.35,
    iterations=15,
    residual=1e-8
)

# Log memory usage
performance_logger.log_memory_usage(
    operation="grid_allocation",
    memory_mb=245.6,
    grid_size=(128, 128, 128)
)
```

### Physics Logger

```python
from israel_stewart.utils.logging_config import physics_logger

# Conservation law validation
physics_logger.log_conservation_check(
    quantity="energy",
    error=1e-10,
    tolerance=1e-8
)

# Solver convergence
physics_logger.log_convergence(
    solver="gmres",
    iterations=25,
    residual=1e-9,
    converged=True
)

# Physics fallbacks
physics_logger.log_physics_fallback(
    operation="fft_computation",
    reason="scipy_unavailable",
    fallback="numpy_fft"
)
```

### Solver Integration

The `@monitor_performance` decorator automatically logs solver operations:

```python
from israel_stewart.core.performance import monitor_performance

@monitor_performance("finite_difference_step")
def evolve_fields(fields, dt):
    # Your solver implementation
    # Automatically logs timing and memory usage
    return updated_fields
```

## Configuration Examples

### Development Setup
Maximum verbosity for debugging:

```python
configure_logging(
    level="DEBUG",
    format_type="console",
    log_file=Path("debug.log"),
    enable_performance=True,
    enable_memory_tracking=True
)
```

### Production Setup
Balanced monitoring:

```python
configure_logging(
    level="INFO",
    format_type="json",
    log_file=Path("production.log"),
    enable_performance=True,
    enable_memory_tracking=False
)
```

### Performance Analysis
Focus on timing and resource usage:

```python
configure_logging(level="WARNING")  # Minimal base logging
enable_performance_logging(True)    # Detailed performance data
enable_memory_debugging(True)       # Memory profiling
```

### Cluster/HPC Setup
Structured logging for automated analysis:

```bash
export ISRAEL_STEWART_LOG_LEVEL=INFO
export ISRAEL_STEWART_LOG_FORMAT=json
export ISRAEL_STEWART_LOG_FILE=/scratch/${SLURM_JOB_ID}/simulation.log
export ISRAEL_STEWART_LOG_PERFORMANCE=true

# Your MPI simulation will log to structured JSON
mpirun -n 64 python simulation.py
```

## Log Analysis

### Performance Monitoring
Key metrics logged automatically:

- **Operation timing**: Function execution time
- **Memory usage**: Peak and current memory consumption
- **Solver convergence**: Iteration counts and residuals
- **Grid operations**: Spatial resolution impact

### Physics Validation
Scientific correctness checks:

- **Conservation laws**: Energy-momentum conservation errors
- **Thermodynamic consistency**: Equation of state violations
- **Numerical stability**: CFL conditions and stability limits
- **Boundary conditions**: Proper boundary treatment

### Example Log Analysis

```python
# Parse structured logs for analysis
import json
from pathlib import Path

log_file = Path("simulation.log")
performance_data = []

for line in log_file.read_text().splitlines():
    if "performance" in line:
        entry = json.loads(line)
        performance_data.append({
            'operation': entry.get('operation'),
            'duration': entry.get('duration_seconds'),
            'memory_mb': entry.get('memory_mb')
        })

# Analyze solver performance trends
import pandas as pd
df = pd.DataFrame(performance_data)
print(df.groupby('operation')['duration'].describe())
```

## Best Practices

### Environment Configuration
- Use environment variables for deployment-specific settings
- Set `ISRAEL_STEWART_LOG_FILE` for persistent logging
- Enable performance logging only when needed for analysis

### Development Workflow
1. Start with `DEBUG` level for initial development
2. Use `INFO` level for integration testing
3. Use `WARNING` level for production runs
4. Enable performance logging for optimization work

### Production Deployment
- Use structured JSON format for automated parsing
- Rotate log files to prevent disk space issues
- Monitor WARNING and ERROR levels for system health
- Disable DEBUG level logging for performance

### Performance Considerations
- Logging overhead is < 50% even with verbose settings
- Disabled loggers have minimal performance impact
- File logging is more expensive than console logging
- JSON formatting has slight overhead vs plain text

## Troubleshooting

### Common Issues

**No log output**:
```python
# Check if logging is configured
from israel_stewart.utils.logging_config import get_logging_status
print(get_logging_status())
```

**Log level too high**:
```python
# Lower the log level to see more messages
set_log_level("DEBUG")
```

**Performance impact**:
```python
# Disable verbose logging for production
set_log_level("WARNING")
enable_performance_logging(False)
```

**File permissions**:
```bash
# Ensure log directory is writable
mkdir -p logs
chmod 755 logs
export ISRAEL_STEWART_LOG_FILE=logs/simulation.log
```

### Integration with Scientific Workflows

The logging system integrates seamlessly with scientific computing workflows:

- **Jupyter notebooks**: Automatic console formatting
- **HPC clusters**: Structured file logging with job IDs
- **CI/CD pipelines**: JSON output for automated testing
- **Performance profiling**: Detailed timing and memory data

For advanced usage and customization, see the API documentation in `israel_stewart.utils.logging_config`.
