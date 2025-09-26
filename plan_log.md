# Proper Logging Implementation Plan

## Current State Analysis
- **No logging framework**: Zero instances of `import logging` or `logger` usage across the entire codebase
- **Heavy print() usage**: Debug print statements scattered throughout production code (6 files with prints)
- **Excessive warnings**: 119+ `warnings.warn()` calls across 21 files masking real issues
- **Debug code in production**: Memory debugging prints, error prints, and validation prints in critical solver paths

## Implementation Plan

### Phase 1: Logging Infrastructure Setup
1. **Create logging configuration module** (`israel_stewart/utils/logging_config.py`)
   - Structured logging with JSON formatter for production
   - Console formatter for development
   - Multiple log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
   - Performance logging for solver operations
   - Memory usage logging with proper formatting

2. **Add logging dependency** to `pyproject.toml`
   - Add `structlog>=23.0.0` for structured logging
   - Configure logging setup in package `__init__.py`

### Phase 2: Replace Debug Prints
3. **Convert solver debug statements**:
   - `implicit.py:520-522` → `logger.debug("Memory tracking", memory_mb=current_memory)`
   - `implicit.py:597-612` → Structured memory leak detection logging
   - Remove all production `print()` statements from solver modules

4. **Convert memory debugging** to proper performance logging:
   - Structured memory usage logs with context
   - Performance metrics with operation timing
   - Optional verbose mode instead of always-on prints

### Phase 3: Improve Warning Management
5. **Categorize and reduce warnings**:
   - Convert physics fallback warnings to proper error handling
   - Keep only critical warnings that indicate real issues
   - Use structured logging for diagnostic information
   - Target reduction: 119 → ~20 critical warnings only

6. **Implement proper error handling**:
   - Replace silent fallbacks with logged degradation
   - Add recovery strategies with logging
   - Proper exception handling with context

### Phase 4: Enhanced Logging Features
7. **Add scientific computing specific features**:
   - Numerical convergence logging with metrics
   - Physics validation logging (conservation checks, etc.)
   - Performance profiling integration
   - Automatic log rotation and archiving

8. **Integration with existing performance monitoring**:
   - Enhance `@monitor_performance` decorator with structured logging
   - Memory optimization logging with detailed metrics
   - Solver stability and stiffness detection logging

### Phase 5: Documentation and Configuration
9. **User-facing logging controls**:
   - Environment variable configuration (`ISRAEL_STEWART_LOG_LEVEL`)
   - Runtime log level adjustment
   - Optional performance logging toggle
   - Memory debugging mode switch

10. **Testing and validation**:
    - Unit tests for logging configuration
    - Integration tests with solver logging
    - Performance impact assessment
    - Documentation and usage examples

## File Structure Changes
```
israel_stewart/
├── utils/
│   ├── __init__.py
│   └── logging_config.py        # New: Centralized logging setup
├── __init__.py                  # Modified: Initialize logging
└── solvers/                     # Modified: Remove prints, add logging
    ├── implicit.py              # ~15 print() statements → structured logging
    ├── spectral.py              # ~24 warnings → selective logging
    ├── finite_difference.py     # ~5 warnings → error handling
    └── splitting.py             # ~4 warnings → recovery logging
```

## Expected Benefits
- **Cleaner codebase**: Remove ~50+ debug print statements
- **Better debugging**: Structured logs with context and metadata
- **Production ready**: Configurable log levels and output formats
- **Performance insight**: Detailed timing and memory usage data
- **Easier maintenance**: Centralized logging configuration and management

## Implementation Time: ~3-4 days
- Day 1: Infrastructure setup and configuration
- Day 2: Replace debug prints in solver modules
- Day 3: Warning management and error handling improvements
- Day 4: Testing, documentation, and integration validation