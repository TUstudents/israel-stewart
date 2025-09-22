# Performance Profiler Suite

This directory contains the enhanced performance profiling infrastructure for the Israel-Stewart spectral solver.

## Files

### Core Implementation
- **Enhanced profiler**: Implemented in `israel_stewart/core/performance.py`
  - `DetailedProfiler`: Hierarchical timing and memory tracking
  - `FFTProfiler`: Specialized FFT operation profiling
  - Context manager API for easy instrumentation

### Profiling Scripts
- **`test_profiler.py`**: Demonstration script showing profiler capabilities
- **`plan_profiling.md`**: Comprehensive 3-phase optimization strategy

### Reports Generated
- **`detailed_performance_report.json`**: Hierarchical timing and memory analysis
- **`fft_efficiency_report.json`**: FFT-specific performance metrics

## Usage Example

```python
from israel_stewart.core.performance import profile_operation, detailed_performance_report

# Profile operations with hierarchical tracking
with profile_operation("spectral_timestep", {"grid_size": (64, 64, 32, 32)}):
    with profile_operation("stress_energy_tensor"):
        # Your computation here
        pass

    with profile_operation("divergence_T"):
        # More computation
        pass

# Generate comprehensive report
report = detailed_performance_report()
```

## Key Features Implemented

### ✅ Phase 1: Detailed Performance Profiling
- **Hierarchical Timing**: Track nested operations with parent-child relationships
- **Memory Allocation Tracking**: Monitor large allocations and memory hotspots
- **FFT-Specific Profiling**: Specialized analysis for spectral operations
- **Optimization Target Identification**: Automated bottleneck detection
- **Scaling Analysis**: Performance scaling with grid size

### 🔄 Phase 2: Memory Optimization (In Progress)
- Memory allocation analysis and pre-allocation strategies
- In-place operation implementation
- Memory-conscious algorithms

### ⏳ Phase 3: Computational Optimization (Planned)
- FFT optimization with FFTW integration
- Redundancy elimination and caching strategies

## Performance Insights from Testing

**Current Bottlenecks Identified:**
- Memory allocations: 3GB+ peak usage per timestep
- FFT operations: 0.6-0.8s per transform
- Spatial derivatives: 90% of divergence computation time

**Optimization Targets:**
1. Large temporary array allocations
2. FFT plan caching and optimization
3. Redundant viscous operator calls

The profiler is now integrated and ready to guide the optimization process in Phases 2 and 3.