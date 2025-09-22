# Plan: Performance Profiling and Optimization for Israel-Stewart Spectral Solver

## Executive Summary

The Israel-Stewart spectral solver has critical performance bottlenecks that prevent scaling to production grid sizes. Current benchmarks show 60+ second timesteps with 216MB memory usage per step on modest 32×32×16×16 grids. This plan provides systematic profiling and optimization strategies to achieve usable performance.

## Current Performance Issues

**Observed Bottlenecks:**
- `spectral_timestep`: 64.90s per timestep
- `stress_energy_tensor`: 3.87s + 128MB memory
- `divergence_T`: 13.68s per call
- `viscous_operator`: 1001 redundant calls per timestep
- `evolution_equations`: 13.76s per call

**Grid Scaling Problems:**
- 16×16×8×8 grid: Manageable but slow
- 32×32×16×16 grid: 16× data, ~16× slower
- 64×64×32×32 grid: Would be 256× slower (impractical)

## Phase 1: Detailed Performance Profiling

### 1.1 Enhanced Profiling Infrastructure

**Current State:** Basic `@monitor_performance` decorators exist but need enhancement

**Needed Improvements:**
- **Hierarchical Timing**: Profile nested operations within timesteps
- **Memory Allocation Tracking**: Track where large arrays are created
- **Call Graph Analysis**: Identify redundant function calls
- **FFT Operation Profiling**: Separate FFT performance from other operations
- **Cache Performance**: Track spectral derivative caching effectiveness

**Implementation:**
```python
# Enhanced profiler with nested operation tracking
class DetailedProfiler:
    def __init__(self):
        self.call_stack = []
        self.nested_timings = {}
        self.memory_snapshots = {}
        self.fft_operations = {}
        self.cache_stats = {}
```

### 1.2 Critical Operation Breakdown

**Target Operations for Detailed Profiling:**

1. **Spectral Derivatives** (`spatial_derivative`, `spatial_gradient`)
   - FFT forward/inverse timing
   - Memory allocation for k-space arrays
   - Cache hit/miss rates

2. **Viscous Operators** (`apply_viscous_operator`, `apply_bulk_viscous_operator`)
   - Tensor contraction performance
   - Temporary array creation
   - Redundant computation detection

3. **Physics Evolution** (`_imex_rk2_step`, `_split_step_advance`)
   - Stage-by-stage timing
   - Memory usage per stage
   - Solver convergence statistics

4. **Tensor Operations** (stress_energy_tensor, divergence_T)
   - Einstein summation performance
   - Memory layout efficiency
   - Vectorization effectiveness

### 1.3 Profiling Output Requirements

**Detailed Reports:**
- **Per-timestep breakdown**: What percentage of time in each major operation
- **Memory hotspots**: Which operations allocate the most memory
- **Scaling analysis**: How operations scale with grid size
- **Optimization targets**: Ranked list of bottlenecks by impact

## Phase 2: Memory Optimization

### 2.1 Memory Allocation Analysis

**Current Issues:**
- 128-216MB per timestep suggests large temporary arrays
- Multiple FFT operations likely creating redundant k-space arrays
- Tensor operations may not be using in-place operations

**Investigation Plan:**
```python
# Memory allocation tracker
def track_large_allocations():
    """Track arrays > 10MB to identify memory hotspots"""
    # Monitor numpy array creation
    # Track FFT workspace allocation
    # Identify temporary tensor storage
```

### 2.2 Memory Reduction Strategies

**Pre-allocation Strategy:**
- Pre-allocate all FFT workspaces during solver initialization
- Reuse k-space arrays across multiple derivative operations
- Implement array pools for temporary tensor storage

**In-place Operations:**
- Convert tensor operations to in-place where possible
- Use views instead of copies for array slicing
- Implement destructive FFT operations where safe

**Memory-conscious Algorithms:**
- Stream FFT operations to reduce peak memory
- Implement block-wise tensor operations for large grids
- Use lower precision (float32) where appropriate

## Phase 3: Computational Optimization

### 3.1 FFT Optimization

**Current Issues:** Standard numpy.fft may not be optimal for repeated operations

**Optimization Targets:**
- **FFTW Integration**: Use scipy.fft with FFTW backend for better performance
- **Plan Caching**: Cache FFT plans across timesteps
- **Parallel FFT**: Use multi-threaded FFT for large transforms
- **FFT Wisdom**: Pre-compute optimal FFT algorithms

**Implementation Priority:**
```python
# Enhanced FFT manager
class OptimizedFFTManager:
    def __init__(self, grid_shape):
        self.fft_plans = {}  # Cache FFT plans
        self.workspace = {}  # Pre-allocated arrays
        self.use_fftw = True  # Enable FFTW backend
```

### 3.3 Redundancy Elimination

**Call Pattern Analysis:**
- Track function call frequencies during timesteps
- Identify cacheable intermediate results
- Eliminate redundant computations in IMEX stages

**Caching Strategy:**
- Cache spectral derivatives when wave vectors unchanged
- Store frequently computed tensor contractions
- Implement smart invalidation for cached results

## Phase 4: Implementation Plan

### 4.1 Profiling Implementation (Week 1)

**Day 1-2: Enhanced Profiler**
- Extend `PerformanceMonitor` with hierarchical timing
- Add memory allocation tracking
- Implement FFT-specific profiling

**Day 3-4: Baseline Measurements**
- Profile current solver on 16×16×8×8 grid
- Profile current solver on 32×32×16×16 grid
- Generate detailed performance reports

**Day 5: Analysis and Prioritization**
- Identify top 5 bottlenecks by time
- Identify top 5 memory consumers
- Create optimization priority matrix

### 4.2 Memory Optimization (Week 2)

**Day 1-2: Memory Allocation Tracking**
- Implement allocation size tracking
- Identify large temporary arrays
- Map memory usage patterns

**Day 3-4: Pre-allocation Implementation**
- Pre-allocate FFT workspaces
- Implement array pools
- Convert to in-place operations

**Day 5: Memory Testing**
- Validate memory reduction
- Test performance impact
- Measure memory scaling improvement

### 4.3 Computational Optimization (Week 3)

**Day 1-2: FFT Optimization**
- Integrate FFTW backend
- Implement FFT plan caching
- Add parallel FFT support

**Day 3-4: Redundancy Elimination**
- Optimize call patterns
- Add caching for intermediate results
- Eliminate redundant computations

**Day 5: Performance Validation**
- Benchmark optimized solver
- Compare against baseline
- Validate numerical accuracy

## Expected Outcomes

### Performance Targets

**Short-term (Phase 1-2):**
- 50% reduction in memory usage per timestep
- 30% reduction in timestep execution time
- Detailed performance understanding for further optimization

**Medium-term (Phase 3):**
- 5× speedup in timestep execution
- Ability to run 64×64×32×32 grids in reasonable time
- Memory usage scaling better than O(N²) with grid size

**Long-term:**
- Production-ready performance for 128×128×64×64 grids
- Memory usage under 1GB for large simulations
- Timesteps under 1 second for typical problems

### Success Metrics

1. **Timestep Performance**: <5 seconds per timestep on 64×64×32×32 grid
2. **Memory Efficiency**: <500MB peak memory usage per timestep
3. **Scaling**: Sub-quadratic scaling with grid resolution
4. **Numerical Accuracy**: No degradation in benchmark results

## Implementation Priority

**Critical Path:**
1. Enhanced profiling → Identify exact bottlenecks
2. Memory optimization → Enable larger grids
3. FFT optimization → Address primary computational bottleneck
4. Full integration testing → Validate improvements

This plan addresses the fundamental performance barriers preventing the Israel-Stewart solver from reaching production readiness while maintaining scientific accuracy.
