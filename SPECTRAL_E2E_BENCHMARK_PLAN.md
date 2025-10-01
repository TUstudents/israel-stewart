# End-to-End Spectral Solver Benchmark Against Analytical Results

## Overview
Create a comprehensive benchmark script that validates the SpectralISHydrodynamics solver against multiple analytical solutions, demonstrating correctness of the 4D spacetime solver architecture and physics implementation.

## Benchmark Structure

### File: `benchmarks/run_spectral_e2e_benchmark.py`

A standalone executable script that runs multiple benchmark scenarios.

## Test Suite Design

### Test 1: Sound Wave Propagation (4D Spacetime)
**Physics**: Linear acoustics in perfect relativistic fluid
**Analytical Solution**: `ρ(t,x) = ρ₀ + A sin(kx - ωt)` with `ω = c_s k`

**Validation**:
1. Initialize entire 4D spacetime with analytical solution
2. Check `∂_μ T^μν ≈ 0` before and after refinement
3. Measure wave properties at multiple time slices
4. Compare to analytical: frequency, phase velocity, amplitude preservation

**Metrics**:
- Frequency error: `|ω_numerical - ω_analytical|/ω_analytical < 1%`
- Phase velocity error: `< 2%`
- Amplitude decay: `< 5%` for ideal fluid
- Conservation violation: `max|∂_μ T^μν| < 10⁻³`

### Test 2: Convergence Study
**Purpose**: Verify spectral accuracy (exponential convergence for smooth solutions)

**Method**:
- Run same sound wave at resolutions: 8³, 16³, 32³, 64³
- Measure L² error vs analytical solution
- Verify exponential convergence: `error ~ exp(-cN)`

**Validation**:
- Plot error vs resolution (log scale)
- Verify convergence rate > 2nd order finite difference

### Test 3: Multi-Mode Superposition
**Physics**: Linearity test with multiple wave modes

**Setup**:
```python
ρ = ρ₀ + Σᵢ Aᵢ sin(kᵢx - ωᵢt)
```

**Validation**:
- Each mode evolves independently
- No spurious mode coupling
- Spectral accuracy maintained

### Test 4: Viscous Damping (Israel-Stewart)
**Physics**: Sound waves with shear/bulk viscosity

**Analytical**: Dispersion relation with viscous corrections:
```
ω(k) = c_s k - iγ(k)
γ(k) = (η + ζ)k²/(2ρ₀) + O(k⁴)  # Navier-Stokes limit
```

**Validation**:
- Measure damping rate from amplitude decay
- Compare to analytical γ(k)
- Verify Israel-Stewart corrections

### Test 5: Long-Time Stability
**Purpose**: Ensure solver doesn't accumulate errors or become unstable

**Method**:
- Evolve sound wave for 100 wave periods
- Monitor energy conservation
- Check field boundedness

**Validation**:
- Energy drift < 1% per 10 periods
- No exponential growth
- Conservation laws maintained

## Implementation Plan

### 1. Create Main Benchmark Script
**File**: `benchmarks/run_spectral_e2e_benchmark.py`

```python
#!/usr/bin/env python3
"""
End-to-End Spectral Solver Benchmark Against Analytical Results

Validates SpectralISHydrodynamics 4D spacetime solver with:
1. Sound wave propagation (4D initialization)
2. Convergence study (spectral accuracy)
3. Multi-mode superposition (linearity)
4. Viscous damping (Israel-Stewart)
5. Long-time stability
"""

class SpectralE2EBenchmark:
    def __init__(self, config: dict):
        """Initialize benchmark with configuration"""

    def run_all_tests(self) -> dict[str, BenchmarkResult]:
        """Run complete benchmark suite"""

    def test_sound_wave_4d(self) -> BenchmarkResult:
        """Test 1: 4D spacetime sound wave"""

    def test_convergence(self) -> BenchmarkResult:
        """Test 2: Convergence study"""

    def test_multimode(self) -> BenchmarkResult:
        """Test 3: Multi-mode superposition"""

    def test_viscous_damping(self) -> BenchmarkResult:
        """Test 4: Viscous damping"""

    def test_stability(self) -> BenchmarkResult:
        """Test 5: Long-time stability"""

    def generate_report(self, results: dict) -> str:
        """Generate markdown report"""
```

### 2. Result Dataclasses
```python
@dataclass
class BenchmarkResult:
    test_name: str
    passed: bool
    error_metric: float
    tolerance: float
    analytical_value: float
    numerical_value: float
    convergence_order: Optional[float]
    computation_time: float
    grid_resolution: tuple
    details: dict
```

### 3. Analytical Solution Helpers
```python
def analytical_sound_wave_4d(
    T: np.ndarray,
    X: np.ndarray,
    k: float,
    amplitude: float,
    rho_0: float = 1.0,
    c_s: float = np.sqrt(1/3)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate analytical sound wave solution for entire 4D spacetime.

    Returns: (rho, u_x, u_t)
    """
```

### 4. Output and Reporting
- **Console**: Live progress with colored output
- **JSON**: `benchmark_results.json` with all metrics
- **Markdown**: `BENCHMARK_REPORT.md` with plots
- **Plots**: Error vs resolution, time series, FFT spectra

## Success Criteria

✅ **Pass**: All 5 tests pass with metrics within tolerance
✅ **Report**: Comprehensive markdown report generated
✅ **Plots**: Visualization of key results
✅ **Documentation**: Clear instructions for running/interpreting

## Timeline

1. **Create benchmark class structure** (30 min)
2. **Implement Test 1: Sound wave 4D** (45 min)
3. **Implement Test 2: Convergence** (30 min)
4. **Implement Tests 3-5** (1 hour)
5. **Add reporting and plotting** (45 min)
6. **Documentation and testing** (30 min)

**Total**: ~4 hours for complete implementation

## Deliverables

1. `benchmarks/run_spectral_e2e_benchmark.py` - Main executable
2. `BENCHMARK_REPORT.md` - Auto-generated results
3. `benchmark_results.json` - Machine-readable results
4. `benchmarks/figures/` - Plots and visualizations
5. Updated `README.md` section on running benchmarks
