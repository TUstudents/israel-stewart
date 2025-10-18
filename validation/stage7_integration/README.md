# Stage 7: Integration Testing

**Status**: 🔴 0% Complete (not started)

**Priority**: LOW (blocked on Stages 3, 5, 6)

## Goal

Comprehensive end-to-end validation of the complete Israel-Stewart implementation under realistic conditions, stress testing, and performance benchmarking.

## Why This Stage Matters

**"Individual components work, but does the full system work together?"**

Stages 1-6 validate components in isolation:
- ✅ Stage 1: Units correct
- ✅ Stage 2: Coefficients correct
- ⚠️ Stage 3: Equations correct (needs tests)
- 🟢 Stage 4: Dispersion relations correct
- ⚠️ Stage 5: Solvers correct (needs tests)
- 🟡 Stage 6: Individual benchmarks correct (2 bugs to fix)

**Stage 7 asks**: What happens when we combine everything?
- Do different parameter combinations work?
- Are there hidden failure modes?
- Does performance degrade at scale?
- Do conservation laws hold over long times?

## Acceptance Criteria

- ❌ All benchmarks pass with multiple parameter sets
- ❌ Regime violation detection works correctly
- ❌ Performance benchmarks documented
- ❌ Conservation holds in long-time evolution
- ❌ No regressions in any stage

## Blocked Dependencies

**Cannot proceed until**:
1. ✅ Stage 1 complete (units) - DONE
2. ✅ Stage 2 complete (coefficients) - DONE
3. ❌ Stage 3 complete (equations) - Need unit tests (6 days)
4. ✅ Stage 4 complete (dispersion) - DONE (with known physical limitation)
5. ❌ Stage 5 complete (solvers) - Need unit tests (5 days)
6. ⚠️ Stage 6 mostly complete (benchmarks) - 2 bugs to fix (5 days)

**Total blocking work**: ~16 days

## Test Categories

### 1. Multi-Parameter Validation (3 days)

**Goal**: Verify benchmarks work across parameter ranges, not just single values

**Sound waves**:
```python
@pytest.mark.parametrize("k", [0.1, 0.5, 1.0, 2.0, 4.0])
@pytest.mark.parametrize("sigma", [1.0, 10.0, 100.0, 1000.0])
@pytest.mark.parametrize("temperature", [0.2, 0.4, 0.8])
def test_sound_waves_parameter_sweep(k, sigma, temperature):
    """Sound waves work across valid parameter space"""
    # Check regime validity
    tau = compute_relaxation_time(sigma, temperature)
    omega = k * (1.0 / np.sqrt(3.0))  # c_s for radiation

    if abs(tau * omega) > 1.0:
        pytest.skip("Outside Israel-Stewart regime")

    # Run benchmark
    benchmark = NumericalSoundWaveBenchmark(...)
    frequency = benchmark.measure_frequency()

    # Should be within 5% across all valid parameters
    assert abs(frequency - omega) / omega < 0.05
```

**Bjorken flow**:
```python
@pytest.mark.parametrize("eta_over_s", [0.1, 0.2, 0.3, 0.4, 0.5])
@pytest.mark.parametrize("tau_0", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("T_0", [0.2, 0.4, 0.6])
def test_bjorken_parameter_sweep(eta_over_s, tau_0, T_0):
    """Bjorken flow works across QGP parameter ranges"""
```

**Diffusion**:
```python
@pytest.mark.parametrize("k", [0.1, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("mu_over_T", [0.0, 0.5, 1.0])
def test_diffusion_parameter_sweep(k, mu_over_T):
    """Diffusion works across chemical potential ranges"""
```

**Tests to create**:
- `multi_parameter/test_sound_wave_sweep.py`
- `multi_parameter/test_bjorken_sweep.py`
- `multi_parameter/test_diffusion_sweep.py`

**Expected outcome**: Identify any parameter combinations that fail unexpectedly

### 2. Stress Testing (2 days)

**Goal**: Find failure modes at extreme (but valid) parameters

**Extreme viscosity**:
```python
def test_extreme_viscosity_low():
    """Nearly-ideal fluid: η/s → 0"""
    eta_over_s = 1e-4  # Extremely small
    # Should approach ideal hydro (Euler equations)

def test_extreme_viscosity_high():
    """Highly viscous: η/s → 1"""
    eta_over_s = 1.0  # Close to unitarity bound
    # Should still be stable (if in regime)
```

**Extreme gradients**:
```python
def test_high_wavenumber():
    """k → k_max = 1/(τ·c_s)"""
    # Should trigger regime warning
    with pytest.warns(UserWarning, match="Outside Israel-Stewart regime"):
        k_max = compute_regime_limit(tau, c_s)
        benchmark = NumericalSoundWaveBenchmark(wavenumber=k_max)
```

**Large amplitudes**:
```python
def test_large_amplitude_sound_wave():
    """A → 0.5 (50% perturbation)"""
    # Should exit linear regime
    # Nonlinear effects should appear
```

**Tests to create**:
- `stress_tests/test_extreme_viscosity.py`
- `stress_tests/test_high_gradients.py`
- `stress_tests/test_large_amplitudes.py`
- `stress_tests/test_long_evolution.py`

**Expected outcome**: Document where implementation breaks down

### 3. Performance Benchmarking (2 days)

**Goal**: Establish performance baselines for different grid sizes and solvers

**Scalability**:
```python
@pytest.mark.parametrize("N", [16, 32, 64, 128])
def test_scaling_with_grid_size(N):
    """Measure time vs grid size"""
    grid = SpaceGrid(grid_points=(N, N, N), ...)

    start = time.time()
    hydro.evolve(t_final=1.0, dt=0.01)
    elapsed = time.time() - start

    # Spectral: O(N³ log N) expected
    # Document actual scaling
```

**Solver comparison**:
```python
def test_solver_performance_comparison():
    """RK4 vs IMEX speed and accuracy"""
    # Same problem, both solvers
    results_rk4 = evolve_with_rk4(...)
    results_imex = evolve_with_imex(...)

    # Compare:
    # - Wall time
    # - Timesteps taken
    # - Final accuracy
```

**Memory profiling**:
```python
def test_memory_usage():
    """Verify 3D architecture memory efficiency"""
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024**2  # MB

    grid = SpaceGrid(grid_points=(64, 64, 64), ...)
    fields = ISFieldConfiguration(grid)

    mem_after = process.memory_info().rss / 1024**2
    mem_used = mem_after - mem_before

    # Should be ~75 MB for 64³ (not ~1500 MB for 4D)
    assert mem_used < 100  # MB
```

**Tests to create**:
- `performance/test_scaling.py`
- `performance/test_solver_comparison.py`
- `performance/test_memory_usage.py`
- `performance/profile_hotspots.py` (using cProfile)

**Expected outcome**: Performance regression tests, optimization targets identified

### 4. Conservation Checks (2 days)

**Goal**: Verify conservation laws hold during evolution, not just initially

**Energy conservation**:
```python
def test_energy_conservation_long_time():
    """∫ρ d³x conserved over long evolution"""
    # Initial energy
    E_0 = np.sum(fields.rho) * grid.dx**3

    # Evolve for 100 relaxation times
    t_final = 100 * coeffs.shear_relaxation_time
    hydro.evolve(t_final=t_final)

    # Final energy
    E_f = np.sum(fields.rho) * grid.dx**3

    # Should conserve to machine precision (for periodic BC)
    assert abs(E_f - E_0) / E_0 < 1e-10
```

**Momentum conservation**:
```python
def test_momentum_conservation():
    """∫(ρu^i) d³x conserved"""
    P_0 = compute_total_momentum(fields)

    hydro.evolve(t_final=10.0)

    P_f = compute_total_momentum(fields)

    np.testing.assert_allclose(P_f, P_0, rtol=1e-10)
```

**Constraint maintenance**:
```python
def test_constraint_preservation():
    """V^μ u_μ = 0, π^μν u_μ = 0, u·u = -1 throughout evolution"""
    for snapshot in trajectory:
        # Check Landau frame conditions
        assert np.allclose(V_dot_u(snapshot), 0.0, atol=1e-12)
        assert np.allclose(pi_dot_u(snapshot), 0.0, atol=1e-12)

        # Check normalization
        assert np.allclose(u_dot_u(snapshot), -1.0, atol=1e-12)
```

**Tests to create**:
- `conservation/test_energy_conservation.py`
- `conservation/test_momentum_conservation.py`
- `conservation/test_constraint_preservation.py`
- `conservation/test_particle_number.py`

**Expected outcome**: Identify any conservation violations (usually indicates solver bugs)

### 5. Regression Testing (1 day)

**Goal**: Ensure changes don't break previously working functionality

**Snapshot comparison**:
```python
def test_regression_sound_wave():
    """Sound wave results match baseline"""
    # Load baseline results (from known-good commit)
    baseline = load_baseline("sound_wave_k05_sigma1000.h5")

    # Run current implementation
    benchmark = NumericalSoundWaveBenchmark(...)
    current = benchmark.run()

    # Compare field-by-field
    np.testing.assert_allclose(
        current.rho,
        baseline.rho,
        rtol=1e-12,
        err_msg="Sound wave regression detected"
    )
```

**Coefficient regression**:
```python
def test_regression_ired_coefficients():
    """IReD coefficients unchanged"""
    # Run full Stage 2 test suite
    # Should always be 26/26 passing
    result = pytest.main([
        "tests/test_ired_coefficients.py",
        "-v"
    ])
    assert result == 0  # All passed
```

**Tests to create**:
- `regression/test_all_benchmarks.py`
- `regression/test_ired_coefficients.py`
- `regression/compare_baselines.py`

**Expected outcome**: CI/CD pipeline can catch regressions automatically

## Test Scripts

### To Be Created (Summary)

**Multi-parameter** (3 days):
- `multi_parameter/test_sound_wave_sweep.py`
- `multi_parameter/test_bjorken_sweep.py`
- `multi_parameter/test_diffusion_sweep.py`

**Stress tests** (2 days):
- `stress_tests/test_extreme_viscosity.py`
- `stress_tests/test_high_gradients.py`
- `stress_tests/test_large_amplitudes.py`
- `stress_tests/test_long_evolution.py`

**Performance** (2 days):
- `performance/test_scaling.py`
- `performance/test_solver_comparison.py`
- `performance/test_memory_usage.py`
- `performance/profile_hotspots.py`

**Conservation** (2 days):
- `conservation/test_energy_conservation.py`
- `conservation/test_momentum_conservation.py`
- `conservation/test_constraint_preservation.py`
- `conservation/test_particle_number.py`

**Regression** (1 day):
- `regression/test_all_benchmarks.py`
- `regression/test_ired_coefficients.py`
- `regression/compare_baselines.py`

**Total**: ~20 test scripts

## Remaining Work

**Cannot start until Stages 3, 5, 6 complete** (blocking: ~16 days)

**Then Stage 7 work** (after unblocked):
- 3 days: Multi-parameter validation
- 2 days: Stress testing
- 2 days: Performance benchmarking
- 2 days: Conservation checks
- 1 day: Regression testing

**Total Stage 7 time**: 10 days (after dependencies resolved)

## Success Metrics

**Before** (2025-10-18):
- Stage 7 not started ✗
- Blocked on Stages 3, 5, 6 ⚠️

**Target**:
- 50+ integration tests passing ✓
- Parameter space mapped ✓
- Stress test limits documented ✓
- Performance baselines established ✓
- Conservation verified over long times ✓
- No regressions in Stages 1-6 ✓

## References

- **Roadmap**: `validation/VALIDATION_ROADMAP.md`
- **Stage 1-6 results**: See individual stage READMEs
- **CI/CD**: `.github/workflows/` (to be created)

## Next Steps

**BLOCKED**: Cannot proceed until:
1. Stage 3 complete (6 days)
2. Stage 5 complete (5 days)
3. Stage 6 bugs fixed (5 days)

**After unblocked**:
1. Create test infrastructure (1 day)
2. Multi-parameter sweep (3 days)
3. Stress testing (2 days)
4. Performance benchmarking (2 days)
5. Conservation verification (2 days)
6. Regression suite (1 day)

**Timeline**: Start ~3 weeks from now (2025-11-08), complete by ~4 weeks from now (2025-11-15)
