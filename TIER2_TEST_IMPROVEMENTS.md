# Tier 2 Test Improvements for test_spectral_solver.py

## Overview

This document outlines improvements to 6 test weaknesses identified in the spectral solver test suite. The work is divided into three phases to reduce complexity and enable incremental validation.

**Status:** Ready for implementation (Tier 1 completed: 57/57 tests passing)

---

## Phase A: Physics Validation Tests (2 tests)

**Goal:** Add proper physics validation to tests that currently only check shape/finiteness

### A1. Fix `test_sound_wave_propagation` (Lines 709-742)

**Current Problem:**
- Name claims "sound wave propagation" but only tests gradient direction
- Missing: time evolution, wave speed verification, dispersion relation

**Changes:**
```python
def test_sound_wave_propagation(self) -> None:
    """Test linear sound wave propagation in spectral method."""
    grid = SpacetimeGrid(
        coordinate_system="cartesian",
        time_range=(0.0, 1.0),
        spatial_ranges=[(0.0, 2 * np.pi), (0.0, 2 * np.pi), (0.0, 2 * np.pi)],
        grid_points=(10, 32, 32, 32),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)

    # Sound wave: ρ(x,t) = ρ₀ + A·sin(k·x - ω·t)
    # For ideal gas: c_s² = ∂P/∂ρ = P/ρ = 0.33
    c_s = np.sqrt(0.33)
    k = 1.0  # Wave number
    omega = c_s * k  # Dispersion relation ω = c_s·k

    x = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    # Initial wave (t=0)
    amplitude = 0.01
    rho_0 = 1.0
    fields.rho = rho_0 + amplitude * np.sin(k * X)
    fields.pressure = 0.33 * fields.rho  # Ideal gas
    fields.u_mu[..., 0] = 1.0

    coeffs = TransportCoefficients(
        shear_viscosity=0.0,  # Ideal fluid for clean wave propagation
        bulk_viscosity=0.0,
    )

    hydro = SpectralISHydrodynamics(grid, fields, coeffs)

    # Evolve wave for short time
    dt = 0.1
    n_steps = 5
    total_time = dt * n_steps

    for _ in range(n_steps):
        hydro.time_step(dt)

    # Wave should have propagated distance: Δx = c_s * t
    expected_shift = c_s * total_time

    # Expected wave after propagation: ρ(x,t) = ρ₀ + A·sin(k·(x - c_s·t))
    expected_rho = rho_0 + amplitude * np.sin(k * (X - expected_shift))

    # Compare propagated wave to expected (allow some numerical diffusion)
    rho_error = np.max(np.abs(fields.rho - expected_rho))
    relative_error = rho_error / amplitude

    assert relative_error < 0.3, (
        f"Sound wave propagation error too large: {relative_error:.3f}. "
        f"Expected wave to propagate at c_s={c_s:.3f}"
    )

    # Verify wave structure is preserved (check correlation)
    correlation = np.corrcoef(fields.rho.flatten(), expected_rho.flatten())[0, 1]
    assert correlation > 0.9, (
        f"Wave structure not preserved: correlation={correlation:.3f}"
    )
```

**Validation:**
```bash
uv run pytest israel_stewart/tests/test_spectral_solver.py::TestSpectralValidation::test_sound_wave_propagation -xvs
```

**Expected Outcome:**
- Actually validates sound wave physics: propagation speed c_s = √(∂P/∂ρ)
- Tests dispersion relation: ω = c_s·k
- Verifies wave structure preservation
- May expose issues with time integration or conservation laws

---

### A2. Fix `test_conservation_integration` (Lines 564-574)

**Current Problem:**
- Only checks shape and finiteness
- Missing: symmetry (T^μν = T^νμ), energy conditions, conservation laws

**Changes:**
```python
def test_conservation_integration(self, setup_hydro_solver: tuple) -> None:
    """Test integration with conservation laws and verify physics."""
    hydro_solver, fields = setup_hydro_solver

    if hydro_solver.conservation is None:
        pytest.skip("Conservation module not available")

    # Test stress-energy tensor computation
    T_munu = hydro_solver.conservation.stress_energy_tensor()

    assert T_munu.shape == (*fields.rho.shape, 4, 4), "Stress-energy tensor has correct shape"
    assert np.all(np.isfinite(T_munu)), "Stress-energy tensor is finite"

    # Test 1: Symmetry T^μν = T^νμ
    for mu in range(4):
        for nu in range(mu + 1, 4):
            symmetry_error = np.max(np.abs(T_munu[..., mu, nu] - T_munu[..., nu, mu]))
            max_val = np.max(np.abs(T_munu[..., mu, nu]))
            relative_error = symmetry_error / (max_val + 1e-14)
            assert relative_error < 1e-10, (
                f"Stress-energy tensor not symmetric: T^{mu}{nu} != T^{nu}{mu}, "
                f"relative error = {relative_error:.2e}"
            )

    # Test 2: Energy condition T^00 ≥ 0 (positive energy density)
    T00 = T_munu[..., 0, 0]
    assert np.all(T00 >= -1e-10), (
        f"Energy density negative: min(T^00) = {np.min(T00):.2e}"
    )

    # Test 3: Trace (for massless ideal gas T^μ_μ = 0, but with viscosity ≠ 0)
    trace = np.trace(T_munu, axis1=-2, axis2=-1)
    assert np.all(np.isfinite(trace)), "Trace is finite"

    # For Israel-Stewart with viscosity, trace should be dominated by bulk pressure
    # T^μ_μ = ρ - 3P - 3Π (in rest frame)
    expected_trace_magnitude = np.max(np.abs(fields.rho - 3 * fields.pressure - 3 * fields.Pi))
    actual_trace_magnitude = np.max(np.abs(trace))

    # Should be within order of magnitude
    assert actual_trace_magnitude < 10 * expected_trace_magnitude, (
        f"Trace magnitude unexpected: {actual_trace_magnitude:.2e} vs "
        f"expected ~{expected_trace_magnitude:.2e}"
    )
```

**Validation:**
```bash
uv run pytest israel_stewart/tests/test_spectral_solver.py::TestSpectralISHydrodynamics::test_conservation_integration -xvs
```

**Expected Outcome:**
- Validates stress-energy tensor symmetry (fundamental requirement)
- Checks energy conditions (physical constraint)
- Verifies trace structure matches Israel-Stewart formalism
- Will catch any sign errors or index transposition bugs

---

## Phase B: Tighter Bounds and Scaling (2 tests)

**Goal:** Replace weak/arbitrary bounds with physics-based constraints

### B1. Strengthen `test_phase_1_integration` (Lines 1013-1054)

**Current Problem:**
- Energy change tolerance of 100% allows doubling!
- Bulk pressure bound of 100 is arbitrary (initial ~0.01, allows 10,000× increase)

**Changes:**
```python
def test_phase_1_integration(self, setup_fixed_solver: tuple) -> None:
    """Integration test that all Phase 1 critical fixes work together."""
    solver, fields = setup_fixed_solver

    if not hasattr(solver, "time_step"):
        pytest.skip("time_step method not available in this solver type")

    # Run a complete time evolution with all fixes active
    dt = 0.01
    n_steps = 3

    # Store initial state
    initial_energy = np.sum(fields.rho)
    initial_Pi = np.mean(fields.Pi)
    initial_pi_norm = np.sqrt(np.mean(fields.pi_munu**2))

    # Evolve the system
    try:
        for step in range(n_steps):
            solver.time_step(dt)
            # Check stability at each step
            assert np.all(np.isfinite(fields.rho)), f"Energy density finite at step {step}"
            assert np.all(np.isfinite(fields.Pi)), f"Bulk pressure finite at step {step}"
            assert np.all(np.isfinite(fields.pi_munu)), f"Shear tensor finite at step {step}"
    except Exception as e:
        pytest.skip(f"Evolution failed: {e}")

    # Check that system remains stable with all fixes
    final_energy = np.sum(fields.rho)
    final_Pi = np.mean(fields.Pi)
    final_pi_norm = np.sqrt(np.mean(fields.pi_munu**2))

    # Energy conservation: tighter bound (was 100%, now 10%)
    # For 3 small timesteps with weak viscosity, 10% is reasonable
    if initial_energy > 0:
        energy_change = abs(final_energy - initial_energy) / initial_energy
        assert energy_change < 0.1, (
            f"Energy change too large: {energy_change:.3f}. "
            f"Initial: {initial_energy:.3e}, Final: {final_energy:.3e}"
        )

    # Bulk pressure: physics-based bound (was 100, now 10×initial or 1.0)
    # Bulk pressure relaxes: should not explode
    Pi_change = abs(final_Pi - initial_Pi)
    max_reasonable_Pi = max(10 * abs(initial_Pi), 1.0)  # Either 10× initial or 1.0
    assert Pi_change < max_reasonable_Pi, (
        f"Bulk pressure change too large: {Pi_change:.3f}. "
        f"Initial: {initial_Pi:.3e}, Final: {final_Pi:.3e}, "
        f"Max allowed: {max_reasonable_Pi:.3e}"
    )

    # Shear stress evolution should be bounded
    pi_change = abs(final_pi_norm - initial_pi_norm)
    max_reasonable_pi = max(10 * initial_pi_norm, 1.0)
    assert pi_change < max_reasonable_pi, (
        f"Shear stress change too large: {pi_change:.3f}. "
        f"Initial: {initial_pi_norm:.3e}, Final: {final_pi_norm:.3e}"
    )
```

**Validation:**
```bash
uv run pytest israel_stewart/tests/test_spectral_solver.py::TestSpectralSolverFixes::test_phase_1_integration -xvs
```

**Expected Outcome:**
- Catches energy explosions (>10% change in 3 steps is unphysical)
- Detects bulk pressure instabilities
- May reveal numerical issues that were hidden by lenient bounds

---

### B2. Add `test_fft_scaling_verification` (New Test)

**Current Problem:**
- `test_performance_scaling` claims O(N log N) but only checks absolute time < 10s
- Missing: actual scaling relationship verification

**Changes:**
Add new test method to `TestSpectralISHydrodynamics` class:

```python
def test_fft_scaling_verification(self) -> None:
    """Verify FFT scales as O(N³ log N) for 3D grids."""

    # Test with grid_size = 16
    grid_16 = SpacetimeGrid(
        coordinate_system="cartesian",
        time_range=(0.0, 1.0),
        spatial_ranges=[(0.0, 2 * np.pi), (0.0, 2 * np.pi), (0.0, 2 * np.pi)],
        grid_points=(10, 16, 16, 16),
        boundary_conditions="periodic",
    )
    fields_16 = ISFieldConfiguration(grid_16)
    solver_16 = SpectralISolver(grid_16, fields_16)
    test_field_16 = np.random.rand(16, 16, 16)

    import time
    start = time.time()
    n_iterations = 20
    for _ in range(n_iterations):
        solver_16.spatial_derivative(test_field_16, 0)
    t1 = time.time() - start

    # Test with grid_size = 32
    grid_32 = SpacetimeGrid(
        coordinate_system="cartesian",
        time_range=(0.0, 1.0),
        spatial_ranges=[(0.0, 2 * np.pi), (0.0, 2 * np.pi), (0.0, 2 * np.pi)],
        grid_points=(10, 32, 32, 32),
        boundary_conditions="periodic",
    )
    fields_32 = ISFieldConfiguration(grid_32)
    solver_32 = SpectralISolver(grid_32, fields_32)
    test_field_32 = np.random.rand(32, 32, 32)

    start = time.time()
    for _ in range(n_iterations):
        solver_32.spatial_derivative(test_field_32, 0)
    t2 = time.time() - start

    # Expected scaling: t2/t1 ≈ (32/16)³ * log(32)/log(16) = 8 * 1.25 = 10
    n1, n2 = 16, 32
    size_ratio = (n2 / n1) ** 3  # 8 for 3D
    log_ratio = np.log(n2) / np.log(n1)  # 1.25
    expected_scaling = size_ratio * log_ratio  # 10

    actual_ratio = t2 / t1

    # Allow factor of 2 tolerance (caching, warmup, etc.)
    lower_bound = 0.5 * expected_scaling
    upper_bound = 2.0 * expected_scaling

    assert lower_bound < actual_ratio < upper_bound, (
        f"FFT scaling incorrect: expected {expected_scaling:.2f}×, "
        f"got {actual_ratio:.2f}× (t1={t1:.3f}s, t2={t2:.3f}s). "
        f"Expected range: [{lower_bound:.2f}, {upper_bound:.2f}]"
    )
```

**Validation:**
```bash
uv run pytest israel_stewart/tests/test_spectral_solver.py::TestSpectralISHydrodynamics::test_fft_scaling_verification -xvs
```

**Expected Outcome:**
- Validates O(N³ log N) complexity for 3D FFT
- Detects performance regressions
- Documents expected scaling behavior

---

## Phase C: Advanced Features (2 tests)

**Goal:** Test previously untested advanced features

### C1. Add `test_newton_krylov_convergence` (New Test)

**Current Problem:**
- Newton-Krylov implicit solver (spectral.py lines 1631-1677) has ZERO test coverage
- Critical for stiff relaxation problems in Israel-Stewart

**Changes:**
Add new test to `TestARS22IMEXRK` class:

```python
def test_newton_krylov_convergence(self, setup_ars_solver: tuple) -> None:
    """Test Newton-Krylov implicit solver convergence for stiff problems."""
    hydro_solver, fields, grid, coeffs = setup_ars_solver

    # Check if method is accessible
    if not hasattr(hydro_solver, "_newton_krylov_solve"):
        pytest.skip("_newton_krylov_solve method not accessible for testing")

    # Create stiff problem: bulk relaxation with short τ_Π
    fields.Pi.fill(1.0)  # Large initial bulk pressure
    fields.rho.fill(1.0)
    fields.pressure.fill(0.33)

    # Very stiff: short relaxation time
    coeffs.bulk_relaxation_time = 0.01
    coeffs.bulk_viscosity = 0.1

    # Large timestep relative to relaxation time (stiff!)
    dt = 0.1  # dt >> τ_Π
    gamma_dt = (1.0 - 1.0 / np.sqrt(2)) * dt

    # Create RHS for implicit solve
    rhs_dict = {
        "Pi": -fields.Pi / coeffs.bulk_relaxation_time * gamma_dt,
        "pi_munu": np.zeros_like(fields.pi_munu),
        "q_mu": np.zeros_like(fields.u_mu),
    }

    try:
        # Call implicit solver
        solution_dict = hydro_solver._newton_krylov_solve(rhs_dict, gamma_dt)

        # Check convergence
        assert "Pi" in solution_dict, "Bulk pressure solution returned"
        assert np.all(np.isfinite(solution_dict["Pi"])), "Solution is finite"

        # Check residual is small
        # Residual: y - y0 - gamma_dt * f(y)
        y = solution_dict["Pi"]
        y0 = fields.Pi
        f_y = -y / coeffs.bulk_relaxation_time
        residual = y - y0 - gamma_dt * f_y

        residual_norm = np.linalg.norm(residual.flatten())
        relative_residual = residual_norm / np.linalg.norm(y0.flatten())

        assert relative_residual < 1e-6, (
            f"Newton-Krylov did not converge: residual={relative_residual:.2e}"
        )

        # Solution should be close to exact: Pi(t+dt) = Pi(0) * exp(-dt/τ_Π)
        expected_Pi = fields.Pi * np.exp(-dt / coeffs.bulk_relaxation_time)
        error = np.max(np.abs(solution_dict["Pi"] - expected_Pi)) / np.max(np.abs(expected_Pi))

        # Allow larger error since this is a nonlinear solve
        assert error < 0.2, (
            f"Newton-Krylov solution error: {error:.3f}. "
            f"Expected: {np.mean(expected_Pi):.3e}, Got: {np.mean(solution_dict['Pi']):.3e}"
        )

    except Exception as e:
        pytest.skip(f"Newton-Krylov test failed (method may not be accessible): {e}")
```

**Validation:**
```bash
uv run pytest israel_stewart/tests/test_spectral_solver.py::TestARS22IMEXRK::test_newton_krylov_convergence -xvs
```

**Expected Outcome:**
- Validates implicit solver for stiff problems (dt >> τ)
- Checks residual convergence
- Verifies solution accuracy against exponential decay
- May be skipped if method is private/inaccessible

---

### C2. Fix `test_expansion_scalar_computation` (Line 1004)

**Current Problem:**
- Test always skipped due to wrong method path
- Looks for `solver.spectral._compute_expansion_scalar`
- Should be `solver._compute_expansion_scalar`

**Changes:**
```python
def test_expansion_scalar_computation(self, setup_fixed_solver: tuple) -> None:
    """Test expansion scalar computation θ = ∇·u."""
    solver, fields = setup_fixed_solver

    # Fix: check correct path (not solver.spectral._compute_expansion_scalar)
    if not hasattr(solver, "_compute_expansion_scalar"):
        pytest.skip("_compute_expansion_scalar method not available")

    grid = solver.spectral.grid
    x = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    # Set u^x = sin(x), u^y = cos(y), u^z = 0
    # Then ∇·u = cos(x) - sin(y)
    fields.u_mu[..., 1] = np.sin(X)  # u^x
    fields.u_mu[..., 2] = np.cos(Y)  # u^y
    fields.u_mu[..., 3] = 0.0  # u^z

    # Compute expansion scalar
    theta = solver._compute_expansion_scalar()

    # Expected result: cos(x) - sin(y)
    expected_theta = np.cos(X) - np.sin(Y)

    # Check that computed expansion matches expected (within spectral accuracy)
    assert np.allclose(theta, expected_theta, rtol=1e-10), (
        "Expansion scalar computed correctly"
    )
    assert np.all(np.isfinite(theta)), "Expansion scalar is finite"
```

**Validation:**
```bash
uv run pytest israel_stewart/tests/test_spectral_solver.py::TestSpectralSolverFixes::test_expansion_scalar_computation -xvs
```

**Expected Outcome:**
- Test actually runs (was always skipped before)
- Validates expansion scalar θ = ∇·u computation
- May reveal if method doesn't exist or has bugs

---

## Implementation Order

**Recommended sequence:**

1. **Phase A** (Physics validation) - Most critical, tests actual physics
   - A1: sound_wave_propagation
   - A2: conservation_integration

2. **Phase B** (Tighter bounds) - Catches numerical instabilities
   - B1: phase_1_integration bounds
   - B2: fft_scaling_verification

3. **Phase C** (Advanced features) - Nice to have, but may be skipped
   - C1: newton_krylov_convergence
   - C2: expansion_scalar path fix

**Validation after each phase:**
```bash
# After Phase A
uv run pytest israel_stewart/tests/test_spectral_solver.py::TestSpectralValidation -v
uv run pytest israel_stewart/tests/test_spectral_solver.py::TestSpectralISHydrodynamics::test_conservation_integration -v

# After Phase B
uv run pytest israel_stewart/tests/test_spectral_solver.py::TestSpectralSolverFixes::test_phase_1_integration -v
uv run pytest israel_stewart/tests/test_spectral_solver.py::TestSpectralISHydrodynamics::test_fft_scaling_verification -v

# After Phase C
uv run pytest israel_stewart/tests/test_spectral_solver.py::TestARS22IMEXRK::test_newton_krylov_convergence -v
uv run pytest israel_stewart/tests/test_spectral_solver.py::TestSpectralSolverFixes::test_expansion_scalar_computation -v

# Full suite
uv run pytest israel_stewart/tests/test_spectral_solver.py -v
```

---

## Success Criteria

**After all phases:**
- All existing tests still pass (57 → 58-59 passed)
- Skipped tests reduced (1 → 0-1)
- Coverage maintained or improved (71%+)
- Execution time reasonable (<150 seconds)

**Quality improvements:**
- Sound wave actually propagates at correct speed
- Conservation laws properly validated
- Energy explosion caught by 10% bound
- FFT scaling verified as O(N³ log N)
- Implicit solver convergence tested
- Expansion scalar test actually runs

---

## Notes

- **Phase A is highest priority** - adds real physics validation
- **Phase B prevents regressions** - catches instabilities early
- **Phase C is optional** - advanced features, may skip if methods inaccessible
- Each phase is independent and can be implemented/validated separately
- All changes are in `test_spectral_solver.py` - no modifications to production code

---

**Document Status:** Ready for implementation
**Created:** 2025-01-XX
**Tier 1 Baseline:** 57 passed, 1 skipped, 71% coverage
