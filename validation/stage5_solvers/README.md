# Stage 5: Numerical Solvers

**Status**: 🔴 20% Complete (no unit tests)

**Priority**: MEDIUM (needed for Stage 6 benchmarks)

## Goal

Validate numerical integrators (Spectral, RK4, IMEX) on simple test problems with known solutions, independent of Israel-Stewart physics.

## Why This Stage Matters

**"If the solver has bugs, even correct equations will give wrong results."**

Current problem: When benchmarks fail, we don't know if it's:
- Physics equations wrong ❌
- Numerical solver wrong ❌
- Both wrong ❌

**Solution**: Test solvers on simple ODEs/PDEs where analytical solutions are known.

## Acceptance Criteria

- ❌ Spectral: FFT derivatives match analytical (< 10⁻¹⁰)
- ❌ RK4: 4th-order convergence demonstrated
- ❌ IMEX: Splitting error O(dt²) verified
- ❌ All solvers conserve energy in simple tests

## Current Status

### ⚠️ Implementation Exists

1. **Spectral Solver** (`israel_stewart/solvers/spectral.py`)
   - FFT-based spatial derivatives
   - Periodic boundary conditions
   - Linear regime detection (|δρ| < 0.1)
   - Working in benchmarks (sound waves, diffusion)
   - **But**: No unit tests for FFT derivatives alone

2. **RK4 Solver** (`israel_stewart/solvers/spectral.py:_evolve_rk4()`)
   - 4th-order Runge-Kutta time integration
   - Used successfully in sound wave tests
   - **But**: No convergence order verification

3. **IMEX Solver** (`israel_stewart/solvers/spectral.py:_evolve_imex()`)
   - Implicit-explicit splitting
   - Handles stiff relaxation terms
   - **But**: No splitting accuracy tests

### ❌ Missing

**No standalone solver tests**:
- Only tested via full IS benchmarks
- Can't isolate solver errors from physics errors
- No convergence rate measurements
- No systematic debugging

## Test Scripts

### To Be Created: Spectral Solver (2 days)

**Priority 1: FFT Derivatives**
- `spectral/test_fft_derivatives.py`

**Test exact derivatives on smooth functions**:
```python
def test_fft_derivative_sine():
    """∂_x sin(kx) = k·cos(kx) exactly"""
    N = 64
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)

    k = 2.0  # Wavenumber
    f = np.sin(k * x)

    # FFT derivative
    df_dx_numerical = fft_derivative(f, L)

    # Analytical
    df_dx_analytical = k * np.cos(k * x)

    # Should be EXACT (< 1e-10 error for k < N/2)
    np.testing.assert_allclose(df_dx_numerical, df_dx_analytical, atol=1e-10)

def test_fft_derivative_nyquist():
    """Test behavior at Nyquist limit k = N/2"""
    # For k > N/2, expect aliasing
    # For k = N/2, expect exact but phase-sensitive
    # For k < N/2, expect machine precision
```

**Priority 2: Boundary Conditions**
- `spectral/verify_boundary_conditions.py`

**Test grid spacing for periodic BC**:
```python
def test_grid_spacing_periodic():
    """Periodic: dx = L/N (NOT L/(N-1))"""
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2*np.pi)] * 3,
        grid_points=(64, 64, 64),
        boundary_conditions="periodic"
    )

    # Check spacing
    expected_dx = (2*np.pi) / 64
    assert abs(grid.dx - expected_dx) < 1e-15

def test_grid_spacing_dirichlet():
    """Dirichlet: dx = L/(N-1)"""
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2*np.pi)] * 3,
        grid_points=(64, 64, 64),
        boundary_conditions="dirichlet"
    )

    expected_dx = (2*np.pi) / 63
    assert abs(grid.dx - expected_dx) < 1e-15
```

**See `EXPANSION_SCALAR_BUG_FIX.md`**: Wrong BC caused 6% error in derivatives.

**Priority 3: Linear Regime Detection**
- `spectral/test_linear_regime.py`

**Test automatic detection of linear perturbations**:
```python
def test_linear_regime_detection():
    """Small perturbations → linearized momentum conversion"""
    # Setup: small density perturbation
    rho_background = 1.0
    delta_rho = 0.05  # 5% perturbation
    fields.rho[:] = rho_background + delta_rho * np.sin(k*x)

    # Solver should detect |δρ|/ρ < 0.1
    assert hydro._is_linear_regime(fields) == True

def test_nonlinear_regime_detection():
    """Large perturbations → full nonlinear conversion"""
    delta_rho = 0.2  # 20% perturbation
    fields.rho[:] = rho_background + delta_rho * np.sin(k*x)

    assert hydro._is_linear_regime(fields) == False
```

**Why this matters**: Nonlinear conversion creates spurious 2nd harmonics in linear regime.

### To Be Created: RK4 Solver (1 day)

**Priority 1: Convergence Order**
- `rk4/test_convergence_order.py`

**Test on simple ODE with known solution**:
```python
def test_rk4_convergence():
    """RK4 should have error ∝ dt⁴"""
    # Simple ODE: dy/dt = -λy, y(0) = 1
    # Exact solution: y(t) = exp(-λt)
    lambda_ = 1.0
    t_final = 1.0

    errors = []
    dts = [0.1, 0.05, 0.025, 0.0125]

    for dt in dts:
        y_numerical = rk4_integrate(lambda y: -lambda_*y, y0=1.0, dt=dt, t_final=t_final)
        y_exact = np.exp(-lambda_ * t_final)
        error = abs(y_numerical - y_exact)
        errors.append(error)

    # Check: error ∝ dt⁴
    for i in range(len(dts)-1):
        ratio = errors[i] / errors[i+1]
        expected_ratio = (dts[i] / dts[i+1])**4  # 4th-order
        # Should be ~16 for dt halving
        assert abs(ratio - expected_ratio) / expected_ratio < 0.1
```

**Priority 2: Stability Region**
- `rk4/verify_stability.py`

**Test stability for stiff problems**:
```python
def test_rk4_stability_limit():
    """RK4 stable for |λ·dt| < 2.785"""
    lambda_ = 10.0  # Stiff problem

    # Stable timestep
    dt_stable = 0.2 / lambda_  # |λdt| = 2.0 < 2.785
    y_stable = rk4_integrate(lambda y: -lambda_*y, y0=1.0, dt=dt_stable, t_final=1.0)
    assert y_stable > 0 and y_stable < 1  # Decaying but stable

    # Unstable timestep
    dt_unstable = 0.3 / lambda_  # |λdt| = 3.0 > 2.785
    y_unstable = rk4_integrate(lambda y: -lambda_*y, y0=1.0, dt=dt_unstable, t_final=1.0)
    # May oscillate or blow up
```

### To Be Created: IMEX Solver (2 days)

**Priority 1: Splitting Accuracy**
- `imex/test_splitting_accuracy.py`

**Test on problem with fast + slow modes**:
```python
def test_imex_splitting():
    """IMEX: fast modes implicit, slow modes explicit"""
    # Test problem: dy/dt = -λ_fast·y - λ_slow·y
    # With λ_fast >> λ_slow

    lambda_fast = 100.0   # Stiff (implicit)
    lambda_slow = 1.0     # Non-stiff (explicit)

    # IMEX should handle this efficiently
    y_imex = imex_integrate(
        explicit_rhs=lambda y: -lambda_slow * y,
        implicit_rhs=lambda y: -lambda_fast * y,
        y0=1.0,
        dt=0.1,  # Large timestep OK for IMEX
        t_final=1.0
    )

    # Compare to exact
    y_exact = np.exp(-(lambda_fast + lambda_slow) * 1.0)

    # IMEX should have O(dt²) error (2nd-order splitting)
    assert abs(y_imex - y_exact) < 0.1 * 0.1**2  # C·dt²
```

**Priority 2: Stiff Stability**
- `imex/verify_stiff_stability.py`

**Test unconditional stability for stiff terms**:
```python
def test_imex_stiff_stability():
    """IMEX should be stable even with dt >> 1/λ_fast"""
    lambda_fast = 1000.0  # Very stiff

    # RK4 would require dt < 0.003
    # IMEX should work with dt = 0.1
    dt_large = 0.1

    y_imex = imex_integrate(
        explicit_rhs=lambda y: 0.0,
        implicit_rhs=lambda y: -lambda_fast * y,
        y0=1.0,
        dt=dt_large,
        t_final=1.0
    )

    # Should decay smoothly without oscillation
    assert 0 < y_imex < 1
```

## Known Issues

### 1. High-k Instability (RESOLVED)

**Issue**: Spectral solver showed exponential growth for k > 4 GeV

**Root cause**: Outside Israel-Stewart regime (|τω| > 1)

**Resolution**: Document in `HIGH_K_INSTABILITY_RESOLUTION.md`, add regime checking

**Test needed**: `spectral/test_regime_validity.py` to verify warning triggers

### 2. Expansion Scalar Bug (FIXED)

**Issue**: 6% error in θ = ∇·v calculation

**Root cause**: Used `boundary_conditions="dirichlet"` (dx=L/(N-1)) instead of `"periodic"` (dx=L/N) for spectral solver

**Fix**: Enforce `boundary_conditions="periodic"` for all spectral tests

**Test needed**: `spectral/verify_boundary_conditions.py` (see above)

### 3. Linear Regime Spurious Harmonics (FIXED)

**Issue**: Nonlinear momentum→velocity conversion created 2nd harmonics in linear sound waves

**Fix**: Auto-detect linear regime (|δρ|<0.1) and use linearized conversion

**Test needed**: `spectral/test_linear_regime.py` (see above)

## Remaining Work

**Time estimate**: 5 days total
- 2 days: Spectral solver tests (FFT derivatives, BC, linear regime)
- 1 day: RK4 tests (convergence order, stability)
- 2 days: IMEX tests (splitting accuracy, stiff stability)

**Breakdown**:
1. Create 7 test scripts (0.5 days)
2. Implement test cases (2.5 days)
3. Debug failures (2 days)

## References

- **Spectral solver**: `israel_stewart/solvers/spectral.py`
- **RK4 implementation**: `israel_stewart/solvers/spectral.py:_evolve_rk4()`
- **IMEX implementation**: `israel_stewart/solvers/spectral.py:_evolve_imex()`
- **Boundary conditions bug**: `EXPANSION_SCALAR_BUG_FIX.md`
- **High-k instability**: `HIGH_K_INSTABILITY_RESOLUTION.md`
- **Linear regime**: `CLAUDE.md` - "Linear Regime Detection" section

## Success Metrics

**Before** (2025-10-18):
- Solvers exist and work in some benchmarks ✓
- **But**: No standalone unit tests ❌
- **But**: No convergence order verification ❌
- **But**: Can't isolate solver bugs from physics bugs ❌

**Target**:
- 15+ solver tests passing ✓
- FFT derivatives < 1e-10 error ✓
- RK4 4th-order convergence verified ✓
- IMEX O(dt²) splitting error verified ✓
- Can debug solver independently from physics ✓

## Next Steps

1. **Create spectral test suite** (2 days)
   - FFT derivative accuracy
   - Boundary condition verification
   - Linear regime detection

2. **Create RK4 test suite** (1 day)
   - Convergence order on simple ODE
   - Stability region verification

3. **Create IMEX test suite** (2 days)
   - Splitting accuracy for stiff+non-stiff
   - Unconditional stability for implicit part

4. **Document findings** (0.5 days)
   - Write `results/solver_validation.md`
   - Update VALIDATION_ROADMAP.md

**After Stage 5 complete**: Can confidently attribute benchmark failures to physics (not numerics).
