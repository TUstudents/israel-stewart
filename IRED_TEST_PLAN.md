# IReD Implementation: Critical Test Analysis & Rigorous Test Plan

## Executive Summary

**Current State**: 66 IReD-specific tests exist, but they test the WRONG things:
- ✅ Coefficient lookup tables (redundantly tested 4+ times)
- ✅ Object creation (smoke tests, no physics)
- ❌ **CRITICAL MISSING**: Regime-valid evolution tests
- ❌ **CRITICAL MISSING**: Conservation laws with IReD
- ❌ **CRITICAL MISSING**: Analytical validation
- ❌ **CRITICAL MISSING**: Regime enforcement

**Problem**: Tests run in invalid regime (|τω| = 10-30 >> 1) or stop after 1-2 timesteps.

---

## Current Test Coverage Analysis

**Last updated**: 2025-10-17 (Phase 4 consolidation complete)

### Existing Tests (58 total after Phase 4 consolidation)

#### 1. `test_ired_coefficients.py` (26 tests, -3 duplicates removed)

**What they cover**:
- ✅ Coefficient values match IReD Tables III-IV
- ✅ Dimensional scaling (η ∝ T/σ, λ ∝ 1/T³)
- ✅ Truncation convergence (14→23→32→41)
- ✅ Input validation (positive T, σ)
- ✅ Physical consistency (positive viscosities)

**What they DON'T cover**:
- ❌ Evolution with these coefficients
- ❌ Numerical stability
- ❌ Physics accuracy

**Duplicates** ~~identified~~ **REMOVED (Phase 4 complete)**:
- ~~`test_shear_viscosity_value`~~ → Removed (covered by `test_validate_against_ired_paper`)
- ~~`test_shear_relaxation_time_value`~~ → Removed (covered by `test_validate_against_ired_paper`)
- ~~`test_diffusion_coefficient_value`~~ → Removed (covered by `test_validate_against_ired_paper`)
- **Result**: Kept comprehensive `test_validate_against_ired_paper`, removed 3 redundant tests ✅

#### 2. `test_ired_benchmarks.py` (30 tests, -5 duplicates removed)

**What they cover**:
- ✅ Factory functions create objects
- ✅ Coefficients transferred correctly
- ✅ Second-order couplings included
- ✅ Smoke tests (1-2 timesteps without crash)

**What they DON'T cover**:
- ❌ **CRITICAL**: Long-time evolution (all tests stop after 1 step)
- ❌ **CRITICAL**: Regime validity (tests acknowledge violations but continue)
- ❌ Conservation during evolution
- ❌ Comparison with analytical solutions

**Test results**:
```
Test                              | Duration | Regime  | Physics Validated?
----------------------------------|----------|---------|-------------------
test_bjorken_with_ired_smoke      | 1 step   | Unknown | NO
test_sound_wave_ired_creation     | 0 steps  | Unknown | NO
test_diffusion_benchmark_creation | 0 steps  | Unknown | NO
```

**Duplicates** ~~identified~~ **REMOVED (Phase 4 complete)**:
- ~~`test_ired_temperature_scaling`~~ → Removed (covered by `test_shear_viscosity_scaling` in coefficients)
- ~~`test_ired_cross_section_scaling`~~ → Removed (covered by `test_shear_viscosity_scaling` in coefficients)
- ~~`test_ired_validation_against_paper`~~ → Removed (covered by comprehensive test in coefficients)
- Creation tests kept (test different APIs: `coefficients` vs `transport_coeffs`, not true duplicates)
- **Result**: Removed 5 redundant tests, kept well-organized creation tests ✅

#### 3. `verify_ired_implementation.py` (12 checks, not in test suite)

**What it covers**:
- ✅ Form B structure (regex on source code)
- ✅ Regime warning exists
- ✅ Required fields present
- ✅ Equilibrium RHS = 0 (static)

**What it DON'T cover**:
- ❌ Evolution correctness
- ❌ Regime ENFORCEMENT (just warns)
- ❌ Numerical stability
- ❌ Comparison with IReD paper results

---

## Critical Gaps Identified

### Gap 1: NO REGIME-VALID EVOLUTION TESTS ❌

**The Problem**:
```
Test File                  | Grid  | τ (fm/c) | k_max (GeV) | |τω|  | Valid? | Tested?
---------------------------|-------|----------|-------------|-------|--------|--------
validate_diffusion_...     | 32³   | 2.1      | 24          | 29.4  | ❌ >>1 | Full (GARBAGE)
test_eigenmode_...         | 32³×16| 1.0      | 24          | 13.9  | ❌ >>1 | Full (FAILED)
validate_bjorken_...       | ?     | ?        | ?           | ?     | ❌     | TIMEOUT
test_ired_benchmarks (all) | varies| varies   | varies      | varies| ❌     | 1-2 steps only
```

**Requirement**: |τω| < 1 (Wagner & Gavassino 2024)

**Current reality**: ALL tests either:
1. Run 1-2 steps (don't test evolution)
2. Run full evolution in INVALID regime (produce garbage)

**Impact**:
- Diffusion validation: 5,771% error (claimed "coupled modes", actually regime violation)
- Eigenmode test: 60% drift (claimed "under investigation", actually regime violation)
- Bjorken: Untested (times out, never validated)

### Gap 2: NO CONSERVATION TESTS WITH IRED ❌

**The Problem**: `test_dynamic_conservation.py` (12 tests) uses PHENOMENOLOGICAL coefficients, not IReD.

**Missing**:
- Energy conservation during IReD evolution
- Momentum conservation with IReD shear viscosity
- Particle number conservation with IReD diffusion

**Why critical**: If conservation fails, ALL physics is wrong.

### Gap 3: NO ANALYTICAL VALIDATION ❌

**The Problem**: No tests verify IReD produces correct physics.

**Missing**:
- Bjorken: T(τ) vs analytical Israel-Stewart
- Sound waves: ω(k) vs dispersion relation
- Diffusion: Γ vs Dk²
- Fick's law: V^i = -D∇^i(μ/T)

**Existing validation scripts** (`validate_*.py`):
- ❌ Not in test suite (manual only)
- ❌ Never successfully run (Bjorken times out)
- ❌ Produce garbage (diffusion 5,771% error)

### Gap 4: NO REGIME ENFORCEMENT ❌

**The Problem**: Tests warn about regime violations but continue anyway.

**Current behavior**:
```python
WARNING: |τω| = 29.42 > 1. Outside Israel-Stewart regime...
[continues to produce garbage results]
[test passes despite invalid physics]
```

**Should be**:
```python
if regime_parameter > 1.0:
    pytest.skip("Outside IS regime: |τω|={:.2f} > 1".format(regime_parameter))
```

### ~~Gap 5: DUPLICATES/REDUNDANCY~~ ✅ **RESOLVED (Phase 4 complete)**

**Duplicates removed**:
1. Coefficient value tests: 3 tests removed (covered by comprehensive validation) ✅
2. Benchmark scaling tests: 2 tests removed (covered by coefficient scaling tests) ✅
3. Benchmark validation test: 1 test removed (covered by coefficient validation) ✅
4. Creation tests: Assessed and kept (test different APIs, not true duplicates) ✅

**Net reduction**: 8 redundant tests removed (-12.5% from 64 → 56 tests) ✅

---

## Regime-Valid Parameter Selection

### The Challenge

For IReD hard sphere gas at T=0.4 GeV, σ=1 fm²:
- Mean free path: λ_mfp ≈ 26 fm
- Shear relaxation: τ_π ≈ 43 fm/c
- Diffusion relaxation: τ_V ≈ 70 fm/c

For regime validity |τω| < 1 with ω ≈ k·c_s (c_s ≈ 0.577):
- Need k < 1/(τ·c_s) ≈ 1/(40 × 0.577) ≈ 0.043 GeV ≈ 0.22 fm⁻¹

For spectral grid N × N × N with domain L:
- k_max = π·N / L

**Options**:

#### Option A: Small τ (Phenomenological, NOT IReD)
```python
τ = 0.05 fm/c (phenomenological)
k_max < 35 → N = 64³ works
```
- ✅ Tests evolution accuracy
- ❌ Not testing IReD coefficients

#### Option B: Coarse Grid (IReD, Marginal)
```python
σ = 100 fm² → τ_V ≈ 0.7 fm/c
N = 8³, L = 2π → k_max ≈ 4 fm⁻¹ ≈ 0.78 GeV
|τω| ≈ 0.7 × 0.577 × 0.78 ≈ 0.3 < 1 ✓
```
- ✅ IReD coefficients
- ✅ Regime valid
- ⚠️ Poor resolution (only 8 points)

#### Option C: Large Domain (IReD, Best)
```python
σ = 100 fm² → τ_V ≈ 0.7 fm/c
N = 32³, L = 20π → k_max ≈ 1.6/L ≈ 0.26 GeV
|τω| ≈ 0.7 × 0.577 × 0.26 ≈ 0.1 < 1 ✓
```
- ✅ IReD coefficients
- ✅ Regime valid
- ✅ Reasonable resolution

**Recommendation**: Use Option C for regime-valid IReD tests

---

## Proposed Rigorous Test Plan

### Phase 1: Fix Regime Violations (IMMEDIATE)

#### Task 1.1: Create regime-valid fixtures
```python
# israel_stewart/tests/conftest.py

@pytest.fixture
def ired_regime_valid_coarse():
    """IReD with 8³ grid, regime-valid."""
    return {
        'temperature': 0.4,
        'cross_section': 100.0,  # Large σ → small τ
        'grid_points': (8, 8, 8),
        'domain_size': 2*np.pi,
        'expected_regime': 0.3,  # |τω| < 1
    }

@pytest.fixture
def ired_regime_valid_large_domain():
    """IReD with 32³ grid, large domain, regime-valid."""
    return {
        'temperature': 0.4,
        'cross_section': 100.0,
        'grid_points': (32, 32, 32),
        'domain_size': 20*np.pi,  # 10× larger
        'expected_regime': 0.1,  # |τω| << 1
    }
```

#### Task 1.2: Add regime validation helper
```python
# israel_stewart/tests/test_helpers.py

def check_regime_validity(grid, transport_coeffs, max_allowed=1.0):
    """Check if parameters are within IS regime."""
    k_max = compute_k_max(grid)
    c_s = 1.0 / np.sqrt(3.0)
    omega_max = k_max * c_s
    tau_max = max(
        transport_coeffs.shear_relaxation_time,
        transport_coeffs.bulk_relaxation_time,
        getattr(transport_coeffs, 'diffusion_relaxation_time', 0)
    )
    regime = abs(tau_max * omega_max)

    if regime > max_allowed:
        pytest.skip(f"Outside IS regime: |τω|={regime:.2f} > {max_allowed}")

    return regime
```

#### Task 1.3: Mark known invalid tests
```python
# tests/test_eigenmode_preservation.py

@pytest.mark.slow
@pytest.mark.xfail(reason="Regime violation: |τω|=13.9 > 1 (32³ grid too fine for τ=1.0)")
def test_eigenmode_ratios_are_preserved():
    ...
```

### Phase 2: Add Conservation Tests (CRITICAL)

**New file**: `israel_stewart/tests/test_ired_conservation.py`

```python
class TestIReDConservation:
    """Test conservation laws with IReD transport coefficients."""

    def test_energy_conservation_regime_valid(self, ired_regime_valid_coarse):
        """Test energy conservation during evolution with IReD."""
        # Setup with regime-valid parameters
        benchmark, ired = create_benchmark_with_ired(**ired_regime_valid_coarse)

        # Check regime
        regime = check_regime_validity(benchmark.grid, benchmark.coefficients)

        # Initial energy
        E0 = compute_total_energy(benchmark.fields, benchmark.grid)

        # Evolve 100 steps
        for _ in range(100):
            benchmark.solver.time_step(dt=0.01)

        # Final energy
        Ef = compute_total_energy(benchmark.solver.fields, benchmark.grid)

        # Conservation check
        relative_change = abs(Ef - E0) / E0
        assert relative_change < 1e-3, f"Energy not conserved: ΔE/E = {relative_change:.2e}"

    def test_particle_conservation_diffusion(self, ired_regime_valid_large_domain):
        """Test particle conservation during diffusion with IReD."""
        # Similar structure...

    def test_momentum_conservation_sound_wave(self, ired_regime_valid_large_domain):
        """Test momentum conservation during sound wave with IReD."""
        # Similar structure...
```

### Phase 3: Add Analytical Validation Tests

**New file**: `israel_stewart/tests/test_ired_analytical.py`

```python
class TestIReDAnalyticalValidation:
    """Compare IReD evolution with analytical predictions."""

    @pytest.mark.slow
    def test_bjorken_temperature_vs_analytical(self, ired_regime_valid_coarse):
        """Test Bjorken T(τ) matches analytical IS solution."""
        # Setup Bjorken with regime-valid IReD
        benchmark, ired = create_bjorken_benchmark_with_ired(
            T0=0.4,
            tau0=0.6,
            cross_section=ired_regime_valid_coarse['cross_section'],
            grid_points=ired_regime_valid_coarse['grid_points']
        )

        # Evolve to τ=3.0 fm/c
        result = benchmark.run_numerical_simulation(
            final_time=3.0,
            timestep=0.1,
            method='rk4'
        )

        # Compare with analytical
        for t, T_num in zip(result['time'], result['temperature']):
            T_analytical = benchmark.analytical.israel_stewart_solution(t)['temperature']
            error = abs(T_num - T_analytical) / T_analytical
            assert error < 0.05, f"T(τ={t:.2f}): error = {error:.1%} > 5%"

    @pytest.mark.slow
    def test_diffusion_decay_rate(self, ired_regime_valid_large_domain):
        """Test diffusion decay rate Γ = Dk²."""
        # Setup diffusion with regime-valid IReD
        benchmark, ired = create_diffusion_benchmark_with_ired(
            wave_number=0.5,  # Low k for regime validity
            **ired_regime_valid_large_domain
        )

        # Evolve and measure decay
        times, amplitudes = evolve_and_extract_amplitude(benchmark, n_periods=5)

        # Fit exponential decay
        Gamma_measured = fit_exponential_decay(times, amplitudes)
        Gamma_expected = benchmark.analytical.damping_rate()

        error = abs(Gamma_measured - Gamma_expected) / Gamma_expected
        assert error < 0.1, f"Decay rate error: {error:.1%} > 10%"
```

### Phase 4: Remove Duplicates ✅ **COMPLETE (2025-10-17)**

**Completed Actions**:
1. ✅ In `test_ired_coefficients.py` (26 tests, was 29):
   - Removed `test_shear_viscosity_value` (covered by `test_validate_against_ired_paper`)
   - Removed `test_shear_relaxation_time_value` (covered by `test_validate_against_ired_paper`)
   - Removed `test_diffusion_coefficient_value` (covered by comprehensive validation)

2. ✅ In `test_ired_benchmarks.py` (30 tests, was 35):
   - Removed `test_ired_temperature_scaling` (covered by coefficient scaling tests)
   - Removed `test_ired_cross_section_scaling` (covered by coefficient scaling tests)
   - Removed `test_ired_validation_against_paper` (covered by coefficient validation)
   - Assessed creation tests: kept (test different APIs, not true duplicates)

3. ✅ **Net reduction**: 8 redundant tests removed (-12.5%)
   - **Before**: 64 tests (29 + 35)
   - **After**: 56 tests (26 + 30)
   - **All tests passing**: 56/56 (100% ✅)

### Phase 5: Stability/Causality Tests

**New file**: `israel_stewart/tests/test_ired_stability.py`

```python
class TestIReDStability:
    """Test IReD stability and causality properties."""

    def test_no_parabolic_instability(self):
        """Verify K=0 structure (no parabolic terms)."""
        # IReD claims K^{μ₁...μℓ} = 0
        # Test: high-k modes should NOT grow exponentially
        ...

    def test_causality_signal_speed(self):
        """Test signal propagation speed < c."""
        # Extract group velocity from dispersion relation
        # Should satisfy v_g < 1
        ...
```

---

## Success Metrics

### Before (Current State)
- **Total tests**: 66
- **Regime-valid evolution tests**: 0 ❌
- **Conservation tests with IReD**: 0 ❌
- **Analytical validation tests**: 0 ❌
- **Regime enforcement**: 0 ❌
- **Duplicate tests**: ~12-15
- **Tests that actually validate physics**: ~10

### After (Target State)
- **Total tests**: ~70 (66 - 12 duplicates + 16 new critical tests)
- **Regime-valid evolution tests**: 3 ✅
- **Conservation tests with IReD**: 3 ✅
- **Analytical validation tests**: 6 ✅
- **Regime enforcement**: All tests check regime ✅
- **Duplicate tests**: 0 ✅
- **Tests that validate physics**: ~40 ✅

### Quality Metrics
- **Test suite runtime**: < 5 minutes (with slow tests: < 30 minutes)
- **Coverage**: Core IReD paths 100%
- **False positives**: 0 (all passing tests mean physics is correct)
- **False negatives**: 0 (all failing tests indicate real bugs)

---

## Implementation Timeline

### Week 1: Critical Fixes
- Day 1-2: Create regime-valid fixtures, add regime checking helper
- Day 3-4: Mark invalid tests with `@pytest.mark.xfail`
- Day 5: Add 3 conservation tests (energy, momentum, particles)

### Week 2: Analytical Validation
- Day 1-2: Bjorken analytical tests (T(τ), π^ηη(τ))
- Day 3-4: Sound wave analytical tests (ω(k), Γ(k))
- Day 5: Diffusion analytical tests (Γ=Dk², Fick's law)

### Week 3: Cleanup & Stability
- Day 1-2: Remove duplicate tests
- Day 3-4: Add stability tests
- Day 5: Documentation update

### Week 4: Validation & Review
- Run full test suite
- Fix any failures
- Document regime-valid parameter space
- Update PHASE_16 status (mark as INCOMPLETE, needs redo)

---

## Immediate Next Steps (Today)

1. **Run baseline** (establish what currently works):
   ```bash
   uv run pytest israel_stewart/tests/test_ired_coefficients.py -v --timeout=300
   uv run pytest israel_stewart/tests/test_ired_benchmarks.py -v --timeout=300
   ```

2. **Create regime-valid fixtures** (`conftest.py`)

3. **Implement regime check helper** (`test_helpers.py`)

4. **Add 1 critical test from each category**:
   - Conservation: `test_energy_conservation_regime_valid`
   - Analytical: `test_bjorken_temperature_vs_analytical`
   - Regime enforcement: `test_regime_violation_skips_test`

5. **Mark known failures**:
   - `test_eigenmode_ratios_are_preserved` → `@pytest.mark.xfail`
   - Update validation scripts with regime warnings

---

## Open Questions / Decisions Needed

1. **Should we keep regime-violating tests?**
   - Option A: Mark `@pytest.mark.xfail` (keep for documentation)
   - Option B: Delete entirely (cleaner)
   - **Recommendation**: Keep with `@xfail` to document what DOESN'T work

2. **Acceptable error tolerances?**
   - Current proposal: 5% for temperature, 10% for viscous stresses, 10% for decay rates
   - These are reasonable for numerical discretization
   - Tighter tolerances (<1%) require convergence studies

3. **Should IReD tests use phenomenological τ for regime validity?**
   - Pro: Tests evolution accuracy
   - Con: Not testing IReD coefficients
   - **Recommendation**: Have BOTH test types, clearly labeled

4. **How to handle computationally expensive tests?**
   - Mark with `@pytest.mark.slow`
   - Run in CI only on nightly builds
   - Provide fast "smoke" versions for rapid iteration

---

## References

- **IReD Paper**: Wagner, Palermo, Ambrus (2022), arXiv:2208.02506
- **Regime Validity**: Wagner & Gavassino (2024), arXiv:2309.14828v2
- **Testing Best Practices**: "Untested code is broken code" - this document
