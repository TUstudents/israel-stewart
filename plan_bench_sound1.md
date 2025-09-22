# Plan: Fix Sound Wave Benchmark Analytical-Numerical Discrepancies

## **Big Picture Problem**
The sound wave benchmark has fundamental issues where analytical and numerical solvers disagree due to three critical bugs:

1. **Sound speed formula underestimates c_s by 25%** (using c_s² = p/(ρ+p) instead of ∂p/∂ε)
2. **Polynomial coefficients in wrong order** for np.roots() calls
3. **Wrong solver API** for time evolution (calling full evolution instead of single timestep)

Neither analytical curves nor numerical measurements can currently be trusted.

## **Phase 1: Fix Sound Speed Calculation (High Priority)**

### Issue: `_estimate_sound_speed()` uses incorrect thermodynamic relation
- **Current**: `cs_squared = p / (rho + p)` → gives 1/4 for radiation background
- **Should be**: `cs_squared = ∂p/∂ε = 1/3` for conformal radiation fluid
- **Impact**: Affects all root-finding initial guesses, phase velocities, dispersion checks

### Fix:
- Replace lines 214-219 in `_estimate_sound_speed()` with proper thermodynamic derivative
- For current background (ρ=1.0, p=1/3): return `1.0/3.0` directly
- Keep viscous corrections separate from ideal sound speed

## **Phase 2: Fix Polynomial Coefficient Ordering (High Priority)**

### Issue: `np.roots()` expects coefficients in descending order [a_n, ..., a_0]
- **Current**: `_build_characteristic_polynomial()` stores as [a_0, a_1, ..., a_n]
- **Impact**: All polynomial-based dispersion solutions are wrong, even ideal fluid limit

### Fix:
- **Preferred**: Delete legacy polynomial path entirely, use only determinant-based solver
- **Alternative**: Reverse coefficient array before calling `np.roots(coeffs[::-1])`
- Remove `solve_exact_dispersion()` and defer everything to `SoundWaveAnalysis`

## **Phase 3: Fix Numerical Time Evolution API (High Priority)**

### Issue: Wrong solver method in benchmark time loop
- **Current**: `self.solver.evolve(dt_cfl)` resets integrator each call
- **Should be**: `self.solver.time_step(dt_cfl)` for single timestep advance

### Fix:
- Replace line 1221: `self.solver.evolve(dt_cfl)` → `self.solver.time_step(dt_cfl)`
- **Alternative**: Use `evolve()` once for full simulation with output callback for monitoring

## **Phase 4: Consolidate Analytical Paths (Medium Priority)**

### Issue: Two analytical approaches giving different results
- **New**: Determinant-based root finding (correct physics, wrong sound speed)
- **Old**: Polynomial method (wrong coefficient order)

### Fix:
- Remove `DispersionRelation.solve_exact_dispersion()` method
- Remove `_build_characteristic_polynomial()` method
- Make all analytical calls go through `SoundWaveAnalysis.analyze_dispersion_relation()`
- Update `WaveTestSuite` to use unified analytical path

## **Phase 5: Validation and Testing (Medium Priority)**

### Test Fixes:
1. **Sound speed verification**: c_s = 1/√3 ≈ 0.5774 for radiation background
2. **Polynomial vs determinant**: Ensure both give same results when fixed
3. **Time evolution**: Verify numerical evolution advances correctly
4. **End-to-end**: Run complete benchmark and verify analytical vs numerical agreement

### Expected Outcomes:
- Sound speed error reduces from 15% to <1%
- Analytical solver gives consistent results across all methods
- Numerical simulation properly evolves and produces meaningful time series
- Complete benchmark shows realistic analytical-numerical comparison

## **Phase 6: Performance and Documentation (Low Priority)**

### Optimizations:
- Remove redundant analytical code paths
- Update documentation to reflect unified approach
- Add physics validation checks for sound speed calculation

### Documentation Updates:
- Update `corrected_sound_wave_assessment.md` with post-fix results
- Document correct API usage for numerical benchmarks
- Add troubleshooting guide for common physics errors

## **Implementation Order:**
1. **Sound speed fix** (immediate impact on all subsequent calculations)
2. **Solver API fix** (enables proper numerical benchmarking)
3. **Remove polynomial path** (eliminates analytical inconsistencies)
4. **Comprehensive testing** (validates all fixes work together)
5. **Documentation update** (records corrected status)

## **Technical Details**

### Sound Speed Calculation Fix
**Location**: `israel_stewart/benchmarks/sound_waves.py:203-221`

**Current problematic code**:
```python
def _estimate_sound_speed(self) -> float:
    # For radiation: c_s^2 = 1/3
    # Include viscous corrections
    rho = np.mean(self.background_fields.rho)
    p = np.mean(self.background_fields.pressure)

    # Ideal gas sound speed - WRONG FORMULA
    cs_squared = p / (rho + p)  # gives 1/4 instead of 1/3
```

**Correct implementation**:
```python
def _estimate_sound_speed(self) -> float:
    """Estimate sound speed from thermodynamic properties."""
    rho = np.mean(self.background_fields.rho)
    p = np.mean(self.background_fields.pressure)

    if rho <= 0:
        return 1.0 / np.sqrt(3.0)

    # Proper thermodynamic sound speed: c_s² = ∂p/∂ε
    # For conformal radiation fluid: p = ε/3 → c_s² = 1/3
    cs_squared = p / rho  # This gives 1/3 for radiation background

    return np.sqrt(max(0.0, min(1.0, cs_squared)))
```

### Solver API Fix
**Location**: `israel_stewart/benchmarks/sound_waves.py:1221`

**Current problematic code**:
```python
# Time evolution loop
for _step in range(n_steps):
    # ... record time series ...

    # Evolve fields one timestep
    try:
        self.solver.evolve(dt_cfl)  # WRONG: resets integrator each call
        current_time += dt_cfl
```

**Correct implementation**:
```python
# Time evolution loop
for _step in range(n_steps):
    # ... record time series ...

    # Evolve fields one timestep
    try:
        self.solver.time_step(dt_cfl)  # CORRECT: single timestep advance
        current_time += dt_cfl
```

### Polynomial Path Removal
**Location**: `israel_stewart/benchmarks/sound_waves.py:504-576`

**Remove these methods**:
- `DispersionRelation.solve_exact_dispersion()`
- `DispersionRelation._build_characteristic_polynomial()`

**Update calls in**:
- `analyze_dispersion_curve()` method
- Any `WaveTestSuite` methods that use polynomial path

This plan addresses the root causes of analytical-numerical disagreement and establishes a reliable foundation for Israel-Stewart hydrodynamics validation.