# Stage 6: Analytical Benchmarks

**Status**: 🟡 50% Complete (3/6 tests passing)

**Priority**: HIGH (validates full system against known solutions)

## Goal

Validate the complete Israel-Stewart implementation (equations + solvers) against analytical solutions for canonical problems: sound waves, Bjorken flow, and particle diffusion.

## Acceptance Criteria

- ✅ Sound wave frequency correct (< 5% error)
- ⚠️ Sound wave damping matches theory (< 15% error)
- ❌ Bjorken T(τ) matches analytical (< 5% error)
- ✅ Bjorken shear stress evolves correctly
- ✅ Diffusion Fick's law V^i = -D∇^i(μ/T) (< 10% error)
- ❌ Diffusion decay rate Γ = Dk² (< factor 2.5×)

## Current Status

### ✅ Passing Tests (3/6)

1. **Sound Wave Frequency** (`test_sound_wave_frequency`)
   - Measures ω(k) from spatial phase shift
   - ✅ Within 5% of analytical ω = k·c_s
   - **Recent fix**: Accept nearly-ideal modes where |Γ/ω| < 1%

2. **Bjorken Shear Stress** (`test_bjorken_shear_stress_evolution`)
   - Validates π^μν evolution in boost-invariant expansion
   - ✅ Follows relaxation equation correctly

3. **Diffusion Fick's Law** (`test_ficks_law`)
   - Validates V^i = -D∇^i(μ/T) in diffusion regime
   - ✅ Within 10% of analytical prediction

### ⚠️ XFAIL: Regime Validity Paradox (1/6)

4. **Sound Wave Damping** (`test_sound_wave_damping`)
   - **Status**: Marked `@pytest.mark.xfail`
   - **Issue**: Cannot simultaneously achieve:
     - |τω| < 1 (regime valid)
     - Γ measurable (Γ > 10⁻⁵ GeV)
   - **Root cause**: Physical limitation of hard sphere gas, NOT a code bug
   - **See**: `stage4_dispersion/README.md` for detailed analysis

**Resolution options** (future work):
- Option B: Find intermediate σ where both conditions met (if exists)
- Option C: Compare to full Israel-Stewart formula (not just Navier-Stokes)

### ❌ Failing Tests (2/6)

5. **Bjorken Temperature Evolution** (`test_bjorken_temperature_evolution`)
   - **Status**: Marked `@pytest.mark.xfail`
   - **Error**: 71% error in T(τ) evolution
   - **Root cause**: Conservation laws not updating temperature field
   - **Fix needed**: Update T from energy density in spectral solver
   - **Time estimate**: 2 days

6. **Diffusion Decay Rate** (`test_diffusion_decay_rate`)
   - **Status**: Marked `@pytest.mark.xfail`
   - **Error**: Long-time numerical instability at t ~ 11 GeV⁻¹
   - **Symptoms**: `overflow encountered in square`, `invalid value in fft`
   - **Root cause**: Numerical instability in extended evolution
   - **Fix needed**: Investigate timestep sensitivity, roundoff accumulation
   - **Time estimate**: 3 days

## Test Scripts

### Existing Tests

**Main test suite**: `tests/test_ired_analytical.py`

Contains all 6 benchmark tests:
- `test_sound_wave_frequency` ✅
- `test_sound_wave_damping` ⚠️ (xfail - Regime Paradox)
- `test_bjorken_shear_stress_evolution` ✅
- `test_bjorken_temperature_evolution` ❌ (xfail - T not evolving)
- `test_ficks_law` ✅
- `test_diffusion_decay_rate` ❌ (xfail - long-time instability)

**Benchmark implementations**:
- `israel_stewart/benchmarks/sound_waves.py` - Linear wave propagation
- `israel_stewart/benchmarks/bjorken_flow.py` - 1D boost-invariant expansion
- `israel_stewart/benchmarks/diffusion.py` - Particle diffusion

### Diagnostic Scripts (to be moved from root)

**Sound waves**:
- `analyze_sound_wave_*.py` - Various sound wave diagnostics
- `compare_sound_wave_*.py` - Compare numerical vs analytical
- `debug_dispersion_k05.py` - Already moved to `stage4_dispersion/`

**Bjorken**:
- `analyze_bjorken_*.py` - Temperature evolution diagnostics
- `check_bjorken_*.py` - Verify boost invariance

**Diffusion**:
- `analyze_diffusion_*.py` - Decay rate diagnostics
- `debug_diffusion_*.py` - Long-time stability analysis

## Detailed Test Analysis

### Test 1: Sound Wave Frequency ✅

**Test**: `test_sound_wave_frequency` in `test_ired_analytical.py:191-234`

**Setup**:
- Small density perturbation: ρ = ρ₀(1 + A·sin(kx))
- Wavenumber k = 0.5 GeV
- Cross-section σ = 10000 fm² (nearly-ideal regime)

**Measurement**: Spatial phase shift between snapshots → ω

**Result**: ω_numerical ≈ 0.289 GeV vs ω_analytical = 0.288 GeV (<5% error) ✅

**Recent fix** (2025-10-18):
- Modified `_is_physical_mode()` in `sound_waves.py:530-568`
- Accept modes where |Γ/ω| < 1% regardless of sign
- Prevents false rejection at numerical precision limit

### Test 2: Sound Wave Damping ⚠️

**Test**: `test_sound_wave_damping` in `test_ired_analytical.py:235-289`

**Status**: `@pytest.mark.xfail` - Physical limitation, not a bug

**Paradox**:
```
Small σ (σ=1 fm²):
  η = 5.071e-04 GeV³ → Γ ≈ 1e-03 GeV (measurable)
  BUT: τω ≈ 4.15 >> 1 (violates Israel-Stewart regime)

Large σ (σ=10000 fm²):
  η = 5.071e-08 GeV³ → Γ ≈ 1e-05 GeV (unmeasurable, numerical noise)
  τω ≈ 0.42 < 1 (regime valid) ✓
```

**Conclusion**: Cannot achieve both conditions with hard sphere gas at single σ.

**Options for future work**:
- Option B: Parameter scan to find intermediate σ (if exists)
- Option C: Use full IS dispersion relation (not just Navier-Stokes Γ_NS)

### Test 3: Bjorken Shear Stress ✅

**Test**: `test_bjorken_shear_stress_evolution` in `test_ired_analytical.py:291-330`

**Setup**: 1D boost-invariant expansion (Bjorken flow)

**Validation**: π^μν follows Israel-Stewart relaxation equation

**Result**: ✅ Shear stress evolution matches analytical prediction

### Test 4: Bjorken Temperature ❌

**Test**: `test_bjorken_temperature_evolution` in `test_ired_analytical.py:332-375`

**Status**: `@pytest.mark.xfail` - Implementation bug

**Error**: 71% error in T(τ) evolution

**Measurement**:
```python
# Analytical (from energy conservation)
T_analytical(τ) = T₀ · (τ₀/τ)^(1/3)

# Numerical (from simulation)
T_numerical(τ) → constant (NOT EVOLVING!)
```

**Root cause**: Conservation law `∇_μ T^μν = 0` computes `d(ρu^μ)/dt` but doesn't update temperature field from energy density.

**Fix needed**:
1. In spectral solver, after each timestep:
   ```python
   # Update T from energy density (radiation EOS)
   fields.temperature[:] = (fields.rho / a)**(1/4)  # Stefan-Boltzmann
   ```
2. Ensure thermodynamic consistency throughout evolution

**Time estimate**: 2 days

### Test 5: Diffusion Fick's Law ✅

**Test**: `test_ficks_law` in `test_ired_analytical.py:377-420`

**Setup**: Particle density gradient → diffusion current

**Validation**: V^i = -D·∇^i(μ/T)

**Result**: ✅ Within 10% of analytical prediction

### Test 6: Diffusion Decay Rate ❌

**Test**: `test_diffusion_decay_rate` in `test_ired_analytical.py:422-470`

**Status**: `@pytest.mark.xfail` - Numerical instability

**Setup**: Gaussian density perturbation → exponential decay

**Expected**: Γ = D·k² (decay rate)

**Issue**: Long-time evolution (t ~ 11 GeV⁻¹) shows numerical overflow

**Error messages**:
```
RuntimeWarning: overflow encountered in square
RuntimeWarning: invalid value encountered in fft
```

**Hypothesis**: Accumulation of roundoff errors in extended evolution

**Fix needed**:
1. Analyze timestep sensitivity (reduce dt?)
2. Check for conservation violation over long times
3. Investigate FFT precision (switch to double precision?)
4. Add numerical damping for high-k modes?

**Time estimate**: 3 days

## Remaining Work

**Time estimate**: 7 days total
- 2 days: Fix Bjorken temperature evolution
- 3 days: Debug diffusion long-time instability
- 2 days: Document all benchmark results

**Breakdown**:

**Task 1: Bjorken Temperature (2 days)**
1. Add temperature update from energy density (0.5 days)
2. Verify thermodynamic consistency (0.5 days)
3. Rerun test and validate (0.5 days)
4. Document fix (0.5 days)

**Task 2: Diffusion Instability (3 days)**
1. Create diagnostic script for long-time evolution (0.5 days)
2. Analyze timestep sensitivity (1 day)
3. Investigate conservation violation (1 day)
4. Implement fix and validate (0.5 days)

**Task 3: Documentation (2 days)**
1. Write detailed benchmark reports in `results/`
2. Update VALIDATION_ROADMAP.md
3. Document known limitations

## Physical Parameter Ranges

**Sound waves**:
- Wavenumber: k ∈ [0.1, 4] GeV (regime valid for τ ~ 0.5)
- Amplitude: A ∈ [0.01, 0.1] (linear regime)
- Temperature: T = 0.4 GeV (400 MeV)
- Cross-section: σ = 1-10000 fm² (explore parameter space)

**Bjorken flow**:
- Initial proper time: τ₀ = 1.0 fm/c
- Evolution time: τ ∈ [1, 10] fm/c
- Temperature: T₀ = 0.4 GeV
- Shear viscosity: η/s ∈ [0.1, 0.5] (QGP range)

**Diffusion**:
- Wavenumber: k ∈ [0.1, 2] GeV
- Evolution time: t ∈ [0, 5] GeV⁻¹ (avoid long-time instability for now)
- Chemical potential: μ/T ∈ [0, 1]
- Diffusion coefficient: D ~ 0.1 GeV²

## References

- **Tests**: `tests/test_ired_analytical.py`
- **Sound waves**: `israel_stewart/benchmarks/sound_waves.py`
- **Bjorken flow**: `israel_stewart/benchmarks/bjorken_flow.py`
- **Diffusion**: `israel_stewart/benchmarks/diffusion.py`
- **Dispersion analysis**: `validation/stage4_dispersion/README.md`
- **Nearly-ideal mode fix**: `CLAUDE.md` - Option A section

## Success Metrics

**Before** (2025-10-18):
- 3/6 tests passing ✓
- 1/6 xfail (Regime Paradox - physical) ⚠️
- 2/6 xfail (implementation bugs) ❌

**Target**:
- 5/6 tests passing ✓
- 1/6 may remain xfail (Regime Paradox - needs physics solution) ⚠️
- Sound frequency ✓
- Bjorken shear ✓
- Diffusion Fick ✓
- **Bjorken T fixed** ✓
- **Diffusion decay fixed** ✓

## Next Steps

1. **Fix Bjorken temperature** (HIGH priority, 2 days)
   - Add T update from ρ in spectral solver
   - Verify thermodynamic consistency
   - Rerun test

2. **Debug diffusion instability** (MEDIUM priority, 3 days)
   - Analyze timestep sensitivity
   - Check conservation over long times
   - Implement fix

3. **Document benchmark results** (2 days)
   - Write `results/sound_wave_validation.md`
   - Write `results/bjorken_validation.md`
   - Write `results/diffusion_validation.md`
   - Update VALIDATION_ROADMAP.md

**After Stage 6 complete**: Full system validated against analytical solutions (within physical limitations).
