# Diffusion Test Failure Investigation

**Date**: 2025-11-01
**Status**: ONGOING - Root cause identified, solution in progress

## Summary

The `test_diffusion_conserves_particles` test fails because particle gradients do not decrease as expected during diffusion. Investigation revealed:

1. ✅ **FIXED**: Temperature was zero → Added `update_temperature_from_eos()` method
2. ❌ **ONGOING**: Diffusion current V^μ only reaches 6% of expected equilibrium value

## Background

Diffusion in Landau frame is driven by chemical potential gradients:
```
dV^μ/dτ = -V^μ/τ_V - D ∇^μ(μ_B/T) + coupling terms
```

At equilibrium: `V_eq ≈ -D ∇(μ/T)`

## Bug #1: Missing Temperature (FIXED ✅)

**Problem**: `ISFieldConfiguration` initializes `temperature = 0` and never updates it.

**Impact**:
- Chemical potential: `μ/T = ln(n/n_eq)` where `n_eq ∝ T^3`
- With T=0: `n_eq ≈ 0`, so `μ/T ≈ 70` (unphysical!)
- This breaks diffusion physics

**Fix**: Added `update_temperature_from_eos()` method to `ISFieldConfiguration`:
```python
def update_temperature_from_eos(self, eos_type: str = "radiation") -> None:
    """Update temperature from energy density using EOS."""
    if eos_type == "radiation":
        # Stefan-Boltzmann: ε = (π^2/30) T^4
        stefan_boltzmann_a = np.pi**2 / 30.0
        rho_safe = np.maximum(self.rho, 1e-15)
        self.temperature[:] = (rho_safe / stefan_boltzmann_a) ** 0.25
```

**Result**:
- Before: T=0, μ/T ≈ 70, n_eq ≈ 10^-31
- After: T=1.32, μ/T ≈ 0.07-0.91, n_eq ≈ 0.28 ✅

## Bug #2: Diffusion Current Too Small (ONGOING ❌)

**Problem**: V^μ only reaches 6% of expected equilibrium value after 100 timesteps.

**Evidence**:
```
Setup:
  - ρ = 1.0 (uniform)
  - n = 0.5 + 0.2 sin(X) → gradient 0.2
  - D = 0.2 (diffusion coefficient)
  - τ_V = 0.1 (relaxation time)
  - dt = 0.002, 100 steps → t = 0.2

Expected:
  - T = 1.32 (radiation EOS) ✅
  - ∇(μ/T) ≈ 0.44 ✅
  - V_eq = -D ∇(μ/T) ≈ -0.087 ✓
  - After t/τ_V = 2: V ≈ 0.86 V_eq ≈ 0.075

Actual:
  - V ≈ 0.0054 (only 6% of expected!)
  - V still growing after 100 steps
```

**Possible Causes**:

1. **Wrong sign in relaxation equation**
   - Expected: `dV/dt = -V/τ_V - D ∇(μ/T)`
   - Check implementation in `relaxation.py:_diffusion_rhs()`

2. **Time integration not applied to V^μ**
   - V^μ may not be included in split-step evolution
   - Check `SpectralISHydrodynamics.time_step()` for V^μ updates

3. **Coupling coefficient error**
   - D or τ_V may have wrong units or scaling
   - Check dimensional analysis

4. **Gradient computation issue**
   - ∇^μ(μ/T) may not be projected correctly
   - Check `_compute_chemical_potential_gradient()` projection

## Diagnostic Scripts Created

1. `diagnose_diffusion_failure.py` - Shows V^μ evolving but too slowly
2. `diagnose_chemical_potential.py` - Confirms T, μ/T, and ∇(μ/T) are now correct

## Next Steps

1. ✅ Check if V^μ is being evolved in time integration
2. ✅ Verify signs in `_diffusion_rhs()` match theory
3. ⬜ Check if relaxation step is actually being called
4. ⬜ Test with explicit time integration (no operator splitting)
5. ⬜ Compare numerical vs analytical solution for simple test case

## Files Modified

- `israel_stewart/core/fields.py`: Added `update_temperature_from_eos()` method
- `israel_stewart/tests/test_dynamic_conservation.py`: Added temperature initialization to diffusion test

## Test Results

- **Before fixes**: 8/9 passing (diffusion test fails, gradient unchanged)
- **After temperature fix**: 8/9 passing (diffusion test still fails, gradient barely changes 0.4%)
- **Target**: 9/9 passing (gradient should decrease by ~10%)

## References

- `israel_stewart/equations/relaxation.py:435-543`: `_diffusion_rhs()` implementation
- `docs/LANDAU_FRAME_FORMULATION.md`: Theoretical background for diffusion
- Wagner & Gavassino (2024) IReD paper, Section 3.2: Diffusion current evolution
