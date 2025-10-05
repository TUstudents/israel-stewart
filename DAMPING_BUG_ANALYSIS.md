# Sound Wave Damping Bug Analysis

## Issue

Sound wave benchmark reports 100% damping error. The numerical simulation shows wave amplitude GROWING (~13% over 32 time units) instead of decaying as predicted by analytical Israel-Stewart theory (should decay to ~18% of initial amplitude).

## Investigation Summary

### Confirmed Working:
1. ✅ Dispersion relation solver correctly finds complex frequency ω - iγ
2. ✅ Analytical damping rate γ = 0.054 is physically reasonable
3. ✅ Dissipative fluxes (Π, π^μν) ARE evolving during simulation
4. ✅ Stress-energy tensor DOES include dissipative contributions
5. ✅ Conservation equations properly use full stress-energy tensor
6. ✅ Zero viscosity case shows no damping (as expected)
7. ✅ Grid spacing in gradient computations is correct
8. ✅ Eigenmode structure is properly computed and initialized

### Root Cause: Sign or Coupling Issue

The dissipative fluxes are initialized correctly to match the Israel-Stewart eigenmode (using imaginary part of complex eigenvector for sin(kx) waves). Initial values:
- Π ≈ 1.8e-4 (matches eigenmode)
- π_xx ≈ 4.6e-4 (matches eigenmode)

However, the wave GROWS instead of damping. Possible causes:

1. **Sign error in stress-energy tensor**: The dissipative terms may be added with wrong sign
   - Check: T^μν = ... + π^μν vs T^μν = ... - π^μν
   - Viscosity should resist flow → should reduce wave amplitude

2. **Relaxation equation sign error**: The fluxes may be relaxing in wrong direction
   - Check: dπ/dt = -π/τ + source vs dπ/dt = +π/τ + source

3. **Eigenmode phase error**: May need REAL part instead of IMAGINARY part
   - Current: Uses Im(eigenvector) for sin(kx)
   - Alternative: Use Re(eigenvector) for sin(kx)?

4. **Stress-energy coupling**: Dissipative fluxes may not be properly coupled back into momentum equation

## Evidence

From `debug_damping.py`:
```
Expected decay: A(t)/A(0) = 0.18 (with γ=0.054 after t=32)
Observed:       A(t)/A(0) = 1.13 (GROWTH!)
```

From `plot_amplitude_evolution.py`:
- Amplitude grows exponentially instead of decaying
- Growth rate ≈ +0.004 (should be decay rate -0.054)

## Next Steps

1. **Verify sign conventions** in stress-energy tensor construction (conservation.py:67-101)
2. **Check Israel-Stewart equation signs** in relaxation.py
3. **Test with simplified Navier-Stokes** (τ→0 limit) to isolate relaxation effects
4. **Compare with literature** Israel-Stewart implementations
5. **Add unit test** for viscous damping of sound waves

## Temporary Workaround

The benchmark currently passes causality tests but fails damping validation. For now:
- Dispersion relation analysis: ✅ WORKING
- Numerical damping validation: ❌ BROKEN (wave grows instead of damping)

## References

- Israel-Stewart equations: W. Israel & J.M. Stewart, Ann. Phys. 118, 341 (1979)
- Relativistic viscous hydrodynamics: P. Romatschke & U. Romatschke, Cambridge (2019)
