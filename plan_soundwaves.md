# Transform Sound Waves Benchmark: Analytical → Numerical Validation

## Current State Analysis
The existing `sound_waves.py` is purely analytical:
- **SoundWaveAnalysis**: Computes theoretical dispersion relations ω(k) by solving linearized equations
- **DispersionRelation**: Builds and solves the characteristic matrix for eigenfrequencies
- **WaveTestSuite**: Validates analytical properties (sound speed recovery, causality, etc.)
- **No Numerical Simulation**: Currently only computes theoretical predictions

## Transformation Plan

### Phase 1: Add Numerical Simulation Infrastructure
**Create `NumericalSoundWaveBenchmark` class**:
- Initialize periodic grid with spectral solver (using new `create_periodic_grid()`)
- Set up initial conditions: sinusoidal perturbation with specific wave number k
- Configure Israel-Stewart field evolution (conservation + relaxation equations)
- Implement time evolution loop with adaptive timesteps

### Phase 2: Initial Condition Setup
**Perturbed Equilibrium State**:
- Background: uniform rest frame with ρ₀, P₀, u^μ = (1,0,0,0)
- Sound wave perturbation: `δρ = A·sin(k·x)`, `δu^x = A'·sin(k·x)`
- Zero initial dissipative fluxes: `Π = π^μν = q^μ = 0`
- Small amplitude A ~ 0.01 for linear regime validity

### Phase 3: Spectral Solver Integration
**Time Evolution Framework**:
- Use `SpectralISHydrodynamics` for complete evolution (conservation + relaxation)
- Leverage new periodic grids for precise wave number k specification
- Monitor field evolution: ρ(x,t), u^x(x,t), dissipative fluxes
- Extract frequency ω and damping rate γ from temporal oscillations

### Phase 4: Frequency/Damping Extraction
**Signal Analysis Methods**:
- **FFT Analysis**: Temporal FFT of ρ(t) at fixed spatial points → measure ω
- **Exponential Fitting**: Fit `ρ(t) = A·e^{-γt}·cos(ωt)` → extract both ω and γ
- **Phase Tracking**: Track wave fronts across periodic domain → verify group velocity
- **Energy Decay**: Monitor total energy dissipation → validate viscous damping

### Phase 5: Analytical Comparison Framework
**Validation Infrastructure**:
- Compare numerical ω_num vs analytical ω_analytical from existing dispersion solver
- Validate damping rate γ_num vs theoretical viscous attenuation
- Test multiple wave numbers: k ∈ [0.1, 5.0] to cover different regimes
- Check second-order transport effects vs analytical predictions

### Phase 6: Comprehensive Benchmark Suite
**Integrated Testing**:
- **Sound Speed Recovery**: ω/k → c_s in ideal limit (η=ζ=0)
- **Viscous Damping**: γ ∝ k² behavior for first-order viscosity
- **Second-Order Effects**: Deviations from simple γ ∝ k² with λ_ππ, ξ₁ ≠ 0
- **Causality Validation**: Ensure v_phase, v_group ≤ c always
- **Convergence Studies**: Numerical accuracy vs grid resolution

### Key Implementation Details
**Numerical Methods**:
- Periodic boundaries essential for clean wave number specification
- Spectral accuracy critical for capturing subtle dispersion effects
- Time integration: IMEX schemes to handle stiff relaxation timescales
- Resolution: N ≥ 64 points per wavelength for spectral accuracy

**Analysis Tools**:
- Windowed FFT for time-resolved frequency analysis
- Complex frequency extraction: ω_complex = ω_real + i·γ
- Statistical analysis over multiple periods for noise reduction
- Error quantification: |ω_num - ω_analytical|/ω_analytical < 1%

This transforms `sound_waves.py` from purely theoretical validation into a true numerical benchmark that validates both the Israel-Stewart physics implementation and the spectral solver accuracy through direct simulation-theory comparison.