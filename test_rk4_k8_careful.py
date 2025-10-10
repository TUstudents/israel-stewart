#!/usr/bin/env -S uv run python
"""
Careful RK4 test for k=8 with small timesteps.

With /τ formulation, source term ~ ζθ/τ_Π, which at k=8 requires dt << τ_Π for stability.
"""

import numpy as np
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark, SoundWaveAnalysis
from israel_stewart.core.fields import TransportCoefficients
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.core.metrics import MinkowskiMetric

k = 8.0
coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=1.0,
    bulk_relaxation_time=0.5,  # τ_Π = 0.5
)

# Analytical mode
grid = SpaceGrid(
    coordinate_system="cartesian",
    spatial_ranges=[(0.0, 2*np.pi)] * 3,
    grid_points=(32, 32, 16),
    boundary_conditions="periodic"
)
metric = MinkowskiMetric()
analytical = SoundWaveAnalysis(grid, metric, coeffs)
wave_vector = np.array([k, 0.0, 0.0])
modes = analytical.analyze_dispersion_relation(wave_vector)
mode = modes[0]

print("=" * 80)
print("CAREFUL RK4 TEST FOR k=8")
print("=" * 80)
print()
print(f"k = {k}, τ_Π = {coeffs.bulk_relaxation_time}")
print(f"Analytical: ω = {mode.frequency:.4f}, γ = {mode.attenuation:.4f}")
print()

# Setup benchmark
benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
)
benchmark.setup_initial_conditions(wave_number=k)

k_idx = 8
rho_fft_0 = np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0]
amp_0 = abs(rho_fft_0)

print("Testing timestep stability...")
print()

# Try different timesteps
for dt in [0.01, 0.005, 0.001]:
    print(f"dt = {dt} (dt/τ_Π = {dt/coeffs.bulk_relaxation_time:.3f})")

    # Reset
    benchmark.setup_initial_conditions(wave_number=k)

    # Try 5 steps
    stable = True
    for i in range(5):
        try:
            benchmark.solver.time_step(dt, method="rk4")
            rho_fft = np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0]
            amp = abs(rho_fft)

            # Check for blow-up
            if amp > 2 * amp_0 or np.isnan(amp) or np.isinf(amp):
                print(f"  Step {i+1}: UNSTABLE (amp = {amp:.2e})")
                stable = False
                break
        except Exception as e:
            print(f"  Step {i+1}: ERROR - {e}")
            stable = False
            break

    if stable:
        print(f"  ✓ Stable after 5 steps (amp = {amp:.2e})")
    else:
        print(f"  ✗ Unstable")

    print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()
print("RK4 with /τ formulation requires very small timesteps at high k")
print("due to the stiff source terms: dΠ/dt ~ -ζθ/τ_Π")
print()
print("For k=8 with τ_Π=0.5, need dt << τ_Π/k ≈ 0.06")
print("This makes RK4 impractical for high wavenumbers.")
print()
print("=" * 80)
