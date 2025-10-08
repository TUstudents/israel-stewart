"""
Check if linear regime detection stays active throughout evolution.

If perturbations grow beyond threshold, solver switches to nonlinear mode,
which may introduce errors.
"""

import numpy as np

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

# Setup
coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=1.0,
    bulk_relaxation_time=0.5,
    lambda_pi_pi=0.0,
    lambda_pi_Pi=0.0,
    xi_1=0.0,
    xi_2=0.0,
)

k = 8.0
benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
)
benchmark.setup_initial_conditions(wave_number=k)

print("=" * 80)
print("LINEAR REGIME DETECTION CHECK")
print("=" * 80)
print()
print("Threshold: max_rho_pert < 0.1 AND max_velocity < 0.1")
print()

# Check at multiple times
times = [0.0, 0.01, 0.05, 0.1]
dt = 0.01

for t in times:
    if t > 0:
        # Evolve to time t
        n_steps = int(t / dt)
        benchmark = NumericalSoundWaveBenchmark(
            domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
        )
        benchmark.setup_initial_conditions(wave_number=k)

        for _ in range(n_steps):
            benchmark.solver.time_step(dt, method="rk4")

    # Check linear regime criteria
    max_rho_pert = np.max(np.abs(benchmark.fields.rho - 1.0))
    max_velocity = np.max(np.abs(benchmark.fields.u_mu[..., 1:4]))

    is_linear = (max_rho_pert < 0.1) and (max_velocity < 0.1)

    print(f"t = {t:.3f}:")
    print(f"  max |ρ - 1|: {max_rho_pert:.6f}  (threshold: 0.1)")
    print(f"  max |v|:      {max_velocity:.6f}  (threshold: 0.1)")
    print(f"  Linear regime: {is_linear}")

    if not is_linear:
        print("  ⚠ WARNING: Linear regime criteria VIOLATED!")
        if max_rho_pert >= 0.1:
            print(f"    - Density perturbation {max_rho_pert:.6f} >= 0.1")
        if max_velocity >= 0.1:
            print(f"    - Velocity {max_velocity:.6f} >= 0.1")

    print()

print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()
print("If linear regime is violated:")
print("  - Solver switches to nonlinear momentum conversion")
print("  - This introduces product rule terms u·dh/dt")
print("  - Creates 2nd harmonics and disrupts eigenmode")
print()
print("If linear regime is maintained:")
print("  - Problem must be elsewhere")
print()
print("=" * 80)
