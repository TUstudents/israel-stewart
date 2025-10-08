"""
Diagnose momentum-to-velocity conversion bug in spectral solver.

The RK4 method uses:
  du^i/dt = (1/h)[d(h·u^i)/dt - u^i·dh/dt]

where h = ε + p is the PERTURBED enthalpy. This introduces nonlinear coupling
u·dh/dt that creates spurious 2nd harmonics, breaking linear eigenmode structure.

For linear eigenmodes: u ~ sin(kx-ωt), dh/dt ~ -ω·cos(kx-ωt)
→ u·dh/dt ~ ω·sin·cos ~ sin(2kx-2ωt)  (WRONG FREQUENCY!)

This script verifies the hypothesis by comparing nonlinear vs linearized conversion.
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

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
)

k = 8.0
benchmark.setup_initial_conditions(wave_number=k)

print("=" * 80)
print("MOMENTUM-TO-VELOCITY CONVERSION DIAGNOSTIC")
print("=" * 80)
print()

# Get initial state (perfect eigenmode)
rho = benchmark.fields.rho.copy()
u_mu = benchmark.fields.u_mu.copy()
Pi = benchmark.fields.Pi.copy()
pi_munu = benchmark.fields.pi_munu.copy()

print("Initial perturbation amplitudes:")
print(f"  |δρ| = {np.max(np.abs(rho - 1.0)):.6e}")
print(f"  |δv| = {np.max(np.abs(u_mu[..., 1])):.6e}")
print(f"  |Π|  = {np.max(np.abs(Pi)):.6e}")
print(f"  |π|  = {np.max(np.abs(pi_munu[..., 1, 1])):.6e}")
print()

# Compute RHS using conservation equations
conservation_rhs = benchmark.solver.conservation.evolution_equations()
drho_dt = conservation_rhs["drho_dt"]
dmom_dt = conservation_rhs["dmom_dt"]

print("Conservation RHS at grid point (8, 0, 0):")
print(f"  dρ/dt        = {drho_dt[8, 0, 0]:.6e}")
print(f"  d(h·u^x)/dt  = {dmom_dt[8, 0, 0, 0]:.6e}")
print()

# Current (buggy) conversion: nonlinear with perturbed h
h_perturbed = rho + benchmark.fields.pressure  # ε + p
dh_dt_perturbed = (4.0 / 3.0) * drho_dt  # For radiation: p = ε/3
u_spatial = u_mu[..., 1:4]

h_safe = np.where(np.abs(h_perturbed) > 1e-14, h_perturbed, 1e-14)
du_dt_nonlinear = (1.0 / h_safe[..., np.newaxis]) * (
    dmom_dt - u_spatial * dh_dt_perturbed[..., np.newaxis]
)

# Linearized conversion: use background h₀ = 4/3
h_background = 4.0 / 3.0
du_dt_linear = dmom_dt / h_background

print("Velocity time derivatives at (8, 0, 0):")
print(f"  du^x/dt (nonlinear): {du_dt_nonlinear[8, 0, 0, 0]:.6e}")
print(f"  du^x/dt (linear):    {du_dt_linear[8, 0, 0, 0]:.6e}")
print(
    f"  Relative diff:       {abs(du_dt_nonlinear[8, 0, 0, 0] - du_dt_linear[8, 0, 0, 0]) / abs(du_dt_linear[8, 0, 0, 0]) * 100:.2f}%"
)
print()

# Compute nonlinear correction term: -u·dh/dt / h
correction_term = -u_spatial * dh_dt_perturbed[..., np.newaxis] / h_safe[..., np.newaxis]

print("Nonlinear correction term -u·dh/dt / h:")
print(f"  At (8, 0, 0):  {correction_term[8, 0, 0, 0]:.6e}")
print(f"  Max over grid: {np.max(np.abs(correction_term)):.6e}")
print()

# Analyze frequency content via FFT
print("Frequency content analysis:")
print()

# FFT of velocity
v_fft = np.fft.fftn(u_mu[..., 1])
print("Velocity FFT amplitudes:")
print(f"  Mode k=8:  {np.abs(v_fft[8, 0, 0]):.6e}  (fundamental)")
print(f"  Mode k=16: {np.abs(v_fft[16, 0, 0]):.6e}  (2nd harmonic)")
print(f"  Ratio:     {np.abs(v_fft[16, 0, 0]) / np.abs(v_fft[8, 0, 0]) * 100:.2f}%")
print()

# FFT of nonlinear correction
correction_fft = np.fft.fftn(correction_term[..., 0])
print("Nonlinear correction FFT amplitudes:")
print(f"  Mode k=8:  {np.abs(correction_fft[8, 0, 0]):.6e}")
print(f"  Mode k=16: {np.abs(correction_fft[16, 0, 0]):.6e}  (2nd harmonic)")
print(f"  Mode k=0:  {np.abs(correction_fft[0, 0, 0]):.6e}  (DC offset)")
print()

# FFT of du_dt difference
diff = du_dt_nonlinear[..., 0] - du_dt_linear[..., 0]
diff_fft = np.fft.fftn(diff)
print("Difference (nonlinear - linear) FFT amplitudes:")
print(f"  Mode k=8:  {np.abs(diff_fft[8, 0, 0]):.6e}")
print(f"  Mode k=16: {np.abs(diff_fft[16, 0, 0]):.6e}  (2nd harmonic)")
print(f"  Mode k=0:  {np.abs(diff_fft[0, 0, 0]):.6e}  (DC offset)")
print()

print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()

# Check if 2nd harmonic is significant
if np.abs(diff_fft[16, 0, 0]) > 1e-10:
    print("✗ PROBLEM: Nonlinear conversion creates significant 2nd harmonic")
    print(
        f"  The term -u·dh/dt introduces k=16 mode with amplitude {np.abs(diff_fft[16, 0, 0]):.2e}"
    )
    print("  This couples the fundamental eigenmode to higher harmonics")
    print("  → Eigenmode structure degrades during evolution")
    print()
    print("Expected behavior:")
    print("  Linear eigenmode at k=8 should evolve as e^(-iωt)")
    print("  All Fourier components should preserve their ratios")
    print("  But nonlinear coupling excites k=16, k=24, etc.")
else:
    print("✓ GOOD: Nonlinear correction is negligible")
    print("  2nd harmonic amplitude < 1e-10")

print()
if (
    abs(du_dt_nonlinear[8, 0, 0, 0] - du_dt_linear[8, 0, 0, 0]) / abs(du_dt_linear[8, 0, 0, 0])
    > 0.01
):
    print(
        f"✗ Nonlinear correction changes du/dt by {abs(du_dt_nonlinear[8, 0, 0, 0] - du_dt_linear[8, 0, 0, 0]) / abs(du_dt_linear[8, 0, 0, 0]) * 100:.1f}%"
    )
    print("  This is significant for eigenmode preservation")
else:
    print("✓ Nonlinear correction is <1% of du/dt")

print()
print("Recommendation:")
print("  For linear stability analysis and eigenmode tests:")
print("  → Use linearized conversion: du/dt = d(h·u)/dt / h₀")
print("  → Only use nonlinear conversion for large-amplitude flows")
print()
print("=" * 80)
