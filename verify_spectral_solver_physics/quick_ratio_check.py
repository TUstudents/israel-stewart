"""
Check if eigenmode ratios are preserved during evolution.

For a perfect linear eigenmode, the ratios v/ρ, Π/ρ, π/ρ should remain constant.
Deviations indicate:
1. Nonlinear effects (finite amplitude)
2. Numerical discretization errors
3. Higher-order Israel-Stewart coupling terms (λ_ππ, λ_πΠ, etc.)
4. Operator splitting errors

Small deviations (~2-3% after 10 timesteps) are expected and acceptable.
Large deviations would indicate a bug in the dispersion relation or time-stepping.
"""

import numpy as np
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

# Setup
coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=0.5,
    bulk_relaxation_time=0.3
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2*np.pi,
    grid_points=(32, 32, 16),
    transport_coeffs=coeffs
)

k = 8.0
benchmark.setup_initial_conditions(wave_number=k)

print("="*80)
print("EIGENMODE RATIO PRESERVATION CHECK")
print("="*80)
print()
print("Testing if eigenmode ratios (v/ρ, Π/ρ, π/ρ) remain constant during evolution.")
print("Expected: Small deviations (~2-5%) due to finite amplitude and numerical effects.")
print()

# Initial state
rho_fft_0 = np.fft.fftn(benchmark.fields.rho - 1.0)
v_fft_0 = np.fft.fftn(benchmark.fields.u_mu[..., 1])
Pi_fft_0 = np.fft.fftn(benchmark.fields.Pi)
pi_fft_0 = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])

k_idx = 8
rho_k_0 = np.abs(rho_fft_0[k_idx, 0, 0])
v_k_0 = np.abs(v_fft_0[k_idx, 0, 0])
Pi_k_0 = np.abs(Pi_fft_0[k_idx, 0, 0])
pi_k_0 = np.abs(pi_fft_0[k_idx, 0, 0])

print(f"Initial Fourier amplitudes:")
print(f"  |ρ_k|  = {rho_k_0:.6e}")
print(f"  |v_k|  = {v_k_0:.6e}")
print(f"  |Π_k|  = {Pi_k_0:.6e}")
print(f"  |π_k|  = {pi_k_0:.6e}")
print()

v_ratio_0 = v_k_0 / rho_k_0
Pi_ratio_0 = Pi_k_0 / rho_k_0
pi_ratio_0 = pi_k_0 / rho_k_0

print(f"Initial ratios:")
print(f"  |v|/|ρ|  = {v_ratio_0:.6f}")
print(f"  |Π|/|ρ|  = {Pi_ratio_0:.6f}")
print(f"  |π|/|ρ|  = {pi_ratio_0:.6f}")
print()

# Evolve and check at multiple timesteps
dt = 0.01
timesteps_to_check = [1, 5, 10]
print(f"Evolving with dt={dt} and checking at steps: {timesteps_to_check}")
print()

for n_steps in timesteps_to_check:
    # Evolve from last checkpoint
    if n_steps == 1:
        current_steps = 1
    else:
        current_steps = n_steps - timesteps_to_check[timesteps_to_check.index(n_steps) - 1]

    for _ in range(current_steps):
        benchmark.solver.time_step(dt, method="split_step")

    # Measure current state
    rho_fft = np.fft.fftn(benchmark.fields.rho - 1.0)
    v_fft = np.fft.fftn(benchmark.fields.u_mu[..., 1])
    Pi_fft = np.fft.fftn(benchmark.fields.Pi)
    pi_fft = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])

    rho_k = np.abs(rho_fft[k_idx, 0, 0])
    v_k = np.abs(v_fft[k_idx, 0, 0])
    Pi_k = np.abs(Pi_fft[k_idx, 0, 0])
    pi_k = np.abs(pi_fft[k_idx, 0, 0])

    v_ratio = v_k / rho_k
    Pi_ratio = Pi_k / rho_k
    pi_ratio = pi_k / rho_k

    print(f"After {n_steps} timesteps (t={n_steps*dt:.3f}):")
    print(f"  |v|/|ρ|  = {v_ratio:.6f}  (change: {(v_ratio - v_ratio_0)/v_ratio_0*100:+.2f}%)")
    print(f"  |Π|/|ρ|  = {Pi_ratio:.6f}  (change: {(Pi_ratio - Pi_ratio_0)/Pi_ratio_0*100:+.2f}%)")
    print(f"  |π|/|ρ|  = {pi_ratio:.6f}  (change: {(pi_ratio - pi_ratio_0)/pi_ratio_0*100:+.2f}%)")
    print()

print("="*80)
print("INTERPRETATION")
print("="*80)
print()

# Assess results
max_change_1 = max(abs((v_ratio - v_ratio_0)/v_ratio_0),
                    abs((Pi_ratio - Pi_ratio_0)/Pi_ratio_0),
                    abs((pi_ratio - pi_ratio_0)/pi_ratio_0)) * 100

if max_change_1 < 5:
    print("✓ GOOD: Ratio changes after 10 timesteps are <5%")
    print("  → Eigenmode structure well preserved")
    print("  → Time-stepping and dispersion relation consistent")
elif max_change_1 < 10:
    print("⚠ MODERATE: Ratio changes after 10 timesteps are 5-10%")
    print("  → Some eigenmode degradation observed")
    print("  → May indicate finite amplitude or numerical effects")
else:
    print("✗ LARGE: Ratio changes after 10 timesteps are >10%")
    print("  → Eigenmode structure NOT well preserved")
    print("  → Possible issues:")
    print("    1. Coupling terms in time-stepping incorrect")
    print("    2. Dispersion relation initialization mismatch")
    print("    3. Nonlinear effects stronger than expected")
    print("    4. Higher-order IS terms not captured in linear analysis")

print()
print("Physical context:")
print("  - Dissipative fields (Π, π) have shorter relaxation times than ρ, v")
print("  - BUT ratios should stay constant for true eigenmode")
print("  - Large ratio changes suggest eigenmode structure breaks down")
print("  - This affects damping predictions (currently ~21% error)")
print()
print("="*80)
