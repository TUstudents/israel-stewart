#!/usr/bin/env python3
"""Test IReD regime only with appropriate timesteps."""

import numpy as np

from israel_stewart.benchmarks.diffusion_flow import create_diffusion_benchmark_with_ired


def fit_exponential_decay(times, amplitudes):
    """Fit exponential decay A(t) = A₀ exp(-Γt) and return Γ."""
    valid = amplitudes > 0
    if np.sum(valid) < 3:
        return np.nan

    t = times[valid]
    A = amplitudes[valid]
    log_A = np.log(A)
    coeffs = np.polyfit(t, log_A, 1)
    Gamma = -coeffs[0]
    return Gamma


print("=" * 80)
print("IRED REGIME ONLY - DIFFUSION DECAY TEST")
print("=" * 80)

# IReD parameters
benchmark, ired_model = create_diffusion_benchmark_with_ired(
    temperature=0.4,
    cross_section=1000.0,
    truncation="41",
    perturbation_amplitude=0.05,
    wave_number=0.5,
    grid_points=(16, 16, 16),
    domain_size=4 * np.pi,
)

fields = benchmark.fields
solver = benchmark.solver

# Transport coefficients
D = ired_model.diffusion_coefficient()
k = 0.5
Gamma_expected = D * k**2

print("\nTransport coefficients:")
print(f"  D = {D:.6e} GeV²")
print(f"  k = {k:.3f} GeV")
print(f"  Γ_expected = Dk² = {Gamma_expected:.6e} GeV")
print(f"  Decay time = 1/Γ = {1/Gamma_expected:.2e} GeV⁻¹")

# Initial state
print("\nInitial state:")
print(f"  n: min={fields.n.min():.6e}, max={fields.n.max():.6e}, mean={fields.n.mean():.6e}")
V_x = fields.V_mu[..., 1]
print(f"  V^x: min={V_x.min():.6e}, max={V_x.max():.6e}, mean={V_x.mean():.6e}")
print(f"  V^x RMS: {np.sqrt(np.mean(V_x**2)):.6e}")

# Track amplitude
times = []
amplitudes = []


def extract_amplitude(t, fields):
    V_x = fields.V_mu[..., 1]
    amplitude = np.sqrt(np.mean(V_x**2))
    times.append(t)
    amplitudes.append(amplitude)

    # Check for NaN
    if not np.isfinite(amplitude):
        print(f"  ⚠ NaN detected at t={t:.3f}!")


# Use SMALL timesteps for stability
dt = 0.1  # Same as diagnostic script (works!)
t_final = 10.0  # Evolve for 10 GeV⁻¹ (0.04% decay expected)
n_steps = int(t_final / dt)

print("\nEvolution parameters:")
print(f"  dt = {dt:.3f} GeV⁻¹")
print(f"  t_final = {t_final:.3f} GeV⁻¹")
print(f"  n_steps = {n_steps}")
print(f"  Expected decay factor: exp(-Γt) = {np.exp(-Gamma_expected * t_final):.6f}")

print("\nEvolving...")
solver.evolve(t_final=t_final, dt=dt, method="rk4", callback=extract_amplitude)

times = np.array(times)
amplitudes = np.array(amplitudes)

print("\nEvolution complete!")
print(f"  Number of snapshots: {len(times)}")
print(f"  All amplitudes finite: {np.all(np.isfinite(amplitudes))}")

if np.all(np.isfinite(amplitudes)):
    print("\n✓ Evolution successful - no NaN produced")

    print("\nAmplitude evolution:")
    print(f"  Initial: {amplitudes[0]:.6e}")
    print(f"  Final: {amplitudes[-1]:.6e}")
    print(f"  Ratio: {amplitudes[-1] / amplitudes[0]:.6f}")
    print(f"  Expected ratio: {np.exp(-Gamma_expected * t_final):.6f}")

    # Fit decay rate
    Gamma_measured = fit_exponential_decay(times, amplitudes)
    error = abs(Gamma_measured - Gamma_expected) / Gamma_expected

    print("\nDecay rate measurement:")
    print(f"  Γ_measured = {Gamma_measured:.6e} GeV")
    print(f"  Γ_expected = {Gamma_expected:.6e} GeV")
    print(f"  Relative error: {error:.1%}")

    if Gamma_measured > 0:
        print("  ✓ Decay (positive Γ)")
    else:
        print("  ✗ GROWTH (negative Γ)!")

    if error < 0.50:
        print("  ✓ Reasonable agreement (< 50% error for short evolution)")
    else:
        print("  ⚠ Large error (> 50%)")

    # Final state
    print("\nFinal state:")
    print(f"  n: min={fields.n.min():.6e}, max={fields.n.max():.6e}, mean={fields.n.mean():.6e}")
    V_x_final = fields.V_mu[..., 1]
    print(f"  V^x: min={V_x_final.min():.6e}, max={V_x_final.max():.6e}")
    print(f"  V^x RMS: {np.sqrt(np.mean(V_x_final**2)):.6e}")
else:
    print("\n✗ Evolution failed - NaN produced")
    first_nan = np.where(~np.isfinite(amplitudes))[0][0]
    print(f"  First NaN at step {first_nan}, t = {times[first_nan]:.3f} GeV⁻¹")
