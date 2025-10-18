#!/usr/bin/env python3
"""Diagnose amplitude growth issue in late-time diffusion evolution."""

import matplotlib.pyplot as plt
import numpy as np

from israel_stewart.benchmarks.diffusion_flow import create_diffusion_benchmark_with_ired

# Create benchmark
benchmark, ired_model = create_diffusion_benchmark_with_ired(
    temperature=0.4,
    cross_section=1000.0,
    truncation="41",
    perturbation_amplitude=0.05,
    wave_number=0.5,
    grid_points=(16, 16, 16),
    domain_size=4 * np.pi,
)

# Transport coefficients
D = ired_model.diffusion_coefficient()
tau_V = ired_model.diffusion_relaxation_time()
k = 0.5
T = 0.4

# Particle density
zeta_3 = 1.2020569
n0 = (zeta_3 / np.pi**2) * T**3

# Expected rates
Gamma_slow = (D * k**2) * (tau_V / n0)
Gamma_fast = 1.0 / tau_V
t_transient = 5.0 * tau_V

print("=" * 80)
print("AMPLITUDE EVOLUTION DIAGNOSTIC")
print("=" * 80)

print("\nExpected decay rates:")
print(f"  Slow mode: Γ_slow = {Gamma_slow:.6e} GeV")
print(f"  Fast mode: Γ_fast = {Gamma_fast:.6e} GeV")
print(f"  Transient time: {t_transient:.3f} GeV⁻¹")

# Track amplitude
times = []
amplitudes = []


def extract_amplitude(t, fields):
    V_x = fields.V_mu[..., 1]
    amplitude = np.sqrt(np.mean(V_x**2))
    times.append(t)
    amplitudes.append(amplitude)

    # Print periodically
    if len(times) % 50 == 0 or t == 0:
        print(f"  t = {t:.3f}: A = {amplitude:.6e}")


# Evolve
t_final = 11.33
dt = 0.05

print(f"\nEvolving for t = 0 → {t_final:.2f} GeV⁻¹ with dt = {dt:.3f}...")
benchmark.solver.evolve(t_final=t_final, dt=dt, method="rk4", callback=extract_amplitude)

# Convert to arrays
times = np.array(times)
amplitudes = np.array(amplitudes)

print(f"\n{'='*80}")
print("AMPLITUDE ANALYSIS")
print("=" * 80)

# Check for NaN or negative
print("\nData quality:")
print(f"  All finite: {np.all(np.isfinite(amplitudes))}")
print(f"  All positive: {np.all(amplitudes > 0)}")
print(f"  Min amplitude: {amplitudes.min():.6e}")
print(f"  Max amplitude: {amplitudes.max():.6e}")

# Early vs late
early_mask = times < t_transient
late_mask = times > t_transient

print(f"\nEarly times (t < {t_transient:.2f}):")
print(f"  Initial: {amplitudes[early_mask][0]:.6e}")
print(f"  Final: {amplitudes[early_mask][-1]:.6e}")
print(f"  Ratio: {amplitudes[early_mask][-1] / amplitudes[early_mask][0]:.6f}")

if np.sum(late_mask) > 0:
    print(f"\nLate times (t > {t_transient:.2f}):")
    print(f"  First: {amplitudes[late_mask][0]:.6e}")
    print(f"  Last: {amplitudes[late_mask][-1]:.6e}")
    print(f"  Ratio: {amplitudes[late_mask][-1] / amplitudes[late_mask][0]:.6f}")

    # Fit late-time
    times_late = times[late_mask]
    amps_late = amplitudes[late_mask]

    log_A = np.log(amps_late)
    coeffs = np.polyfit(times_late, log_A, deg=1)
    slope = coeffs[0]
    Gamma_measured = -slope

    print("\nLate-time linear fit:")
    print(f"  Slope of log(A) vs t: {slope:.6e}")
    print(f"  Gamma = -slope: {Gamma_measured:.6e} GeV")

    if Gamma_measured < 0:
        print("  ⚠️ NEGATIVE GAMMA = GROWTH!")
    else:
        print("  ✓ Positive Gamma = decay")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Linear plot
ax1.plot(times, amplitudes, "b.-", label="Numerical")
ax1.axvline(t_transient, color="r", linestyle="--", alpha=0.5, label="Transient end")
ax1.set_xlabel("Time (GeV⁻¹)")
ax1.set_ylabel("Amplitude (RMS of V^x)")
ax1.set_title("Amplitude Evolution")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Log plot
ax2.semilogy(times, amplitudes, "b.-", label="Numerical")
ax2.axvline(t_transient, color="r", linestyle="--", alpha=0.5, label="Transient end")
ax2.set_xlabel("Time (GeV⁻¹)")
ax2.set_ylabel("Amplitude (log scale)")
ax2.set_title("Amplitude Evolution (log scale)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("amplitude_evolution_diagnostic.png", dpi=150)
print("\n✓ Plot saved to amplitude_evolution_diagnostic.png")

print(f"\n{'='*80}")
print("DIAGNOSIS")
print("=" * 80)

if not np.all(np.isfinite(amplitudes)):
    print("\n✗ NaN or Inf detected in amplitudes - numerical instability!")
elif Gamma_measured < 0:
    print("\n✗ Measured negative decay rate - amplitude is GROWING!")
    print("  Possible causes:")
    print("    - Numerical instability in long-time evolution")
    print("    - Incorrect late-time fitting (need to check monotonicity)")
    print("    - Physical instability in IReD equations")
else:
    print("\n✓ Amplitude decaying as expected")
    print(f"  Measured Γ = {Gamma_measured:.6e} vs expected {Gamma_slow:.6e}")
    print(f"  Factor: {Gamma_measured / Gamma_slow:.2f}×")
