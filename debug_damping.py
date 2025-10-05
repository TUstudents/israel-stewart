#!/usr/bin/env python3
"""
Debug script for investigating 100% damping error in sound wave benchmark.

This script runs a minimal sound wave simulation and analyzes the damping
extraction to identify why measured_damping is 0 while analytical_damping > 0.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

# Create benchmark with standard settings
transport_coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=0.5,
    bulk_relaxation_time=0.3,
    lambda_pi_pi=0.1,
    lambda_pi_Pi=0.05,
    xi_1=0.2,
    xi_2=0.1,
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi,
    grid_points=(32, 32, 16),
    transport_coeffs=transport_coeffs,
)

# Run simulation for k=1.0
wave_number = 1.0
print("=" * 80)
print("DAMPING EXTRACTION DEBUG")
print("=" * 80)
print(f"Wave number k: {wave_number}")
print()

# Get analytical prediction
wave_vector = np.array([wave_number, 0.0, 0.0])
analytical_modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)

if analytical_modes:
    mode = analytical_modes[0]
    print("Analytical Predictions:")
    print(f"  Frequency (ω):      {mode.frequency:.6f}")
    print(f"  Damping (γ):        {mode.attenuation:.6f}")
    print(f"  Sound speed (c_s):  {mode.sound_speed:.6f}")
    print()
else:
    print("ERROR: Could not find analytical mode!")
    sys.exit(1)

# Run short simulation to inspect time series
print("Running simulation...")
result = benchmark.run_simulation(
    wave_number=wave_number,
    simulation_time=10.0,
    n_periods=3,
)

print()
print("Numerical Results:")
print(f"  Measured frequency: {result.measured_frequency:.6f}")
print(f"  Measured damping:   {result.measured_damping_rate:.6f}")
print(f"  Frequency error:    {result.frequency_error:.1%}")
print(f"  Damping error:      {result.damping_error:.1%}")
print()

# Inspect time series
time = result.time_series_data["time"]
density = result.time_series_data["density"]

print(f"Time series length: {len(time)} points")
print(f"Time range: {time[0]:.3f} to {time[-1]:.3f}")
print(f"Density range: {density.min():.6f} to {density.max():.6f}")
print()

# Analyze damping extraction
signal_ac = density - np.mean(density)
envelope = np.abs(signal_ac)

print("Signal Statistics:")
print(f"  Mean density:       {np.mean(density):.6f}")
print(f"  AC amplitude:       {np.max(np.abs(signal_ac)):.6f}")
print(f"  Initial envelope:   {envelope[0]:.6f}")
print(f"  Final envelope:     {envelope[-1]:.6f}")
print(f"  Envelope ratio:     {envelope[-1]/envelope[0]:.6f}" if envelope[0] > 0 else "  Envelope ratio:     undefined")
print()

# Manual damping extraction
if envelope[0] > 0 and envelope[-1] > 0:
    # Expected decay: A(t) = A(0) * exp(-γ*t)
    # ln(A(t)/A(0)) = -γ*t
    # γ = -ln(A(t)/A(0)) / t

    # Try with initial vs final
    gamma_simple = -np.log(envelope[-1] / envelope[0]) / (time[-1] - time[0])
    print(f"Simple damping estimate (initial→final): {gamma_simple:.6f}")

    # Try log-linear fit
    valid_mask = envelope > 0.01 * np.max(envelope)
    if np.sum(valid_mask) > 5:
        valid_time = time[valid_mask]
        valid_envelope = envelope[valid_mask]
        log_envelope = np.log(valid_envelope)

        # Linear fit: log(A) = log(A0) - γ*t
        coeffs = np.polyfit(valid_time, log_envelope, 1)
        gamma_fit = -coeffs[0]

        print(f"Log-linear fit damping:                  {gamma_fit:.6f}")
        print(f"Fit R²: {1 - np.var(log_envelope - np.polyval(coeffs, valid_time))/np.var(log_envelope):.4f}")
    print()

# Plot time series and envelope
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Time series
ax = axes[0]
ax.plot(time, density, 'b-', alpha=0.7, label='Density')
ax.plot(time, np.mean(density) + envelope, 'r--', alpha=0.5, label='Envelope')
ax.plot(time, np.mean(density) - envelope, 'r--', alpha=0.5)
ax.set_xlabel('Time')
ax.set_ylabel('Density')
ax.set_title(f'Sound Wave Time Series (k={wave_number})')
ax.legend()
ax.grid(True, alpha=0.3)

# Envelope in log scale
ax = axes[1]
valid_mask = envelope > 0.01 * np.max(envelope)
ax.semilogy(time[valid_mask], envelope[valid_mask], 'b-', alpha=0.7, label='Envelope')

# Overlay analytical decay
if analytical_modes:
    analytical_decay = envelope[0] * np.exp(-mode.attenuation * time)
    ax.semilogy(time, analytical_decay, 'r--', alpha=0.5, label=f'Analytical (γ={mode.attenuation:.4f})')

# Overlay measured decay
if result.measured_damping_rate > 0:
    measured_decay = envelope[0] * np.exp(-result.measured_damping_rate * time)
    ax.semilogy(time, measured_decay, 'g--', alpha=0.5, label=f'Measured (γ={result.measured_damping_rate:.4f})')

ax.set_xlabel('Time')
ax.set_ylabel('Envelope (log scale)')
ax.set_title('Damping Envelope Analysis')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_damping.png', dpi=150)
print("Saved plot to debug_damping.png")

# Root cause analysis
print("=" * 80)
print("ROOT CAUSE ANALYSIS")
print("=" * 80)

if result.measured_damping_rate == 0.0:
    print("⚠️  ISSUE: Measured damping rate is exactly 0.0")
    print()
    print("Possible causes:")
    print("1. Signal envelope is not decaying (viscosity not working)")
    print("2. Damping extraction failed (numerical issue in fit)")
    print("3. Simulation time too short for measurable decay")
    print()

    # Check which one
    decay_ratio = envelope[-1] / envelope[0] if envelope[0] > 0 else 1.0
    print(f"Decay ratio: {decay_ratio:.6f}")

    if decay_ratio > 0.9:
        print("→ Signal barely decayed! Viscosity may not be applied correctly.")
    elif decay_ratio < 0.01:
        print("→ Signal decayed too much! May have numerical issues in log fit.")
    else:
        print("→ Signal decayed moderately. Extraction algorithm may have failed.")
        print("   Check _extract_frequency_damping implementation.")
else:
    damping_ratio = result.measured_damping_rate / mode.attenuation
    print(f"Damping ratio (measured/analytical): {damping_ratio:.6f}")

    if damping_ratio < 0.1:
        print("→ Measured damping much smaller than expected")
        print("   Viscosity may not be fully active in simulation")
    elif damping_ratio > 10.0:
        print("→ Measured damping much larger than expected")
        print("   May have spurious numerical damping")
    else:
        print("✓ Damping is in reasonable range")
        print(f"  Error should be ~{abs(1-damping_ratio)*100:.1f}%, not 100%")

print("=" * 80)
