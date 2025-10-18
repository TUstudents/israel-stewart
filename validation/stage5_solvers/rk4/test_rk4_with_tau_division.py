#!/usr/bin/env -S uv run python
"""
Test frequency and phase evolution with RK4 integration.

Using dispersion-consistent formulation: dΠ/dt = -Π/τ_Π - ζθ/τ_Π
with RK4 (fully explicit) instead of IMEX.
"""

import numpy as np
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark, SoundWaveAnalysis
from israel_stewart.core.fields import TransportCoefficients
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.core.metrics import MinkowskiMetric

# Test parameters
k = 8.0
coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=1.0,
    bulk_relaxation_time=0.5,
)

# Get analytical eigenmode
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
print("RK4 INTEGRATION TEST (with /τ formulation)")
print("=" * 80)
print()
print(f"Wave number: k = {k}")
print(f"Integration: RK4 (fully explicit, 4th order)")
print()
print(f"Analytical eigenmode:")
print(f"  ω = {mode.frequency:.6f}")
print(f"  γ = {mode.attenuation:.6f}")
print()

# Get eigenmode ratios
omega_complex = complex(mode.frequency, -mode.attenuation)
dispersion_matrix = analytical._build_dispersion_matrix(omega_complex, wave_vector)
U, s, Vh = np.linalg.svd(dispersion_matrix)
eigenvector = Vh[-1, :].conj()
if abs(eigenvector[0]) > 1e-12:
    eigenvector = eigenvector / eigenvector[0]

Pi_ratio = eigenvector[2]
pi_xx_ratio = eigenvector[3]

Pi_phase_analytical = np.angle(Pi_ratio)
pi_phase_analytical = np.angle(pi_xx_ratio)

print(f"Eigenmode ratios:")
print(f"  δΠ/δρ    = {Pi_ratio.real:.4f} + {Pi_ratio.imag:+.4f}j, phase = {Pi_phase_analytical*180/np.pi:.2f}°")
print(f"  δπ_xx/δρ = {pi_xx_ratio.real:.4f} + {pi_xx_ratio.imag:+.4f}j, phase = {pi_phase_analytical*180/np.pi:.2f}°")
print()

# Run benchmark with RK4
benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
)
benchmark.setup_initial_conditions(wave_number=k)

# Get initial state
k_idx = 8
rho_fft_0 = np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0]
Pi_fft_0 = np.fft.fftn(benchmark.fields.Pi)[k_idx, 0, 0]
pi_fft_0 = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])[k_idx, 0, 0]

# Track amplitude for frequency/damping
amplitudes = [abs(rho_fft_0)]
times = [0.0]

print("EVOLUTION (RK4)")
print("=" * 80)

# Evolve with RK4
dt = 0.025
n_steps = 100  # t=2.5

for i in range(n_steps):
    benchmark.solver.time_step(dt, method="rk4")

    if (i + 1) % 20 == 0:
        t = (i + 1) * dt
        rho_fft = np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0]
        amplitudes.append(abs(rho_fft))
        times.append(t)
        print(f"  t = {t:.2f}, |ρ_k| = {abs(rho_fft):.6f}")

print()

# Final state
rho_fft_f = np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0]
Pi_fft_f = np.fft.fftn(benchmark.fields.Pi)[k_idx, 0, 0]
pi_fft_f = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])[k_idx, 0, 0]

Pi_ratio_f = Pi_fft_f / rho_fft_f
pi_ratio_f = pi_fft_f / rho_fft_f

Pi_phase_f = np.angle(Pi_ratio_f)
pi_phase_f = np.angle(pi_ratio_f)

# Measure frequency and damping
times = np.array(times)
amplitudes = np.array(amplitudes)

# Fit exp(-γt) * cos(ωt)
# Use log for damping
log_amp = np.log(amplitudes)
gamma_measured = -(log_amp[-1] - log_amp[0]) / (times[-1] - times[0])

# Frequency from phase
phase = np.angle(rho_fft_f) - np.angle(rho_fft_0)
# Unwrap if needed
if phase < -np.pi:
    phase += 2*np.pi
elif phase > np.pi:
    phase -= 2*np.pi
omega_measured = phase / (times[-1] - times[0])

print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

print("FREQUENCY:")
print(f"  Analytical:  ω = {mode.frequency:.6f}")
print(f"  Measured:    ω = {omega_measured:.6f}")
print(f"  Error:       {abs(omega_measured - mode.frequency)/mode.frequency * 100:.2f}%")
print()

print("DAMPING:")
print(f"  Analytical:  γ = {mode.attenuation:.6f}")
print(f"  Measured:    γ = {gamma_measured:.6f}")
damping_error = abs(gamma_measured - mode.attenuation) / mode.attenuation * 100
if gamma_measured > 0:
    print(f"  Error:       {damping_error:.2f}%")
    print(f"  ✓ Positive damping (stable)")
else:
    print(f"  ✗ NEGATIVE DAMPING (unstable)")
print()

print("PHASE EVOLUTION:")
print(f"  δΠ/δρ:    {Pi_phase_analytical*180/np.pi:.2f}° → {Pi_phase_f*180/np.pi:.2f}° (drift: {(Pi_phase_f - Pi_phase_analytical)*180/np.pi:+.2f}°)")
print(f"  δπ_xx/δρ: {pi_phase_analytical*180/np.pi:.2f}° → {pi_phase_f*180/np.pi:.2f}° (drift: {(pi_phase_f - pi_phase_analytical)*180/np.pi:+.2f}°)")
print()

# Check sign flips
Pi_ratio_0 = Pi_fft_0 / rho_fft_0
pi_ratio_0 = pi_fft_0 / rho_fft_0

Pi_sign_flip = (Pi_ratio_0.imag * Pi_ratio_f.imag) < 0
pi_sign_flip = (pi_ratio_0.imag * pi_ratio_f.imag) < 0

if Pi_sign_flip:
    print("✗ Π imaginary part FLIPPED SIGN")
else:
    print("✓ Π imaginary part maintained sign")

if pi_sign_flip:
    print("✗ π imaginary part FLIPPED SIGN")
else:
    print("✓ π imaginary part maintained sign")

print()
print("=" * 80)
print("SUMMARY: RK4 with /τ formulation")
print("=" * 80)
print()

if abs(omega_measured - mode.frequency)/mode.frequency < 0.05:
    print(f"✓ Frequency error < 5%: {abs(omega_measured - mode.frequency)/mode.frequency * 100:.2f}%")
else:
    print(f"⚠  Frequency error > 5%: {abs(omega_measured - mode.frequency)/mode.frequency * 100:.2f}%")

if gamma_measured > 0 and damping_error < 20:
    print(f"✓ Damping stable with error < 20%: {damping_error:.2f}%")
elif gamma_measured > 0:
    print(f"⚠  Damping stable but error > 20%: {damping_error:.2f}%")
else:
    print(f"✗ Damping UNSTABLE (negative)")

if not Pi_sign_flip and not pi_sign_flip:
    print(f"✓ Phase signs preserved")
else:
    print(f"✗ Phase signs flipped")

print()
print("Conclusion:")
if gamma_measured > 0 and not Pi_sign_flip and not pi_sign_flip:
    print("  RK4 with /τ formulation provides STABLE, PHYSICS-CORRECT evolution")
    print("  → This is the correct implementation!")
else:
    print("  RK4 with /τ formulation still has issues")

print()
print("=" * 80)
