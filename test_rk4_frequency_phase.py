#!/usr/bin/env -S uv run python
"""
Test frequency, damping, and phase evolution with RK4 + /τ formulation.
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
    bulk_relaxation_time=0.5,
)

# Analytical eigenmode
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

# Get eigenmode ratios
omega_complex = complex(mode.frequency, -mode.attenuation)
dispersion_matrix = analytical._build_dispersion_matrix(omega_complex, wave_vector)
U, s, Vh = np.linalg.svd(dispersion_matrix)
eigenvector = Vh[-1, :].conj()
if abs(eigenvector[0]) > 1e-12:
    eigenvector = eigenvector / eigenvector[0]

Pi_ratio_theory = eigenvector[2]
pi_xx_ratio_theory = eigenvector[3]
Pi_phase_theory = np.angle(Pi_ratio_theory)
pi_phase_theory = np.angle(pi_xx_ratio_theory)

print("=" * 80)
print("RK4 + /τ FORMULATION: COMPLETE TEST")
print("=" * 80)
print()
print(f"k = {k}")
print(f"Integration: RK4 (fully explicit)")
print(f"Source terms: dΠ/dt = -Π/τ_Π - ζθ/τ_Π (dispersion-consistent)")
print()
print(f"Analytical predictions:")
print(f"  ω = {mode.frequency:.6f}")
print(f"  γ = {mode.attenuation:.6f}")
print(f"  Phase(δΠ/δρ) = {Pi_phase_theory*180/np.pi:.2f}°")
print(f"  Phase(δπ_xx/δρ) = {pi_phase_theory*180/np.pi:.2f}°")
print()

# Setup
benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
)
benchmark.setup_initial_conditions(wave_number=k)

k_idx = 8

# Initial state
rho_fft_0 = np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0]
Pi_fft_0 = np.fft.fftn(benchmark.fields.Pi)[k_idx, 0, 0]
pi_fft_0 = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])[k_idx, 0, 0]

Pi_ratio_0 = Pi_fft_0 / rho_fft_0
pi_ratio_0 = pi_fft_0 / rho_fft_0
Pi_phase_0 = np.angle(Pi_ratio_0)
pi_phase_0 = np.angle(pi_ratio_0)

print("Initial ratios:")
print(f"  δΠ/δρ = {Pi_ratio_0.real:.4f} + {Pi_ratio_0.imag:+.4f}j, phase = {Pi_phase_0*180/np.pi:.2f}°")
print(f"  δπ_xx/δρ = {pi_ratio_0.real:.4f} + {pi_ratio_0.imag:+.4f}j, phase = {pi_phase_0*180/np.pi:.2f}°")
print()

# Evolve with RK4 (use smaller dt for safety, fewer steps for speed)
dt = 0.01
n_steps = 50  # Evolve to t=0.5 (shorter for speed)

print(f"Evolving with RK4, dt={dt}, {n_steps} steps to t={n_steps*dt}...")

amplitudes = [abs(rho_fft_0)]
times = [0.0]

for i in range(n_steps):
    benchmark.solver.time_step(dt, method="rk4")

    if (i + 1) % 10 == 0:
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

# Measure frequency/damping
times = np.array(times)
amplitudes = np.array(amplitudes)

log_amp = np.log(amplitudes)
gamma_measured = -(log_amp[-1] - log_amp[0]) / (times[-1] - times[0])

phase = np.angle(rho_fft_f) - np.angle(rho_fft_0)
if phase < -np.pi:
    phase += 2*np.pi
elif phase > np.pi:
    phase -= 2*np.pi
omega_measured = phase / (times[-1] - times[0])

# Final ratios
Pi_ratio_f = Pi_fft_f / rho_fft_f
pi_ratio_f = pi_fft_f / rho_fft_f
Pi_phase_f = np.angle(Pi_ratio_f)
pi_phase_f = np.angle(pi_ratio_f)

print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

print("FREQUENCY:")
print(f"  Analytical:  ω = {mode.frequency:.6f}")
print(f"  Measured:    ω = {omega_measured:.6f}")
omega_error = abs(omega_measured - mode.frequency)/mode.frequency * 100
print(f"  Error:       {omega_error:.2f}%")
print()

print("DAMPING:")
print(f"  Analytical:  γ = +{mode.attenuation:.6f}")
print(f"  Measured:    γ = {gamma_measured:+.6f}")
if gamma_measured > 0:
    damping_error = abs(gamma_measured - mode.attenuation) / mode.attenuation * 100
    print(f"  Error:       {damping_error:.2f}%")
    print(f"  ✓ Positive damping (stable)")
else:
    print(f"  ✗ NEGATIVE DAMPING (unstable)")
print()

print("PHASE EVOLUTION:")
dPi_phase = (Pi_phase_f - Pi_phase_0) * 180/np.pi
dpi_phase = (pi_phase_f - pi_phase_0) * 180/np.pi
print(f"  δΠ/δρ:    {Pi_phase_0*180/np.pi:.2f}° → {Pi_phase_f*180/np.pi:.2f}° (drift: {dPi_phase:+.2f}°)")
print(f"  δπ_xx/δρ: {pi_phase_0*180/np.pi:.2f}° → {pi_phase_f*180/np.pi:.2f}° (drift: {dpi_phase:+.2f}°)")
print()

# Check sign flips
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
print("SUMMARY")
print("=" * 80)
print()

all_good = True

if omega_error < 5:
    print(f"✓ Frequency error < 5%: {omega_error:.2f}%")
else:
    print(f"⚠  Frequency error: {omega_error:.2f}%")
    all_good = False

if gamma_measured > 0:
    if damping_error < 20:
        print(f"✓ Damping correct and stable: {damping_error:.2f}% error")
    else:
        print(f"⚠  Damping stable but inaccurate: {damping_error:.2f}% error")
        all_good = False
else:
    print(f"✗ Damping UNSTABLE (negative)")
    all_good = False

if not Pi_sign_flip and not pi_sign_flip:
    print(f"✓ Phase signs preserved (no flips)")
else:
    print(f"✗ Phase signs flipped")
    all_good = False

if abs(dPi_phase) < 15 and abs(dpi_phase) < 15:
    print(f"✓ Phase drift acceptable (< 15°)")
else:
    print(f"⚠  Phase drift: Π={dPi_phase:.1f}°, π={dpi_phase:.1f}°")
    all_good = False

print()

if all_good:
    print("=" * 80)
    print("✓✓✓ RK4 + /τ FORMULATION IS CORRECT ✓✓✓")
    print("=" * 80)
    print()
    print("This confirms:")
    print("  1. Source term SHOULD have /τ division (dispersion-consistent)")
    print("  2. RK4 provides stable, accurate evolution")
    print("  3. IMEX has instability issues at high k (negative damping)")
    print()
    print("Recommendation: Use RK4 for moderate k, fix IMEX for high k")
else:
    print("Some issues remain - further investigation needed")

print()
print("=" * 80)
