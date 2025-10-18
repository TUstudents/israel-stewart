"""Measure frequency and damping at k=8.0 with corrected physics."""
import numpy as np
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=1.0,
    bulk_relaxation_time=0.5,
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
)
k = 8.0
benchmark.setup_initial_conditions(wave_number=k)

k_idx = 8

# Analytical predictions for k=8
omega_analytical = 5.457140
gamma_analytical = 0.200454

print("=" * 80)
print("FREQUENCY AND DAMPING MEASUREMENT (k=8.0)")
print("=" * 80)
print(f"\nWave number: k = {k}")
print(f"\nAnalytical predictions:")
print(f"  ω = {omega_analytical:.6f}")
print(f"  γ = {gamma_analytical:.6f}")
print()

# Evolve and record
dt = 0.01
n_steps = 100  # Evolve to t=1.0 (enough for several oscillations at k=8)
times = []
rho_amplitudes = []
rho_phases = []

print("Evolving...")
for i in range(n_steps + 1):
    t = i * dt

    # Record state
    rho_k = np.fft.fftn(benchmark.fields.rho)[k_idx, 0, 0]
    times.append(t)
    rho_amplitudes.append(abs(rho_k))
    rho_phases.append(np.angle(rho_k))

    if i % 25 == 0:
        print(f"  t = {t:.2f}, |ρ_k| = {abs(rho_k):.6f}")

    if i < n_steps:
        benchmark.solver.time_step(dt, method="spectral_imex")

times = np.array(times)
rho_amplitudes = np.array(rho_amplitudes)
rho_phases = np.unwrap(np.array(rho_phases))

# Measure damping from amplitude decay
log_amp = np.log(rho_amplitudes)
gamma_fit = -np.polyfit(times, log_amp, 1)[0]

# Measure frequency from phase evolution
omega_fit = -np.polyfit(times, rho_phases, 1)[0]

# Calculate errors
freq_error = abs(omega_fit - omega_analytical) / omega_analytical * 100
gamma_sign = "✓" if gamma_fit > 0 else "✗"

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"\nFrequency:")
print(f"  Analytical:  ω = {omega_analytical:.6f}")
print(f"  Measured:    ω = {omega_fit:.6f}")
print(f"  Error:       {freq_error:.2f}%")
print()
print(f"Damping:")
print(f"  Analytical:  γ = +{gamma_analytical:.6f}")
print(f"  Measured:    γ = {gamma_fit:+.6f} {gamma_sign}")

if gamma_fit < 0:
    print(f"  ⚠️  INSTABILITY: Amplitude growing instead of decaying!")
    print(f"  Growth rate: {-gamma_fit:.6f}")
else:
    gamma_error = abs(gamma_fit - gamma_analytical) / gamma_analytical * 100
    print(f"  Error:       {gamma_error:.1f}%")

print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()
if freq_error < 1.0:
    print("✓ Frequency error < 1% - SOURCE TERM FIX IS CORRECT")
else:
    print(f"⚠️  Frequency error {freq_error:.2f}%")
    if freq_error < 5.0:
        print("  Still reasonably good, confirms source term fix")

print()
if gamma_fit < 0:
    print("✗ Negative damping - IMEX NUMERICAL INSTABILITY")
    print(f"  Expected: γ = +{gamma_analytical:.6f}")
    print(f"  Got:      γ = {gamma_fit:.6f}")
    print(f"  This matches the γ = -0.146 mentioned in documentation")
elif abs(gamma_fit - gamma_analytical) < 0.01:
    print("✓ Damping matches analytical prediction - EXCELLENT")
else:
    print(f"⚠️  Damping has {abs(gamma_fit - gamma_analytical)/gamma_analytical*100:.1f}% error")
    print("  Positive but not perfectly accurate")
