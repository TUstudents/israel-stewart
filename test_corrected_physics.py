"""Quick test of corrected physics - measure frequency and damping."""
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
    domain_size=2 * np.pi,
    grid_points=(32, 32, 16),
    transport_coeffs=coeffs
)
k = 1.0
benchmark.setup_initial_conditions(wave_number=k)

k_idx = int(k)

# Analytical eigenmode
omega_analytical = 0.599320
gamma_analytical = 0.045869

print("=" * 80)
print("CORRECTED PHYSICS TEST")
print("=" * 80)
print(f"\nAnalytical predictions (from dispersion relation):")
print(f"  ω = {omega_analytical:.6f}")
print(f"  γ = {gamma_analytical:.6f}")
print()

# Get initial amplitude
rho_k_0 = np.fft.fftn(benchmark.fields.rho)[k_idx, 0, 0]
Pi_k_0 = np.fft.fftn(benchmark.fields.Pi)[k_idx, 0, 0]
amplitude_0 = abs(rho_k_0)

# Evolve for short time
dt = 0.01
n_steps = 300
times = []
amplitudes = []
phases = []

print("Evolving for 300 steps (t = 3.0)...")
for i in range(n_steps):
    t = i * dt

    # Record every 10 steps
    if i % 10 == 0:
        rho_k = np.fft.fftn(benchmark.fields.rho)[k_idx, 0, 0]
        times.append(t)
        amplitudes.append(abs(rho_k))
        phases.append(np.angle(rho_k))

    benchmark.solver.time_step(dt, method="spectral_imex")

times = np.array(times)
amplitudes = np.array(amplitudes)
phases = np.unwrap(np.array(phases))

# Measure damping from amplitude
log_amp = np.log(amplitudes)
gamma_fit = -np.polyfit(times, log_amp, 1)[0]

# Measure frequency from phase
omega_fit = -np.polyfit(times, phases, 1)[0]

# Calculate errors
freq_error = abs(omega_fit - omega_analytical) / omega_analytical * 100
gamma_error = abs(gamma_fit - gamma_analytical) / gamma_analytical * 100

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
print(f"  Analytical:  γ = {gamma_analytical:.6f}")
print(f"  Measured:    γ = {gamma_fit:.6f}")
print(f"  Error:       {gamma_error:.2f}%")
print()

# Check if unstable
if gamma_fit < 0:
    print("⚠️  WARNING: Negative damping detected - numerical instability!")
    print(f"    γ = {gamma_fit:.6f} (should be positive)")
else:
    print("✓ Damping is positive (stable)")

print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()
if freq_error < 1.0:
    print("✓ Frequency accuracy < 1% confirms source term fix is correct")
else:
    print(f"✗ Frequency error {freq_error:.2f}% suggests physics issue")

if gamma_fit < 0:
    print("✗ Negative damping indicates IMEX numerical instability")
    print("  This is a separate issue from the source term fix")
elif abs(gamma_fit - gamma_analytical) > 0.05:
    print(f"⚠️  Damping error is large but still positive")
else:
    print("✓ Damping matches analytical prediction")
