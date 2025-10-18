"""Test IMEX stability with detailed diagnostics."""
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

# Track k=8 mode amplitude over time
k_idx = 8
def get_amplitude(fields):
    rho_fft = np.fft.fftn(fields.rho - 1.0)
    return abs(rho_fft[k_idx, 0, 0])

print("=" * 80)
print("IMEX STABILITY DIAGNOSTIC")
print("=" * 80)
print()

A_0 = get_amplitude(benchmark.fields)
print(f"Initial amplitude: {A_0:.6e}")
print()

# Evolve for short time with very small steps
times = []
amplitudes = []
dt = 0.01

for i in range(20):
    t = i * dt
    A = get_amplitude(benchmark.fields)
    times.append(t)
    amplitudes.append(A)
    
    if i % 5 == 0:
        ratio = A / A_0
        print(f"t = {t:.3f}: A/A_0 = {ratio:.6f}")
    
    benchmark.solver.time_step(dt, method="spectral_imex")

# Check if growing or decaying
print()
A_final = amplitudes[-1]
ratio = A_final / A_0
print(f"Final amplitude ratio: {ratio:.6f}")
if ratio > 1.01:
    print("⚠️  GROWING (UNSTABLE)")
elif ratio < 0.99:
    print("✓ DECAYING (STABLE)")
else:
    print("≈ CONSTANT")

# Compute growth rate
t_final = times[-1]
gamma_measured = np.log(A_final / A_0) / t_final
print(f"Measured γ: {gamma_measured:.6f}")
print(f"Expected γ: +0.200454 (positive = decay)")
print()
