"""Pinpoint when instability appears."""
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
A_0 = abs(np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0])

print("Tracking when amplitude starts growing...")
print("t\t|ρ_k|\t\tRatio")
print("-" * 50)

dt = 0.005  # Same as before
t_max = 2.0  # Check first 2 seconds
for i in range(int(t_max/dt) + 1):
    t = i * dt
    A = abs(np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0])
    ratio = A / A_0
    
    if i % 40 == 0:  # Every 0.2 seconds
        status = "GROWING" if ratio > 1.01 else ("DECAY" if ratio < 0.99 else "CONST")
        print(f"{t:.2f}\t{A:.2f}\t\t{ratio:.4f}\t{status}")
    
    if i < int(t_max/dt):
        benchmark.solver.time_step(dt, method="spectral_imex")
