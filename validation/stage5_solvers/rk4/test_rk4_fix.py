"""Test RK4 after removing intermediate normalization."""
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

print("Testing RK4 after fix...")
print()

dt = 0.01
n_steps = int(1.0 / dt)

for i in range(n_steps):
    benchmark.solver.time_step(dt, method="rk4")

A_final = abs(np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0])
ratio = A_final / A_0
gamma = -np.log(ratio) / 1.0

print(f"A(0):   {A_0:.2f}")
print(f"A(1.0): {A_final:.2f}")
print(f"Ratio:  {ratio:.4f}")
print(f"γ:      {gamma:.6f} (expected: 0.200454)")
print()

if ratio > 1.05:
    print("⚠️  STILL UNSTABLE (growing)")
elif ratio < 0.95 and abs(gamma - 0.200454) < 0.02:
    print("✓ STABLE and accurate!")
else:
    print("✓ STABLE but damping not accurate")
