"""Test if normalization is causing instability."""
import numpy as np
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

# Monkey-patch to disable normalization
original_normalize = None

def disable_normalization(self):
    """Do nothing instead of normalizing."""
    pass

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

# Disable normalization
from israel_stewart.core.fields import ISFieldConfiguration
original_normalize = ISFieldConfiguration.normalize_four_velocity
ISFieldConfiguration.normalize_four_velocity = disable_normalization

k_idx = 8
A_0 = abs(np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0])

print("Testing RK4 WITHOUT four-velocity normalization...")
print()

for i in range(100):
    benchmark.solver.time_step(0.01, method="rk4")

A_final = abs(np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0])
ratio = A_final / A_0
gamma = -np.log(ratio) / 1.0

print(f"A(0):   {A_0:.2f}")
print(f"A(1.0): {A_final:.2f}")
print(f"Ratio:  {ratio:.4f}")
print(f"γ:      {gamma:.6f}")
print()

if ratio > 1.05:
    print("⚠️  STILL UNSTABLE")
else:
    print("✓ STABLE")

# Restore original
ISFieldConfiguration.normalize_four_velocity = original_normalize
