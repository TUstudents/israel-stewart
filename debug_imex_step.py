"""Debug a single IMEX step to find sign errors."""
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

# Get initial Fourier mode
k_idx = 8
rho_k_0 = np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0]
Pi_k_0 = np.fft.fftn(benchmark.fields.Pi)[k_idx, 0, 0]
pi_k_0 = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])[k_idx, 0, 0]

print("INITIAL STATE (k=8 mode):")
print(f"  ρ_k = {rho_k_0}")
print(f"  Π_k = {Pi_k_0}")
print(f"  π_k = {pi_k_0}")
print()

# Take ONE IMEX step
dt = 0.01
benchmark.solver.time_step(dt, method="spectral_imex")

# Check after one step
rho_k_1 = np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0]
Pi_k_1 = np.fft.fftn(benchmark.fields.Pi)[k_idx, 0, 0]
pi_k_1 = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])[k_idx, 0, 0]

print("AFTER ONE STEP (dt=0.01):")
print(f"  ρ_k = {rho_k_1}")
print(f"  Π_k = {Pi_k_1}")
print(f"  π_k = {pi_k_1}")
print()

# Compute expected evolution based on analytical
omega = 5.457140
gamma = 0.200454
omega_complex = complex(omega, -gamma)

# Expected: ρ_k(t) = ρ_k(0) * exp(-i*omega*t) * exp(-gamma*t)
# At small t: ρ_k(dt) ≈ ρ_k(0) * (1 - i*omega*dt) * (1 - gamma*dt)
expected_factor = np.exp(-1j * omega_complex * dt)
rho_k_expected = rho_k_0 * expected_factor

print("EXPECTED EVOLUTION:")
print(f"  ρ_k expected = {rho_k_expected}")
print()

print("COMPARISON:")
print(f"  |ρ_k| actual:   {abs(rho_k_1):.6f}")
print(f"  |ρ_k| expected: {abs(rho_k_expected):.6f}")
print(f"  Ratio: {abs(rho_k_1)/abs(rho_k_expected):.6f}")

if abs(rho_k_1) > abs(rho_k_0):
    print("  ⚠️  AMPLITUDE GROWING!")
elif abs(rho_k_1) < abs(rho_k_0) * 0.99:
    print("  ✓ AMPLITUDE DECAYING")
else:
    print("  ≈ AMPLITUDE CONSTANT")
