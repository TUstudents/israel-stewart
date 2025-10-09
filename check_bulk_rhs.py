"""Check bulk relaxation RHS after IMEX step."""
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

# Initial state
Pi_k_0 = np.fft.fftn(benchmark.fields.Pi)[k_idx, 0, 0]
v_k_0 = np.fft.fftn(benchmark.fields.u_mu[..., 1])[k_idx, 0, 0]

print(f"Initial Π_k = {Pi_k_0}")
print(f"Initial v_k = {v_k_0}")
print()

# Take step
dt = 0.01
benchmark.solver.time_step(dt, method="spectral_imex")

# Get updated state
Pi_k_1 = np.fft.fftn(benchmark.fields.Pi)[k_idx, 0, 0]
v_k_1 = np.fft.fftn(benchmark.fields.u_mu[..., 1])[k_idx, 0, 0]
theta_k_1 = 1j * k * v_k_1

print(f"After step:")
print(f"  Π_k = {Pi_k_1}")
print(f"  v_k = {v_k_1}")
print(f"  θ_k = {theta_k_1}")
print()

# Compute relaxation RHS at updated state
relax_rhs = benchmark.solver.relaxation.compute_relaxation_rhs(benchmark.fields)
Pi_size = benchmark.fields.Pi.size
dPi_dt = relax_rhs[:Pi_size].reshape(benchmark.fields.Pi.shape)
dPi_k = np.fft.fftn(dPi_dt)[k_idx, 0, 0]

# Expected: dΠ/dt = -Π/τ_Π - ζθ
tau_Pi = 0.5
zeta = 0.04
expected_dPi_k = -Pi_k_1/tau_Pi - zeta*theta_k_1

print(f"Bulk relaxation RHS:")
print(f"  dΠ/dt expected = {expected_dPi_k}")
print(f"  dΠ/dt actual   = {dPi_k}")
print(f"  Error: {abs(dPi_k - expected_dPi_k)/abs(expected_dPi_k)*100:.4f}%")
print()

# Analytical eigenmode evolution
omega = 5.457140
gamma = 0.200454
omega_c = complex(omega, -gamma)

Pi_expected = Pi_k_0 * np.exp(-1j * omega_c * dt)
dPi_analytical = -1j * omega_c * Pi_expected

print(f"For perfect eigenmode:")
print(f"  Π_k should be = {Pi_expected}")
print(f"  Π_k actually  = {Pi_k_1}")
print(f"  Error: {abs(Pi_k_1 - Pi_expected)/abs(Pi_expected)*100:.4f}%")
print()
print(f"  dΠ/dt should be = {dPi_analytical}")
print(f"  dΠ/dt actually  = {dPi_k}")
print(f"  Error: {abs(dPi_k - dPi_analytical)/abs(dPi_analytical)*100:.4f}%")
