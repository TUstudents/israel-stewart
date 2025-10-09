"""Check theta after IMEX step."""
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

# Get initial velocity
v_k_0 = np.fft.fftn(benchmark.fields.u_mu[..., 1])[k_idx, 0, 0]
print(f"Initial v_k = {v_k_0}")

# Expected after dt
omega = 5.457140
gamma = 0.200454
dt = 0.01
v_k_expected = v_k_0 * np.exp(-1j * complex(omega, -gamma) * dt)
print(f"Expected v_k after dt = {v_k_expected}")

# Take IMEX step
benchmark.solver.time_step(dt, method="spectral_imex")

# Check actual velocity
v_k_actual = np.fft.fftn(benchmark.fields.u_mu[..., 1])[k_idx, 0, 0]
print(f"Actual v_k after dt = {v_k_actual}")
print(f"Velocity error: {abs(v_k_actual - v_k_expected)/abs(v_k_expected)*100:.4f}%")
print()

# Now compute θ from this velocity
velocity = benchmark.fields.u_mu[..., 1:4]
theta_solver = benchmark.solver.spectral.spatial_divergence(velocity)
theta_k = np.fft.fftn(theta_solver)[k_idx, 0, 0]

# Expected theta
theta_k_expected = 1j * k * v_k_expected
print(f"Expected θ_k = ik·v_k = {theta_k_expected}")
print(f"Actual θ_k (from solver) = {theta_k}")
print(f"Expansion error: {abs(theta_k - theta_k_expected)/abs(theta_k_expected)*100:.4f}%")
print()

# Manual computation
vx_k = np.fft.fftn(velocity[..., 0])[k_idx, 0, 0]
theta_k_manual = 1j * k * vx_k
print(f"Manual θ_k = ik·v^x_k = {theta_k_manual}")
print(f"Manual matches solver: {np.allclose(theta_k, theta_k_manual)}")
