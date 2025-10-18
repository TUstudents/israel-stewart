"""Debug expansion scalar calculation in detail."""
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

print("=" * 80)
print("EXPANSION SCALAR DEBUG")
print("=" * 80)
print()

# Get velocity field
velocity = benchmark.fields.u_mu[..., 1:4]
vx = velocity[..., 0]
vy = velocity[..., 1]
vz = velocity[..., 2]

# FFT
vx_k = np.fft.fftn(vx)
vy_k = np.fft.fftn(vy)
vz_k = np.fft.fftn(vz)

k_idx = 8
print(f"Velocity Fourier modes at k={k}:")
print(f"  v^x_k = {vx_k[k_idx, 0, 0]}")
print(f"  v^y_k = {vy_k[k_idx, 0, 0]}")
print(f"  v^z_k = {vz_k[k_idx, 0, 0]}")
print()

# Manual computation: θ_k = ik_x·v^x_k + ik_y·v^y_k + ik_z·v^z_k
theta_k_manual = 1j * k * vx_k[k_idx, 0, 0]  # k_y = k_z = 0
print(f"Manual θ_k calculation:")
print(f"  θ_k = i·{k}·v^x_k = {theta_k_manual}")
print()

# Using solver's spatial_divergence
theta_solver = benchmark.solver.spectral.spatial_divergence(velocity)
theta_k_solver = np.fft.fftn(theta_solver)[k_idx, 0, 0]

print(f"Solver's spatial_divergence:")
print(f"  θ (real space computed, then FFT)")
print(f"  θ_k = {theta_k_solver}")
print()

# Compare
print(f"Comparison:")
print(f"  Manual:  {theta_k_manual}")
print(f"  Solver:  {theta_k_solver}")
print(f"  Ratio:   {theta_k_solver / theta_k_manual}")
print(f"  Error:   {abs(theta_k_solver - theta_k_manual)/abs(theta_k_manual)*100:.2f}%")
print()

# Check real space theta directly
print("Checking real space θ at specific points:")
# At x=0, the velocity is maximum
idx_x0 = 0
print(f"  At x=0, y=0, z=0:")
print(f"    v^x = {vx[idx_x0, 0, 0]:.6e}")
print(f"    θ   = {theta_solver[idx_x0, 0, 0]:.6e}")
print()

# θ should oscillate with same wavelength as velocity
# For cos(kx) velocity, θ = -k·sin(kx)  
# So θ and v should be 90° out of phase
print("Phase check:")
print(f"  |v^x_k| = {abs(vx_k[k_idx, 0, 0]):.4f}")
print(f"  |θ_k|   = {abs(theta_k_solver):.4f}")
print(f"  phase(v^x_k) = {np.angle(vx_k[k_idx, 0, 0]):.4f} rad")
print(f"  phase(θ_k)   = {np.angle(theta_k_solver):.4f} rad")
print(f"  phase difference = {np.angle(theta_k_solver) - np.angle(vx_k[k_idx, 0, 0]):.4f} rad")
print(f"  (should be π/2 = {np.pi/2:.4f})")
