"""
Check if stress-energy tensor is computed correctly for momentum conservation.

The momentum conservation equation is:
  ∂_t(T^0j) + ∂_i(T^ij) = 0
  → ∂_t(T^0j) = -∂_i(T^ij)

For a relativistic fluid, the momentum density is:
  T^0j = (ε+p)u^0·u^j ≈ h·u^j  (in rest frame where u^0 ≈ 1)

NOT ρ·u^j!

This script checks if the conservation equations use the correct momentum density.
"""

import numpy as np

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

# Setup
coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=1.0,
    bulk_relaxation_time=0.5,
    lambda_pi_pi=0.0,
    lambda_pi_Pi=0.0,
    xi_1=0.0,
    xi_2=0.0,
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
)

k = 8.0
benchmark.setup_initial_conditions(wave_number=k)

print("=" * 80)
print("STRESS-ENERGY TENSOR CHECK")
print("=" * 80)
print()

# Compute stress-energy tensor
T = benchmark.solver.conservation.stress_energy_tensor()

# At k=8, check T^0x and T^xx
k_idx = 8

T_0x = T[k_idx, 0, 0, 0, 1]  # T^0x at (k_idx, 0, 0)
T_xx = T[k_idx, 0, 0, 1, 1]  # T^xx at (k_idx, 0, 0)

print(f"At grid point (k_idx={k_idx}, 0, 0):")
print(f"  T^0x (momentum density): {T_0x:.6e}")
print(f"  T^xx (xx stress):        {T_xx:.6e}")
print()

# Check what it should be
rho = benchmark.fields.rho[k_idx, 0, 0]
p = benchmark.fields.pressure[k_idx, 0, 0]
v_x = benchmark.fields.u_mu[k_idx, 0, 0, 1]
Pi = benchmark.fields.Pi[k_idx, 0, 0]
pi_xx = benchmark.fields.pi_munu[k_idx, 0, 0, 1, 1]

h = rho + p  # Enthalpy

print("Field values:")
print(f"  ρ  = {rho:.6e}")
print(f"  p  = {p:.6e}")
print(f"  h  = {h:.6e}")
print(f"  v_x = {v_x:.6e}")
print(f"  Π  = {Pi:.6e}")
print(f"  π_xx = {pi_xx:.6e}")
print()

# Perfect fluid contribution to T^0x
# In rest frame: T^0x = (ε+p)·u^0·u^x ≈ h·v_x (since u^0 ≈ 1, u^x ≈ v_x)
T_0x_perfect = h * v_x

print("T^0x components:")
print(f"  Perfect fluid (h·v_x): {T_0x_perfect:.6e}")
print(f"  Actual T^0x:           {T_0x:.6e}")
print(f"  Match: {np.allclose(T_0x, T_0x_perfect, rtol=1e-6)}")
print()

# T^xx components
T_xx_perfect = p  # In rest frame, spatial part of T^μν is p·g^ij
T_xx_with_diss = T_xx_perfect + Pi + pi_xx

print("T^xx components:")
print(f"  Perfect fluid (p):       {T_xx_perfect:.6e}")
print(f"  + Bulk (Π):              {Pi:.6e}")
print(f"  + Shear (π_xx):          {pi_xx:.6e}")
print(f"  Total (expected):        {T_xx_with_diss:.6e}")
print(f"  Actual T^xx:             {T_xx:.6e}")
print(f"  Match: {np.allclose(T_xx, T_xx_with_diss, rtol=1e-6)}")
print()

# Now check the divergence
print("=" * 80)
print("MOMENTUM CONSERVATION")
print("=" * 80)
print()

# The conservation equation is:
# ∂_t(T^0x) = -∂_i(T^ix) = -∂_x(T^xx) - ∂_y(T^yx) - ∂_z(T^zx)

# Compute spatial divergence of T^ix
T_flux_x = T[..., 1:4, 1]  # (T^1x, T^2x, T^3x)

# Compute divergence
div_T_x = benchmark.solver.spectral.spatial_divergence(T_flux_x)

print(f"At grid point (k_idx={k_idx}, 0, 0):")
print(f"  ∇·T^ix = {div_T_x[k_idx, 0, 0]:.6e}")
print(f"  ∂_t(T^0x) should be: {-div_T_x[k_idx, 0, 0]:.6e}")
print()

# FFT to check k=8 mode
T_0x_fft = np.fft.fftn(T[..., 0, 1])
div_T_x_fft = np.fft.fftn(div_T_x)

k_idx_fft = 8
T_0x_k = T_0x_fft[k_idx_fft, 0, 0]
div_T_x_k = div_T_x_fft[k_idx_fft, 0, 0]

print(f"Fourier mode k={k}:")
print(f"  T^0x(k) = {T_0x_k}")
print(f"  ∂_t(T^0x)(k) = -∇·T^ix(k) = {-div_T_x_k}")
print()

# This should match d(h·v)/dt
# Get analytical expectation
wave_vector = np.array([k, 0.0, 0.0])
modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)
mode = modes[0]
omega = complex(mode.frequency, -mode.attenuation)

v_fft = np.fft.fftn(benchmark.fields.u_mu[..., 1])
v_k = v_fft[k_idx_fft, 0, 0]

# For linearized: h ≈ h₀ = 4/3
# T^0x = h·v ≈ h₀·v
# ∂_t(T^0x) = h₀·∂_t(v) = h₀·(-iω·v)
h_0 = 4.0 / 3.0
d_T0x_dt_analytical = h_0 * (-1j * omega * v_k)

print("Analytical expectation:")
print(f"  ∂_t(T^0x)(k) = h₀·(-iω·v_k) = {d_T0x_dt_analytical}")
print()

print("Comparison:")
print(f"  From ∇·T:     {-div_T_x_k}")
print(f"  Analytical:   {d_T0x_dt_analytical}")
print(
    f"  Ratio:        {-div_T_x_k / d_T0x_dt_analytical if abs(d_T0x_dt_analytical) > 1e-14 else 'N/A'}"
)
print()

if np.allclose(-div_T_x_k, d_T0x_dt_analytical, rtol=0.01):
    print("✓ Conservation equation gives correct ∂_t(T^0x)")
else:
    error = abs((-div_T_x_k - d_T0x_dt_analytical) / d_T0x_dt_analytical) * 100
    print(f"✗ Conservation equation is WRONG by {error:.2f}%")

print()
print("=" * 80)
