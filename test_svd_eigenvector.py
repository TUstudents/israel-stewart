#!/usr/bin/env python
"""Test if SVD gives the correct nullspace vector."""

import numpy as np

# Create a simple test matrix with known nullspace
# M = [[1, 2], [2, 4]] has nullspace [-2, 1]
M_test = np.array([[1.0, 2.0], [2.0, 4.0]])

U, s, Vh = np.linalg.svd(M_test)
print("Test matrix M:")
print(M_test)
print()
print(f"Singular values: {s}")
print(f"Smallest: {s[-1]:.6e}")
print()

# Nullspace vector from SVD
v_svd = Vh[-1, :]
print(f"SVD nullspace vector: {v_svd}")
print(f"Expected: [-0.894, 0.447] (normalized [-2, 1])")
print()

residual = M_test @ v_svd
print(f"Residual M·v: {residual}")
print(f"||M·v||: {np.linalg.norm(residual):.6e}")
print()

# Now test the actual dispersion matrix
print("="*80)
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=0.5,
    bulk_relaxation_time=0.3
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2*np.pi,
    grid_points=(32, 32, 16),
    transport_coeffs=coeffs
)

k = 8.0
wave_vector = np.array([k, 0.0, 0.0])
analytical_modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)
mode = analytical_modes[0]

omega_complex = complex(mode.frequency, -mode.attenuation)
matrix = benchmark.analytical._build_dispersion_matrix(omega_complex, wave_vector)

print("Dispersion matrix:")
print(matrix)
print()

# Compute SVD carefully
U, s, Vh = np.linalg.svd(matrix)
print(f"Singular values: {s}")
print()

# The nullspace vector is the COLUMN of V (or row of V^H) corresponding to smallest singular value
print("V^H (rows are right singular vectors):")
for i in range(4):
    print(f"  v_{i} (s={s[i]:.2e}): {Vh[i, :]}")
print()

# Check each singular vector
for i in range(4):
    v_i = Vh[i, :]
    residual_i = matrix @ v_i
    print(f"Residual for v_{i}:")
    print(f"  ||M·v||: {np.linalg.norm(residual_i):.6e}")
    print(f"  Expected: {s[i]:.6e} (should match singular value from M = U·S·V^H)")
    print(f"  Ratio: {np.linalg.norm(residual_i) / s[i]:.6e} (should be ~1 for SVD property)")
    print()

# WAIT: For SVD, M·v_i = s_i * u_i, NOT zero!
# To get nullspace, we need s_i ≈ 0, then M·v_i ≈ 0
print("="*80)
print("KEY INSIGHT:")
print("For SVD: M = U·S·V^H")
print("Property: M·v_i = s_i * u_i")
print()
print("For nullspace: need s_i ≈ 0, then M·v_i ≈ 0")
print(f"Here: s_3 = {s[-1]:.2e} ≈ 0, so v_3 is nullspace vector")
print()

# Verify
v_null = Vh[-1, :]
residual_null = matrix @ v_null
expected_residual = s[-1] * U[:, -1]  # Should be s_3 * u_3

print(f"Nullspace vector: {v_null}")
print(f"Residual M·v: {residual_null}")
print(f"Expected (s*u): {expected_residual}")
print(f"||M·v||: {np.linalg.norm(residual_null):.6e}")
print(f"||s*u||: {np.linalg.norm(expected_residual):.6e}")
print()

# AH! The issue is that we're computing ||M·v|| incorrectly!
# Let me recalculate
print("Recalculating residual magnitude:")
print(f"  M·v components: {residual_null}")
res_abs = [abs(x) for x in residual_null]
print(f"  |M·v_i|: {res_abs}")
print(f"  ||M·v|| = sqrt(sum |M·v_i|^2) = {np.sqrt(sum(x**2 for x in res_abs)):.6e}")
print()

# The ||M·v|| = 1.3 we saw earlier must be wrong. Let me verify:
print("From earlier debug:")
print("  eigenvector_rotated components:", eigenvector_rotated := Vh[-1, :] * np.exp(4.696522j) )
