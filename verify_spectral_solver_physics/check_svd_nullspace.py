"""
Check if the SVD eigenvector is actually a null vector of the dispersion matrix.

For a true eigenmode, the dispersion matrix should be singular, meaning:
  M·v = 0

If M is NOT exactly singular (due to numerical errors or approximations),
then M·v ≠ 0, and the initialized fields are not an exact eigenmode.
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

k = 8.0
benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
)

# Get analytical eigenmode
wave_vector = np.array([k, 0.0, 0.0])
modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)
mode = modes[0]
omega = complex(mode.frequency, -mode.attenuation)

print("=" * 80)
print("SVD NULLSPACE CHECK")
print("=" * 80)
print()
print(f"Testing k={k}, ω={omega}")
print()

# Build dispersion matrix
dispersion_matrix = benchmark.analytical._build_dispersion_matrix(omega, wave_vector)

print("Dispersion matrix:")
print(dispersion_matrix)
print()

# SVD decomposition
U, s, Vh = np.linalg.svd(dispersion_matrix)

print("Singular values:")
for i, sv in enumerate(s):
    print(f"  s[{i}] = {sv:.6e}")
print()

# Check if smallest singular value is actually small
if abs(s[-1]) < 1e-10:
    print(f"✓ Smallest singular value {s[-1]:.2e} is very small")
    print("  Matrix is nearly singular (good for eigenmode)")
else:
    print(f"⚠ Smallest singular value {s[-1]:.2e} is NOT small")
    print("  Matrix is not singular - eigenmode might be inaccurate!")

print()

# Get eigenvector
eigenvector = Vh[-1, :].conj()
eigenvector_normalized = eigenvector / eigenvector[0]

print("Eigenvector (normalized so δρ = 1):")
print(f"  δρ:    {eigenvector_normalized[0]}")
print(f"  δv_x:  {eigenvector_normalized[1]}")
print(f"  δΠ:    {eigenvector_normalized[2]}")
print(f"  δπ_xx: {eigenvector_normalized[3]}")
print()

# Check nullspace: M·v should be ~0
residual = dispersion_matrix @ eigenvector_normalized

print("Nullspace residual M·v:")
print(f"  |M·v| = {np.linalg.norm(residual):.6e}")
print("Components:")
for i, r in enumerate(residual):
    print(f"  (M·v)[{i}] = {r:.6e}")
print()

# Check individual equation residuals relative to terms
print("Relative residuals:")
# For each row of M·v, compare to the magnitude of M·v_j for each component
for i in range(4):
    row_contributions = abs(dispersion_matrix[i, :] * eigenvector_normalized)
    max_term = np.max(row_contributions)
    rel_residual = abs(residual[i]) / max_term if max_term > 1e-14 else 0
    print(f"  Equation {i}: {rel_residual*100:.2f}% (abs residual: {abs(residual[i]):.2e})")

print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()
print("If M·v is very small (< 1e-12):")
print("  - Eigenvector is accurate nullspace vector")
print("  - Initialization is correct eigenmode")
print()
print("If M·v is NOT small:")
print("  - Dispersion matrix has numerical errors")
print("  - OR: discretization doesn't exactly match analytical theory")
print("  - Fields will drift immediately upon evolution")
print()
print("=" * 80)
