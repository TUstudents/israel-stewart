#!/usr/bin/env python
"""Check if we can refine the dispersion relation root to reduce condition number."""

import numpy as np
from scipy.optimize import fsolve
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

# Setup
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

# Get current root
analytical_modes = benchmark.analytical.analyze_dispersion_relation(
    wave_vector=np.array([k, 0.0, 0.0])
)
mode = analytical_modes[0]
omega_initial = complex(mode.frequency, -mode.attenuation)

print("="*80)
print("DISPERSION ROOT REFINEMENT")
print("="*80)
print(f"Initial root: ω = {omega_initial}")
print()

# Check determinant
det_initial = benchmark.analytical._determinant_function(omega_initial, k)
print(f"det(M) at initial root: {det_initial}")
print(f"|det(M)|: {abs(det_initial):.6e}")
print()

# Refine using fsolve
def det_real_imag(omega_vec):
    """Determinant as real/imag pair for fsolve."""
    omega = complex(omega_vec[0], omega_vec[1])
    det = benchmark.analytical._determinant_function(omega, k)
    return [det.real, det.imag]

omega_vec_initial = [omega_initial.real, omega_initial.imag]
omega_vec_refined, info, ier, msg = fsolve(det_real_imag, omega_vec_initial, full_output=True)

print("Root refinement result:")
print(f"  Initial: ω = {omega_initial}")
print(f"  Refined: ω = {complex(omega_vec_refined[0], omega_vec_refined[1])}")
print(f"  Success: {ier == 1}")
print(f"  Message: {msg}")
print()

omega_refined = complex(omega_vec_refined[0], omega_vec_refined[1])
det_refined = benchmark.analytical._determinant_function(omega_refined, k)
print(f"det(M) at refined root: {det_refined}")
print(f"|det(M)|: {abs(det_refined):.6e}")
print()

# Check if refinement helped
print("Improvement:")
print(f"  |det| initial: {abs(det_initial):.6e}")
print(f"  |det| refined: {abs(det_refined):.6e}")
print(f"  Improvement factor: {abs(det_initial) / abs(det_refined):.2f}×")
print()

# Build matrix with refined root
matrix_refined = benchmark.analytical._build_dispersion_matrix(omega_refined, np.array([k, 0, 0]))
U, s, Vh = np.linalg.svd(matrix_refined)

print("Singular values (refined root):")
print(f"  s = {s}")
print(f"  s_min = {s[-1]:.6e}")
print(f"  Condition number: {s[0] / s[-1]:.6e}")
print()

# Check residual
v_null = Vh[-1, :]
residual = matrix_refined @ v_null
print(f"Nullspace residual:")
print(f"  ||M·v|| = {np.linalg.norm(residual):.6e}")
print(f"  Expected = {s[-1]:.6e}")
print(f"  Ratio: {np.linalg.norm(residual) / s[-1]:.2e}")
print()

# THE REAL QUESTION: Is the high residual due to numerical precision limits,
# or is there a bug in the eigenmode extraction code?

# Let me try a different approach: use eigenvalue decomposition instead of SVD
print("="*80)
print("ALTERNATIVE: Eigenvalue decomposition")
print("="*80)

# For M·v = 0, v is eigenvector of M with eigenvalue 0
# Let's check eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(matrix_refined)
print(f"Eigenvalues: {eigenvalues}")
print(f"Smallest |λ|: {min(abs(eigenvalues)):.6e}")
print()

# Find eigenvector for smallest eigenvalue
idx_min = np.argmin(np.abs(eigenvalues))
v_eig = eigenvectors[:, idx_min]
residual_eig = matrix_refined @ v_eig

print(f"Eigenvector for smallest eigenvalue:")
print(f"  λ = {eigenvalues[idx_min]:.6e}")
print(f"  ||M·v||: {np.linalg.norm(residual_eig):.6e}")
print()

# Compare SVD vs eigen approaches
print("Comparison:")
print(f"  SVD:  ||M·v|| = {np.linalg.norm(residual):.6e}")
print(f"  Eigen: ||M·v|| = {np.linalg.norm(residual_eig):.6e}")
print()

print("="*80)
print("CONCLUSION")
print("="*80)

if np.linalg.norm(residual) < 1e-10 and np.linalg.norm(residual_eig) < 1e-10:
    print("✓ Both methods give good nullspace vectors")
    print("  The eigenmode extraction is correct")
elif np.linalg.norm(residual) > 0.1:
    print("✗ Large residual indicates numerical issues")
    print("  Possible causes:")
    print("  1. Root finding didn't converge to exact root")
    print("  2. Matrix is extremely ill-conditioned")
    print("  3. There may be a bug in matrix construction")
else:
    print("⚠ Moderate residual")
    print("  May be acceptable for initialization")

print("="*80)
