#!/usr/bin/env python3
"""
Diagnostic script to identify expansion scalar divergence bug.

Tests FFT-based spatial derivative computation for simple analytical cases.
"""

import sys
from pathlib import Path

import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacetime_grid import SpacetimeGrid
from israel_stewart.solvers.spectral import SpectralISHydrodynamics

print("=" * 70)
print("EXPANSION SCALAR DIVERGENCE BUG DIAGNOSTIC")
print("=" * 70)

# Create test setup matching the failing test
grid = SpacetimeGrid(
    coordinate_system="cartesian",
    time_range=(0.0, 1.0),
    spatial_ranges=[(0.0, 2 * np.pi), (0.0, 2 * np.pi), (0.0, 2 * np.pi)],
    grid_points=(10, 16, 16, 16),  # (nt, nx, ny, nz)
)

fields = ISFieldConfiguration(grid)
fields.rho.fill(1.0)
fields.pressure.fill(0.33)
fields.u_mu[..., 0] = 1.0
fields.Pi.fill(0.01)
fields.pi_munu.fill(0.005)

coeffs = TransportCoefficients(
    shear_viscosity=0.1,
    bulk_viscosity=0.05,
    shear_relaxation_time=0.5,
    bulk_relaxation_time=0.3,
)

solver = SpectralISHydrodynamics(grid, fields, coeffs)

print(f"\nGrid shape: {grid.shape}")
print(
    f"Grid dimensions: nt={solver.spectral.nt}, nx={solver.spectral.nx}, ny={solver.spectral.ny}, nz={solver.spectral.nz}"
)
print(
    f"Grid spacing: dx={solver.spectral.dx:.4f}, dy={solver.spectral.dy:.4f}, dz={solver.spectral.dz:.4f}"
)

# Set up test velocity field
x = np.linspace(0, 2 * np.pi, 16, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

print("\nTest field setup:")
print(f"X shape: {X.shape}, varies along axis 0")
print(f"Y shape: {Y.shape}, varies along axis 1")
print(f"Z shape: {Z.shape}, varies along axis 2")

# Set velocity field: u^x = sin(X), u^y = cos(Y), u^z = 0
fields.u_mu[..., 1] = np.sin(X)  # u^x
fields.u_mu[..., 2] = np.cos(Y)  # u^y
fields.u_mu[..., 3] = 0.0  # u^z

print("\nVelocity field:")
print(f"fields.u_mu shape: {fields.u_mu.shape}")
print(f"u^x = sin(X): range [{np.min(fields.u_mu[..., 1]):.3f}, {np.max(fields.u_mu[..., 1]):.3f}]")
print(f"u^y = cos(Y): range [{np.min(fields.u_mu[..., 2]):.3f}, {np.max(fields.u_mu[..., 2]):.3f}]")
print(f"u^z = 0: range [{np.min(fields.u_mu[..., 3]):.3f}, {np.max(fields.u_mu[..., 3]):.3f}]")

# Expected divergence: d(sin(X))/dx + d(cos(Y))/dy + d(0)/dz = cos(X) - sin(Y)
expected_theta = np.cos(X) - np.sin(Y)
print("\nExpected θ = ∇·u = cos(X) - sin(Y):")
print(f"  Shape: {expected_theta.shape}")
print(f"  Range: [{np.min(expected_theta):.3f}, {np.max(expected_theta):.3f}]")
print(f"  Varies with z? {np.std(expected_theta, axis=2).max() > 1e-10} (should be False)")

# Test 1: Direct FFT derivative of sin(X)
print("\n" + "-" * 70)
print("TEST 1: Direct FFT derivative of sin(X) → should give cos(X)")
print("-" * 70)

test_field = np.sin(X)
print(f"Input field shape: {test_field.shape}")
print(f"Input range: [{np.min(test_field):.3f}, {np.max(test_field):.3f}]")

# Compute derivative using spectral method
deriv_x = solver.spectral.spatial_derivative(test_field, direction=0)
print(f"Derivative shape: {deriv_x.shape}")
print(f"Derivative range: [{np.min(deriv_x):.3f}, {np.max(deriv_x):.3f}]")

expected_deriv = np.cos(X)
error = np.max(np.abs(deriv_x - expected_deriv))
print(f"Expected (cos(X)) range: [{np.min(expected_deriv):.3f}, {np.max(expected_deriv):.3f}]")
print(f"Max error: {error:.2e}")
print("✓ PASS" if error < 1e-10 else "✗ FAIL")

# Test 2: Compute expansion scalar using solver method
print("\n" + "-" * 70)
print("TEST 2: Expansion scalar θ = ∇·u using solver method")
print("-" * 70)

# Extract velocity before calling _compute_expansion_scalar
velocity_spatial = fields.u_mu[..., 1:4]
print(f"velocity_spatial shape: {velocity_spatial.shape} (should be (10, 16, 16, 16, 3))")

# Check what spatial_divergence receives
print("\nChecking spatial_divergence input handling:")
print(f"  velocity_spatial.ndim = {velocity_spatial.ndim} (expecting 5)")
print(f"  velocity_spatial.shape[0] = {velocity_spatial.shape[0]} (expecting {solver.spectral.nt})")

if velocity_spatial.ndim == 5 and velocity_spatial.shape[0] == solver.spectral.nt:
    time_sliced = velocity_spatial[-1, :, :, :, :]
    print(f"  ✓ Time-slicing should occur: shape after = {time_sliced.shape}")
    vx = time_sliced[..., 0]
    vy = time_sliced[..., 1]
    vz = time_sliced[..., 2]
    print(f"  Component shapes: vx={vx.shape}, vy={vy.shape}, vz={vz.shape}")

    # Check FFT shapes
    vx_k = solver.spectral.adaptive_fft(vx)
    print(f"\n  FFT output shape: {vx_k.shape}")
    print("  Expected for real FFT: (16, 16, 9)")
    print("  Expected for complex FFT: (16, 16, 16)")

    # Check k-vector shapes
    if vx_k.shape != solver.spectral.k_vectors[0].shape:
        print("\n  Using cached k-vectors for real FFT")
        cached_k = solver.spectral.get_cached_k_vectors(vx_k.shape)
        print(f"  Cached kx shape: {cached_k['kx'].shape}")
        print(f"  Cached ky shape: {cached_k['ky'].shape}")
        print(f"  Cached kz shape: {cached_k['kz'].shape}")
    else:
        print("\n  Using precomputed k-vectors")
        print(f"  kx shape: {solver.spectral.k_vectors[0].shape}")
        print(f"  ky shape: {solver.spectral.k_vectors[1].shape}")
        print(f"  kz shape: {solver.spectral.k_vectors[2].shape}")

# Now compute theta using solver
theta = solver._compute_expansion_scalar()
print(f"\nComputed θ shape: {theta.shape}")
print(f"Computed θ range: [{np.min(theta):.3f}, {np.max(theta):.3f}]")
print(f"Expected θ range: [{np.min(expected_theta):.3f}, {np.max(expected_theta):.3f}]")

# Check z-variation (this is the bug symptom!)
theta_z_std = np.std(theta, axis=2)
expected_z_std = np.std(expected_theta, axis=2)
print("\nZ-axis variation check:")
print(f"  Computed θ std along z: max={np.max(theta_z_std):.2e}, mean={np.mean(theta_z_std):.2e}")
print(
    f"  Expected θ std along z: max={np.max(expected_z_std):.2e}, mean={np.mean(expected_z_std):.2e}"
)
print(
    f"  Computed SHOULD be ~0 (independent of z): {'✓ PASS' if np.max(theta_z_std) < 1e-10 else '✗ FAIL'}"
)

# Overall error
error_theta = np.max(np.abs(theta - expected_theta))
rel_error = error_theta / np.max(np.abs(expected_theta))
print("\nOverall error:")
print(f"  Max absolute error: {error_theta:.2e}")
print(f"  Max relative error: {rel_error:.2e}")
print(f"  Should be < 1e-10: {'✓ PASS' if error_theta < 1e-10 else '✗ FAIL'}")

# Correlation check (current test)
correlation = np.corrcoef(theta.flatten(), expected_theta.flatten())[0, 1]
print(f"  Correlation: {correlation:.6f}")
print(f"  Current test threshold > 0.5: {'✓ PASS (weak)' if correlation > 0.5 else '✗ FAIL'}")

# Test 3: Manual divergence computation to isolate bug
print("\n" + "-" * 70)
print("TEST 3: Manual component-by-component derivative")
print("-" * 70)

vx_field = fields.u_mu[-1, :, :, :, 1]  # Last time slice, u^x component
vy_field = fields.u_mu[-1, :, :, :, 2]  # Last time slice, u^y component
vz_field = fields.u_mu[-1, :, :, :, 3]  # Last time slice, u^z component

print(f"vx_field shape: {vx_field.shape}")
print(f"vy_field shape: {vy_field.shape}")
print(f"vz_field shape: {vz_field.shape}")

dvx_dx = solver.spectral.spatial_derivative(vx_field, direction=0)
dvy_dy = solver.spectral.spatial_derivative(vy_field, direction=1)
dvz_dz = solver.spectral.spatial_derivative(vz_field, direction=2)

print(f"\ndvx/dx shape: {dvx_dx.shape}")
print(f"dvy/dy shape: {dvy_dy.shape}")
print(f"dvz/dz shape: {dvz_dz.shape}")

manual_theta = dvx_dx + dvy_dy + dvz_dz

print("\nManual θ = dvx/dx + dvy/dy + dvz/dz:")
print(f"  Range: [{np.min(manual_theta):.3f}, {np.max(manual_theta):.3f}]")
print(f"  Z-std: max={np.max(np.std(manual_theta, axis=2)):.2e}")

error_manual = np.max(np.abs(manual_theta - expected_theta))
print(f"  Error vs expected: {error_manual:.2e}")
print(f"  Match solver result? {np.allclose(manual_theta, theta)}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
