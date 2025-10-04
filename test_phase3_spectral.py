"""Test Phase 3: SpectralISolver with pure 3D SpaceGrid."""

import numpy as np

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.solvers.spectral import SpectralISHydrodynamics, SpectralISolver


def test_spectral_solver_initialization():
    """Test SpectralISolver initializes with SpaceGrid."""
    print("Test 1: SpectralISolver initialization with SpaceGrid...")

    # Create 3D spatial grid
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2 * np.pi)] * 3,
        grid_points=(16, 16, 16),
        boundary_conditions="periodic",
    )

    # Initialize fields
    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 1.0
    fields.u_mu[..., 0] = 1.0

    # Create solver
    solver = SpectralISolver(grid, fields)

    # Verify attributes
    assert solver.nx == 16
    assert solver.ny == 16
    assert solver.nz == 16
    assert not hasattr(solver, "nt"), "Solver should not have nt attribute"
    assert not hasattr(solver, "dt"), "Solver should not have dt attribute"
    assert solver.k_vectors[0].shape == (16, 16, 16)

    print("✓ SpectralISolver initialized correctly with SpaceGrid")
    return True


def test_spatial_derivative_3d():
    """Test spatial derivatives on pure 3D fields."""
    print("\nTest 2: Spatial derivatives on pure 3D fields...")

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2 * np.pi)] * 3,
        grid_points=(32, 32, 32),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)
    solver = SpectralISolver(grid, fields)

    # Create test field: sin(x)
    x, y, z = grid.coordinates["x"], grid.coordinates["y"], grid.coordinates["z"]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    field = np.sin(X)

    # Compute derivative (should be cos(x))
    df_dx = solver.spatial_derivative(field, direction=0)
    expected = np.cos(X)

    # Check shape
    assert df_dx.shape == (32, 32, 32), f"Expected (32,32,32), got {df_dx.shape}"

    # Check accuracy
    error = np.max(np.abs(df_dx - expected))
    assert error < 1e-10, f"Derivative error too large: {error}"

    print(f"✓ Spatial derivative: max error = {error:.2e}")
    return True


def test_laplacian_3d():
    """Test Laplacian computation on pure 3D fields."""
    print("\nTest 3: Laplacian on pure 3D fields...")

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2 * np.pi)] * 3,
        grid_points=(32, 32, 32),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)
    coeffs = TransportCoefficients(shear_viscosity=0.1, bulk_viscosity=0.05)
    hydro = SpectralISHydrodynamics(grid, fields, coeffs)

    # Create test field: sin(x) + sin(y) + sin(z)
    x, y, z = grid.coordinates["x"], grid.coordinates["y"], grid.coordinates["z"]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    field = np.sin(X) + np.sin(Y) + np.sin(Z)

    # Compute Laplacian (should be -sin(x) - sin(y) - sin(z))
    laplacian = hydro._compute_laplacian(field)
    expected = -field

    # Check shape
    assert laplacian.shape == (32, 32, 32), f"Expected (32,32,32), got {laplacian.shape}"

    # Check accuracy
    error = np.max(np.abs(laplacian - expected))
    assert error < 1e-10, f"Laplacian error too large: {error}"

    print(f"✓ Laplacian computation: max error = {error:.2e}")
    return True


def test_spatial_divergence_3d():
    """Test spatial divergence on pure 3D vector fields."""
    print("\nTest 4: Spatial divergence on pure 3D vector fields...")

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2 * np.pi)] * 3,
        grid_points=(32, 32, 32),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)
    solver = SpectralISolver(grid, fields)

    # Create test vector field: v = (sin(x), sin(y), sin(z))
    x, y, z = grid.coordinates["x"], grid.coordinates["y"], grid.coordinates["z"]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    vector_field = np.zeros((32, 32, 32, 3))
    vector_field[..., 0] = np.sin(X)
    vector_field[..., 1] = np.sin(Y)
    vector_field[..., 2] = np.sin(Z)

    # Compute divergence (should be cos(x) + cos(y) + cos(z))
    div = solver.spatial_divergence(vector_field)
    expected = np.cos(X) + np.cos(Y) + np.cos(Z)

    # Check shape
    assert div.shape == (32, 32, 32), f"Expected (32,32,32), got {div.shape}"

    # Check accuracy
    error = np.max(np.abs(div - expected))
    assert error < 1e-10, f"Divergence error too large: {error}"

    print(f"✓ Spatial divergence: max error = {error:.2e}")
    return True


def test_no_4d_support():
    """Test that 4D fields are rejected."""
    print("\nTest 5: Verify 4D fields are rejected...")

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(8, 8, 8),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)
    solver = SpectralISolver(grid, fields)

    # Try to pass 4D field (should fail)
    field_4d = np.random.randn(10, 8, 8, 8)

    try:
        solver.spatial_derivative(field_4d, direction=0)
        print("✗ Should have rejected 4D field!")
        return False
    except ValueError as e:
        if "pure 3D" in str(e):
            print(f"✓ Correctly rejected 4D field: {e}")
            return True
        else:
            print(f"✗ Wrong error message: {e}")
            return False


def test_integration_with_fields():
    """Test integration with ISFieldConfiguration."""
    print("\nTest 6: Integration with ISFieldConfiguration...")

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2 * np.pi)] * 3,
        grid_points=(16, 16, 16),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)
    coeffs = TransportCoefficients(
        shear_viscosity=0.1,
        bulk_viscosity=0.05,
        shear_relaxation_time=0.5,
        bulk_relaxation_time=0.3,
    )

    # Initialize with physical values
    fields.rho[:] = 1.0
    fields.pressure[:] = 1.0 / 3.0
    fields.u_mu[..., 0] = 1.0  # Rest frame

    # Create hydrodynamics solver
    hydro = SpectralISHydrodynamics(grid, fields, coeffs)

    # Verify solver created correctly
    assert hydro.spectral.nx == 16
    assert hydro.spectral.ny == 16
    assert hydro.spectral.nz == 16

    # Verify fields are pure 3D
    assert fields.rho.shape == (16, 16, 16)
    assert fields.u_mu.shape == (16, 16, 16, 4)
    assert fields.pi_munu.shape == (16, 16, 16, 4, 4)

    print("✓ Successfully integrated with ISFieldConfiguration")
    print(f"  - Scalar field shape: {fields.rho.shape}")
    print(f"  - Vector field shape: {fields.u_mu.shape}")
    print(f"  - Tensor field shape: {fields.pi_munu.shape}")
    return True


def test_memory_reduction():
    """Demonstrate memory reduction from 4D to 3D."""
    print("\nTest 7: Memory reduction demonstration...")

    # 3D grid
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(64, 64, 64),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)

    # Calculate memory for one scalar field
    field_3d_size = fields.rho.nbytes / (1024**2)  # MB

    # Compare to hypothetical 4D (nt=20)
    nt = 20
    field_4d_size = field_3d_size * nt

    reduction = (1 - field_3d_size / field_4d_size) * 100

    print("✓ Memory comparison (64³ grid, one scalar field):")
    print(f"  - 3D storage: {field_3d_size:.2f} MB")
    print(f"  - 4D storage (nt={nt}): {field_4d_size:.2f} MB")
    print(f"  - Reduction: {reduction:.1f}%")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3: SpectralISolver Pure 3D Validation")
    print("=" * 60)

    tests = [
        test_spectral_solver_initialization,
        test_spatial_derivative_3d,
        test_laplacian_3d,
        test_spatial_divergence_3d,
        test_no_4d_support,
        test_integration_with_fields,
        test_memory_reduction,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("✓ ALL TESTS PASSED - Phase 3 complete!")
        exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        exit(1)
