"""Test Phase 1: SpaceGrid implementation."""

import numpy as np

from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.core.spacetime_grid import SpacetimeGrid


def test_basic_initialization():
    """Test basic SpaceGrid initialization."""
    print("Test 1: Basic initialization...")
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(8, 8, 8),
        boundary_conditions="periodic",
    )

    assert grid.shape == (8, 8, 8)
    assert grid.ndim == 3
    assert grid.nx == 8
    assert grid.ny == 8
    assert grid.nz == 8
    print("✓ Basic initialization passed")


def test_periodic_vs_dirichlet():
    """Test spacing differences for boundary conditions."""
    print("\nTest 2: Periodic vs Dirichlet spacing...")

    grid_periodic = SpaceGrid("cartesian", [(0, 1)] * 3, (8, 8, 8), boundary_conditions="periodic")
    grid_dirichlet = SpaceGrid(
        "cartesian", [(0, 1)] * 3, (8, 8, 8), boundary_conditions="dirichlet"
    )

    # Periodic: dx = L/N
    assert np.isclose(grid_periodic.dx, 1.0 / 8)

    # Dirichlet: dx = L/(N-1)
    assert np.isclose(grid_dirichlet.dx, 1.0 / 7)

    print(f"✓ Periodic dx = {grid_periodic.dx:.6f}")
    print(f"✓ Dirichlet dx = {grid_dirichlet.dx:.6f}")


def test_meshgrid():
    """Test meshgrid creation."""
    print("\nTest 3: Meshgrid creation...")
    grid = SpaceGrid("cartesian", [(0, 1)] * 3, (8, 8, 8))
    X, Y, Z = grid.meshgrid()

    assert X.shape == (8, 8, 8)
    assert Y.shape == (8, 8, 8)
    assert Z.shape == (8, 8, 8)
    print("✓ Meshgrid shapes correct")


def test_gradient():
    """Test gradient computation."""
    print("\nTest 4: Gradient computation...")
    grid = SpaceGrid(
        "cartesian", [(0, 2 * np.pi)] * 3, (32, 32, 32), boundary_conditions="periodic"
    )

    X, Y, Z = grid.meshgrid()
    field = np.sin(X)

    grad = grid.gradient(field, axis=0)
    expected = np.cos(X)

    error = np.max(np.abs(grad - expected))
    assert error < 1e-2  # Finite difference accuracy
    print(f"✓ Gradient error: {error:.2e}")


def test_divergence():
    """Test divergence computation."""
    print("\nTest 5: Divergence computation...")
    grid = SpaceGrid(
        "cartesian", [(0, 2 * np.pi)] * 3, (32, 32, 32), boundary_conditions="periodic"
    )

    X, Y, Z = grid.meshgrid()
    vector_field = np.zeros((*grid.shape, 3))
    vector_field[..., 0] = np.sin(X)
    vector_field[..., 1] = np.sin(Y)
    vector_field[..., 2] = np.sin(Z)

    div = grid.divergence(vector_field)
    expected = np.cos(X) + np.cos(Y) + np.cos(Z)

    error = np.max(np.abs(div - expected))
    assert error < 1e-2
    print(f"✓ Divergence error: {error:.2e}")


def test_laplacian():
    """Test Laplacian computation."""
    print("\nTest 6: Laplacian computation...")
    grid = SpaceGrid(
        "cartesian", [(0, 2 * np.pi)] * 3, (32, 32, 32), boundary_conditions="periodic"
    )

    X, Y, Z = grid.meshgrid()
    field = np.sin(X) + np.sin(Y) + np.sin(Z)

    laplacian = grid.laplacian(field)
    expected = -field

    error = np.max(np.abs(laplacian - expected))
    assert error < 1e-2
    print(f"✓ Laplacian error: {error:.2e}")


def test_type_checking():
    """Test that SpaceGrid is distinct from SpacetimeGrid."""
    print("\nTest 7: Type checking...")

    space_grid = SpaceGrid("cartesian", [(0, 1)] * 3, (8, 8, 8))

    assert isinstance(space_grid, SpaceGrid)
    assert not isinstance(space_grid, SpacetimeGrid)
    assert not hasattr(space_grid, "nt")
    assert not hasattr(space_grid, "dt")

    print("✓ SpaceGrid is distinct from SpacetimeGrid")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1: SpaceGrid Validation Tests")
    print("=" * 60)

    tests = [
        test_basic_initialization,
        test_periodic_vs_dirichlet,
        test_meshgrid,
        test_gradient,
        test_divergence,
        test_laplacian,
        test_type_checking,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)

    if passed == len(tests):
        print("✓ ALL PHASE 1 TESTS PASSED!")
        exit(0)
    else:
        exit(1)
