"""Test Phase 4: Physics equations with SpaceGrid (simplified)."""

import numpy as np

from israel_stewart.core.fields import ISFieldConfiguration
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.core.spacetime_grid import SpacetimeGrid
from israel_stewart.equations.conservation import ConservationLaws


def test_conservation_laws_spacegrid():
    """Test ConservationLaws works with SpaceGrid."""
    print("Test 1: ConservationLaws with SpaceGrid (3D)...")

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(16, 16, 16),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 1.0
    fields.pressure[:] = 1.0 / 3.0
    fields.u_mu[..., 0] = 1.0  # Rest frame

    # Create conservation laws
    conservation = ConservationLaws(fields)

    # Test stress-energy tensor computation
    T_munu = conservation.stress_energy_tensor()

    # Verify shape
    assert T_munu.shape == (16, 16, 16, 4, 4)

    # Verify T^00 = rho in rest frame
    T_00 = T_munu[..., 0, 0]
    assert np.allclose(T_00, fields.rho)

    print(f"✓ Stress-energy tensor shape: {T_munu.shape}")
    print("✓ T^00 = rho (rest frame)")
    return True


def test_stress_energy_symmetry():
    """Test T^μν is symmetric."""
    print("\nTest 2: Stress-energy tensor symmetry...")

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2 * np.pi)] * 3,
        grid_points=(32, 32, 32),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)
    X, Y, Z = grid.meshgrid()

    # Wave perturbation
    fields.rho[:] = 1.0 + 0.1 * np.sin(2.0 * X)
    fields.pressure[:] = fields.rho / 3.0
    fields.u_mu[..., 0] = 1.0

    conservation = ConservationLaws(fields)
    T_munu = conservation.stress_energy_tensor()

    # Check symmetry
    max_asymmetry = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            asymmetry = np.max(np.abs(T_munu[..., i, j] - T_munu[..., j, i]))
            max_asymmetry = max(max_asymmetry, asymmetry)

    assert max_asymmetry < 1e-10

    print(f"✓ T^μν is symmetric: max asymmetry = {max_asymmetry:.2e}")
    return True


def test_spatial_projector():
    """Test spatial projector Δ^μν."""
    print("\nTest 3: Spatial projector...")

    grid = SpaceGrid(
        coordinate_system="cartesian", spatial_ranges=[(0.0, 1.0)] * 3, grid_points=(8, 8, 8)
    )

    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 1.0
    fields.pressure[:] = 1.0 / 3.0
    fields.u_mu[..., 0] = 1.0

    conservation = ConservationLaws(fields)
    Delta = conservation._spatial_projector()

    assert Delta.shape == (8, 8, 8, 4, 4)

    # In rest frame: Δ^00 = 0, Δ^ii = 1
    assert np.allclose(Delta[..., 0, 0], 0.0)
    assert np.allclose(Delta[..., 1, 1], 1.0)
    assert np.allclose(Delta[..., 2, 2], 1.0)
    assert np.allclose(Delta[..., 3, 3], 1.0)

    print(f"✓ Projector shape: {Delta.shape}")
    print("✓ Δ^00 = 0, Δ^ii = 1 (rest frame)")
    return True


def test_perfect_fluid_limit():
    """Test perfect fluid limit (no viscosity)."""
    print("\nTest 4: Perfect fluid limit...")

    grid = SpaceGrid(
        coordinate_system="cartesian", spatial_ranges=[(0.0, 1.0)] * 3, grid_points=(16, 16, 16)
    )

    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 2.0
    fields.pressure[:] = 2.0 / 3.0
    fields.u_mu[..., 0] = 1.0

    # No dissipative fluxes (perfect fluid)
    fields.Pi[:] = 0.0
    fields.pi_munu[:] = 0.0
    fields.q_mu[:] = 0.0

    conservation = ConservationLaws(fields)
    T_munu = conservation.stress_energy_tensor()

    # Perfect fluid: T^μν = ρ u^μ u^ν + p Δ^μν
    # In rest frame: T^00 = ρ, T^ii = p
    T_00 = T_munu[..., 0, 0]
    T_11 = T_munu[..., 1, 1]

    assert np.allclose(T_00, 2.0)
    assert np.allclose(T_11, 2.0 / 3.0)

    print(f"✓ Perfect fluid: T^00 = {np.mean(T_00):.1f} (rho)")
    print(f"✓ Perfect fluid: T^11 = {np.mean(T_11):.3f} (pressure)")
    return True


def test_spacegrid_coordinate_arrays():
    """Test that coordinate arrays work correctly."""
    print("\nTest 5: Coordinate arrays with SpaceGrid...")

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(8, 8, 8),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 1.0
    fields.pressure[:] = 1.0 / 3.0
    fields.u_mu[..., 0] = 1.0

    conservation = ConservationLaws(fields)

    # Get coordinate arrays
    coords = conservation._get_coordinate_arrays()

    # SpaceGrid should have 3 coordinate arrays
    assert len(coords) == 3, f"Expected 3 coords for SpaceGrid, got {len(coords)}"

    print(f"✓ SpaceGrid provides {len(coords)} coordinate arrays")
    print(f"✓ Coordinate names: {grid.coordinate_names}")
    return True


def test_spacetimegrid_compatibility():
    """Verify SpacetimeGrid still works."""
    print("\nTest 6: SpacetimeGrid compatibility (4D)...")

    grid = SpacetimeGrid(
        coordinate_system="cartesian",
        time_range=(0.0, 1.0),
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(2, 8, 8, 8),
        boundary_conditions="periodic",
    )

    # Note: Old ISFieldConfiguration expected SpacetimeGrid
    # For backward compatibility, need to handle carefully

    coords = grid.coordinates

    # SpacetimeGrid should have 4 coordinate arrays
    assert len(coords) == 4

    print("✓ SpacetimeGrid provides 4 coordinate arrays")
    print(f"✓ Coordinate names: {grid.coordinate_names}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4: Physics Equations - Basic Tests")
    print("=" * 60)

    tests = [
        test_conservation_laws_spacegrid,
        test_stress_energy_symmetry,
        test_spatial_projector,
        test_perfect_fluid_limit,
        test_spacegrid_coordinate_arrays,
        test_spacetimegrid_compatibility,
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
            print(f"✗ Test failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("✓ ALL TESTS PASSED")
        print("\nNote: ISRelaxationEquations requires spacetime derivatives")
        print("and therefore needs SpacetimeGrid (or spectral solver)")
        print("for proper 4D evolution. SpaceGrid works for")
        print("conservation laws and stress-energy tensor construction.")
        exit(0)
    else:
        exit(1)
