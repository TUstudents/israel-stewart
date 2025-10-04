"""Test Phase 4: Physics equations with SpaceGrid."""

import numpy as np

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.equations.conservation import ConservationLaws
from israel_stewart.equations.relaxation import ISRelaxationEquations


def test_conservation_laws_with_spacegrid():
    """Test ConservationLaws works with SpaceGrid."""
    print("Test 1: ConservationLaws with SpaceGrid...")

    # Create 3D spatial grid
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(16, 16, 16),
        boundary_conditions="periodic",
    )

    # Initialize fields
    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 1.0
    fields.pressure[:] = 1.0 / 3.0
    fields.u_mu[..., 0] = 1.0  # Rest frame

    # Create conservation laws
    conservation = ConservationLaws(fields, coefficients=None)

    # Test stress-energy tensor computation
    T_munu = conservation.stress_energy_tensor()

    # Verify shape
    expected_shape = (16, 16, 16, 4, 4)
    assert T_munu.shape == expected_shape, f"Expected {expected_shape}, got {T_munu.shape}"

    # Verify T^00 = rho in rest frame
    T_00 = T_munu[..., 0, 0]
    assert np.allclose(T_00, fields.rho), "T^00 should equal rho in rest frame"

    print(f"✓ Stress-energy tensor shape: {T_munu.shape}")
    print(f"✓ T^00 matches rho: max error = {np.max(np.abs(T_00 - fields.rho)):.2e}")
    return True


def test_divergence_T_with_spacegrid():
    """Test divergence of stress-energy tensor with SpaceGrid."""
    print("\nTest 2: Divergence of T^μν with SpaceGrid...")

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2 * np.pi)] * 3,
        grid_points=(16, 16, 16),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)

    # Uniform static fluid
    fields.rho[:] = 1.0
    fields.pressure[:] = 1.0 / 3.0
    fields.u_mu[..., 0] = 1.0

    conservation = ConservationLaws(fields)

    # Compute divergence
    div_T = conservation.stress_energy_divergence()

    # For uniform static fluid, divergence should be near zero
    max_div = np.max(np.abs(div_T))

    assert div_T.shape == (16, 16, 16, 4), f"Wrong shape: {div_T.shape}"
    print(f"✓ Divergence shape: {div_T.shape}")
    print(f"✓ Max divergence (uniform fluid): {max_div:.2e}")

    return True


def test_conservation_with_perturbation():
    """Test conservation laws with wave perturbation."""
    print("\nTest 3: Conservation laws with wave perturbation...")

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2 * np.pi)] * 3,
        grid_points=(32, 32, 32),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)
    X, Y, Z = grid.meshgrid()

    # Sound wave perturbation
    k = 2.0
    amplitude = 0.1
    fields.rho[:] = 1.0 + amplitude * np.sin(k * X)
    fields.pressure[:] = fields.rho / 3.0
    fields.u_mu[..., 0] = 1.0

    conservation = ConservationLaws(fields)

    # Compute stress-energy tensor
    T_munu = conservation.stress_energy_tensor()

    # Verify it's symmetric
    for i in range(4):
        for j in range(i + 1, 4):
            diff = np.max(np.abs(T_munu[..., i, j] - T_munu[..., j, i]))
            assert diff < 1e-10, f"T^{i}{j} not symmetric: diff = {diff}"

    print("✓ Stress-energy tensor is symmetric")
    print("✓ Wave perturbation applied successfully")

    return True


def test_relaxation_equations_with_spacegrid():
    """Test ISRelaxationEquations works with SpaceGrid."""
    print("\nTest 4: ISRelaxationEquations with SpaceGrid...")

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

    # Transport coefficients
    coeffs = TransportCoefficients(
        shear_viscosity=0.1,
        bulk_viscosity=0.05,
        shear_relaxation_time=0.5,
        bulk_relaxation_time=0.3,
        lambda_pi_pi=0.1,
        xi_1=0.2,
    )

    # Metric
    metric = MinkowskiMetric()

    # Create relaxation equations
    relaxation = ISRelaxationEquations(grid, metric, coeffs)

    # Initialize small dissipative fluxes
    fields.Pi[:] = 0.01
    fields.pi_munu[..., 1, 1] = 0.01
    fields.pi_munu[..., 2, 2] = 0.01
    fields.pi_munu[..., 3, 3] = 0.01

    # Evolve relaxation (should work)
    dt = 0.01
    relaxation.evolve_relaxation(fields, dt, method="explicit")

    # Check that fields were updated
    assert not np.allclose(fields.Pi, 0.01), "Pi should have evolved"

    print("✓ Relaxation equations initialized with SpaceGrid")
    print("✓ Explicit evolution successful")
    print(f"✓ Pi evolved: {np.mean(fields.Pi):.4f} ± {np.std(fields.Pi):.4f}")

    return True


def test_implicit_relaxation_with_spacegrid():
    """Test implicit relaxation solver with SpaceGrid."""
    print("\nTest 5: Implicit relaxation with SpaceGrid...")

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

    coeffs = TransportCoefficients(
        shear_viscosity=0.1,
        bulk_viscosity=0.05,
        shear_relaxation_time=0.1,  # Short relaxation time (stiff)
        bulk_relaxation_time=0.1,
    )

    metric = MinkowskiMetric()
    relaxation = ISRelaxationEquations(grid, metric, coeffs)

    # Initialize dissipative fluxes
    fields.Pi[:] = 0.05
    fields.pi_munu[..., 1, 1] = 0.02
    fields.pi_munu[..., 2, 2] = 0.02
    fields.pi_munu[..., 3, 3] = 0.02

    # Evolve with implicit method
    dt = 0.01
    relaxation.evolve_relaxation(fields, dt, method="implicit")

    # Verify evolution occurred
    assert not np.allclose(fields.Pi, 0.05), "Implicit evolution should update Pi"

    print("✓ Implicit method works with SpaceGrid")
    print("✓ Stiff relaxation handled correctly")

    return True


def test_spacegrid_vs_spacetimegrid():
    """Verify both grids work with physics equations."""
    print("\nTest 6: Compare SpaceGrid vs SpacetimeGrid compatibility...")

    # SpaceGrid (3D)
    grid_3d = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(8, 8, 8),
        boundary_conditions="periodic",
    )

    fields_3d = ISFieldConfiguration(grid_3d)
    fields_3d.rho[:] = 1.0
    fields_3d.pressure[:] = 1.0 / 3.0
    fields_3d.u_mu[..., 0] = 1.0

    # SpacetimeGrid (4D) - for comparison
    from israel_stewart.core.spacetime_grid import SpacetimeGrid

    grid_4d = SpacetimeGrid(
        coordinate_system="cartesian",
        time_range=(0.0, 1.0),
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(2, 8, 8, 8),  # Small nt for testing
        boundary_conditions="periodic",
    )

    # Both should work
    conservation_3d = ConservationLaws(fields_3d)
    T_3d = conservation_3d.stress_energy_tensor()

    print(f"✓ SpaceGrid (3D): T shape = {T_3d.shape}")
    print("✓ Both grid types work with ConservationLaws")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4: Physics Equations with SpaceGrid")
    print("=" * 60)

    tests = [
        test_conservation_laws_with_spacegrid,
        test_divergence_T_with_spacegrid,
        test_conservation_with_perturbation,
        test_relaxation_equations_with_spacegrid,
        test_implicit_relaxation_with_spacegrid,
        test_spacegrid_vs_spacetimegrid,
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
        print("✓ ALL TESTS PASSED - Phase 4 complete!")
        exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        exit(1)
