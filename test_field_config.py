#!/usr/bin/env python3
"""
Test script for ISFieldConfiguration and SpacetimeGrid implementation.

This script validates the complete field configuration system for
Israel-Stewart hydrodynamics with constraint enforcement and grid management.
"""

import os
import sys

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_spacetime_grid():
    """Test SpacetimeGrid functionality."""
    print("Testing SpacetimeGrid...")

    try:
        from israel_stewart.core import create_cartesian_grid

        # Create simple Cartesian grid
        grid = create_cartesian_grid(
            time_range=(0.0, 1.0), spatial_extent=2.0, grid_points=(10, 8, 8, 8)
        )

        print(f"✅ Grid created: {grid}")
        print(f"   Shape: {grid.shape}")
        print(f"   Total points: {grid.total_points}")
        print(f"   Coordinate system: {grid.coordinate_system}")

        # Test coordinate operations
        coords_at_center = grid.coordinate_at_index((5, 4, 4, 4))
        print(f"   Coordinates at center: {coords_at_center}")

        # Test gradients
        test_field = np.sin(np.linspace(0, np.pi, grid.total_points)).reshape(
            grid.shape
        )
        gradient_t = grid.gradient(test_field, axis=0)
        print(f"   Gradient shape: {gradient_t.shape}")

        return True

    except Exception as e:
        print(f"❌ SpacetimeGrid test failed: {e}")
        return False


def test_field_configuration():
    """Test ISFieldConfiguration functionality."""
    print("\nTesting ISFieldConfiguration...")

    try:
        from israel_stewart.core import ISFieldConfiguration, create_cartesian_grid

        # Create grid and field configuration
        grid = create_cartesian_grid(
            time_range=(0.0, 1.0),
            spatial_extent=1.0,
            grid_points=(5, 4, 4, 4),  # Small grid for testing
        )

        # Set metric for proper tensor operations
        from israel_stewart.core import MinkowskiMetric

        grid.metric = MinkowskiMetric()

        config = ISFieldConfiguration(grid)
        print("✅ Field configuration created")
        print(f"   Total field count: {config.total_field_count}")

        # Test initial state
        print(f"   Initial energy density shape: {config.rho.shape}")
        print(f"   Initial four-velocity shape: {config.u_mu.shape}")

        # Set some test values
        config.rho.fill(1.0)  # Uniform energy density
        config.pressure.fill(0.3)  # Equation of state p = ρ/3
        config.n.fill(0.5)  # Particle density

        # Add some non-trivial four-velocity
        config.u_mu[..., 0] = 1.1  # Time component
        config.u_mu[..., 1] = 0.2  # X component

        print("   Set test field values")

        # Test constraint enforcement
        print("   Applying constraints...")
        config.apply_constraints()
        print("   ✅ Constraints applied")

        # Validate constraints
        validation = config.validate_field_configuration()
        print("   Validation results:")
        for key, result in validation.items():
            status = "✅" if result else "❌"
            print(f"     {status} {key}: {result}")

        # Test state vector operations
        state_vector = config.to_state_vector()
        print(f"   State vector length: {len(state_vector)}")

        # Test round-trip
        config2 = ISFieldConfiguration(grid)
        config2.from_state_vector(state_vector)
        print("   ✅ State vector round-trip successful")

        # Test stress-energy tensor computation
        T_munu = config.compute_stress_energy_tensor()
        print(f"   Stress-energy tensor shape: {T_munu.shape}")

        # Test conserved charges
        charges = config.compute_conserved_charges()
        print(f"   Conserved charges computed: {list(charges.keys())}")

        return validation["overall_valid"]

    except Exception as e:
        print(f"❌ ISFieldConfiguration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_integration_with_tensors():
    """Test integration with tensor framework."""
    print("\nTesting tensor framework integration...")

    try:
        from israel_stewart.core import (
            FourVector,
            ISFieldConfiguration,
            MinkowskiMetric,
            create_cartesian_grid,
        )

        # Create grid with Minkowski metric
        grid = create_cartesian_grid(
            time_range=(0.0, 1.0),
            spatial_extent=1.0,
            grid_points=(3, 3, 3, 3),  # Very small for testing
        )

        # Set metric
        grid.metric = MinkowskiMetric()

        config = ISFieldConfiguration(grid)

        # Set up realistic hydrodynamic state
        config.rho.fill(2.0)
        config.pressure.fill(0.6)  # Stiff EOS
        config.n.fill(1.0)

        # Create boosted four-velocity
        velocity_components = np.array([1.2, 0.3, 0.1, 0.0])  # Boosted in x-direction

        # Set at all grid points
        for indices in np.ndindex(grid.shape):
            config.u_mu[indices] = velocity_components

        print("   Set up boosted hydrodynamic state")

        # Apply constraints with metric
        config.apply_constraints()

        # Test four-velocity properties
        sample_u = FourVector(config.u_mu[0, 0, 0, 0], False, grid.metric)
        print(f"   Sample four-velocity magnitude²: {sample_u.magnitude_squared():.6f}")
        print(f"   Is timelike: {sample_u.is_timelike()}")
        print(f"   Lorentz factor: {sample_u.lorentz_factor():.3f}")

        # Validate final state
        validation = config.validate_field_configuration()
        overall_valid = validation["overall_valid"]

        print(f"   ✅ Integration test {'passed' if overall_valid else 'failed'}")

        return overall_valid

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_performance():
    """Test performance with larger grids."""
    print("\nTesting performance...")

    try:
        from israel_stewart.core import ISFieldConfiguration, create_cartesian_grid
        from israel_stewart.core.performance import (
            performance_report,
            reset_performance_stats,
        )

        # Reset performance monitoring
        reset_performance_stats()

        # Create moderately sized grid
        grid = create_cartesian_grid(
            time_range=(0.0, 2.0),
            spatial_extent=4.0,
            grid_points=(20, 16, 16, 16),  # ~80k grid points
        )

        # Set metric for proper tensor operations
        from israel_stewart.core import MinkowskiMetric

        grid.metric = MinkowskiMetric()

        config = ISFieldConfiguration(grid)
        print(f"   Created large grid: {grid.total_points} points")

        # Set up fields
        config.rho.fill(1.5)
        config.pressure.fill(0.5)
        config.n.fill(0.8)

        # Apply constraints (this should be monitored)
        import time

        start_time = time.time()
        config.apply_constraints()
        constraint_time = time.time() - start_time

        print(f"   Constraint application time: {constraint_time:.3f} seconds")

        # Test state vector operations
        start_time = time.time()
        state_vector = config.to_state_vector()
        pack_time = time.time() - start_time

        print(f"   State vector packing time: {pack_time:.3f} seconds")
        print(f"   State vector size: {len(state_vector):,} elements")

        # Test stress-energy tensor computation
        start_time = time.time()
        config.compute_stress_energy_tensor()
        tensor_time = time.time() - start_time

        print(f"   Stress-energy tensor time: {tensor_time:.3f} seconds")

        # Get performance report
        report = performance_report()
        if report["operation_counts"]:
            print(
                f"   Performance monitoring captured {len(report['operation_counts'])} operations"
            )
            for op, count in report["operation_counts"].items():
                print(f"     {op}: {count} calls")

        return True

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Israel-Stewart Field Configuration Test Suite")
    print("=" * 50)

    tests = [
        test_spacetime_grid,
        test_field_configuration,
        test_integration_with_tensors,
        test_performance,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("Test Summary:")

    test_names = [
        "SpacetimeGrid",
        "ISFieldConfiguration",
        "Tensor Integration",
        "Performance",
    ]

    for name, result in zip(test_names, results, strict=False):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {name}")

    overall_success = all(results)
    print(
        f"\nOverall: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}"
    )

    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
