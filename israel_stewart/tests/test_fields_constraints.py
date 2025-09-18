"""
Tests for fields.py constraint fixes focusing on critical bug fixes.

This module tests the two major bug fixes:
1. None metric case handling in constraint enforcement methods
2. Symbolic metric compatibility (SymPy vs NumPy attribute differences)

All tests focus on ensuring production-ready reliability for Israel-Stewart
hydrodynamics simulations with different metric configurations.
"""

import numpy as np
import pytest
import sympy as sp

from israel_stewart.core import (
    FLRWMetric,
    MilneMetric,
    MinkowskiMetric,
    SpacetimeGrid,
)
from israel_stewart.core.fields import ISFieldConfiguration


class TestNoneMetricConstraints:
    """Test constraint enforcement with None metric (default Minkowski case)."""

    @pytest.fixture
    def default_grid(self) -> SpacetimeGrid:
        """Create SpacetimeGrid with None metric (most common case)."""
        return SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            grid_points=(3, 3, 3, 3),
            metric=None,  # Default Minkowski case
        )

    @pytest.fixture
    def field_config(self, default_grid) -> ISFieldConfiguration:
        """Create field configuration on default grid."""
        return ISFieldConfiguration(default_grid)

    def test_apply_constraints_none_metric(self, field_config):
        """Test that apply_constraints works with None metric (main bug fix)."""
        # Setup realistic field values
        field_config.rho.fill(1.0)  # Energy density
        field_config.pressure.fill(0.33)  # Pressure

        # Set non-trivial four-velocity (will be normalized)
        field_config.u_mu[..., 0] = 1.1  # Slightly off-normalized time component
        field_config.u_mu[..., 1] = 0.1  # Small spatial velocity

        # Set non-trivial dissipative fields
        field_config.Pi.fill(0.01)  # Bulk pressure
        field_config.pi_munu[..., 0, 1] = 0.005  # Shear component
        field_config.q_mu[..., 1] = 0.002  # Heat flux

        # This should NOT crash (was the main bug)
        try:
            field_config.apply_constraints()
        except ValueError as e:
            if "Cannot project" in str(e) and "without metric" in str(e):
                pytest.fail(f"None metric constraint bug not fixed: {e}")
            else:
                raise

        # Verify constraints are properly enforced
        assert field_config._constraints_enforced

    def test_four_velocity_normalization_none_metric(self, field_config):
        """Test four-velocity normalization with None metric."""
        # Set denormalized four-velocity
        field_config.u_mu[..., 0] = 2.0  # Too large time component
        field_config.u_mu[..., 1] = 0.5  # Spatial component

        field_config.apply_constraints()

        # Check normalization: u^μ u_μ = -1 for mostly-plus Minkowski
        u_squared = -(field_config.u_mu[..., 0] ** 2) + np.sum(
            field_config.u_mu[..., 1:4] ** 2, axis=-1
        )
        expected_norm = -1.0

        np.testing.assert_allclose(u_squared, expected_norm, rtol=1e-12)

    def test_shear_tensor_projection_none_metric(self, field_config):
        """Test shear tensor projection with None metric."""
        # Set non-orthogonal, non-traceless shear tensor
        field_config.pi_munu[..., 0, 0] = 0.1  # Trace component
        field_config.pi_munu[..., 0, 1] = 0.05  # Non-orthogonal component
        field_config.pi_munu[..., 1, 1] = 0.1  # Diagonal component

        field_config.apply_constraints()

        # Verify orthogonality: π^μν u_μ = 0
        # In Minkowski: u_μ = g_μν u^ν = (-u^0, u^1, u^2, u^3)
        u_lower = field_config.u_mu.copy()
        u_lower[..., 0] *= -1  # Lower time index in mostly-plus

        contraction = np.einsum("...ij,...i->...j", field_config.pi_munu, u_lower)
        np.testing.assert_allclose(contraction, 0.0, atol=1e-14)

        # Verify tracelessness: π^μ_μ = g_μν π^μν = 0
        g_diag = np.array([-1, 1, 1, 1])  # Minkowski diagonal
        trace = np.einsum("...ii,i->...", field_config.pi_munu, g_diag)
        np.testing.assert_allclose(trace, 0.0, atol=1e-14)

    def test_heat_flux_projection_none_metric(self, field_config):
        """Test heat flux projection with None metric."""
        # Set non-orthogonal heat flux
        field_config.q_mu[..., 0] = 0.1  # Time component (should be projected out)
        field_config.q_mu[..., 1] = 0.05  # Spatial component
        field_config.q_mu[..., 2] = 0.03  # Spatial component

        field_config.apply_constraints()

        # Verify orthogonality: q^μ u_μ = 0
        # In Minkowski: u_μ = (-u^0, u^1, u^2, u^3)
        u_lower = field_config.u_mu.copy()
        u_lower[..., 0] *= -1  # Lower time index in mostly-plus

        dot_product = np.einsum("...i,...i->...", field_config.q_mu, u_lower)
        np.testing.assert_allclose(dot_product, 0.0, atol=1e-14)


class TestSymbolicMetricConstraints:
    """Test constraint enforcement with symbolic metrics (SymPy matrices)."""

    @pytest.fixture
    def milne_grid(self) -> SpacetimeGrid:
        """Create SpacetimeGrid with Milne metric (symbolic)."""
        return SpacetimeGrid(
            coordinate_system="milne",
            time_range=(0.1, 1.0),  # τ > 0 for Milne coordinates
            spatial_ranges=[(-1.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            grid_points=(3, 3, 3, 3),
            metric=MilneMetric(),
        )

    @pytest.fixture
    def flrw_grid(self) -> SpacetimeGrid:
        """Create SpacetimeGrid with FLRW metric (symbolic)."""
        return SpacetimeGrid(
            coordinate_system="spherical",
            time_range=(0.1, 1.0),
            spatial_ranges=[(0.1, 1.0), (0.0, np.pi), (0.0, 2 * np.pi)],
            grid_points=(3, 3, 3, 3),
            metric=FLRWMetric(),
        )

    def test_milne_metric_constraint_enforcement(self, milne_grid):
        """Test constraints work with Milne metric (SymPy matrices)."""
        field_config = ISFieldConfiguration(milne_grid)

        # Set realistic initial conditions
        field_config.rho.fill(1.0)
        field_config.pressure.fill(0.33)

        # Set four-velocity (will be normalized)
        field_config.u_mu[..., 0] = 1.0
        field_config.u_mu[..., 1] = 0.1

        # Set dissipative fields
        field_config.Pi.fill(0.01)
        field_config.pi_munu[..., 1, 2] = 0.005
        field_config.q_mu[..., 2] = 0.002

        # This should NOT crash (was the symbolic metric bug)
        try:
            field_config.apply_constraints()
        except AttributeError as e:
            if "ndim" in str(e):
                pytest.fail(f"Symbolic metric ndim bug not fixed: {e}")
            else:
                raise

        assert field_config._constraints_enforced

    def test_flrw_metric_constraint_enforcement(self, flrw_grid):
        """Test constraints work with FLRW metric (SymPy matrices)."""
        field_config = ISFieldConfiguration(flrw_grid)

        # Set realistic initial conditions
        field_config.rho.fill(1.0)
        field_config.pressure.fill(0.33)

        # Set four-velocity
        field_config.u_mu[..., 0] = 1.0
        field_config.u_mu[..., 3] = 0.05  # Small angular velocity

        # Set dissipative fields
        field_config.Pi.fill(0.01)
        field_config.pi_munu[..., 0, 3] = 0.003
        field_config.q_mu[..., 1] = 0.001

        # Should work without ndim AttributeError
        field_config.apply_constraints()
        assert field_config._constraints_enforced

    def test_symbolic_metric_inverse_handling(self, milne_grid):
        """Test that symbolic metric inverse is properly converted."""
        field_config = ISFieldConfiguration(milne_grid)

        # Check that metric inverse is accessible
        assert hasattr(milne_grid.metric, "inverse")
        g_inv = milne_grid.metric.inverse

        # Should be SymPy matrix initially
        assert hasattr(g_inv, "shape")  # SymPy matrices have shape
        assert not hasattr(g_inv, "ndim")  # But not ndim

        # After constraint application, should not crash
        field_config.apply_constraints()

        # Verify that constraints were applied successfully
        assert field_config._constraints_enforced


class TestNumericalMetricConstraints:
    """Test constraint enforcement with numerical metrics (NumPy arrays)."""

    @pytest.fixture
    def minkowski_grid(self) -> SpacetimeGrid:
        """Create SpacetimeGrid with explicit MinkowskiMetric."""
        return SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            grid_points=(3, 3, 3, 3),
            metric=MinkowskiMetric(),
        )

    def test_numerical_metric_constraints(self, minkowski_grid):
        """Test constraints with numerical MinkowskiMetric."""
        field_config = ISFieldConfiguration(minkowski_grid)

        # Set initial conditions
        field_config.rho.fill(1.0)
        field_config.u_mu[..., 0] = 1.1
        field_config.u_mu[..., 1] = 0.2
        field_config.Pi.fill(0.01)
        field_config.pi_munu[..., 1, 2] = 0.005
        field_config.q_mu[..., 2] = 0.002

        # Should work fine (numerical metrics have ndim)
        field_config.apply_constraints()
        assert field_config._constraints_enforced

        # Verify numerical accuracy is maintained
        u_squared = -(field_config.u_mu[..., 0] ** 2) + np.sum(
            field_config.u_mu[..., 1:4] ** 2, axis=-1
        )
        np.testing.assert_allclose(u_squared, -1.0, rtol=1e-12)


class TestEdgeCasesAndRegression:
    """Test edge cases and regression prevention."""

    def test_backward_compatibility(self):
        """Ensure fixes don't break existing functionality."""
        # Test various grid configurations that should still work
        configs = [
            # None metric (most common case)
            SpacetimeGrid("cartesian", (0, 1), [(0, 1), (0, 1), (0, 1)], (2, 2, 2, 2), None),
            # Explicit Minkowski
            SpacetimeGrid(
                "cartesian", (0, 1), [(0, 1), (0, 1), (0, 1)], (2, 2, 2, 2), MinkowskiMetric()
            ),
            # Symbolic metric
            SpacetimeGrid(
                "milne", (0.1, 1), [(-1, 1), (0, 1), (0, 1)], (2, 2, 2, 2), MilneMetric()
            ),
        ]

        for grid in configs:
            field_config = ISFieldConfiguration(grid)
            field_config.rho.fill(1.0)
            field_config.u_mu[..., 0] = 1.0

            # Should not crash for any configuration
            field_config.apply_constraints()
            assert field_config._constraints_enforced

    def test_constraint_methods_accessible(self):
        """Test that individual constraint methods are accessible and functional."""
        grid = SpacetimeGrid("cartesian", (0, 1), [(0, 1), (0, 1), (0, 1)], (3, 3, 3, 3))
        field_config = ISFieldConfiguration(grid)

        # Set initial conditions
        field_config.rho.fill(1.0)
        field_config.u_mu[..., 0] = 1.1
        field_config.u_mu[..., 1] = 0.1
        field_config.Pi.fill(0.01)
        field_config.pi_munu[..., 1, 2] = 0.005
        field_config.q_mu[..., 2] = 0.002

        # Test that individual constraint methods work
        field_config._normalize_four_velocity()
        field_config._project_shear_tensor()
        field_config._project_heat_flux()

        # Should not crash and should apply some constraints
        # Check four-velocity normalization
        u_squared = -(field_config.u_mu[..., 0] ** 2) + np.sum(
            field_config.u_mu[..., 1:4] ** 2, axis=-1
        )
        np.testing.assert_allclose(u_squared, -1.0, rtol=1e-10)

    def test_minimal_grid_constraints(self):
        """Test constraints work on minimal grids."""
        # Very small grid
        grid = SpacetimeGrid("cartesian", (0, 1), [(0, 1), (0, 1), (0, 1)], (2, 2, 2, 2))
        field_config = ISFieldConfiguration(grid)

        field_config.rho.fill(1.0)
        field_config.u_mu[..., 0] = 1.0

        # Should work even on minimal grids
        field_config.apply_constraints()
        assert field_config._constraints_enforced


if __name__ == "__main__":
    pytest.main([__file__])
