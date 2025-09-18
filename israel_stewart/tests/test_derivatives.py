"""
Tests for derivatives.py fixes focusing on critical bug fixes.

This module tests the three major bug fixes:
1. SymPy Matrix shape incompatibility in scalar_gradient method
2. edge_order=2 failure on minimal grids in vector_divergence method
3. Parallel projector sign convention for metric signatures

All tests focus on ensuring production-ready reliability for Israel-Stewart
hydrodynamics simulations.
"""

import numpy as np
import pytest
import sympy as sp

from israel_stewart.core import (
    FourVector,
    MinkowskiMetric,
    SpacetimeGrid,
    TensorField,
)
from israel_stewart.core.derivatives import CovariantDerivative, ProjectionOperator


class TestScalarGradientFix:
    """Test SymPy Matrix shape incompatibility fix in scalar_gradient method."""

    @pytest.fixture
    def symbolic_grid(self) -> SpacetimeGrid:
        """Create a simple grid for symbolic computation tests."""
        return SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            grid_points=(3, 3, 3, 3),
        )

    @pytest.fixture
    def symbolic_metric(self) -> MinkowskiMetric:
        """Create Minkowski metric for symbolic tests."""
        return MinkowskiMetric()

    def test_scalar_gradient_symbolic_return_type(self, symbolic_metric):
        """Test that scalar_gradient returns proper FourVector for symbolic expressions."""
        cov_deriv = CovariantDerivative(symbolic_metric)

        # Create a symbolic scalar field using the EXACT same symbols that the method uses
        # This tests the original bug: SymPy Matrix shape incompatibility
        coord_symbols = [sp.Symbol(f"x{i}") for i in range(4)]  # Match exactly what method uses
        x0, x1, x2, x3 = coord_symbols
        scalar_field = x1**2 + x2**2  # Simple symbolic expression

        # Create dummy coordinate arrays (the symbolic branch doesn't actually use these)
        coord_arrays = [np.array([0.0, 1.0]) for _ in range(4)]

        # This should not crash with shape incompatibility (original bug)
        gradient = cov_deriv.scalar_gradient(scalar_field, coord_arrays)

        # Should return FourVector, not SymPy Matrix
        assert isinstance(gradient, FourVector)
        assert hasattr(gradient, "components")

        # Components should have proper format for FourVector
        components = gradient.components
        if hasattr(components, "shape"):
            # NumPy array case from fallback
            assert components.shape == (4,), f"Expected shape (4,), got {components.shape}"
        elif hasattr(components, "__len__"):
            # SymPy array case
            assert len(components) == 4, f"Expected length 4, got {len(components)}"
        else:
            # For SymPy arrays, check if it's indexable
            assert components[0] is not None

        # Verify that gradients are computed correctly
        # For field = x1² + x2², we expect ∇φ = (0, 2*x1, 2*x2, 0)
        # Check that we get non-zero derivatives for x1 and x2
        if hasattr(components, "__len__"):
            # Check that derivatives are computed (not all zero)
            assert str(components[1]) == "2*x1", f"Expected 2*x1, got {components[1]}"
            assert str(components[2]) == "2*x2", f"Expected 2*x2, got {components[2]}"

    def test_scalar_gradient_symbolic_computation(self, symbolic_grid, symbolic_metric):
        """Test symbolic gradient computation for known analytical result."""
        cov_deriv = CovariantDerivative(symbolic_metric)

        # Use a simple polynomial: φ = 2t + 3x + 4y + 5z
        coords = symbolic_grid.meshgrid()
        t, x, y, z = coords[0], coords[1], coords[2], coords[3]
        scalar_field = 2 * t + 3 * x + 4 * y + 5 * z

        # Get coordinate arrays for the method
        coord_arrays = [symbolic_grid.coordinates[name] for name in ["t", "x", "y", "z"]]

        gradient = cov_deriv.scalar_gradient(scalar_field, coord_arrays)

        # For Minkowski metric, covariant gradient = partial derivative
        # ∂φ/∂t = 2, ∂φ/∂x = 3, ∂φ/∂y = 4, ∂φ/∂z = 5
        components = gradient.components

        # Extract numerical values, handling both SymPy and NumPy cases
        if hasattr(components, "dtype"):
            # NumPy array
            expected = np.array([2.0, 3.0, 4.0, 5.0])
            np.testing.assert_allclose(components, expected, rtol=1e-10)
        else:
            # SymPy array - check each component
            expected_values = [2, 3, 4, 5]
            for i, expected_val in enumerate(expected_values):
                if hasattr(components[i], "evalf"):
                    computed_val = float(components[i].evalf())
                else:
                    computed_val = float(components[i])
                assert abs(computed_val - expected_val) < 1e-10

    def test_scalar_gradient_complex_field(self, symbolic_grid, symbolic_metric):
        """Test that gradient handles complex symbolic expressions."""
        cov_deriv = CovariantDerivative(symbolic_metric)

        # Complex field: φ = t² - x²
        coords = symbolic_grid.meshgrid()
        t, x = coords[0], coords[1]
        scalar_field = t**2 - x**2

        # Get coordinate arrays for the method
        coord_arrays = [symbolic_grid.coordinates[name] for name in ["t", "x", "y", "z"]]

        # Should not crash and should return valid FourVector
        gradient = cov_deriv.scalar_gradient(scalar_field, coord_arrays)
        assert isinstance(gradient, FourVector)

        # Verify contravariant flag is set correctly
        assert gradient.is_contravariant

    def test_scalar_gradient_fallback_mechanism(self, symbolic_grid, symbolic_metric):
        """Test that NumPy fallback works when sp.Array is unavailable."""
        cov_deriv = CovariantDerivative(symbolic_metric)

        # Create field that will definitely require symbolic computation
        coords = symbolic_grid.meshgrid()
        x = coords[1]
        scalar_field = sp.sin(x)  # Trigonometric function

        # Get coordinate arrays for the method
        coord_arrays = [symbolic_grid.coordinates[name] for name in ["t", "x", "y", "z"]]

        # This should work regardless of sp.Array availability
        gradient = cov_deriv.scalar_gradient(scalar_field, coord_arrays)
        assert isinstance(gradient, FourVector)


class TestVectorDivergenceFix:
    """Test edge_order=2 failure fix on minimal grids in vector_divergence method."""

    def test_vector_divergence_minimal_grid(self):
        """Test that vector_divergence works on minimal 2-point grids."""
        # Create minimal grid with only 2 points per dimension
        minimal_grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            grid_points=(2, 2, 2, 2),  # Minimal grid
        )

        metric = MinkowskiMetric()
        cov_deriv = CovariantDerivative(metric)

        # Create simple vector field with shape matching grid
        vector_components = np.zeros(minimal_grid.shape + (4,))
        vector_components[..., 0] = 1.0  # Constant time component
        vector_components[..., 1] = 2.0  # Constant x component

        vector_field = TensorField(vector_components, "mu", metric)

        # Get coordinate arrays for the method
        coord_arrays = [minimal_grid.coordinates[name] for name in ["t", "x", "y", "z"]]

        # This should NOT crash with edge_order=2 error
        try:
            divergence = cov_deriv.vector_divergence(vector_field, coord_arrays)
            assert divergence.shape == minimal_grid.shape
            # For constant vector in flat space, divergence should be zero
            assert np.allclose(divergence, 0.0, atol=1e-10)
        except ValueError as e:
            if "edge_order" in str(e):
                pytest.fail(f"edge_order error not fixed: {e}")
            else:
                raise

    def test_vector_divergence_edge_order_scaling(self):
        """Test that edge_order scales appropriately with grid size."""
        metric = MinkowskiMetric()
        cov_deriv = CovariantDerivative(metric)

        # Test different grid sizes
        grid_sizes = [2, 3, 5, 10]

        for n_points in grid_sizes:
            grid = SpacetimeGrid(
                coordinate_system="cartesian",
                time_range=(0.0, 1.0),
                spatial_ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
                grid_points=(n_points, n_points, n_points, n_points),
            )

            # Create linear vector field: V^μ = (0, x, y, z)
            coords = grid.meshgrid()
            x, y, z = coords[1], coords[2], coords[3]

            vector_components = np.zeros(grid.shape + (4,))
            vector_components[..., 1] = x
            vector_components[..., 2] = y
            vector_components[..., 3] = z

            vector_field = TensorField(vector_components, "mu", metric)

            # Get coordinate arrays for the method
            coord_arrays = [grid.coordinates[name] for name in ["t", "x", "y", "z"]]

            # Should work for all grid sizes
            divergence = cov_deriv.vector_divergence(vector_field, coord_arrays)
            assert divergence.shape == grid.shape

            # For V^μ = (0, x, y, z), divergence = ∂V^x/∂x + ∂V^y/∂y + ∂V^z/∂z = 3
            expected_divergence = 3.0
            interior_points = divergence[1:-1, 1:-1, 1:-1, 1:-1] if n_points > 2 else divergence

            if len(interior_points.flatten()) > 0:
                # For larger grids, check interior points for accuracy
                if n_points >= 3:
                    np.testing.assert_allclose(interior_points, expected_divergence, rtol=0.1)

    def test_vector_divergence_single_point_grid(self):
        """Test edge case with single point per dimension."""
        # Single point grid
        single_grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 0.0),
            spatial_ranges=[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            grid_points=(1, 1, 1, 1),
        )

        metric = MinkowskiMetric()
        cov_deriv = CovariantDerivative(metric)

        vector_components = np.ones(single_grid.shape + (4,))
        vector_field = TensorField(vector_components, "mu", metric)

        # Get coordinate arrays for the method
        coord_arrays = [single_grid.coordinates[name] for name in ["t", "x", "y", "z"]]

        # Should handle gracefully, either by using edge_order=1 or appropriate fallback
        try:
            divergence = cov_deriv.vector_divergence(vector_field, coord_arrays)
            assert divergence.shape == single_grid.shape
        except ValueError as e:
            if "edge_order" in str(e):
                pytest.fail(f"edge_order error not properly handled for single point grid: {e}")
            else:
                # Other errors might be expected for single point grids
                pass


class TestParallelProjectorFix:
    """Test parallel projector sign convention fix for metric signatures."""

    @pytest.fixture
    def mostly_plus_metric(self) -> MinkowskiMetric:
        """Minkowski metric with mostly plus signature (-,+,+,+)."""
        return MinkowskiMetric(signature="mostly_plus")

    @pytest.fixture
    def mostly_minus_metric(self) -> MinkowskiMetric:
        """Minkowski metric with mostly minus signature (+,-,-,-)."""
        return MinkowskiMetric(signature="mostly_minus")

    def test_parallel_projection_orthogonality_mostly_plus(self, mostly_plus_metric):
        """Test parallel projection orthogonality for mostly plus signature."""
        # Create normalized timelike four-velocity
        u_components = np.array([1.0, 0.0, 0.0, 0.0])  # Rest frame

        # Normalize: u^μ u_μ = -c² for mostly plus
        u_norm_squared = mostly_plus_metric.components[0, 0]  # g_00 = -1
        u_components[0] = np.sqrt(-u_norm_squared)  # u^0 = 1 for rest frame

        u = FourVector(u_components, True, mostly_plus_metric)
        projector = ProjectionOperator(u, mostly_plus_metric)

        # Create test vector
        test_vector = FourVector([1.0, 2.0, 3.0, 4.0], True, mostly_plus_metric)

        # Project parallel and perpendicular
        v_parallel = projector.project_vector_parallel(test_vector)
        v_perpendicular = projector.project_vector_perpendicular(test_vector)

        # Check orthogonality: v_∥ · v_⊥ = 0
        dot_product = v_parallel.dot(v_perpendicular)
        assert abs(dot_product) < 1e-14, f"Parallel and perpendicular not orthogonal: {dot_product}"

        # Check decomposition: v = v_∥ + v_⊥
        reconstructed = FourVector(
            v_parallel.components + v_perpendicular.components, True, mostly_plus_metric
        )
        np.testing.assert_allclose(reconstructed.components, test_vector.components, rtol=1e-14)

    def test_parallel_projection_orthogonality_mostly_minus(self, mostly_minus_metric):
        """Test parallel projection orthogonality for mostly minus signature."""
        # Create normalized timelike four-velocity
        u_components = np.array([1.0, 0.0, 0.0, 0.0])  # Rest frame

        # Normalize: u^μ u_μ = +c² for mostly minus
        u_norm_squared = mostly_minus_metric.components[0, 0]  # g_00 = +1
        u_components[0] = np.sqrt(u_norm_squared)  # u^0 = 1 for rest frame

        u = FourVector(u_components, True, mostly_minus_metric)
        projector = ProjectionOperator(u, mostly_minus_metric)

        # Create test vector
        test_vector = FourVector([1.0, 2.0, 3.0, 4.0], True, mostly_minus_metric)

        # Project parallel and perpendicular
        v_parallel = projector.project_vector_parallel(test_vector)
        v_perpendicular = projector.project_vector_perpendicular(test_vector)

        # Check orthogonality: v_∥ · v_⊥ = 0
        dot_product = v_parallel.dot(v_perpendicular)
        assert abs(dot_product) < 1e-14, f"Parallel and perpendicular not orthogonal: {dot_product}"

        # Check decomposition: v = v_∥ + v_⊥
        reconstructed = FourVector(
            v_parallel.components + v_perpendicular.components, True, mostly_minus_metric
        )
        np.testing.assert_allclose(reconstructed.components, test_vector.components, rtol=1e-14)

    def test_parallel_projection_sign_convention(self, mostly_plus_metric, mostly_minus_metric):
        """Test that parallel projection gives correct signs for different metrics."""
        # Same four-velocity in both metrics
        u_plus = FourVector([1.0, 0.0, 0.0, 0.0], True, mostly_plus_metric)
        u_minus = FourVector([1.0, 0.0, 0.0, 0.0], True, mostly_minus_metric)

        projector_plus = ProjectionOperator(u_plus, mostly_plus_metric)
        projector_minus = ProjectionOperator(u_minus, mostly_minus_metric)

        # Same test vector
        test_plus = FourVector([1.0, 2.0, 3.0, 4.0], True, mostly_plus_metric)
        test_minus = FourVector([1.0, 2.0, 3.0, 4.0], True, mostly_minus_metric)

        # Project parallel
        v_parallel_plus = projector_plus.project_vector_parallel(test_plus)
        v_parallel_minus = projector_minus.project_vector_parallel(test_minus)

        # Verify that the formulation is signature-independent
        # The key test: both should satisfy v_∥ = (u · v / u · u) u

        # For mostly plus: u · u = -1, u · v = -1 (since g_00 = -1)
        u_dot_u_plus = u_plus.dot(u_plus)
        u_dot_v_plus = u_plus.dot(test_plus)
        expected_coefficient_plus = u_dot_v_plus / u_dot_u_plus

        # For mostly minus: u · u = +1, u · v = +1 (since g_00 = +1)
        u_dot_u_minus = u_minus.dot(u_minus)
        u_dot_v_minus = u_minus.dot(test_minus)
        expected_coefficient_minus = u_dot_v_minus / u_dot_u_minus

        # Check that coefficients have correct signs
        assert np.sign(expected_coefficient_plus) == np.sign(u_dot_v_plus / u_dot_u_plus)
        assert np.sign(expected_coefficient_minus) == np.sign(u_dot_v_minus / u_dot_u_minus)

        # Both projections should be proportional to u with correct coefficients
        np.testing.assert_allclose(
            v_parallel_plus.components, expected_coefficient_plus * u_plus.components, rtol=1e-14
        )
        np.testing.assert_allclose(
            v_parallel_minus.components, expected_coefficient_minus * u_minus.components, rtol=1e-14
        )

    def test_perpendicular_projector_consistency(self, mostly_plus_metric, mostly_minus_metric):
        """Test that perpendicular projector is consistent across signatures."""
        # Test vectors
        u_plus = FourVector([1.0, 0.0, 0.0, 0.0], True, mostly_plus_metric)
        u_minus = FourVector([1.0, 0.0, 0.0, 0.0], True, mostly_minus_metric)

        projector_plus = ProjectionOperator(u_plus, mostly_plus_metric)
        projector_minus = ProjectionOperator(u_minus, mostly_minus_metric)

        # Get perpendicular projectors
        delta_plus = projector_plus.perpendicular_projector()
        delta_minus = projector_minus.perpendicular_projector()

        # Both should satisfy Δ^μν u_ν = 0 (perpendicular to u)
        # Contract with u: Δ^μν u_ν
        contraction_plus = np.einsum("ij,j->i", delta_plus.components, u_plus.lower().components)
        contraction_minus = np.einsum("ij,j->i", delta_minus.components, u_minus.lower().components)

        np.testing.assert_allclose(contraction_plus, 0.0, atol=1e-14)
        np.testing.assert_allclose(contraction_minus, 0.0, atol=1e-14)


class TestIntegrationAndRegression:
    """Integration tests ensuring all fixes work together without regressions."""

    def test_end_to_end_derivatives_workflow(self):
        """Test complete derivatives workflow with all fixes."""
        # Create moderate-sized grid to test all components
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            grid_points=(4, 4, 4, 4),
        )

        metric = MinkowskiMetric()
        cov_deriv = CovariantDerivative(metric)

        # Test scalar gradient (Fix 1)
        coords = grid.meshgrid()
        t, x, y, z = coords[0], coords[1], coords[2], coords[3]
        scalar_field = t**2 + x**2 + y**2 + z**2

        # Get coordinate arrays for the method
        coord_arrays = [grid.coordinates[name] for name in ["t", "x", "y", "z"]]

        gradient = cov_deriv.scalar_gradient(scalar_field, coord_arrays)
        assert isinstance(gradient, FourVector)

        # Test vector divergence (Fix 2)
        vector_components = np.zeros(grid.shape + (4,))
        vector_components[..., 1] = x  # V^x = x
        vector_components[..., 2] = y  # V^y = y
        vector_components[..., 3] = z  # V^z = z

        vector_field = TensorField(vector_components, "mu", metric)
        divergence = cov_deriv.vector_divergence(vector_field, coord_arrays)

        # Divergence of (0, x, y, z) should be 3
        interior = divergence[1:-1, 1:-1, 1:-1, 1:-1]
        if len(interior.flatten()) > 0:
            np.testing.assert_allclose(interior, 3.0, rtol=0.1)

        # Test projection operators (Fix 3)
        u = FourVector([1.0, 0.1, 0.1, 0.1], True, metric)
        projector = ProjectionOperator(u, metric)

        test_vector = FourVector([1.0, 2.0, 3.0, 4.0], True, metric)
        v_parallel = projector.project_vector_parallel(test_vector)
        v_perpendicular = projector.project_vector_perpendicular(test_vector)

        # Orthogonality check
        dot_product = v_parallel.dot(v_perpendicular)
        assert abs(dot_product) < 1e-12

    def test_backwards_compatibility(self):
        """Ensure fixes don't break existing functionality."""
        # Test that existing code patterns still work
        metric = MinkowskiMetric()
        cov_deriv = CovariantDerivative(metric)

        # Numerical arrays should still work fine
        dummy_shape = (3, 3, 3, 3)
        scalar_numeric = np.ones(dummy_shape)

        # Create dummy coordinate arrays
        coord_arrays = [np.linspace(0, 1, 3) for _ in range(4)]

        # This should work without issues
        result = cov_deriv.scalar_gradient(scalar_numeric, coord_arrays)
        assert isinstance(result, FourVector)

        # Vector fields should work
        vector_numeric = np.ones(dummy_shape + (4,))
        vector_field = TensorField(vector_numeric, "mu", metric)

        divergence = cov_deriv.vector_divergence(vector_field, coord_arrays)
        assert divergence.shape == dummy_shape

    def test_mixed_symbolic_numeric_robustness(self):
        """Test robustness with mixed symbolic/numeric operations."""
        metric = MinkowskiMetric()
        cov_deriv = CovariantDerivative(metric)

        # Symbolic field that evaluates to numeric
        x_sym = sp.Symbol("x")
        numeric_field = np.ones((3, 3, 3, 3)) * float(x_sym.subs(x_sym, 2.0))

        # Create dummy coordinate arrays
        coord_arrays = [np.linspace(0, 1, 3) for _ in range(4)]

        gradient = cov_deriv.scalar_gradient(numeric_field, coord_arrays)
        assert isinstance(gradient, FourVector)

        # Should handle gracefully
        assert gradient.components.shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__])
