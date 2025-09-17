"""
Tests for Christoffel symbol computation and related functionality.

This module tests:
- Numerical vs symbolic Christoffel symbol computation
- Specialized coordinate system implementations
- Integration with derivative operators
- Performance and accuracy validation
"""

import warnings

import numpy as np
import pytest
import sympy as sp

from israel_stewart.core import (
    BJorkenMetric,
    FLRWMetric,
    GeneralMetric,
    MilneMetric,
    MinkowskiMetric,
    SchwarzschildMetric,
    create_cartesian_grid,
)
from israel_stewart.core.derivatives import CovariantDerivative
from israel_stewart.core.metrics import (
    CoordinateError,
    MetricError,
)


class TestMinkowskiChristoffel:
    """Test Christoffel symbols for Minkowski metric."""

    def test_minkowski_christoffel_zeros(self) -> None:
        """Minkowski metric should have zero Christoffel symbols."""
        metric = MinkowskiMetric()
        christoffel = metric.christoffel_symbols

        assert christoffel.shape == (4, 4, 4)
        assert np.allclose(christoffel, 0.0)

    def test_minkowski_different_signatures(self) -> None:
        """Test both metric signatures give zero Christoffel symbols."""
        metric_plus = MinkowskiMetric(signature="mostly_plus")
        metric_minus = MinkowskiMetric(signature="mostly_minus")

        christoffel_plus = metric_plus.christoffel_symbols
        christoffel_minus = metric_minus.christoffel_symbols

        assert np.allclose(christoffel_plus, 0.0)
        assert np.allclose(christoffel_minus, 0.0)

    def test_minkowski_numerical_christoffel(self) -> None:
        """Test numerical Christoffel computation with coordinate arrays."""
        metric = MinkowskiMetric()

        # Create coordinate arrays
        t = np.linspace(0, 1, 10)
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        z = np.linspace(-1, 1, 10)
        coords = [t, x, y, z]

        christoffel_numerical = metric.christoffel_symbols_numerical(coords)
        assert christoffel_numerical.shape == (4, 4, 4)
        assert np.allclose(christoffel_numerical, 0.0)

    def test_minkowski_grid_integration(self) -> None:
        """Test Christoffel computation with SpacetimeGrid."""
        metric = MinkowskiMetric()
        grid = create_cartesian_grid((0, 1), 2.0, (5, 5, 5, 5))

        christoffel_from_grid = metric.christoffel_symbols_from_grid(grid)
        assert christoffel_from_grid.shape == (4, 4, 4)
        assert np.allclose(christoffel_from_grid, 0.0)


class TestSpecializedMetrics:
    """Test specialized metric implementations."""

    def test_milne_metric_properties(self) -> None:
        """Test Milne metric basic properties."""
        metric = MilneMetric()

        assert metric.coordinates == ["tau", "eta", "x", "y"]
        assert metric.signature == (1, -1, -1, -1)
        assert not metric.is_constant()

        components = metric.components
        assert components.shape == (4, 4)

    def test_bjorken_metric_inheritance(self) -> None:
        """Test BJorkenMetric inherits from MilneMetric correctly."""
        bjorken = BJorkenMetric()
        milne = MilneMetric()

        assert bjorken.coordinates == ["tau", "eta", "x", "y"]
        assert bjorken.signature == milne.signature
        assert isinstance(bjorken, MilneMetric)

    def test_flrw_metric_properties(self) -> None:
        """Test FLRW metric basic properties."""
        metric = FLRWMetric()

        assert metric.coordinates == ["t", "r", "theta", "phi"]
        assert metric.signature == (1, -1, -1, -1)
        assert not metric.is_constant()

    def test_schwarzschild_metric_properties(self) -> None:
        """Test Schwarzschild metric basic properties."""
        metric = SchwarzschildMetric()

        assert metric.coordinates == ["t", "r", "theta", "phi"]
        assert metric.signature == (1, -1, -1, -1)
        assert not metric.is_constant()

    def test_flrw_curvature_parameters(self) -> None:
        """Test FLRW metric with different curvature parameters."""
        # Flat universe
        flat_flrw = FLRWMetric(curvature_param=0)
        assert flat_flrw.k == 0

        # Closed universe
        closed_flrw = FLRWMetric(curvature_param=1)
        assert closed_flrw.k == 1

        # Open universe
        open_flrw = FLRWMetric(curvature_param=-1)
        assert open_flrw.k == -1

    def test_schwarzschild_radius_parameter(self) -> None:
        """Test Schwarzschild metric with different radii."""
        metric1 = SchwarzschildMetric(schwarzschild_radius=1.0)
        metric2 = SchwarzschildMetric(schwarzschild_radius=2.0)

        assert metric1.rs == 1.0
        assert metric2.rs == 2.0


class TestNumericalChristoffel:
    """Test numerical Christoffel symbol computation."""

    def test_constant_metric_derivatives(self) -> None:
        """Test that constant metrics have zero derivatives."""
        # Create a constant non-Minkowski metric
        components = np.array([[2, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
        metric = GeneralMetric(components)

        t = np.linspace(0, 1, 10)
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        z = np.linspace(-1, 1, 10)
        coords = [t, x, y, z]

        christoffel = metric.christoffel_symbols_numerical(coords)
        assert np.allclose(christoffel, 0.0)

    def test_coordinate_validation(self) -> None:
        """Test coordinate array validation."""
        metric = MinkowskiMetric()

        # Test wrong number of coordinates
        with pytest.raises(CoordinateError):
            metric.christoffel_symbols_numerical([np.linspace(0, 1, 10)])

        with pytest.raises(CoordinateError):
            metric.christoffel_symbols_numerical([np.linspace(0, 1, 10) for _ in range(5)])

    def test_warning_for_no_coordinates(self) -> None:
        """Test warning when no coordinates provided for numerical computation."""
        # Create a coordinate-dependent metric (symbolic)
        t = sp.Symbol("t")
        g = sp.Matrix([[t**2, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
        metric = GeneralMetric(g)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metric._compute_christoffel_numerical()
            # Filter out other warnings (like metric validation warnings)
            coord_warnings = [
                warning for warning in w if "coordinate arrays" in str(warning.message)
            ]
            assert len(coord_warnings) == 1
            assert "coordinate arrays" in str(coord_warnings[0].message)

    def test_different_grid_shapes(self) -> None:
        """Test numerical computation with different grid shapes."""
        metric = MinkowskiMetric()

        # 1D grid
        coords_1d = [np.linspace(0, 1, 10), np.array([0]), np.array([0]), np.array([0])]
        christoffel_1d = metric.christoffel_symbols_numerical(coords_1d)
        assert christoffel_1d.shape == (4, 4, 4)

        # 2D grid
        coords_2d = [
            np.linspace(0, 1, 5),
            np.linspace(-1, 1, 5),
            np.array([0]),
            np.array([0]),
        ]
        christoffel_2d = metric.christoffel_symbols_numerical(coords_2d)
        assert christoffel_2d.shape == (4, 4, 4)


class TestChristoffelContractions:
    """Test Christoffel symbol contractions in covariant derivatives."""

    def test_vector_contraction(self) -> None:
        """Test Christoffel contraction for vectors."""
        metric = MinkowskiMetric()
        cov_deriv = CovariantDerivative(metric)
        christoffel = metric.christoffel_symbols

        # Test vector
        vector = np.array([1, 2, 3, 4])

        # Contravariant contraction
        result_contra = cov_deriv._contract_christoffel(
            vector, christoffel, tensor_index_pos=0
        )
        assert result_contra.shape == (4, 4)  # Result has shape (4_deriv, 4_orig)
        assert np.allclose(result_contra, 0.0)  # Zero for Minkowski

        # Covariant contraction
        result_cov = cov_deriv._contract_christoffel_covariant(
            vector, christoffel, tensor_index_pos=0
        )
        assert result_cov.shape == (4, 4)  # Result has shape (4_deriv, 4_orig)
        assert np.allclose(result_cov, 0.0)  # Zero for Minkowski

    def test_rank2_tensor_contraction(self) -> None:
        """Test Christoffel contraction for rank-2 tensors."""
        metric = MinkowskiMetric()
        cov_deriv = CovariantDerivative(metric)
        christoffel = metric.christoffel_symbols

        # Test rank-2 tensor
        tensor = np.random.rand(4, 4)

        # Test contraction on first index
        result = cov_deriv._contract_christoffel(
            tensor, christoffel, tensor_index_pos=0
        )
        assert result.shape == (4, 4, 4)  # (4_deriv, 4_orig_0, 4_orig_1)
        assert np.allclose(result, 0.0)  # Zero for Minkowski

        # Test contraction on second index
        result = cov_deriv._contract_christoffel(
            tensor, christoffel, tensor_index_pos=1
        )
        assert result.shape == (4, 4, 4)  # (4_orig_0, 4_deriv, 4_orig_1)
        assert np.allclose(result, 0.0)  # Zero for Minkowski

    def test_higher_rank_tensor_contraction(self) -> None:
        """Test Christoffel contraction for higher-rank tensors."""
        metric = MinkowskiMetric()
        cov_deriv = CovariantDerivative(metric)
        christoffel = metric.christoffel_symbols

        # Test rank-3 tensor
        tensor3 = np.random.rand(4, 4, 4)
        result = cov_deriv._contract_christoffel(
            tensor3, christoffel, tensor_index_pos=1
        )
        assert result.shape == (4, 4, 4, 4)  # (4_orig_0, 4_deriv, 4_orig_1, 4_orig_2)
        assert np.allclose(result, 0.0)  # Zero for Minkowski

        # Test rank-4 tensor
        tensor4 = np.random.rand(4, 4, 4, 4)
        result = cov_deriv._contract_christoffel(
            tensor4, christoffel, tensor_index_pos=3
        )
        assert result.shape == (4, 4, 4, 4, 4)  # (4_orig_0, 4_orig_1, 4_orig_2, 4_deriv, 4_orig_3)
        assert np.allclose(result, 0.0)  # Zero for Minkowski

    def test_scalar_contraction(self) -> None:
        """Test Christoffel contraction for scalars (should return zero)."""
        metric = MinkowskiMetric()
        cov_deriv = CovariantDerivative(metric)
        christoffel = metric.christoffel_symbols

        scalar = np.array(5.0)
        # Scalars have no tensor indices, so contraction should fail or return zero
        # For a scalar, there are no tensor indices to contract
        # This test should be skipped or modified since scalars don't have tensor indices
        with pytest.raises((IndexError, ValueError)):
            result = cov_deriv._contract_christoffel(
                scalar, christoffel, tensor_index_pos=0
            )


class TestSymbolicChristoffel:
    """Test symbolic Christoffel symbol computation."""

    def test_minkowski_symbolic_christoffel(self) -> None:
        """Test symbolic computation for Minkowski metric."""
        # Create symbolic Minkowski metric
        g = sp.Matrix([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        metric = GeneralMetric(g)

        christoffel = metric.christoffel_symbols
        # All components should be zero for constant metric
        is_zero = all(
            christoffel[i, j, k] == 0 for i in range(4) for j in range(4) for k in range(4)
        )
        assert is_zero

    def test_coordinate_dependent_metric(self) -> None:
        """Test symbolic computation for coordinate-dependent metric."""
        # Simple time-dependent metric g_00 = t^2
        t, x, y, z = sp.symbols("t x y z", real=True)

        g = sp.Matrix([[t**2, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
        metric = GeneralMetric(g)

        christoffel = metric.christoffel_symbols

        # Should have non-zero components
        is_zero = all(
            christoffel[i, j, k] == 0 for i in range(4) for j in range(4) for k in range(4)
        )
        assert not is_zero


class TestMetricValidation:
    """Test metric validation and error handling."""

    def test_invalid_metric_shape(self) -> None:
        """Test error for invalid metric tensor shape."""
        with pytest.raises(CoordinateError):
            GeneralMetric(np.ones((3, 3)))  # Wrong size

        with pytest.raises(CoordinateError):
            GeneralMetric(np.ones((4, 3)))  # Non-square

    def test_asymmetric_metric(self) -> None:
        """Test error for asymmetric metric tensor."""
        asymmetric = np.array(
            [
                [1, 1, 0, 0],
                [0, -1, 0, 0],  # Should be [1, -1, 0, 0] for symmetry
                [0, 0, -1, 0],
                [0, 0, 0, -1],
            ]
        )

        with pytest.raises(MetricError):
            metric = GeneralMetric(asymmetric)
            metric.validate_components()

    def test_singular_metric_warning(self) -> None:
        """Test warning for nearly singular metric."""
        # Create nearly singular metric
        epsilon = 1e-16
        singular = np.array(
            [
                [1, 0, 0, 0],
                [0, epsilon, 0, 0],  # Very small eigenvalue
                [0, 0, -1, 0],
                [0, 0, 0, -1],
            ]
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metric = GeneralMetric(singular)
            _ = metric.inverse  # This should trigger warning
            assert len(w) > 0
            assert any("conditioned" in str(warning.message) for warning in w)

    def test_determinant_warning(self) -> None:
        """Test warning for positive determinant (unusual for spacetime)."""
        positive_det = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],  # Positive - unusual for spacetime metric
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metric = GeneralMetric(positive_det)
            _ = metric.determinant  # This should trigger warning
            assert len(w) > 0
            assert any("non-negative" in str(warning.message) for warning in w)


class TestPerformanceAndAccuracy:
    """Test performance and accuracy of Christoffel computations."""

    def test_christoffel_caching(self) -> None:
        """Test that Christoffel symbols are properly cached."""
        metric = MinkowskiMetric()

        # First computation
        christoffel1 = metric.christoffel_symbols

        # Second computation should use cache
        christoffel2 = metric.christoffel_symbols

        # Should be the same object (cached)
        assert christoffel1 is christoffel2

    def test_numerical_vs_symbolic_consistency(self) -> None:
        """Test consistency between numerical and symbolic computations."""
        # Use a simple symbolic metric that can be evaluated numerically
        g_symbolic = sp.Matrix([[2, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])

        g_numerical = np.array(
            [[2, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=float
        )

        metric_sym = GeneralMetric(g_symbolic)
        metric_num = GeneralMetric(g_numerical)

        # Both should give zero Christoffel symbols (constant metric)
        christoffel_sym = metric_sym.christoffel_symbols
        christoffel_num = metric_num.christoffel_symbols

        # Both should give zero for constant metrics
        sym_is_zero = all(
            christoffel_sym[i, j, k] == 0 for i in range(4) for j in range(4) for k in range(4)
        )
        assert sym_is_zero
        assert np.allclose(christoffel_num, 0.0)

    @pytest.mark.slow
    def test_large_grid_performance(self) -> None:
        """Test performance with larger coordinate grids."""
        metric = MinkowskiMetric()

        # Large grid
        n_points = 50
        coords = [np.linspace(-1, 1, n_points) for _ in range(4)]

        # Should complete without timeout (marked as slow test)
        christoffel = metric.christoffel_symbols_numerical(coords)
        assert christoffel.shape == (4, 4, 4)
        assert np.allclose(christoffel, 0.0)

    def test_memory_efficiency(self) -> None:
        """Test memory efficiency for different tensor ranks."""
        metric = MinkowskiMetric()
        cov_deriv = CovariantDerivative(metric)
        christoffel = metric.christoffel_symbols

        # Test memory usage doesn't grow excessively with tensor rank
        for rank in range(1, 5):
            shape = tuple(4 for _ in range(rank))
            tensor = np.random.rand(*shape)

            result = cov_deriv._contract_christoffel(
                tensor, christoffel, tensor_index_pos=0
            )
            # Result should have one additional dimension for derivative index
            expected_shape = (4,) + shape  # Add derivative dimension
            assert result.shape == expected_shape
            # For Minkowski metric, result should be zero
            assert np.allclose(result, 0.0)


if __name__ == "__main__":
    pytest.main([__file__])
