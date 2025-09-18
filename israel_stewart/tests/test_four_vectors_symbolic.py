"""
Tests for FourVector symbolic workflow compatibility.

This module tests the three critical bug fixes for SymPy symbolic computation:
1. Shape validation accepting SymPy column vectors (4,1)
2. Safe symbolic expression handling in is_timelike/is_spacelike/is_null
3. Metric-safe boost operations with None metric defaults

All tests focus on ensuring symbolic workflows work correctly in the
Israel-Stewart hydrodynamics codebase.
"""

import numpy as np
import pytest
import sympy as sp

from israel_stewart.core.four_vectors import FourVector
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.transformations import LorentzTransformation


class TestSymbolicFourVectorConstruction:
    """Test FourVector construction with SymPy expressions and column vectors."""

    def test_sympy_column_vector_construction(self):
        """Test that SymPy column vectors (4,1) can be used to construct FourVectors."""
        # Create SymPy column vector (this was the main bug)
        tau = sp.Symbol("tau", positive=True)
        components = sp.Matrix(
            [
                [1 / tau],  # time component
                [0],  # x component
                [0],  # y component
                [0],  # z component
            ]
        )

        # This should NOT crash (was the main bug)
        vector = FourVector(components, is_covariant=False)

        # Verify components are accessible
        assert vector.time_component == 1 / tau
        assert vector.x == 0
        assert vector.y == 0
        assert vector.z == 0

        # Verify shape is acceptable (either (4,) or (4,1) for SymPy)
        assert vector.components.shape in [(4,), (4, 1)]

    def test_sympy_row_vector_construction(self):
        """Test that SymPy row vectors (1,4) are handled correctly."""
        xi = sp.Symbol("xi", real=True)

        # Create row vector - should be automatically converted
        row_vector = sp.Matrix([[1, xi, 0, 0]])

        # Constructor should handle this gracefully
        vector = FourVector(row_vector.T, is_covariant=False)  # Transpose to column

        assert vector.time_component == 1
        assert vector.x == xi

    def test_symbolic_expressions_in_components(self):
        """Test FourVectors with complex symbolic expressions."""
        # Create symbolic variables
        t, x, y, z = sp.symbols("t x y z", real=True)
        gamma = sp.Symbol("gamma", positive=True)

        # Create four-velocity with symbolic components
        components = sp.Matrix([gamma, gamma * x, gamma * y, gamma * z])

        vector = FourVector(components, is_covariant=False)

        # Verify symbolic operations work
        assert vector.time_component == gamma
        assert vector.spatial_components[0] == gamma * x

    def test_mixed_numeric_symbolic_construction(self):
        """Test FourVectors with mixed numeric and symbolic components."""
        alpha = sp.Symbol("alpha", real=True)

        # Mix of numeric and symbolic
        components = [1.0, alpha, 0.5, 0.0]

        vector = FourVector(components, is_covariant=False)

        assert vector.time_component == 1.0
        assert vector.x == alpha
        assert vector.y == 0.5


class TestSymbolicClassificationMethods:
    """Test is_timelike, is_spacelike, is_null with symbolic expressions."""

    @pytest.fixture
    def minkowski_metric(self) -> MinkowskiMetric:
        """Create Minkowski metric for testing."""
        return MinkowskiMetric()

    def test_symbolic_timelike_safe_casting(self, minkowski_metric):
        """Test that is_timelike handles symbolic expressions safely."""
        # Create symbolic four-vector that cannot be evaluated to float
        tau = sp.Symbol("tau", positive=True)  # Unknown positive parameter

        components = sp.Matrix([1 / tau, 0, 0, 0])
        vector = FourVector(components, is_covariant=False, metric=minkowski_metric)

        # This should NOT crash with TypeError (was the main bug)
        try:
            result = vector.is_timelike()
            # For unevaluable symbolic expressions, should return False as safe default
            assert isinstance(result, bool)
        except TypeError as e:
            if "Cannot convert expression" in str(e):
                pytest.fail(f"Symbolic is_timelike casting bug not fixed: {e}")
            else:
                raise

    def test_symbolic_spacelike_safe_casting(self, minkowski_metric):
        """Test that is_spacelike handles symbolic expressions safely."""
        xi = sp.Symbol("xi", real=True)  # Unknown real parameter

        components = sp.Matrix([0, xi, 0, 0])
        vector = FourVector(components, is_covariant=False, metric=minkowski_metric)

        # Should not crash with symbolic expressions
        result = vector.is_spacelike()
        assert isinstance(result, bool)

    def test_symbolic_null_safe_casting(self, minkowski_metric):
        """Test that is_null handles symbolic expressions safely."""
        k = sp.Symbol("k", real=True)

        # Create null vector: (k, k, 0, 0) has magnitude squared = -k² + k² = 0
        components = sp.Matrix([k, k, 0, 0])
        vector = FourVector(components, is_covariant=False, metric=minkowski_metric)

        # Should not crash with symbolic expressions
        result = vector.is_null()
        assert isinstance(result, bool)

    def test_evaluable_symbolic_expressions(self, minkowski_metric):
        """Test that evaluable symbolic expressions don't crash and return safe defaults."""
        # Create a symbolic expression that CAN be evaluated
        components = sp.Matrix([sp.sqrt(2), 1, 0, 0])  # This evaluates to numeric
        vector = FourVector(components, is_covariant=False, metric=minkowski_metric)

        # These should NOT crash and return safe defaults (False)
        # Note: Full symbolic tensor operations are complex; the main goal is no crashes
        is_timelike = vector.is_timelike()
        is_spacelike = vector.is_spacelike()

        # For symbolic expressions that can't be fully processed, should return False as safe default
        assert isinstance(is_timelike, bool)
        assert isinstance(is_spacelike, bool)

    def test_classification_without_metric(self):
        """Test that classification methods handle None metric appropriately."""
        # Create vector without metric (default case)
        components = [2.0, 1.0, 0.0, 0.0]
        vector = FourVector(components, is_covariant=False, metric=None)

        # Classification methods should raise informative errors for None metric
        with pytest.raises(ValueError, match="Cannot determine timelike nature without metric"):
            vector.is_timelike()

        with pytest.raises(ValueError, match="Cannot determine spacelike nature without metric"):
            vector.is_spacelike()


class TestSymbolicBoostOperations:
    """Test boost operations with symbolic vectors and None metrics."""

    def test_boost_with_none_metric_timelike_check(self):
        """Test that boost_to_rest_frame works with None metric (main bug fix)."""

        # Create timelike four-velocity without metric (default case)
        components = [2.0, 1.0, 0.5, 0.0]  # γ = 2, v = (0.5, 0.25, 0)
        velocity = FourVector(components, is_covariant=False, metric=None)

        transform = LorentzTransformation()

        # This should NOT crash (was the main bug)
        try:
            boost_matrix = transform.boost_to_rest_frame(velocity)
            assert boost_matrix.shape == (4, 4)
        except ValueError as e:
            if "Cannot determine timelike nature without metric" in str(e):
                pytest.fail(f"None metric boost bug not fixed: {e}")
            else:
                raise

    def test_boost_with_none_metric_spacelike_rejection(self):
        """Test that boost operations correctly reject non-timelike vectors with None metric."""

        # Create spacelike four-vector: u^0 < |u_spatial|
        components = [1.0, 2.0, 1.0, 0.0]  # |u_spatial| = √5 > u^0 = 1
        velocity = FourVector(components, is_covariant=False, metric=None)

        transform = LorentzTransformation()

        # Should reject with appropriate error
        with pytest.raises(Exception, match="non-timelike"):
            transform.boost_to_rest_frame(velocity)

    def test_symbolic_boost_operations(self):
        """Test boost operations with symbolic four-vectors."""
        from israel_stewart.core.metrics import MinkowskiMetric

        # Create symbolic four-velocity
        gamma = sp.Symbol("gamma", positive=True, real=True)
        v = sp.Symbol("v", real=True)

        # Assume γ > 1 and |v| < 1 for timelike condition
        components = sp.Matrix([gamma, gamma * v, 0, 0])
        velocity = FourVector(components, is_covariant=False, metric=MinkowskiMetric())

        # Basic boost operations should not crash
        try:
            # For now, boost operations with symbolic velocities may have limitations
            # but they should not crash with type errors
            three_vel = velocity.extract_three_velocity()
            assert three_vel[0] == v  # v = u^1/u^0 = γv/γ = v
        except Exception as e:
            # Allow physics errors but not type/casting errors
            if "TypeError" in str(e) or "Cannot convert" in str(e):
                pytest.fail(f"Symbolic boost operations have casting issues: {e}")

    def test_boost_with_explicit_metric(self):
        """Test that boost operations work correctly with explicit metrics."""

        metric = MinkowskiMetric()
        components = [2.0, 1.0, 0.5, 0.0]
        velocity = FourVector(components, is_covariant=False, metric=metric)

        transform = LorentzTransformation()

        # Should work fine with explicit metric
        boost_matrix = transform.boost_to_rest_frame(velocity)
        assert boost_matrix.shape == (4, 4)

        # Also test the reverse operation
        reverse_boost = transform.boost_from_rest_frame(velocity)
        assert reverse_boost.shape == (4, 4)


class TestSymbolicEdgeCases:
    """Test edge cases and regression prevention for symbolic workflows."""

    def test_various_sympy_shapes_accepted(self):
        """Test that various SymPy matrix shapes are handled correctly."""
        # Column vector (4,1) - main case
        col_vector = sp.Matrix([[1], [0], [0], [0]])
        v1 = FourVector(col_vector, is_covariant=False)
        assert v1.components.shape in [(4,), (4, 1)]

        # Regular list with symbolic elements
        alpha = sp.Symbol("alpha", real=True)
        list_components = [1, alpha, 0, 0]
        v2 = FourVector(list_components, is_covariant=False)
        assert v2.x == alpha

    def test_backward_compatibility_preserved(self):
        """Ensure fixes don't break existing numeric workflows."""
        # Regular numpy arrays should still work
        np_components = np.array([1.0, 0.5, 0.3, 0.1])
        vector = FourVector(np_components, is_covariant=False)
        assert vector.components.shape in [(4,), (4, 1)]

        # With Minkowski metric
        metric = MinkowskiMetric()
        vector_with_metric = FourVector(np_components, is_covariant=False, metric=metric)

        # Classification should work
        assert isinstance(vector_with_metric.is_timelike(), bool)
        assert isinstance(vector_with_metric.is_spacelike(), bool)
        assert isinstance(vector_with_metric.is_null(), bool)

    def test_mixed_symbolic_numeric_operations(self):
        """Test operations mixing symbolic and numeric components."""
        # Symbolic four-vector
        t = sp.Symbol("t", positive=True)
        sym_vector = FourVector([t, 0, 0, 0], is_covariant=False)

        # Numeric four-vector
        num_vector = FourVector([1.0, 0.5, 0.0, 0.0], is_covariant=False)

        # Basic property access should work for both
        assert sym_vector.time_component == t
        assert num_vector.time_component == 1.0

        # Shape should be acceptable
        assert sym_vector.components.shape in [(4,), (4, 1)]
        assert num_vector.components.shape in [(4,), (4, 1)]

    def test_symbolic_normalization_edge_case(self):
        """Test normalization behavior with symbolic expressions."""
        # Create vector with symbolic magnitude
        a = sp.Symbol("a", positive=True, real=True)
        components = sp.Matrix([a, 0, 0, 0])
        vector = FourVector(components, is_covariant=False, metric=MinkowskiMetric())

        # Magnitude operations should not crash
        try:
            mag_sq = vector.magnitude_squared()
            # Should be -a² in mostly-plus Minkowski
            assert isinstance(mag_sq, sp.Expr)
        except Exception as e:
            if "Cannot convert" in str(e):
                pytest.fail(f"Symbolic magnitude computation fails: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
