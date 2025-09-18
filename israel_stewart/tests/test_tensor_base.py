"""
Tests for the TensorField base class functionality.

This module tests the core tensor operations including grid-based tensors,
index operations, contractions, and SymPy functionality.
"""

import numpy as np
import pytest
import sympy as sp

from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.tensor_base import TensorField


class TestTensorFieldValidation:
    """Test tensor field validation functionality."""

    def test_grid_based_vector_validation(self) -> None:
        """Test that vectors on grids are properly validated."""
        # Create a vector field on a 2x3x4x5 grid
        grid_shape = (2, 3, 4, 5)
        vector_shape = (*grid_shape, 4)
        components = np.random.random(vector_shape)

        # This should work fine
        tensor = TensorField(components, "mu")
        assert tensor.rank == 1
        assert tensor.components.shape == vector_shape

    def test_grid_based_matrix_validation(self) -> None:
        """Test that rank-2 tensors on grids are properly validated."""
        # Create a rank-2 tensor field on a 2x3x4x5 grid
        grid_shape = (2, 3, 4, 5)
        matrix_shape = (*grid_shape, 4, 4)
        components = np.random.random(matrix_shape)

        # This should work fine
        tensor = TensorField(components, "mu nu")
        assert tensor.rank == 2
        assert tensor.components.shape == matrix_shape

    def test_invalid_tensor_dimensions(self) -> None:
        """Test that invalid tensor dimensions are rejected."""
        # Wrong tensor index dimensions
        components = np.random.random((10, 10, 3, 3))  # 3x3 instead of 4x4

        with pytest.raises(ValueError, match="Tensor index dimensions must be 4"):
            TensorField(components, "mu nu")

    def test_rank_mismatch(self) -> None:
        """Test that rank mismatches are detected."""
        # Test with insufficient total dimensions for the specified rank
        components_1d = np.random.random((4,))  # Only 1D, but we'll claim it's rank-2
        with pytest.raises(
            ValueError, match="indices specify rank 2.*but components have only 1 dimensions"
        ):
            TensorField(components_1d, "mu nu")

        # Test with insufficient tensor dimensions
        components_bad = np.random.random((10, 3))  # Only 3 in last dim, not 4
        with pytest.raises(ValueError, match="Tensor index dimensions must be 4"):
            TensorField(components_bad, "mu")

    def test_sympy_validation(self) -> None:
        """Test SymPy tensor validation."""
        # Valid 4x4 SymPy matrix
        components = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f"T_{i}{j}"))
        tensor = TensorField(components, "mu nu")
        assert tensor.rank == 2

        # Invalid dimensions for SymPy
        components = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"T_{i}{j}"))
        with pytest.raises(ValueError, match="SymPy tensor dimensions must be 4"):
            TensorField(components, "mu nu")


class TestTensorOperations:
    """Test tensor operations and contractions."""

    def test_grid_vector_operations(self) -> None:
        """Test operations on grid-based vectors."""
        grid_shape = (2, 3)
        vector_shape = (*grid_shape, 4)

        # Create two vectors
        v1_comp = np.random.random(vector_shape)
        v2_comp = np.random.random(vector_shape)

        v1 = TensorField(v1_comp, "mu")
        v2 = TensorField(v2_comp, "_mu")  # Covariant

        # Test contraction (should work now)
        result = v1.contract(v2, 0, 0)
        # Scalar contraction should return numpy array, not TensorField
        assert isinstance(result, np.ndarray)
        # Result should have grid shape only (no tensor indices)
        assert result.shape == grid_shape

    def test_grid_matrix_operations(self) -> None:
        """Test operations on grid-based matrices."""
        grid_shape = (2, 3)
        matrix_shape = (*grid_shape, 4, 4)

        # Create a matrix tensor
        components = np.random.random(matrix_shape)
        tensor = TensorField(components, "mu nu")

        # Test transpose (only on tensor indices, not grid)
        transposed = tensor.transpose()
        assert transposed.rank == 2
        assert transposed.components.shape == matrix_shape

        # Test symmetrization
        symmetric = tensor.symmetrize()
        assert symmetric.rank == 2
        assert symmetric.components.shape == matrix_shape

    def test_manual_contraction_fix(self) -> None:
        """Test that the manual contraction bug is fixed."""
        # Create SymPy matrix and vector
        matrix_comp = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f"M_{i}{j}"))
        vector_comp = sp.Matrix(4, 1, lambda i, j: sp.Symbol(f"v_{i}"))

        matrix_tensor = TensorField(matrix_comp, "mu nu")
        vector_tensor = TensorField(vector_comp, "_mu")

        # Contract first index of matrix with vector
        result = matrix_tensor.contract(vector_tensor, 0, 0)
        assert result.rank == 1

        # Verify the contraction is correct by checking first component
        # Should be: sum_i M_i0 * v_i
        expected_first = sum(matrix_comp[i, 0] * vector_comp[i] for i in range(4))
        assert result.components[0] == expected_first


class TestExtendedSymPyFunctionality:
    """Test extended SymPy functionality for higher-rank tensors."""

    def test_sympy_transpose_vector(self) -> None:
        """Test SymPy vector transposition."""
        components = sp.Matrix([sp.Symbol(f"v_{i}") for i in range(4)])
        tensor = TensorField(components, "mu")

        # Vector transpose should work
        transposed = tensor.transpose()
        assert transposed.rank == 1

    def test_higher_rank_index_operations(self) -> None:
        """Test index raising/lowering for higher rank tensors."""
        # Create a rank-3 tensor
        components = np.random.random((4, 4, 4))
        tensor = TensorField(components, "mu nu rho")

        # Set up Minkowski metric
        metric = MinkowskiMetric()
        tensor.metric = metric

        # Test raising index at different positions
        raised_0 = tensor.raise_index(0)
        assert raised_0.rank == 3
        assert not raised_0.indices[0][0]  # Should be contravariant

        raised_1 = tensor.raise_index(1)
        assert raised_1.rank == 3
        assert not raised_1.indices[1][0]  # Should be contravariant

    def test_rank_4_tensor_operations(self) -> None:
        """Test operations on rank-4 tensors."""
        components = np.random.random((4, 4, 4, 4))
        tensor = TensorField(components, "mu nu rho sigma")

        # Set up metric
        metric = MinkowskiMetric()
        tensor.metric = metric

        # Test index operations
        raised = tensor.raise_index(2)  # Raise third index
        assert raised.rank == 4
        assert not raised.indices[2][0]  # Should be contravariant

        lowered = raised.lower_index(2)  # Lower it back
        assert lowered.rank == 4
        assert lowered.indices[2][0]  # Should be covariant again


class TestIndexValidation:
    """Test index validation and compatibility."""

    def test_index_compatibility_validation(self) -> None:
        """Test that index compatibility is properly validated."""
        # Create tensors with compatible indices
        v1_comp = np.random.random((4,))
        v2_comp = np.random.random((4,))

        v1 = TensorField(v1_comp, "mu")  # Contravariant
        v2 = TensorField(v2_comp, "_mu")  # Covariant

        # This should work (contravariant with covariant)
        result = v1.contract(v2, 0, 0)
        # Should return scalar (numpy array), not TensorField
        assert isinstance(result, np.ndarray | float | complex)

    def test_trace_operations(self) -> None:
        """Test trace operations on various tensor ranks."""
        # Rank-2 tensor
        components_2 = np.random.random((4, 4))
        tensor_2 = TensorField(components_2, "mu nu")

        trace_2 = tensor_2.trace()
        assert isinstance(trace_2, float | complex | np.floating)

        # Rank-3 tensor on grid
        grid_shape = (2, 3)
        components_3 = np.random.random((*grid_shape, 4, 4, 4))
        tensor_3 = TensorField(components_3, "mu nu rho")

        # Trace over first two indices
        trace_3 = tensor_3.trace((0, 1))
        # Result should have grid shape plus remaining tensor dimension
        expected_shape = (*grid_shape, 4)
        assert isinstance(trace_3, np.ndarray)
        assert trace_3.shape == expected_shape


class TestPerformanceAndWarnings:
    """Test performance monitoring and warning systems."""

    def test_high_rank_warning(self) -> None:
        """Test that high-rank tensors generate warnings."""
        # Create a rank-5 tensor (should trigger warning)
        components = np.random.random((4, 4, 4, 4, 4))

        with pytest.warns(UserWarning, match="High-rank tensor.*performance issues"):
            TensorField(components, "mu nu rho sigma tau")

    def test_sympy_conversion_warning(self) -> None:
        """Test that SymPy conversion operations generate warnings."""
        # Skip this test for now as SymPy Arrays aren't fully supported yet
        pytest.skip("SymPy Array support not yet implemented")


if __name__ == "__main__":
    pytest.main([__file__])
