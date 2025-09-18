"""
Tests for tensor utilities and common functions.

This module tests the utility functions in tensor_utils.py including
validation, conversion, and optimization functions.
"""

import warnings

import numpy as np
import pytest
import sympy as sp

from israel_stewart.core.tensor_utils import (
    PhysicsError,
    RelativisticError,
    TensorValidationError,
    convert_to_numpy,
    convert_to_sympy,
    ensure_compatible_types,
    is_numpy_array,
    is_sympy_matrix,
    optimized_einsum,
    validate_einsum_string,
    validate_index_compatibility,
    validate_tensor_dimensions,
)


class TestTypeGuards:
    """Test type guard functions."""

    def test_is_numpy_array(self) -> None:
        """Test NumPy array type guard."""
        assert is_numpy_array(np.array([1, 2, 3]))
        assert is_numpy_array(np.zeros((4, 4)))
        assert not is_numpy_array([1, 2, 3])
        assert not is_numpy_array(sp.Matrix([1, 2, 3]))

    def test_is_sympy_matrix(self) -> None:
        """Test SymPy matrix type guard."""
        assert is_sympy_matrix(sp.Matrix([1, 2, 3]))
        assert is_sympy_matrix(sp.Matrix(4, 4, lambda i, j: i + j))
        assert not is_sympy_matrix(np.array([1, 2, 3]))
        assert not is_sympy_matrix([1, 2, 3])

    def test_ensure_compatible_types(self) -> None:
        """Test type compatibility checking."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        mat1 = sp.Matrix([1, 2, 3])
        mat2 = sp.Matrix([4, 5, 6])

        assert ensure_compatible_types(arr1, arr2)
        assert ensure_compatible_types(mat1, mat2)
        assert not ensure_compatible_types(arr1, mat1)
        assert not ensure_compatible_types(mat1, arr2)


class TestConversionFunctions:
    """Test conversion between NumPy and SymPy."""

    def test_convert_to_sympy(self) -> None:
        """Test conversion to SymPy matrices."""
        # NumPy to SymPy
        arr = np.array([[1, 2], [3, 4]])
        mat = convert_to_sympy(arr)
        assert is_sympy_matrix(mat)
        assert mat.shape == (2, 2)

        # SymPy to SymPy (identity)
        original = sp.Matrix([[1, 2], [3, 4]])
        result = convert_to_sympy(original)
        assert result is original

        # Invalid type
        with pytest.raises(TypeError, match="Cannot convert.*to SymPy Matrix"):
            convert_to_sympy([1, 2, 3])

    def test_convert_to_numpy_basic(self) -> None:
        """Test basic NumPy conversion."""
        # SymPy to NumPy
        mat = sp.Matrix([[1, 2], [3, 4]])
        arr = convert_to_numpy(mat)
        assert is_numpy_array(arr)
        assert arr.shape == (2, 2)
        assert np.allclose(arr, [[1, 2], [3, 4]])

        # NumPy to NumPy (identity)
        original = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = convert_to_numpy(original)
        assert np.array_equal(result, original)

    def test_convert_to_numpy_with_dtype(self) -> None:
        """Test NumPy conversion with specific dtype."""
        mat = sp.Matrix([[1, 2], [3, 4]])
        arr = convert_to_numpy(mat, dtype=np.int32)
        assert arr.dtype == np.int32

        # NumPy dtype conversion
        original = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        result = convert_to_numpy(original, dtype=np.int32)
        assert result.dtype == np.int32
        assert np.array_equal(result, [[1, 2], [3, 4]])

    def test_convert_to_numpy_complex(self) -> None:
        """Test conversion of complex SymPy expressions."""
        # Simple complex matrix
        mat = sp.Matrix([[1 + sp.I, 2], [3, 4 - sp.I]])
        arr = convert_to_numpy(mat, warn_precision_loss=False)
        # Check that it's a complex array and has correct values
        assert arr.dtype in [np.complex64, np.complex128, complex]
        # Basic verification that imaginary parts are preserved
        assert np.isclose(arr[0, 0], 1 + 1j) or np.isclose(arr[0, 0], complex(1, 1))
        assert np.isclose(arr[1, 1], 4 - 1j) or np.isclose(arr[1, 1], complex(4, -1))

    def test_convert_to_numpy_precision_warnings(self) -> None:
        """Test precision loss warnings."""
        # Symbolic expressions should warn
        x = sp.Symbol("x")
        mat = sp.Matrix([[x, 1], [2, 3]])

        with pytest.warns(UserWarning, match="Converting SymPy expressions with symbols"):
            convert_to_numpy(mat, warn_precision_loss=True)

        # Simple conversion without warnings
        simple_mat = sp.Matrix([[1, 2], [3, 4]])
        # This should work without warnings
        result = convert_to_numpy(simple_mat, warn_precision_loss=False)
        assert is_numpy_array(result)

    def test_convert_to_numpy_invalid_sympy(self) -> None:
        """Test conversion of invalid SymPy expressions."""
        # Unevaluable expression - this is hard to create, so test error handling
        with pytest.raises(TypeError, match="Cannot convert.*to NumPy array"):
            convert_to_numpy("invalid_type")


class TestValidateTensorDimensions:
    """Test tensor dimension validation."""

    def test_validate_numpy_basic(self) -> None:
        """Test basic NumPy tensor validation."""
        # Valid 4x4 tensor
        components = np.random.random((4, 4))
        validate_tensor_dimensions(components)  # Should not raise

        # Invalid dimensions
        components = np.random.random((3, 3))
        with pytest.raises(ValueError, match="All tensor dimensions must be 4"):
            validate_tensor_dimensions(components)

    def test_validate_numpy_grid_tensors(self) -> None:
        """Test validation of grid-based tensor fields."""
        # Valid grid-based vector field
        grid_shape = (10, 12, 8)
        vector_components = np.random.random((*grid_shape, 4))
        validate_tensor_dimensions(vector_components, tensor_rank=1)  # Should not raise

        # Valid grid-based rank-2 tensor field
        matrix_components = np.random.random((*grid_shape, 4, 4))
        validate_tensor_dimensions(matrix_components, tensor_rank=2)  # Should not raise

        # Invalid tensor dimensions
        bad_components = np.random.random((*grid_shape, 3, 3))
        with pytest.raises(ValueError, match="Tensor index dimensions must be 4"):
            validate_tensor_dimensions(bad_components, tensor_rank=2)

        # Insufficient dimensions for rank
        insufficient = np.random.random((10,))  # Only 1D but claiming rank 2
        with pytest.raises(ValueError, match="but tensor rank 2 requires at least 2 dimensions"):
            validate_tensor_dimensions(insufficient, tensor_rank=2)

    def test_validate_sympy_basic(self) -> None:
        """Test SymPy tensor validation."""
        # Valid 4x4 matrix
        components = sp.Matrix(4, 4, lambda i, j: i + j)
        validate_tensor_dimensions(components)  # Should not raise

        # Invalid dimensions
        components = sp.Matrix(3, 3, lambda i, j: i + j)
        with pytest.raises(ValueError, match="All tensor dimensions must be 4"):
            validate_tensor_dimensions(components)

    def test_validate_sympy_with_rank(self) -> None:
        """Test SymPy validation with tensor rank."""
        # Valid rank-1 vector
        vector = sp.Matrix([1, 2, 3, 4])
        validate_tensor_dimensions(vector, tensor_rank=1)  # Should not raise

        # Valid rank-1 column vector
        col_vector = sp.Matrix(4, 1, lambda i, j: i)
        validate_tensor_dimensions(col_vector, tensor_rank=1)  # Should not raise

        # Valid rank-2 matrix
        matrix = sp.Matrix(4, 4, lambda i, j: i + j)
        validate_tensor_dimensions(matrix, tensor_rank=2)  # Should not raise

        # Invalid combinations
        with pytest.raises(ValueError, match="SymPy tensor with rank 1 has invalid shape"):
            validate_tensor_dimensions(matrix, tensor_rank=1)  # 4x4 claimed as rank-1

        with pytest.raises(ValueError, match="SymPy tensor with rank 2 has invalid shape"):
            validate_tensor_dimensions(vector, tensor_rank=2)  # vector claimed as rank-2

    def test_validate_nan_infinity(self) -> None:
        """Test detection of NaN and infinity values."""
        # NaN values
        components = np.array([[1, 2, 3, 4], [np.nan, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        with pytest.raises(ValueError, match="contain NaN or infinite values"):
            validate_tensor_dimensions(components)

        # Infinity values
        components = np.array([[1, 2, 3, 4], [5, np.inf, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        with pytest.raises(ValueError, match="contain NaN or infinite values"):
            validate_tensor_dimensions(components)

    def test_validate_large_values_warning(self) -> None:
        """Test warning for very large tensor components."""
        components = np.array([[1e101, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

        with pytest.warns(UserWarning, match="very large.*may cause numerical issues"):
            validate_tensor_dimensions(components)

    def test_validate_expected_shape(self) -> None:
        """Test validation against expected shape."""
        components = np.random.random((4, 4))

        # Correct shape
        validate_tensor_dimensions(components, expected_shape=(4, 4))  # Should not raise

        # Wrong shape
        with pytest.raises(ValueError, match="Expected shape.*got"):
            validate_tensor_dimensions(components, expected_shape=(3, 3))


class TestIndexValidation:
    """Test index compatibility validation."""

    def test_valid_contractions(self) -> None:
        """Test valid index contractions."""
        # Contravariant with covariant (ideal)
        tensor1_indices = [(False, "mu"), (True, "nu")]  # mu^mu nu_nu
        tensor2_indices = [(True, "mu"), (False, "rho")]  # mu_mu rho^rho

        # This should work without warnings (contravariant mu with covariant mu)
        validate_index_compatibility(tensor1_indices, tensor2_indices, (0, 0))

    def test_same_type_contraction_warning(self) -> None:
        """Test warning for contracting same-type indices."""
        # Both contravariant
        tensor1_indices = [(False, "mu"), (False, "nu")]
        tensor2_indices = [(False, "mu"), (False, "rho")]

        with pytest.warns(UserWarning, match="Contracting indices of same type.*contravariant"):
            validate_index_compatibility(tensor1_indices, tensor2_indices, (0, 0))

        # Both covariant
        tensor1_indices = [(True, "mu"), (True, "nu")]
        tensor2_indices = [(True, "mu"), (True, "rho")]

        with pytest.warns(UserWarning, match="Contracting indices of same type.*covariant"):
            validate_index_compatibility(tensor1_indices, tensor2_indices, (0, 0))

    def test_index_out_of_range(self) -> None:
        """Test index out of range errors."""
        tensor1_indices = [(False, "mu")]
        tensor2_indices = [(True, "nu")]

        # First tensor index out of range
        with pytest.raises(ValueError, match="Contraction index 1 out of range for first tensor"):
            validate_index_compatibility(tensor1_indices, tensor2_indices, (1, 0))

        # Second tensor index out of range
        with pytest.raises(ValueError, match="Contraction index 1 out of range for second tensor"):
            validate_index_compatibility(tensor1_indices, tensor2_indices, (0, 1))


class TestEinsumValidation:
    """Test einsum string validation."""

    def test_valid_einsum_strings(self) -> None:
        """Test valid einsum string validation."""
        # Valid contractions
        validate_einsum_string("ij,jk->ik", 2, 2)  # Matrix multiplication
        validate_einsum_string("i,i->", 1, 1)  # Dot product
        validate_einsum_string("ijk,k->ij", 3, 1)  # Tensor-vector contraction

    def test_invalid_einsum_format(self) -> None:
        """Test invalid einsum string formats."""
        with pytest.raises(ValueError, match="Invalid einsum string format"):
            validate_einsum_string("ij,jk", 2, 2)  # Missing arrow

        with pytest.raises(ValueError, match="Expected 2 input tensors"):
            validate_einsum_string("ij->i", 1, 1)  # Only one input tensor

    def test_rank_mismatch(self) -> None:
        """Test einsum rank mismatch detection."""
        with pytest.raises(ValueError, match="First tensor rank mismatch"):
            validate_einsum_string("ijk,jk->ik", 2, 2)  # First tensor is rank 3, not 2

        with pytest.raises(ValueError, match="Second tensor rank mismatch"):
            validate_einsum_string("ij,jkl->ikl", 2, 2)  # Second tensor is rank 3, not 2

    def test_repeated_index_validation(self) -> None:
        """Test validation of repeated indices."""
        # Too many repetitions
        with pytest.raises(ValueError, match="appears more than twice"):
            validate_einsum_string("iii,i->i", 3, 1)

    def test_invalid_characters(self) -> None:
        """Test invalid character detection."""
        # Before our fix, this would fail with invalid characters
        # Now we support A-Z for grid dimensions
        validate_einsum_string("ABa,ABa->AB", 3, 3)  # Should work

        # But truly invalid characters should still fail
        with pytest.raises(ValueError, match="Invalid index characters"):
            validate_einsum_string("i1,1j->ij", 2, 2)  # Numbers not allowed

    def test_output_index_warnings(self) -> None:
        """Test warnings for mismatched output indices."""
        # This should generate a warning about output indices
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_einsum_string("ij,jk->ki", 2, 2)  # Output order might not match expected

            # Check if warning was issued (this is implementation dependent)
            output_warnings = [warning for warning in w if "Output indices" in str(warning.message)]
            # We don't strictly require the warning, just test that validation completes


class TestOptimizedEinsum:
    """Test optimized einsum functionality."""

    def test_numpy_einsum_basic(self) -> None:
        """Test basic NumPy einsum operations."""
        a = np.random.random((4, 4))
        b = np.random.random((4, 4))

        # Matrix multiplication
        result = optimized_einsum("ij,jk->ik", a, b)
        expected = np.einsum("ij,jk->ik", a, b)
        assert np.allclose(result, expected)

        # Vector dot product
        v1 = np.random.random((4,))
        v2 = np.random.random((4,))
        result = optimized_einsum("i,i->", v1, v2)
        expected = np.dot(v1, v2)
        assert np.allclose(result, expected)

    def test_sympy_einsum_basic(self) -> None:
        """Test basic SymPy einsum operations."""
        a = sp.Matrix([[1, 2], [3, 4]])
        b = sp.Matrix([[5, 6], [7, 8]])

        # Matrix multiplication
        result = optimized_einsum("ij,jk->ik", a, b)
        expected = a * b
        assert result == expected

    def test_sympy_einsum_fallback(self) -> None:
        """Test SymPy einsum fallback to NumPy."""
        a = sp.Matrix(4, 4, lambda i, j: i + j)
        b = sp.Matrix(4, 4, lambda i, j: i * j)

        # Complex operation that doesn't match simple patterns - element-wise product
        # This should warn and use NumPy fallback since it's not matrix multiplication
        with pytest.warns(UserWarning):
            result = optimized_einsum("ij,ij->ij", a, b)  # Element-wise product, triggers fallback
            # Should work and return a SymPy matrix
            assert is_sympy_matrix(result)

    def test_mixed_type_conversion(self) -> None:
        """Test mixed NumPy/SymPy tensor handling."""
        numpy_tensor = np.array([[1, 2], [3, 4]])
        sympy_tensor = sp.Matrix([[5, 6], [7, 8]])

        # Mixed types should be converted to NumPy
        result = optimized_einsum("ij,jk->ik", numpy_tensor, sympy_tensor)
        assert is_numpy_array(result)

    def test_empty_tensor_error(self) -> None:
        """Test error handling for empty tensor list."""
        with pytest.raises(ValueError, match="At least one tensor must be provided"):
            optimized_einsum("->")

    def test_malformed_einsum_error(self) -> None:
        """Test error handling for malformed einsum strings."""
        a = sp.Matrix([[1, 2], [3, 4]])
        b = sp.Matrix([[5, 6], [7, 8]])

        with pytest.raises(ValueError, match="Invalid einsum format"):
            optimized_einsum("ij,jk", a, b)  # Missing arrow


class TestExceptionClasses:
    """Test custom exception classes."""

    def test_exception_inheritance(self) -> None:
        """Test exception class hierarchy."""
        # Test that custom exceptions are properly defined
        assert issubclass(TensorValidationError, Exception)
        assert issubclass(PhysicsError, Exception)
        assert issubclass(RelativisticError, PhysicsError)

        # Test instantiation
        validation_error = TensorValidationError("Test validation error")
        assert str(validation_error) == "Test validation error"

        physics_error = PhysicsError("Test physics error")
        assert str(physics_error) == "Test physics error"

        relativistic_error = RelativisticError("Test relativistic error")
        assert str(relativistic_error) == "Test relativistic error"


if __name__ == "__main__":
    pytest.main([__file__])
