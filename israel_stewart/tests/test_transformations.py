"""
Tests for the Lorentz and coordinate transformations module.

This module tests critical transformation functionality including tensor
transformations, boost operations, and Thomas-Wigner rotations.
"""

import warnings

import numpy as np
import pytest
import sympy as sp

from israel_stewart.core.four_vectors import FourVector
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.tensor_base import TensorField
from israel_stewart.core.tensor_utils import PhysicsError
from israel_stewart.core.transformations import CoordinateTransformation, LorentzTransformation


class TestLorentzTransformation:
    """Test Lorentz transformation functionality."""

    def test_boost_matrix_basic(self) -> None:
        """Test basic boost matrix construction."""
        # Test zero velocity (should be identity)
        boost = LorentzTransformation.boost_matrix([0, 0, 0])
        assert np.allclose(boost, np.eye(4))

        # Test small boost in x-direction
        v = 0.1  # 0.1c
        boost = LorentzTransformation.boost_matrix([v, 0, 0])
        gamma = 1.0 / np.sqrt(1.0 - v**2)

        # Check time-time component
        assert np.isclose(boost[0, 0], gamma)
        # Check time-space components
        assert np.isclose(boost[0, 1], -gamma * v)
        assert np.isclose(boost[1, 0], -gamma * v)

    def test_boost_matrix_superluminal(self) -> None:
        """Test that superluminal velocities raise errors."""
        with pytest.raises(PhysicsError, match="less than speed of light"):
            LorentzTransformation.boost_matrix([1.1, 0, 0])

        with pytest.raises(PhysicsError, match="less than speed of light"):
            LorentzTransformation.boost_matrix([0.9, 0.9, 0])

    def test_rotation_matrix(self) -> None:
        """Test spatial rotation matrices."""
        # Test rotation around z-axis
        angle = np.pi / 4  # 45 degrees
        rotation = LorentzTransformation.rotation_matrix([0, 0, 1], angle)

        # Time component should be unchanged
        assert np.allclose(rotation[0, :], [1, 0, 0, 0])
        assert np.allclose(rotation[:, 0], [1, 0, 0, 0])

        # Check rotation in xy-plane
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        expected_xy = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        assert np.allclose(rotation[1:3, 1:3], expected_xy)

    def test_transform_tensor_numpy_contravariant(self) -> None:
        """Test NumPy tensor transformation for contravariant tensors."""
        metric = MinkowskiMetric()
        transformer = LorentzTransformation(metric)

        # Create a simple contravariant tensor
        components = np.random.random((4, 4))
        tensor = TensorField(components, "mu nu", metric)

        # Apply small boost
        boost = LorentzTransformation.boost_matrix([0.1, 0, 0])
        transformed = transformer.transform_tensor(tensor, boost)

        # Check that transformation preserves tensor structure
        assert transformed.rank == 2
        assert transformed.components.shape == (4, 4)
        assert not transformed.indices[0][0]  # Still contravariant
        assert not transformed.indices[1][0]  # Still contravariant

    def test_transform_tensor_numpy_covariant(self) -> None:
        """Test NumPy tensor transformation for covariant tensors."""
        metric = MinkowskiMetric()
        transformer = LorentzTransformation(metric)

        # Create a covariant tensor
        components = np.random.random((4, 4))
        tensor = TensorField(components, "_mu _nu", metric)

        # Apply small boost
        boost = LorentzTransformation.boost_matrix([0.1, 0, 0])
        transformed = transformer.transform_tensor(tensor, boost)

        # Check that transformation preserves tensor structure
        assert transformed.rank == 2
        assert transformed.components.shape == (4, 4)
        assert transformed.indices[0][0]  # Still covariant
        assert transformed.indices[1][0]  # Still covariant

    def test_transform_tensor_numpy_mixed(self) -> None:
        """Test NumPy tensor transformation for mixed tensors."""
        metric = MinkowskiMetric()
        transformer = LorentzTransformation(metric)

        # Create mixed tensors (contravariant-covariant)
        components = np.random.random((4, 4))
        tensor1 = TensorField(components, "mu _nu", metric)

        # Create mixed tensors (covariant-contravariant)
        tensor2 = TensorField(components, "_mu nu", metric)

        # Apply small boost
        boost = LorentzTransformation.boost_matrix([0.1, 0, 0])

        transformed1 = transformer.transform_tensor(tensor1, boost)
        transformed2 = transformer.transform_tensor(tensor2, boost)

        # Check that transformations preserve index structure
        assert not transformed1.indices[0][0] and transformed1.indices[1][0]  # μ_ν
        assert transformed2.indices[0][0] and not transformed2.indices[1][0]  # _μν

    def test_transform_tensor_sympy_contravariant(self) -> None:
        """Test SymPy tensor transformation for contravariant tensors."""
        metric = MinkowskiMetric()
        transformer = LorentzTransformation(metric)

        # Create SymPy contravariant tensor
        components = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f"T_{i}{j}"))
        tensor = TensorField(components, "mu nu", metric)

        # Apply transformation
        boost = LorentzTransformation.boost_matrix([0.1, 0, 0])
        transformed = transformer.transform_tensor(tensor, boost)

        # Check structure
        assert transformed.rank == 2
        assert isinstance(transformed.components, sp.Matrix)
        assert transformed.components.shape == (4, 4)

    def test_transform_tensor_sympy_covariant(self) -> None:
        """Test SymPy tensor transformation for covariant tensors."""
        metric = MinkowskiMetric()
        transformer = LorentzTransformation(metric)

        # Create SymPy covariant tensor
        components = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f"T_{i}{j}"))
        tensor = TensorField(components, "_mu _nu", metric)

        # Apply transformation
        boost = LorentzTransformation.boost_matrix([0.1, 0, 0])
        transformed = transformer.transform_tensor(tensor, boost)

        # Check structure
        assert transformed.rank == 2
        assert isinstance(transformed.components, sp.Matrix)
        assert transformed.indices[0][0] and transformed.indices[1][0]

    def test_transform_tensor_sympy_mixed(self) -> None:
        """Test SymPy tensor transformation for mixed tensors."""
        metric = MinkowskiMetric()
        transformer = LorentzTransformation(metric)

        # Test both mixed cases
        components = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f"T_{i}{j}"))

        tensor1 = TensorField(components, "mu _nu", metric)  # Contravariant-covariant
        tensor2 = TensorField(components, "_mu nu", metric)  # Covariant-contravariant

        boost = LorentzTransformation.boost_matrix([0.1, 0, 0])

        transformed1 = transformer.transform_tensor(tensor1, boost)
        transformed2 = transformer.transform_tensor(tensor2, boost)

        # Check structure preservation
        assert not transformed1.indices[0][0] and transformed1.indices[1][0]
        assert transformed2.indices[0][0] and not transformed2.indices[1][0]

    def test_transform_vector(self) -> None:
        """Test four-vector transformation."""
        metric = MinkowskiMetric()
        transformer = LorentzTransformation(metric)

        # Create test four-vector
        components = np.array([1.0, 0.5, 0.3, 0.1])
        vector = FourVector(components, False, metric)

        # Apply boost
        boost = LorentzTransformation.boost_matrix([0.2, 0, 0])
        transformed = transformer.transform_vector(vector, boost)

        # Check structure
        assert transformed.components.shape == (4,)
        assert isinstance(transformed, FourVector)

    def test_inverse_transformation(self) -> None:
        """Test inverse transformation calculation."""
        boost = LorentzTransformation.boost_matrix([0.3, 0.2, 0.1])
        inverse = LorentzTransformation.inverse_transformation(boost)

        # Check that boost * inverse = identity
        product = np.dot(boost, inverse)
        assert np.allclose(product, np.eye(4), atol=1e-12)

    def test_composition(self) -> None:
        """Test transformation composition."""
        boost1 = LorentzTransformation.boost_matrix([0.1, 0, 0])
        boost2 = LorentzTransformation.boost_matrix([0, 0.2, 0])

        composed = LorentzTransformation.composition(boost1, boost2)

        # Check that composition is a valid 4x4 matrix
        assert composed.shape == (4, 4)

        # Manual composition should match
        manual = np.dot(boost2, boost1)
        assert np.allclose(composed, manual)

    def test_boost_to_rest_frame(self) -> None:
        """Test boost to rest frame calculation."""
        metric = MinkowskiMetric()
        transformer = LorentzTransformation(metric)

        # Test with a simple velocity in x-direction for easier verification
        v_simple = np.array([0.3, 0, 0])  # Only x-component
        gamma_simple = 1.0 / np.sqrt(1.0 - np.dot(v_simple, v_simple))
        four_vel_simple = np.array([gamma_simple, gamma_simple * v_simple[0], 0, 0])
        four_velocity_simple = FourVector(four_vel_simple, False, metric)

        # Get boost to rest frame
        boost = transformer.boost_to_rest_frame(four_velocity_simple)

        # Apply boost - should give rest frame four-velocity
        transformed = transformer.transform_vector(four_velocity_simple, boost)

        # In rest frame, spatial components should be much smaller
        # Use more relaxed tolerance for this complex calculation
        spatial_magnitude = np.linalg.norm(transformed.spatial_components)
        assert (
            spatial_magnitude < 1e-10
        ), f"Spatial components not small enough: {transformed.spatial_components}"
        assert transformed.time_component > 0

        # Verify that boost is a valid Lorentz transformation
        assert transformer.validate_transformation(boost)

    def test_validate_transformation(self) -> None:
        """Test transformation validation."""
        metric = MinkowskiMetric()
        transformer = LorentzTransformation(metric)

        # Valid transformation (proper Lorentz boost)
        boost = LorentzTransformation.boost_matrix([0.3, 0.2, 0.1])
        assert transformer.validate_transformation(boost)

        # Invalid transformation (non-square)
        invalid = np.random.random((3, 4))
        assert not transformer.validate_transformation(invalid)

        # Invalid transformation (doesn't preserve metric)
        non_lorentz = np.random.random((4, 4))
        assert not transformer.validate_transformation(non_lorentz)

    def test_active_vs_passive_transformation(self) -> None:
        """Test active vs passive transformations."""
        metric = MinkowskiMetric()
        transformer = LorentzTransformation(metric)

        # Create test tensor
        components = np.random.random((4, 4))
        tensor = TensorField(components, "mu nu", metric)

        boost = LorentzTransformation.boost_matrix([0.1, 0, 0])

        # Active and passive should give different results
        active = transformer.active_vs_passive_transformation(tensor, boost, active=True)
        passive = transformer.active_vs_passive_transformation(tensor, boost, active=False)

        assert not np.allclose(active.components, passive.components)


class TestThomasWignerRotation:
    """Test Thomas-Wigner rotation functionality and error handling."""

    def test_small_velocities(self) -> None:
        """Test Thomas-Wigner rotation for small velocities."""
        v1 = np.array([1e-4, 0, 0])  # Very small
        v2 = np.array([0, 1e-4, 0])  # Very small

        # Should not raise warnings
        rotation = LorentzTransformation.thomas_wigner_rotation(v1, v2)

        # Should be a valid 4x4 rotation matrix
        assert rotation.shape == (4, 4)

        # Should preserve time component
        assert np.allclose(rotation[0, :], [1, 0, 0, 0])
        assert np.allclose(rotation[:, 0], [1, 0, 0, 0])

    def test_moderate_velocities_warning(self) -> None:
        """Test that moderate velocities trigger warnings."""
        v1 = np.array([0.15, 0, 0])  # 0.15c (above 0.1c threshold)
        v2 = np.array([0, 0.12, 0])  # 0.12c (above 0.1c threshold)

        with pytest.warns(UserWarning, match="approximation may be inaccurate"):
            rotation = LorentzTransformation.thomas_wigner_rotation(v1, v2)

        # Should still return a valid matrix
        assert rotation.shape == (4, 4)

    def test_high_velocities_error(self) -> None:
        """Test that high velocities raise NotImplementedError."""
        v1 = np.array([0.6, 0, 0])  # 0.6c
        v2 = np.array([0, 0.4, 0])  # 0.4c

        with pytest.raises(NotImplementedError, match="high velocities"):
            LorentzTransformation.thomas_wigner_rotation(v1, v2)

    def test_superluminal_velocities_error(self) -> None:
        """Test that superluminal velocities raise PhysicsError."""
        v1 = np.array([1.1, 0, 0])  # > c
        v2 = np.array([0, 0.3, 0])

        with pytest.raises(PhysicsError, match="less than speed of light"):
            LorentzTransformation.thomas_wigner_rotation(v1, v2)

    def test_invalid_dimensions(self) -> None:
        """Test that invalid velocity dimensions raise ValueError."""
        v1 = np.array([0.1, 0.2])  # Only 2D
        v2 = np.array([0.3, 0.4, 0.5])

        with pytest.raises(ValueError, match="must be 3-dimensional"):
            LorentzTransformation.thomas_wigner_rotation(v1, v2)

    def test_parallel_velocities(self) -> None:
        """Test Thomas-Wigner rotation for parallel velocities."""
        v1 = np.array([0.001, 0, 0])
        v2 = np.array([0.002, 0, 0])  # Parallel to v1

        rotation = LorentzTransformation.thomas_wigner_rotation(v1, v2)

        # Should be identity (no rotation for parallel boosts)
        assert np.allclose(rotation, np.eye(4))

    def test_perpendicular_velocities(self) -> None:
        """Test Thomas-Wigner rotation for perpendicular velocities."""
        v1 = np.array([0.001, 0, 0])
        v2 = np.array([0, 0.001, 0])  # Perpendicular to v1

        rotation = LorentzTransformation.thomas_wigner_rotation(v1, v2)

        # Should have non-trivial rotation around z-axis
        assert rotation.shape == (4, 4)
        # Time components should be unchanged
        assert np.allclose(rotation[0, :], [1, 0, 0, 0])
        assert np.allclose(rotation[:, 0], [1, 0, 0, 0])

    def test_list_input_conversion(self) -> None:
        """Test that list inputs are properly converted."""
        v1 = [0.001, 0, 0]  # List input
        v2 = [0, 0.001, 0]  # List input

        # Should work without errors
        rotation = LorentzTransformation.thomas_wigner_rotation(v1, v2)
        assert rotation.shape == (4, 4)


class TestCoordinateTransformation:
    """Test coordinate transformation functionality."""

    def test_jacobian_matrix(self) -> None:
        """Test Jacobian matrix calculation."""
        metric = MinkowskiMetric()
        transformer = CoordinateTransformation(metric)

        # Define simple coordinate transformation
        t, x, y, z = sp.symbols("t x y z", real=True)
        old_coords = [t, x, y, z]
        new_coords = [t, x + t, y, z]  # Boost-like transformation

        jacobian = transformer.jacobian_matrix(old_coords, new_coords)

        # Check structure
        assert jacobian.shape == (4, 4)
        assert jacobian[0, 0] == 1  # dt'/dt = 1
        assert jacobian[1, 0] == 1  # dx'/dt = 1
        assert jacobian[1, 1] == 1  # dx'/dx = 1

    def test_standard_transformations(self) -> None:
        """Test standard coordinate transformations."""
        # Test spherical coordinates
        spherical = CoordinateTransformation.cartesian_to_spherical()
        assert "coordinates" in spherical
        assert len(spherical["coordinates"]) == 4

        # Test cylindrical coordinates
        cylindrical = CoordinateTransformation.cartesian_to_cylindrical()
        assert "coordinates" in cylindrical
        assert len(spherical["coordinates"]) == 4

        # Test Milne coordinates
        milne = CoordinateTransformation.milne_coordinates()
        assert "coordinates" in milne
        assert len(milne["coordinates"]) == 4

    def test_numerical_jacobian(self) -> None:
        """Test numerical Jacobian calculation."""
        metric = MinkowskiMetric()
        transformer = CoordinateTransformation(metric)

        # Define simple transformation function
        def transform_func(coords: np.ndarray) -> np.ndarray:
            t, x, y, z = coords
            return np.array([t, x + 0.1 * t, y, z])

        # Test point
        coords = np.array([1.0, 2.0, 3.0, 4.0])

        # Calculate numerical Jacobian
        jacobian = transformer.numerical_jacobian(transform_func, coords)

        # Check structure
        assert jacobian.shape == (4, 4)
        # Check known derivatives
        assert np.isclose(jacobian[0, 0], 1.0, atol=1e-6)  # dt'/dt = 1
        assert np.isclose(jacobian[1, 0], 0.1, atol=1e-6)  # dx'/dt = 0.1
        assert np.isclose(jacobian[1, 1], 1.0, atol=1e-6)  # dx'/dx = 1


class TestTransformationConsistency:
    """Test consistency between NumPy and SymPy implementations."""

    def test_numpy_sympy_consistency_contravariant(self) -> None:
        """Test that NumPy and SymPy give consistent results for contravariant tensors."""
        metric = MinkowskiMetric()
        transformer = LorentzTransformation(metric)

        # Create simple numeric tensor components
        components_np = np.array(
            [[1.0, 2.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )

        # Create corresponding SymPy tensor
        components_sp = sp.Matrix(components_np)

        # Create tensors
        tensor_np = TensorField(components_np, "mu nu", metric)
        tensor_sp = TensorField(components_sp, "mu nu", metric)

        # Apply same transformation
        boost = LorentzTransformation.boost_matrix([0.1, 0, 0])

        transformed_np = transformer.transform_tensor(tensor_np, boost)
        transformed_sp = transformer.transform_tensor(tensor_sp, boost)

        # Convert SymPy result to NumPy for comparison
        transformed_sp_np = np.array(transformed_sp.components).astype(float)

        # Should be approximately equal
        assert np.allclose(transformed_np.components, transformed_sp_np, rtol=1e-10)

    def test_numpy_sympy_consistency_covariant(self) -> None:
        """Test consistency for covariant tensors."""
        metric = MinkowskiMetric()
        transformer = LorentzTransformation(metric)

        components_np = np.array(
            [[1.0, 0.5, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )

        components_sp = sp.Matrix(components_np)

        tensor_np = TensorField(components_np, "_mu _nu", metric)
        tensor_sp = TensorField(components_sp, "_mu _nu", metric)

        boost = LorentzTransformation.boost_matrix([0.1, 0, 0])

        transformed_np = transformer.transform_tensor(tensor_np, boost)
        transformed_sp = transformer.transform_tensor(tensor_sp, boost)

        transformed_sp_np = np.array(transformed_sp.components).astype(float)

        assert np.allclose(transformed_np.components, transformed_sp_np, rtol=1e-10)

    def test_numpy_sympy_consistency_mixed(self) -> None:
        """Test consistency for mixed tensors."""
        metric = MinkowskiMetric()
        transformer = LorentzTransformation(metric)

        components_np = np.diag([2.0, 1.0, 1.0, 1.0])  # Simple diagonal tensor
        components_sp = sp.Matrix(components_np)

        # Test both mixed cases
        tensor_np1 = TensorField(components_np, "mu _nu", metric)
        tensor_sp1 = TensorField(components_sp, "mu _nu", metric)

        tensor_np2 = TensorField(components_np, "_mu nu", metric)
        tensor_sp2 = TensorField(components_sp, "_mu nu", metric)

        boost = LorentzTransformation.boost_matrix([0.1, 0, 0])

        # Test first mixed case
        transformed_np1 = transformer.transform_tensor(tensor_np1, boost)
        transformed_sp1 = transformer.transform_tensor(tensor_sp1, boost)
        transformed_sp_np1 = np.array(transformed_sp1.components).astype(float)
        assert np.allclose(transformed_np1.components, transformed_sp_np1, rtol=1e-10)

        # Test second mixed case
        transformed_np2 = transformer.transform_tensor(tensor_np2, boost)
        transformed_sp2 = transformer.transform_tensor(tensor_sp2, boost)
        transformed_sp_np2 = np.array(transformed_sp2.components).astype(float)
        assert np.allclose(transformed_np2.components, transformed_sp_np2, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__])
