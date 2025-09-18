"""
Tests for FourVector class and Lorentz transformations.

This module tests the relativistic four-vector implementation,
focusing on proper handling of covariant vs contravariant transformations.
"""

import numpy as np
import pytest

from israel_stewart.core.four_vectors import FourVector
from israel_stewart.core.metrics import MinkowskiMetric


class TestFourVectorBoost:
    """Test Lorentz boost transformations for four-vectors."""

    @pytest.fixture
    def minkowski_metric(self) -> MinkowskiMetric:
        """Create Minkowski metric for testing."""
        return MinkowskiMetric()

    def test_contravariant_boost_identity(self, minkowski_metric: MinkowskiMetric) -> None:
        """Test that zero boost leaves contravariant vector unchanged."""
        # Create contravariant four-vector
        components = np.array([1.0, 0.5, 0.3, 0.1])
        vector = FourVector(components, is_covariant=False, metric=minkowski_metric)

        # Apply zero boost
        zero_velocity = np.array([0.0, 0.0, 0.0])
        boosted = vector.boost(zero_velocity)

        # Should be unchanged
        np.testing.assert_allclose(boosted.components, components, rtol=1e-14)
        assert not boosted.indices[0][0]  # Still contravariant

    def test_covariant_boost_identity(self, minkowski_metric: MinkowskiMetric) -> None:
        """Test that zero boost leaves covariant vector unchanged."""
        # Create covariant four-vector
        components = np.array([1.0, 0.5, 0.3, 0.1])
        vector = FourVector(components, is_covariant=True, metric=minkowski_metric)

        # Apply zero boost
        zero_velocity = np.array([0.0, 0.0, 0.0])
        boosted = vector.boost(zero_velocity)

        # Should be unchanged
        np.testing.assert_allclose(boosted.components, components, rtol=1e-14)
        assert boosted.indices[0][0]  # Still covariant

    def test_contravariant_x_boost(self, minkowski_metric: MinkowskiMetric) -> None:
        """Test contravariant vector boost in x-direction."""
        # Create contravariant four-vector at rest
        components = np.array([1.0, 0.0, 0.0, 0.0])
        vector = FourVector(components, is_covariant=False, metric=minkowski_metric)

        # Boost in x-direction with v = 0.6c
        velocity = np.array([0.6, 0.0, 0.0])
        gamma = 1.0 / np.sqrt(1.0 - 0.6**2)  # γ = 1.25
        boosted = vector.boost(velocity)

        # Expected result: u'^0 = γ(u^0 - v·u) = γ(1 - 0.6*0) = γ
        #                  u'^1 = γ(u^1 - v*u^0) = γ(0 - 0.6*1) = -γv
        expected = np.array([gamma, -gamma * 0.6, 0.0, 0.0])
        np.testing.assert_allclose(boosted.components, expected, rtol=1e-14)

    def test_covariant_x_boost(self, minkowski_metric: MinkowskiMetric) -> None:
        """Test covariant vector boost in x-direction."""
        # Create covariant four-vector
        components = np.array([1.0, 0.0, 0.0, 0.0])
        vector = FourVector(components, is_covariant=True, metric=minkowski_metric)

        # Boost in x-direction with v = 0.6c
        velocity = np.array([0.6, 0.0, 0.0])
        gamma = 1.0 / np.sqrt(1.0 - 0.6**2)  # γ = 1.25
        boosted = vector.boost(velocity)

        # For covariant vectors, we use the inverse of the boost matrix
        # u'_μ = (Λ^-1)_μ^ν u_ν where Λ^-1 is the inverse boost matrix
        # For u_μ = [1, 0, 0, 0] and boost in x-direction with v = 0.6:
        # The inverse boost matrix gives: u'_μ = [γ, γv, 0, 0]
        expected = np.array([gamma, gamma * 0.6, 0.0, 0.0])
        np.testing.assert_allclose(boosted.components, expected, rtol=1e-14)

    def test_boost_consistency_dot_product(self, minkowski_metric: MinkowskiMetric) -> None:
        """Test that dot product is preserved under Lorentz boosts."""
        # Create contravariant and covariant versions of the same physical vector
        contravariant_components = np.array([2.0, 1.0, 0.5, 0.2])
        covariant_components = minkowski_metric.components @ contravariant_components

        vector_contra = FourVector(
            contravariant_components, is_covariant=False, metric=minkowski_metric
        )
        vector_cov = FourVector(covariant_components, is_covariant=True, metric=minkowski_metric)

        # Boost both vectors
        velocity = np.array([0.3, 0.2, 0.1])
        boosted_contra = vector_contra.boost(velocity)
        boosted_cov = vector_cov.boost(velocity)

        # Dot product should be preserved
        original_dot = vector_contra.dot(vector_cov)
        boosted_dot = boosted_contra.dot(boosted_cov)

        np.testing.assert_allclose(boosted_dot, original_dot, rtol=1e-12)

    def test_boost_inverse_symmetry(self, minkowski_metric: MinkowskiMetric) -> None:
        """Test that boost followed by inverse boost returns to original."""
        # Create contravariant vector
        components = np.array([3.0, 1.5, 0.8, 0.4])
        vector = FourVector(components, is_covariant=False, metric=minkowski_metric)

        # Apply boost
        velocity = np.array([0.4, 0.3, 0.2])
        boosted = vector.boost(velocity)

        # Apply inverse boost
        inverse_velocity = -velocity
        recovered = boosted.boost(inverse_velocity)

        # Should recover original vector
        np.testing.assert_allclose(recovered.components, components, rtol=1e-12)

    def test_covariant_boost_inverse_symmetry(self, minkowski_metric: MinkowskiMetric) -> None:
        """Test boost inverse symmetry for covariant vectors."""
        # Create covariant vector
        components = np.array([3.0, 1.5, 0.8, 0.4])
        vector = FourVector(components, is_covariant=True, metric=minkowski_metric)

        # Apply boost
        velocity = np.array([0.4, 0.3, 0.2])
        boosted = vector.boost(velocity)

        # Apply inverse boost
        inverse_velocity = -velocity
        recovered = boosted.boost(inverse_velocity)

        # Should recover original vector
        np.testing.assert_allclose(recovered.components, components, rtol=1e-12)

    def test_covariant_transformation_formula(self, minkowski_metric: MinkowskiMetric) -> None:
        """Test that covariant vectors transform with the correct mathematical formula.

        This test specifically verifies the fix for the bug where covariant vectors
        were incorrectly transformed using left multiplication instead of right multiplication.
        """
        # Create covariant four-vector
        components = np.array([1.0, 2.0, 0.0, 0.0])
        vector = FourVector(components, is_covariant=True, metric=minkowski_metric)

        # Apply boost in x-direction
        velocity = np.array([0.5, 0.0, 0.0])
        gamma = 1.0 / np.sqrt(1.0 - 0.5**2)  # γ = 2/√3 ≈ 1.1547

        # Get the boost transformation
        boosted = vector.boost(velocity)

        # Manually compute expected result using correct covariant formula
        # u'_μ = u_ν (Λ^{-1})^ν_μ
        boost_matrix = vector._lorentz_boost_matrix(velocity, gamma)
        inv_boost_matrix = np.linalg.inv(boost_matrix)

        # For covariant vectors, we use right multiplication: u_μ @ (Λ^{-1})
        expected = components @ inv_boost_matrix

        np.testing.assert_allclose(boosted.components, expected, rtol=1e-14)

    def test_boost_composition(self, minkowski_metric: MinkowskiMetric) -> None:
        """Test that successive boosts work correctly."""
        # Create vector
        components = np.array([2.0, 0.0, 0.0, 0.0])
        vector = FourVector(components, is_covariant=False, metric=minkowski_metric)

        # Apply two successive boosts in x-direction
        v1 = np.array([0.3, 0.0, 0.0])
        v2 = np.array([0.2, 0.0, 0.0])

        # First boost
        boosted1 = vector.boost(v1)
        # Second boost
        boosted2 = boosted1.boost(v2)

        # Result should be physically consistent
        assert np.all(np.isfinite(boosted2.components))

        # Time component should increase due to cumulative boost
        assert boosted2.components[0] > boosted1.components[0]

    def test_boost_physical_limits(self, minkowski_metric: MinkowskiMetric) -> None:
        """Test boost behavior near speed of light."""
        components = np.array([1.0, 0.0, 0.0, 0.0])
        vector = FourVector(components, is_covariant=False, metric=minkowski_metric)

        # Test high velocity boost (but still subluminal)
        high_velocity = np.array([0.99, 0.0, 0.0])
        boosted = vector.boost(high_velocity)

        # Should produce finite, large Lorentz factor
        gamma_expected = 1.0 / np.sqrt(1.0 - 0.99**2)
        assert np.isfinite(boosted.components[0])
        assert boosted.components[0] > 7.0  # γ ≈ 7.09 for v = 0.99c

        # Test that superluminal velocity raises error
        superluminal_velocity = np.array([1.1, 0.0, 0.0])
        with pytest.raises(ValueError, match="less than speed of light"):
            vector.boost(superluminal_velocity)

    def test_three_dimensional_boost(self, minkowski_metric: MinkowskiMetric) -> None:
        """Test boost in arbitrary 3D direction."""
        components = np.array([2.0, 1.0, 0.5, 0.2])
        vector_contra = FourVector(components, is_covariant=False, metric=minkowski_metric)
        vector_cov = FourVector(components, is_covariant=True, metric=minkowski_metric)

        # Boost in arbitrary direction
        velocity = np.array([0.2, 0.3, 0.1])
        boosted_contra = vector_contra.boost(velocity)
        boosted_cov = vector_cov.boost(velocity)

        # Results should be different for covariant vs contravariant
        assert not np.allclose(boosted_contra.components, boosted_cov.components)

        # But both should be finite and physically reasonable
        assert np.all(np.isfinite(boosted_contra.components))
        assert np.all(np.isfinite(boosted_cov.components))

    def test_boost_matrix_properties(self, minkowski_metric: MinkowskiMetric) -> None:
        """Test mathematical properties of the boost matrix."""
        components = np.array([1.0, 0.0, 0.0, 0.0])
        vector = FourVector(components, is_covariant=False, metric=minkowski_metric)

        velocity = np.array([0.5, 0.0, 0.0])
        gamma = 1.0 / np.sqrt(1.0 - 0.5**2)

        # Access the boost matrix
        boost_matrix = vector._lorentz_boost_matrix(velocity, gamma)

        # Boost matrix should be 4x4
        assert boost_matrix.shape == (4, 4)

        # Determinant should be 1 (proper orthochronous Lorentz transformation)
        # Note: Spatial rotations have det = 1, time reversal has det = -1
        det = np.linalg.det(boost_matrix)
        np.testing.assert_allclose(det, 1.0, rtol=1e-10)

        # For pure boosts, the matrix should be symmetric
        # (This is specific to boost transformations, not general Lorentz transformations)
        np.testing.assert_allclose(boost_matrix, boost_matrix.T, rtol=1e-14)

        # Verify orthogonality: Λ^T η Λ = η where η is Minkowski metric
        eta = minkowski_metric.components
        product = boost_matrix.T @ eta @ boost_matrix
        np.testing.assert_allclose(product, eta, rtol=1e-10, atol=1e-15)


class TestFourVectorDotProduct:
    """Test dot product functionality with proper index handling."""

    @pytest.fixture
    def minkowski_metric(self) -> MinkowskiMetric:
        """Create Minkowski metric for testing."""
        return MinkowskiMetric()

    def test_dot_product_mixed_indices(self, minkowski_metric: MinkowskiMetric) -> None:
        """Test dot product between contravariant and covariant vectors."""
        # Create contravariant vector
        contra_components = np.array([2.0, 1.0, 0.5, 0.2])
        vector_contra = FourVector(contra_components, is_covariant=False, metric=minkowski_metric)

        # Create corresponding covariant vector
        cov_components = minkowski_metric.components @ contra_components
        vector_cov = FourVector(cov_components, is_covariant=True, metric=minkowski_metric)

        # Dot product should work
        dot_result = vector_contra.dot(vector_cov)

        # Manually compute expected result: u^μ u_μ = η_μν u^μ u^ν
        expected = contra_components @ minkowski_metric.components @ contra_components
        np.testing.assert_allclose(dot_result, expected, rtol=1e-14)

    def test_dot_product_same_indices_auto_conversion(
        self, minkowski_metric: MinkowskiMetric
    ) -> None:
        """Test that dot product automatically converts indices when needed."""
        # Create two contravariant vectors
        components1 = np.array([2.0, 1.0, 0.5, 0.2])
        components2 = np.array([1.5, 0.8, 0.3, 0.1])

        vector1 = FourVector(components1, is_covariant=False, metric=minkowski_metric)
        vector2 = FourVector(components2, is_covariant=False, metric=minkowski_metric)

        # Dot product should automatically handle index conversion
        dot_result = vector1.dot(vector2)

        # Expected: convert one to covariant and compute dot product
        expected = components1 @ minkowski_metric.components @ components2
        np.testing.assert_allclose(dot_result, expected, rtol=1e-14)
