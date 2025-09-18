"""
Four-vector operations for relativistic hydrodynamics.

This module provides specialized four-vector classes with physics-specific
operations like Lorentz boosts, dot products, and relativistic transformations.
"""

# Forward reference for metrics
from typing import TYPE_CHECKING, Optional

import numpy as np
import sympy as sp

from .performance import monitor_performance

# Import base tensor and utilities
from .tensor_base import TensorField
from .tensor_utils import (
    PhysicsError,
    validate_tensor_dimensions,
)

if TYPE_CHECKING:
    from .metrics import MetricBase


class FourVector(TensorField):
    """
    Specialized class for four-vectors in relativity.

    Provides physics-specific operations like Lorentz boosts,
    dot products, and time/space component access.
    """

    def __init__(
        self,
        components: np.ndarray | sp.Matrix | list[float],
        is_covariant: bool = False,
        metric: Optional["MetricBase"] = None,
    ):
        """
        Initialize four-vector.

        Args:
            components: Four components [t, x, y, z] or [0, 1, 2, 3]
            is_covariant: Whether components are covariant (default: contravariant)
            metric: Metric tensor for operations
        """
        if isinstance(components, list):
            components = np.array(components)

        # Validate four-vector components
        validate_tensor_dimensions(components, expected_shape=(4,))

        # Set up index notation
        index_str = "_mu" if is_covariant else "mu"
        super().__init__(components, index_str, metric)

    @property
    def time_component(self) -> float | sp.Expr:
        """Time component (index 0)."""
        return self.components[0]

    @property
    def spatial_components(self) -> np.ndarray | sp.Matrix:
        """Spatial components (indices 1, 2, 3)."""
        return self.components[1:4]

    @property
    def x(self) -> float | sp.Expr:
        """X-component (index 1)."""
        return self.components[1]

    @property
    def y(self) -> float | sp.Expr:
        """Y-component (index 2)."""
        return self.components[2]

    @property
    def z(self) -> float | sp.Expr:
        """Z-component (index 3)."""
        return self.components[3]

    @monitor_performance("four_vector_dot")
    def dot(self, other: "FourVector") -> float | sp.Expr:
        """
        Lorentz-invariant dot product.

        Args:
            other: Another four-vector

        Returns:
            Scalar dot product g_μν u^μ v^ν
        """
        if self.metric is None:
            raise ValueError("Cannot compute dot product without metric tensor")

        # Ensure one vector is covariant, one contravariant
        if self.indices[0][0] == other.indices[0][0]:  # Same type
            other_converted = (
                other.lower_index(0) if not self.indices[0][0] else other.raise_index(0)
            )
        else:
            other_converted = other

        # Compute dot product
        if isinstance(self.components, np.ndarray):
            return np.dot(self.components, other_converted.components)
        else:
            return self.components.dot(other_converted.components)

    def magnitude_squared(self) -> float | sp.Expr:
        """
        Compute magnitude squared |u|² = g_μν u^μ u^ν.

        Returns:
            Scalar magnitude squared
        """
        return self.dot(self)

    def magnitude(self) -> float | sp.Expr:
        """
        Compute magnitude |u| = √(g_μν u^μ u^ν).

        Returns:
            Scalar magnitude
        """
        mag_sq = self.magnitude_squared()
        if isinstance(mag_sq, np.ndarray):
            return np.sqrt(np.abs(mag_sq))
        else:
            return sp.sqrt(sp.Abs(mag_sq))

    @monitor_performance("four_vector_normalize")
    def normalize(self) -> "FourVector":
        """
        Normalize four-vector to unit magnitude.

        For null vectors (magnitude = 0), returns the original vector unchanged
        since normalization is undefined.

        Returns:
            Normalized four-vector

        Raises:
            PhysicsError: If vector magnitude is zero (null vector)
        """
        mag = self.magnitude()

        # Handle null vectors (magnitude ≈ 0)
        tolerance = 1e-15
        if isinstance(mag, np.ndarray):
            is_null = bool(np.abs(mag) < tolerance)
        else:
            is_null = abs(float(mag)) < tolerance

        if is_null:
            raise PhysicsError(
                "Cannot normalize null vector (magnitude = 0). "
                "Use is_null() to check for null vectors before normalizing."
            )

        # Safe normalization for non-null vectors
        if isinstance(self.components, np.ndarray):
            normalized_components = self.components / mag
        else:
            normalized_components = self.components / mag

        return FourVector(normalized_components, self.indices[0][0], self.metric)

    def is_timelike(self, tolerance: float = 1e-10) -> bool:
        """
        Check if four-vector is timelike.

        For mostly-plus signature (-,+,+,+): timelike vectors have g_μν u^μ u^ν < 0

        Args:
            tolerance: Numerical tolerance for comparison

        Returns:
            True if timelike
        """
        if self.metric is None:
            raise ValueError("Cannot determine timelike nature without metric")

        mag_sq = self.magnitude_squared()

        # Check metric signature to determine timelike condition
        signature = getattr(self.metric, "signature", (-1, 1, 1, 1))
        if signature[0] < 0:  # Mostly-plus signature
            timelike_condition = mag_sq < -tolerance
        else:  # Mostly-minus signature
            timelike_condition = mag_sq > tolerance

        if isinstance(mag_sq, np.ndarray):
            return bool(timelike_condition)
        else:
            return bool(
                float(mag_sq) < -tolerance if signature[0] < 0 else float(mag_sq) > tolerance
            )

    def is_spacelike(self, tolerance: float = 1e-10) -> bool:
        """
        Check if four-vector is spacelike.

        For mostly-plus signature (-,+,+,+): spacelike vectors have g_μν u^μ u^ν > 0

        Args:
            tolerance: Numerical tolerance for comparison

        Returns:
            True if spacelike
        """
        if self.metric is None:
            raise ValueError("Cannot determine spacelike nature without metric")

        mag_sq = self.magnitude_squared()

        # Check metric signature to determine spacelike condition
        signature = getattr(self.metric, "signature", (-1, 1, 1, 1))
        if signature[0] < 0:  # Mostly-plus signature
            spacelike_condition = mag_sq > tolerance
        else:  # Mostly-minus signature
            spacelike_condition = mag_sq < -tolerance

        if isinstance(mag_sq, np.ndarray):
            return bool(spacelike_condition)
        else:
            return bool(
                float(mag_sq) > tolerance if signature[0] < 0 else float(mag_sq) < -tolerance
            )

    def is_null(self, tolerance: float = 1e-10) -> bool:
        """
        Check if four-vector is null (lightlike).

        Args:
            tolerance: Numerical tolerance for comparison

        Returns:
            True if null (magnitude squared ≈ 0)
        """
        mag_sq = self.magnitude_squared()
        if isinstance(mag_sq, np.ndarray):
            return bool(np.abs(mag_sq) < tolerance)
        else:
            return abs(float(mag_sq)) < tolerance

    @monitor_performance("lorentz_boost")
    def boost(self, velocity: np.ndarray | list[float]) -> "FourVector":
        """
        Apply Lorentz boost transformation.

        Args:
            velocity: 3-velocity vector [vx, vy, vz]

        Returns:
            Boosted four-vector
        """
        if isinstance(velocity, list):
            velocity = np.array(velocity)

        if len(velocity) != 3:
            raise ValueError("Velocity must be 3-dimensional")

        # Compute boost parameters
        v_squared = np.dot(velocity, velocity)
        if v_squared >= 1.0:
            raise ValueError("Velocity must be less than speed of light (in units where c=1)")

        gamma = 1.0 / np.sqrt(1.0 - v_squared)

        # Build boost matrix
        boost_matrix = self._lorentz_boost_matrix(velocity, gamma)

        # Check if vector is covariant or contravariant
        is_covariant = self.indices[0][0]  # True for covariant, False for contravariant

        # Apply correct transformation based on vector type
        if is_covariant:
            # Covariant vectors transform with inverse-transpose: u'_μ = (Λ^-1)_μ^ν u_ν = Λ_ν^μ u_ν
            # This preserves the duality relationship: if u_μ = η_μν u^ν, then u'_μ = η_μν u'^ν
            transform_matrix = np.linalg.inv(boost_matrix)
        else:
            # Contravariant vectors transform with standard matrix: u'^μ = Λ^μ_ν u^ν
            transform_matrix = boost_matrix

        # Apply transformation
        if isinstance(self.components, np.ndarray):
            boosted_components = np.dot(transform_matrix, self.components)
        else:
            transform_matrix_sp = sp.Matrix(transform_matrix)
            boosted_components = transform_matrix_sp * self.components

        return FourVector(boosted_components, self.indices[0][0], self.metric)

    def _lorentz_boost_matrix(self, velocity: np.ndarray, gamma: float) -> np.ndarray:
        """
        Construct Lorentz boost matrix.

        Args:
            velocity: 3-velocity vector
            gamma: Lorentz factor

        Returns:
            4x4 boost matrix
        """
        vx, vy, vz = velocity
        v_squared = np.dot(velocity, velocity)

        # Boost matrix components
        matrix = np.eye(4)

        # Time-time component
        matrix[0, 0] = gamma

        # Time-space components
        matrix[0, 1] = -gamma * vx
        matrix[0, 2] = -gamma * vy
        matrix[0, 3] = -gamma * vz

        # Space-time components
        matrix[1, 0] = -gamma * vx
        matrix[2, 0] = -gamma * vy
        matrix[3, 0] = -gamma * vz

        # Space-space components
        if v_squared > 1e-10:  # Avoid division by zero
            factor = (gamma - 1.0) / v_squared

            matrix[1, 1] = 1.0 + factor * vx * vx
            matrix[1, 2] = factor * vx * vy
            matrix[1, 3] = factor * vx * vz

            matrix[2, 1] = factor * vy * vx
            matrix[2, 2] = 1.0 + factor * vy * vy
            matrix[2, 3] = factor * vy * vz

            matrix[3, 1] = factor * vz * vx
            matrix[3, 2] = factor * vz * vy
            matrix[3, 3] = 1.0 + factor * vz * vz

        return matrix

    @staticmethod
    def from_three_velocity(
        three_velocity: np.ndarray | list[float], metric: Optional["MetricBase"] = None
    ) -> "FourVector":
        """
        Construct four-velocity from three-velocity.

        Args:
            three_velocity: 3-velocity vector [vx, vy, vz]
            metric: Spacetime metric

        Returns:
            Four-velocity u^μ = γ(1, v)
        """
        if isinstance(three_velocity, list):
            three_velocity = np.array(three_velocity)

        v_squared = np.dot(three_velocity, three_velocity)
        if v_squared >= 1.0:
            raise PhysicsError("Three-velocity must be less than speed of light")

        gamma = 1.0 / np.sqrt(1.0 - v_squared)

        # Four-velocity components: u^μ = γ(1, v⃗)
        four_velocity_components = np.zeros(4)
        four_velocity_components[0] = gamma
        four_velocity_components[1:4] = gamma * three_velocity

        return FourVector(four_velocity_components, False, metric)

    @staticmethod
    def zero_vector(metric: Optional["MetricBase"] = None) -> "FourVector":
        """Create zero four-vector."""
        return FourVector(np.zeros(4), False, metric)

    @staticmethod
    def time_vector(metric: Optional["MetricBase"] = None) -> "FourVector":
        """Create unit time vector (1, 0, 0, 0)."""
        components = np.array([1.0, 0.0, 0.0, 0.0])
        return FourVector(components, False, metric)

    @staticmethod
    def spatial_vector(
        spatial_components: np.ndarray | list[float],
        metric: Optional["MetricBase"] = None,
    ) -> "FourVector":
        """
        Create spatial four-vector (0, x, y, z).

        Args:
            spatial_components: Spatial components [x, y, z]
            metric: Spacetime metric

        Returns:
            Spatial four-vector
        """
        if isinstance(spatial_components, list):
            spatial_components = np.array(spatial_components)

        if len(spatial_components) != 3:
            raise ValueError("Spatial components must be 3-dimensional")

        components = np.zeros(4)
        components[1:4] = spatial_components
        return FourVector(components, False, metric)

    def extract_three_velocity(self) -> np.ndarray | sp.Matrix:
        """
        Extract three-velocity from four-velocity.

        For four-velocity u^μ = γ(1, v⃗), returns v⃗ = u⃗/u^0

        Returns:
            Three-velocity array [vx, vy, vz]
        """
        if abs(self.time_component) < 1e-15:
            raise PhysicsError("Cannot extract three-velocity from spatial four-vector")

        return self.spatial_components / self.time_component

    def lorentz_factor(self) -> float | sp.Expr:
        """
        Compute Lorentz factor γ from four-velocity.

        For normalized four-velocity, γ = |u^0|

        Returns:
            Lorentz factor
        """
        return abs(self.time_component)

    def rapidity(self) -> float | sp.Expr:
        """
        Compute rapidity φ from four-velocity.

        Rapidity is defined by γ = cosh(φ), v = tanh(φ)

        Returns:
            Rapidity φ
        """
        gamma = self.lorentz_factor()
        if isinstance(gamma, np.ndarray):
            return np.arccosh(gamma)
        else:
            return sp.acosh(gamma)

    def relativistic_energy(self, rest_mass: float | sp.Expr = 1.0) -> float | sp.Expr:
        """
        Compute relativistic energy E = γmc².

        Args:
            rest_mass: Rest mass (default: 1 in natural units)

        Returns:
            Relativistic energy
        """
        gamma = self.lorentz_factor()
        return gamma * rest_mass  # c=1 in natural units

    def relativistic_momentum_magnitude(self, rest_mass: float | sp.Expr = 1.0) -> float | sp.Expr:
        """
        Compute magnitude of relativistic momentum |p⃗| = γm|v⃗|.

        Args:
            rest_mass: Rest mass (default: 1 in natural units)

        Returns:
            Momentum magnitude
        """
        three_vel = self.extract_three_velocity()
        gamma = self.lorentz_factor()

        if isinstance(three_vel, np.ndarray):
            v_mag = np.sqrt(np.dot(three_vel, three_vel))
        elif isinstance(three_vel, sp.Matrix):
            v_mag = sp.sqrt(sum(three_vel[i] ** 2 for i in range(3)))
        else:
            raise TypeError(f"Unsupported type for three_velocity: {type(three_vel)}")

        return gamma * rest_mass * v_mag
