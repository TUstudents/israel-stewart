"""
Lorentz transformations and coordinate transformations for relativistic hydrodynamics.

This module provides transformation matrices, boost operations, and general
coordinate transformations for tensor fields in curved spacetime.
"""

# Forward reference for metrics
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import numpy as np
import sympy as sp

from .four_vectors import FourVector
from .performance import monitor_performance

# Import base classes and utilities
from .tensor_base import TensorField
from .tensor_utils import PhysicsError, optimized_einsum

if TYPE_CHECKING:
    from .metrics import MetricBase


class LorentzTransformation:
    """
    Lorentz transformations for special and general relativity.

    Handles boosts, rotations, and general coordinate transformations
    for tensor fields in curved spacetime.
    """

    def __init__(self, metric: Optional["MetricBase"] = None):
        """
        Initialize Lorentz transformation handler.

        Args:
            metric: Spacetime metric (Minkowski if None)
        """
        self.metric = metric

    @staticmethod
    def boost_matrix(velocity: np.ndarray | list[float]) -> np.ndarray:
        """
        Construct Lorentz boost matrix Λ^μ_ν.

        Args:
            velocity: 3-velocity [vx, vy, vz] in units where c = 1

        Returns:
            4x4 boost matrix
        """
        if isinstance(velocity, list):
            velocity = np.array(velocity)

        if len(velocity) != 3:
            raise ValueError("Velocity must be 3-dimensional")

        v_squared = np.dot(velocity, velocity)
        if v_squared >= 1.0:
            raise PhysicsError("Velocity must be less than speed of light")

        gamma = 1.0 / np.sqrt(1.0 - v_squared)

        # Build boost matrix
        boost = np.eye(4)

        # Time components
        boost[0, 0] = gamma
        boost[0, 1:4] = -gamma * velocity
        boost[1:4, 0] = -gamma * velocity

        # Spatial components
        if v_squared > 1e-12:  # Avoid division by zero
            factor = (gamma - 1.0) / v_squared
            for i in range(3):
                for j in range(3):
                    boost[i + 1, j + 1] = (1.0 if i == j else 0.0) + factor * velocity[
                        i
                    ] * velocity[j]

        return boost

    @staticmethod
    def rotation_matrix(axis: np.ndarray | list[float], angle: float) -> np.ndarray:
        """
        Construct spatial rotation matrix.

        Args:
            axis: Rotation axis (unit vector)
            angle: Rotation angle in radians

        Returns:
            4x4 rotation matrix (identity in time)
        """
        if isinstance(axis, list):
            axis = np.array(axis)

        # Normalize axis
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            raise ValueError("Rotation axis cannot be zero")
        axis = axis / axis_norm

        # Rodrigues rotation formula
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # 3x3 rotation matrix using Rodrigues formula
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )  # Skew-symmetric matrix

        rotation_3d = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)

        # Embed in 4x4 matrix
        rotation_4d = np.eye(4)
        rotation_4d[1:4, 1:4] = rotation_3d

        return rotation_4d

    @monitor_performance("vector_transformation")
    def transform_vector(self, vector: FourVector, transformation: np.ndarray) -> FourVector:
        """
        Apply Lorentz transformation to four-vector: V'^μ = Λ^μ_ν V^ν.

        Args:
            vector: Input four-vector
            transformation: 4x4 transformation matrix Λ^μ_ν

        Returns:
            Transformed four-vector
        """
        if transformation.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4")

        # Apply transformation
        if isinstance(vector.components, np.ndarray):
            transformed_components = optimized_einsum("mn,n->m", transformation, vector.components)
        else:
            transformation_sp = sp.Matrix(transformation)
            transformed_components = transformation_sp * vector.components

        return FourVector(transformed_components, vector.indices[0][0], self.metric)

    @monitor_performance("tensor_transformation")
    def transform_tensor(self, tensor: TensorField, transformation: np.ndarray) -> TensorField:
        """
        Apply Lorentz transformation to rank-2 tensor: T'^μν = Λ^μ_α Λ^ν_β T^αβ.

        Args:
            tensor: Input rank-2 tensor
            transformation: 4x4 transformation matrix

        Returns:
            Transformed tensor
        """
        if tensor.rank != 2:
            raise NotImplementedError("Transformation only implemented for rank-2 tensors")

        if transformation.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4")

        # For contravariant indices: T'^μν = Λ^μ_α Λ^ν_β T^αβ
        # For mixed indices, need inverse transformation for covariant parts
        inv_transformation = np.linalg.inv(transformation)

        if isinstance(tensor.components, np.ndarray):
            # Determine transformation based on index types
            if not tensor.indices[0][0] and not tensor.indices[1][0]:  # Both contravariant
                transformed = optimized_einsum(
                    "ma,nb,ab->mn", transformation, transformation, tensor.components
                )
            elif tensor.indices[0][0] and tensor.indices[1][0]:  # Both covariant
                transformed = optimized_einsum(
                    "am,bn,mn->ab",
                    inv_transformation,
                    inv_transformation,
                    tensor.components,
                )
            else:  # Mixed indices
                if not tensor.indices[0][0]:  # First contravariant, second covariant
                    transformed = optimized_einsum(
                        "ma,nb,ab->mn",
                        transformation,
                        inv_transformation,
                        tensor.components,
                    )
                else:  # First covariant, second contravariant
                    transformed = optimized_einsum(
                        "am,nb,mn->ab",
                        inv_transformation,
                        transformation,
                        tensor.components,
                    )
        else:
            # SymPy version - simplified
            transformation_sp = sp.Matrix(transformation)
            if not tensor.indices[0][0] and not tensor.indices[1][0]:  # Both contravariant
                transformed = transformation_sp * tensor.components * transformation_sp.T
            else:
                # More complex transformation for mixed/covariant indices
                inv_transformation_sp = sp.Matrix(inv_transformation)
                transformed = inv_transformation_sp * tensor.components * inv_transformation_sp.T

        return TensorField(transformed, tensor._index_string(), self.metric)

    @staticmethod
    def inverse_transformation(transformation: np.ndarray) -> np.ndarray:
        """
        Compute inverse Lorentz transformation.

        Args:
            transformation: Forward transformation matrix

        Returns:
            Inverse transformation matrix
        """
        return np.linalg.inv(transformation)

    @staticmethod
    def composition(transform1: np.ndarray, transform2: np.ndarray) -> np.ndarray:
        """
        Compose two Lorentz transformations.

        Args:
            transform1: First transformation
            transform2: Second transformation

        Returns:
            Composed transformation (transform2 ∘ transform1)
        """
        return optimized_einsum("ij,jk->ik", transform2, transform1)

    def boost_to_rest_frame(self, four_velocity: FourVector) -> np.ndarray:
        """
        Construct boost to rest frame of given four-velocity.

        Args:
            four_velocity: Four-velocity to boost to rest

        Returns:
            Boost transformation matrix
        """
        # Extract 3-velocity from four-velocity
        # For normalized four-velocity u^μ = γ(1, v⃗), we have u^0 = γ, u^i = γv^i
        gamma = abs(four_velocity.time_component)
        if gamma < 1e-12:
            raise PhysicsError("Cannot extract velocity from null four-velocity")

        # Three-velocity: v^i = u^i / u^0 for timelike vectors
        if not four_velocity.is_timelike():
            raise PhysicsError("Can only extract three-velocity from timelike four-velocity")

        three_velocity = four_velocity.spatial_components / gamma

        # Return inverse boost (to go to rest frame)
        boost = self.boost_matrix(-three_velocity)  # Negative to go to rest frame
        return boost

    def boost_from_rest_frame(self, four_velocity: FourVector) -> np.ndarray:
        """
        Construct boost from rest frame to lab frame.

        Args:
            four_velocity: Four-velocity of final frame

        Returns:
            Boost transformation matrix
        """
        # Extract 3-velocity from four-velocity
        gamma = abs(four_velocity.time_component)
        if gamma < 1e-12:
            raise PhysicsError("Cannot extract velocity from null four-velocity")

        if not four_velocity.is_timelike():
            raise PhysicsError("Can only extract three-velocity from timelike four-velocity")

        three_velocity = four_velocity.spatial_components / gamma

        # Return forward boost (from rest frame to lab frame)
        return self.boost_matrix(three_velocity)

    @staticmethod
    def thomas_wigner_rotation(
        velocity1: np.ndarray | list[float], velocity2: np.ndarray | list[float]
    ) -> np.ndarray:
        """
        Compute Thomas-Wigner rotation for successive boosts.

        When two boosts are applied successively in different directions,
        the result includes a spatial rotation (Thomas-Wigner rotation).

        Args:
            velocity1: First boost velocity
            velocity2: Second boost velocity

        Returns:
            4x4 rotation matrix for Thomas-Wigner rotation
        """
        # Convert to numpy arrays if needed (type narrowing)
        if not isinstance(velocity1, np.ndarray):
            velocity1 = np.array(velocity1)
        if not isinstance(velocity2, np.ndarray):
            velocity2 = np.array(velocity2)

        # Simplified Thomas-Wigner rotation calculation
        # Full calculation involves relativistic velocity addition

        # For small velocities, the rotation angle is approximately
        # θ ≈ (v₁ × v₂) / (2c²) in the small velocity limit

        v1_squared = np.dot(velocity1, velocity1)
        v2_squared = np.dot(velocity2, velocity2)

        if v1_squared < 1e-6 and v2_squared < 1e-6:
            # Small velocity approximation
            cross_product = np.cross(velocity1, velocity2)
            angle = np.linalg.norm(cross_product) / 2.0

            if angle > 1e-12:
                axis = cross_product / np.linalg.norm(cross_product)
                return LorentzTransformation.rotation_matrix(axis, float(angle))

        # For larger velocities, return identity (full calculation is complex)
        return np.eye(4)

    def validate_transformation(self, transformation: np.ndarray) -> bool:
        """
        Validate that transformation preserves metric signature.

        For Lorentz transformations: Λ^T g Λ = g

        Args:
            transformation: Transformation matrix to validate

        Returns:
            True if transformation preserves metric
        """
        if transformation.shape != (4, 4):
            return False

        if self.metric is None:
            # Use Minkowski metric for validation
            minkowski = np.diag([-1, 1, 1, 1])
            preserved_metric = optimized_einsum(
                "ij,jk,kl->il", transformation.T, minkowski, transformation
            )
            return np.allclose(preserved_metric, minkowski, rtol=1e-10)
        else:
            # Use provided metric
            metric_components = self.metric.components
            preserved_metric = optimized_einsum(
                "ij,jk,kl->il", transformation.T, metric_components, transformation
            )
            return np.allclose(preserved_metric, metric_components, rtol=1e-10)

    def active_vs_passive_transformation(
        self, tensor: TensorField, transformation: np.ndarray, active: bool = True
    ) -> TensorField:
        """
        Apply active or passive transformation to tensor.

        Active transformation: changes the tensor components
        Passive transformation: changes the coordinate system

        Args:
            tensor: Input tensor
            transformation: Transformation matrix
            active: True for active, False for passive transformation

        Returns:
            Transformed tensor
        """
        if active:
            # Active transformation: T'^μν = Λ^μ_α Λ^ν_β T^αβ
            return self.transform_tensor(tensor, transformation)  # type: ignore[no-any-return]
        else:
            # Passive transformation: T'^μν = (Λ^-1)^μ_α (Λ^-1)^ν_β T^αβ
            inverse_transform = self.inverse_transformation(transformation)
            return self.transform_tensor(tensor, inverse_transform)  # type: ignore[no-any-return]


class CoordinateTransformation:
    """
    General coordinate transformations for curved spacetime.

    Handles transformations between different coordinate systems
    and provides Jacobian matrices for tensor transformations.
    """

    def __init__(self, metric: "MetricBase"):
        """
        Initialize coordinate transformation handler.

        Args:
            metric: Spacetime metric tensor
        """
        self.metric = metric

    def jacobian_matrix(
        self, old_coordinates: list[sp.Symbol], new_coordinates: list[sp.Expr]
    ) -> sp.Matrix:
        """
        Compute Jacobian matrix ∂x'^μ/∂x^ν for coordinate transformation.

        Args:
            old_coordinates: Original coordinate symbols [t, x, y, z]
            new_coordinates: New coordinate expressions in terms of old

        Returns:
            4x4 Jacobian matrix
        """
        if len(old_coordinates) != 4 or len(new_coordinates) != 4:
            raise ValueError("Coordinate transformations must be 4-dimensional")

        jacobian = sp.zeros(4, 4)

        for mu in range(4):
            for nu in range(4):
                jacobian[mu, nu] = sp.diff(new_coordinates[mu], old_coordinates[nu])

        return jacobian

    def inverse_jacobian(self, jacobian: sp.Matrix) -> sp.Matrix:
        """
        Compute inverse Jacobian matrix.

        Args:
            jacobian: Forward Jacobian matrix

        Returns:
            Inverse Jacobian matrix
        """
        return jacobian.inv()

    def transform_metric(self, jacobian: sp.Matrix, old_metric: sp.Matrix) -> sp.Matrix:
        """
        Transform metric tensor: g'_μν = (∂x^α/∂x'^μ)(∂x^β/∂x'^ν) g_αβ.

        Args:
            jacobian: Jacobian matrix ∂x'^μ/∂x^ν
            old_metric: Original metric tensor components

        Returns:
            Transformed metric tensor
        """
        inverse_jacobian = self.inverse_jacobian(jacobian)

        # Transform metric: g'_μν = (J^-1)^α_μ (J^-1)^β_ν g_αβ
        transformed_metric = inverse_jacobian.T * old_metric * inverse_jacobian

        return transformed_metric

    @staticmethod
    def cartesian_to_spherical() -> dict[str, sp.Expr]:
        """
        Standard Cartesian to spherical coordinate transformation.

        Returns:
            Dictionary with coordinate transformation expressions
        """
        # Define coordinate symbols
        t, x, y, z = sp.symbols("t x y z", real=True)

        # Spherical coordinates
        r = sp.sqrt(x**2 + y**2 + z**2)
        theta = sp.acos(z / r)  # Polar angle
        phi = sp.atan2(y, x)  # Azimuthal angle

        return {
            "t_new": t,
            "r": r,
            "theta": theta,
            "phi": phi,
            "coordinates": [t, r, theta, phi],
            "old_coordinates": [t, x, y, z],
        }

    @staticmethod
    def cartesian_to_cylindrical() -> dict[str, sp.Expr]:
        """
        Standard Cartesian to cylindrical coordinate transformation.

        Returns:
            Dictionary with coordinate transformation expressions
        """
        # Define coordinate symbols
        t, x, y, z = sp.symbols("t x y z", real=True)

        # Cylindrical coordinates
        rho = sp.sqrt(x**2 + y**2)  # Radial distance in xy-plane
        phi = sp.atan2(y, x)  # Azimuthal angle
        z_cyl = z  # Height (unchanged)

        return {
            "t_new": t,
            "rho": rho,
            "phi": phi,
            "z": z_cyl,
            "coordinates": [t, rho, phi, z_cyl],
            "old_coordinates": [t, x, y, z],
        }

    @staticmethod
    def milne_coordinates() -> dict[str, sp.Expr]:
        """
        Minkowski to Milne (Rindler-like) coordinate transformation.

        Useful for relativistic hydrodynamics simulations.

        Returns:
            Dictionary with coordinate transformation expressions
        """
        # Define coordinate symbols
        t, x, y, z = sp.symbols("t x y z", real=True)

        # Milne coordinates: τ = √(t² - x²), η = (1/2)ln((t+x)/(t-x))
        tau = sp.sqrt(t**2 - x**2)  # Proper time
        eta = sp.Rational(1, 2) * sp.log((t + x) / (t - x))  # Rapidity
        y_milne = y  # Transverse coordinates unchanged
        z_milne = z

        return {
            "tau": tau,
            "eta": eta,
            "y": y_milne,
            "z": z_milne,
            "coordinates": [tau, eta, y_milne, z_milne],
            "old_coordinates": [t, x, y, z],
        }

    def numerical_jacobian(
        self,
        transformation_func: Callable[[np.ndarray], np.ndarray],
        coordinates: np.ndarray,
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """
        Compute numerical Jacobian matrix for coordinate transformation.

        Args:
            transformation_func: Function that transforms coordinates
            coordinates: Point at which to evaluate Jacobian
            epsilon: Step size for numerical derivatives

        Returns:
            4x4 numerical Jacobian matrix
        """
        jacobian = np.zeros((4, 4))

        for nu in range(4):
            # Forward difference approximation
            coords_plus = coordinates.copy()
            coords_plus[nu] += epsilon

            coords_minus = coordinates.copy()
            coords_minus[nu] -= epsilon

            # Compute derivatives
            transform_plus: np.ndarray = transformation_func(coords_plus)
            transform_minus: np.ndarray = transformation_func(coords_minus)

            jacobian[:, nu] = (transform_plus - transform_minus) / (2 * epsilon)

        return jacobian
