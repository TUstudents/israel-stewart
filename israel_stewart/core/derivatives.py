"""
Covariant derivatives and projection operators for relativistic hydrodynamics.

This module provides covariant differentiation in curved spacetime and
projection operators for fluid decomposition in Israel-Stewart formalism.
"""

import warnings
from functools import cached_property

# Forward reference for metrics
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp

from .four_vectors import FourVector
from .performance import monitor_performance

# Import base classes and utilities
from .tensor_base import TensorField
from .tensor_utils import optimized_einsum

if TYPE_CHECKING:
    from .metrics import MetricBase


class CovariantDerivative:
    """
    Covariant derivative operator ∇_μ for tensor fields in curved spacetime.

    Handles derivatives of scalars, vectors, and higher-rank tensors using
    Christoffel symbols from the metric tensor.
    """

    def __init__(self, metric: "MetricBase"):
        """
        Initialize covariant derivative operator.

        Args:
            metric: Metric tensor providing Christoffel symbols
        """
        self.metric = metric

    @cached_property
    def christoffel_symbols(self) -> np.ndarray:
        """Get Christoffel symbols Γ^λ_μν from metric."""
        return self.metric.christoffel_symbols

    @monitor_performance("scalar_gradient")
    def scalar_gradient(
        self,
        scalar_field: np.ndarray | sp.Expr,
        coordinates: np.ndarray | list[np.ndarray],
    ) -> FourVector:
        """
        Compute covariant gradient ∇_μ φ = ∂_μ φ of scalar field.

        For scalars, covariant derivative equals ordinary partial derivative.

        Args:
            scalar_field: Scalar field φ(x^μ)
            coordinates: Coordinate grid [t, x, y, z]

        Returns:
            Gradient four-vector ∇_μ φ
        """
        if isinstance(scalar_field, sp.Expr):
            # Symbolic gradient
            gradient_components = []
            coord_symbols = [sp.Symbol(f"x{i}") for i in range(4)]

            for mu in range(4):
                gradient_components.append(sp.diff(scalar_field, coord_symbols[mu]))

            return FourVector(sp.Matrix(gradient_components), True, self.metric)

        else:
            # Numerical gradient using numpy
            if isinstance(coordinates, list):
                coordinates = np.array(coordinates)

            gradient_components = np.gradient(scalar_field, *coordinates)
            return FourVector(np.array(gradient_components), True, self.metric)

    @monitor_performance("vector_divergence")
    def vector_divergence(
        self,
        vector_field: FourVector,
        coordinates: list[np.ndarray],
    ) -> np.ndarray:
        """
        Compute covariant divergence ∇_μ V^μ of a vector field across the entire grid.

        Returns a scalar field (grid) representing the divergence at each point.
        """
        christoffel = self.christoffel_symbols
        components = vector_field.components  # Shape: (..., 4)

        # 1. Compute all partial derivatives ∂_μ V^ν at once.
        # The result `grad_V` will have shape (4, ..., 4)
        # where the first axis is the derivative index μ.
        grad_V = np.stack(
            [
                np.gradient(components[..., nu], axis=mu, edge_order=2)
                for mu in range(4)
                for nu in range(4)
            ],
            axis=0,
        ).reshape(4, 4, *components.shape[:-1])

        # We need the trace of the partial derivatives: ∂_μ V^μ
        partial_deriv_trace = np.einsum("ii...->...", grad_V)  # Sums over the diagonal

        # 2. Compute the Christoffel term Γ^μ_μλ V^λ, vectorized over the grid.
        # christoffel shape: (4, 4, 4)
        # components shape: (..., 4)
        # result shape: (...)
        christoffel_term = np.einsum("iil,...l->...", christoffel, components)

        # 3. Return the sum as a scalar field
        result: np.ndarray = partial_deriv_trace + christoffel_term
        return result

    def material_derivative(
        self,
        tensor_field: TensorField,
        four_velocity: FourVector,
        coordinates: np.ndarray | list[np.ndarray],
    ) -> TensorField:
        """
        Compute material derivative D = u^μ ∇_μ along fluid flow.

        Args:
            tensor_field: Tensor field to differentiate
            four_velocity: Fluid four-velocity u^μ
            coordinates: Coordinate grid

        Returns:
            Material derivative of tensor field
        """
        # For each component μ, compute u^μ ∇_μ T
        material_deriv_components = np.zeros_like(tensor_field.components)

        for mu in range(4):
            # Compute ∇_μ T
            cov_deriv = self.tensor_covariant_derivative(tensor_field, mu, coordinates)

            # Contract with u^μ: u^μ ∇_μ T
            material_deriv_components += four_velocity.components[mu] * cov_deriv.components

        return TensorField(material_deriv_components, tensor_field._index_string(), self.metric)

    @monitor_performance("tensor_covariant_derivative")
    def tensor_covariant_derivative(
        self,
        tensor_field: TensorField,
        coordinates: list[np.ndarray],
    ) -> TensorField:
        """
        Compute covariant derivative ∇_μ T^α...β... of a general tensor field.
        Returns a new tensor field with one additional covariant index (the derivative index).
        """
        christoffel = self.christoffel_symbols
        components = tensor_field.components

        # 1. Compute all partial derivatives ∂_μ T^..._...
        partial_derivatives = np.stack(
            [self._partial_derivative(components, mu, coordinates) for mu in range(4)], axis=-1
        )

        correction = np.zeros_like(partial_derivatives)

        # 2. Add correction terms for each index
        for i, (is_cov, _) in enumerate(tensor_field.indices):
            if is_cov:
                # Covariant index correction: -Γ^λ_μi T_...λ...
                correction -= self._contract_christoffel_covariant(components, christoffel, i)
            else:
                # Contravariant index correction: +Γ^i_μλ T^...λ...
                correction += self._contract_christoffel(components, christoffel, i)

        covariant_derivative = partial_derivatives + correction

        # Build new index string
        new_indices = tensor_field._index_string() + " _d"

        return TensorField(covariant_derivative, new_indices, self.metric)

    def _partial_derivative(
        self,
        tensor_components: np.ndarray,
        index: int,
        coordinates: list[np.ndarray],
    ) -> np.ndarray:
        """Compute partial derivative ∂_μ T along a specified coordinate axis."""
        result: np.ndarray = np.gradient(
            tensor_components, coordinates[index], axis=index, edge_order=2
        )
        return result

    def _contract_christoffel(
        self,
        tensor_components: np.ndarray,
        christoffel: np.ndarray,
        tensor_index_pos: int,
    ) -> np.ndarray:
        """Computes the contravariant correction term: Γ^α_μλ T^...λ..."""
        # Move the contracted axis to the end
        moved_tensor = np.moveaxis(tensor_components, tensor_index_pos, -1)
        # Contract with Christoffel symbols
        # Γ^α_μλ T^...λ -> result^...α_μ
        # (a, m, l) * (..., l) -> (..., a, m)
        correction = np.tensordot(moved_tensor, christoffel, axes=([-1], [2]))
        # Move the new axes to their correct positions
        return np.moveaxis(correction, [-2, -1], [tensor_index_pos, tensor_components.ndim])

    def _contract_christoffel_covariant(
        self,
        tensor_components: np.ndarray,
        christoffel: np.ndarray,
        tensor_index_pos: int,
    ) -> np.ndarray:
        """Computes the covariant correction term: Γ^λ_μα T_...λ..."""
        # Move the contracted axis to the end
        moved_tensor = np.moveaxis(tensor_components, tensor_index_pos, -1)
        # Contract with Christoffel symbols
        # Γ^λ_μα T_...λ -> result_...α_μ
        # (l, m, a) * (..., l) -> (..., m, a)
        correction = np.tensordot(moved_tensor, christoffel, axes=([-1], [0]))
        # Move the new axes to their correct positions
        return np.moveaxis(correction, [-2, -1], [tensor_components.ndim, tensor_index_pos])

    def laplacian(
        self,
        scalar_field: np.ndarray | sp.Expr,
        coordinates: np.ndarray | list[np.ndarray],
    ) -> float | sp.Expr:
        """
        Compute covariant Laplacian ∇^μ ∇_μ φ = g^μν ∇_μ ∇_ν φ.

        Args:
            scalar_field: Scalar field φ
            coordinates: Coordinate grid

        Returns:
            Laplacian scalar
        """
        # First compute gradient
        gradient = self.scalar_gradient(scalar_field, coordinates)

        # Then compute divergence of gradient (raise index first)
        gradient_contravariant = gradient.raise_index(0)

        return self.vector_divergence(gradient_contravariant, coordinates)

    def lie_derivative(
        self,
        tensor_field: TensorField,
        vector_field: FourVector,
        coordinates: np.ndarray | list[np.ndarray],
    ) -> TensorField:
        """
        Compute Lie derivative L_V T of tensor along vector field.

        Args:
            tensor_field: Tensor to differentiate
            vector_field: Vector field V^μ
            coordinates: Coordinate grid

        Returns:
            Lie derivative L_V T
        """
        # Lie derivative: L_V T = V^μ ∇_μ T + (∇_μ V^ν) T terms

        # Material derivative part: V^μ ∇_μ T
        material_part = self.material_derivative(tensor_field, vector_field, coordinates)

        # Additional terms depend on tensor rank - simplified implementation
        # Full implementation would include all index contractions

        return material_part


class ProjectionOperator:
    """
    Projection operators for fluid decomposition in relativistic hydrodynamics.

    Decomposes tensors into components parallel and perpendicular to the
    fluid four-velocity, essential for Israel-Stewart formalism.
    """

    def __init__(self, four_velocity: FourVector, metric: "MetricBase"):
        """
        Initialize projection operators.

        Args:
            four_velocity: Fluid four-velocity u^μ (assumed normalized)
            metric: Spacetime metric tensor
        """
        self.u = four_velocity
        self.metric = metric

        # Validate four-velocity normalization with correct signature handling
        mag_sq = self.u.magnitude_squared()
        signature = getattr(metric, "signature", (-1, 1, 1, 1))

        if signature[0] < 0:  # Mostly-plus signature (-,+,+,+)
            expected_norm = -1.0
            if abs(mag_sq - expected_norm) > 1e-10:
                warnings.warn(
                    f"Four-velocity should be normalized: u^μ u_μ = -1, got {mag_sq:.2e}",
                    stacklevel=2,
                )
        else:  # Mostly-minus signature (+,-,-,-)
            expected_norm = 1.0
            if abs(mag_sq - expected_norm) > 1e-10:
                warnings.warn(
                    f"Four-velocity should be normalized: u^μ u_μ = +1, got {mag_sq:.2e}",
                    stacklevel=2,
                )

    def parallel_projector(self) -> TensorField:
        """
        Compute parallel projection tensor P^μν = u^μ u^ν.

        Projects onto direction parallel to four-velocity.

        Returns:
            Parallel projection tensor
        """
        # Outer product of four-velocity with itself
        u_outer = np.outer(self.u.components, self.u.components)

        return TensorField(u_outer, "mu nu", self.metric)

    @monitor_performance("perpendicular_projector")
    def perpendicular_projector(self) -> TensorField:
        """
        Compute perpendicular projection tensor Δ^μν = g^μν + u^μ u^ν.

        Projects onto 3-space orthogonal to four-velocity.

        Returns:
            Perpendicular projection tensor Δ^μν
        """
        # Get inverse metric g^μν
        g_inverse = self.metric.inverse

        # Compute u^μ u^ν
        u_outer = np.outer(self.u.components, self.u.components)

        # Perpendicular projector: Δ^μν = g^μν + sign * u^μ u^ν
        sign = self.metric.signature[0]
        delta = g_inverse + sign * u_outer

        return TensorField(delta, "mu nu", self.metric)

    def project_vector_parallel(self, vector: FourVector) -> FourVector:
        """
        Project vector parallel to four-velocity: V_∥^μ = -(u · V) u^μ.

        Args:
            vector: Input four-vector

        Returns:
            Parallel component of vector
        """
        # Compute u · V = u_μ V^μ
        dot_product = self.u.dot(vector)

        # Parallel projection: V_∥^μ = -(u · V) u^μ
        parallel_components = -dot_product * self.u.components

        return FourVector(parallel_components, False, self.metric)

    @monitor_performance("project_vector_perpendicular")
    def project_vector_perpendicular(self, vector: FourVector) -> FourVector:
        """
        Project vector perpendicular to four-velocity: V_⊥^μ = Δ^μν V_ν.

        Args:
            vector: Input four-vector

        Returns:
            Perpendicular component of vector
        """
        delta = self.perpendicular_projector()
        vector_lowered = vector.lower_index(0)

        # Project: V_⊥^μ = Δ^μν V_ν
        if isinstance(vector.components, np.ndarray):
            perp_components = optimized_einsum(
                "mn,n->m", delta.components, vector_lowered.components
            )
        else:
            perp_components = delta.components * vector_lowered.components

        return FourVector(perp_components, False, self.metric)

    def project_tensor_spatial(self, tensor: TensorField) -> TensorField:
        """
        Project tensor into spatial hypersurface: T_spatial^μν = Δ^μα Δ^νβ T_αβ.

        Args:
            tensor: Input rank-2 tensor

        Returns:
            Spatially projected tensor
        """
        if tensor.rank != 2:
            raise ValueError("Spatial projection only implemented for rank-2 tensors")

        delta = self.perpendicular_projector()
        tensor_lowered = tensor.lower_index(0).lower_index(1)

        # Double projection: T_spatial^μν = Δ^μα Δ^νβ T_αβ
        if isinstance(tensor.components, np.ndarray):
            spatial = optimized_einsum(
                "ma,nb,ab->mn",
                delta.components,
                delta.components,
                tensor_lowered.components,
            )
        else:
            # SymPy version - simplified
            spatial = delta.components * tensor_lowered.components * delta.components

        return TensorField(spatial, "mu nu", self.metric)

    def extract_scalar_density(self, tensor: TensorField) -> float | sp.Expr:
        """
        Extract scalar density from tensor: ρ = u_μ u_ν T^μν.

        Args:
            tensor: Input rank-2 tensor

        Returns:
            Scalar density
        """
        if tensor.rank != 2:
            raise ValueError("Scalar extraction only works for rank-2 tensors")

        u_cov = self.u.lower_index(0)

        # Contract: ρ = u_μ u_ν T^μν
        if isinstance(tensor.components, np.ndarray):
            scalar = optimized_einsum(
                "i,j,ij->", u_cov.components, u_cov.components, tensor.components
            )
        else:
            scalar = 0
            for mu in range(4):
                for nu in range(4):
                    scalar += (
                        u_cov.components[mu] * u_cov.components[nu] * tensor.components[mu, nu]
                    )

        return scalar

    def extract_vector_density(self, tensor: TensorField) -> FourVector:
        """
        Extract vector density from tensor: j^μ = -Δ^μν u_α T^α_ν.

        Args:
            tensor: Input rank-2 tensor

        Returns:
            Vector density (spatial)
        """
        if tensor.rank != 2:
            raise ValueError("Vector extraction only works for rank-2 tensors")

        delta = self.perpendicular_projector()
        u_cov = self.u.lower_index(0)
        tensor_mixed = tensor.lower_index(1)  # T^α_ν

        # Contract: j^μ = -Δ^μν u_α T^α_ν
        if isinstance(tensor.components, np.ndarray):
            vector = -optimized_einsum(
                "mn,a,an->m",
                delta.components,
                u_cov.components,
                tensor_mixed.components,
            )
        else:
            vector = sp.zeros(4, 1)
            for mu in range(4):
                for nu in range(4):
                    for alpha in range(4):
                        vector[mu] -= (
                            delta.components[mu, nu]
                            * u_cov.components[alpha]
                            * tensor_mixed.components[alpha, nu]
                        )

        return FourVector(vector, False, self.metric)

    def decompose_tensor(
        self, tensor: TensorField
    ) -> dict[str, float | sp.Expr | FourVector | TensorField]:
        """
        Complete 3+1 decomposition of rank-2 tensor.

        Decomposes T^μν into:
        - Scalar density: ρ = u_μ u_ν T^μν
        - Vector density: j^μ = -Δ^μν u_α T^α_ν
        - Pressure tensor: P^μν = Δ^μα Δ^νβ T_αβ

        Args:
            tensor: Rank-2 tensor to decompose

        Returns:
            Dictionary with decomposed components
        """
        if tensor.rank != 2:
            raise ValueError("Tensor decomposition only works for rank-2 tensors")

        return {
            "scalar_density": self.extract_scalar_density(tensor),
            "vector_density": self.extract_vector_density(tensor),
            "pressure_tensor": self.project_tensor_spatial(tensor),
        }

    def fluid_frame_components(
        self, tensor: TensorField
    ) -> dict[str, float | np.ndarray | sp.Matrix]:
        """
        Extract components in fluid rest frame.

        Args:
            tensor: Rank-2 tensor

        Returns:
            Dictionary with frame components
        """
        if tensor.rank != 2:
            raise ValueError("Frame extraction only works for rank-2 tensors")

        # Energy density: T^00 in fluid frame
        energy_density = self.extract_scalar_density(tensor)

        # Momentum density: T^0i in fluid frame (should be zero for perfect fluid)
        momentum_density = self.extract_vector_density(tensor)

        # Stress tensor: T^ij in fluid frame
        stress_tensor = self.project_tensor_spatial(tensor)

        return {
            "energy_density": energy_density,
            "momentum_density": momentum_density.spatial_components,
            "stress_tensor": stress_tensor.components[1:4, 1:4],  # 3x3 spatial part
        }
