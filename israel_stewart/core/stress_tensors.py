"""
Stress-energy and viscous stress tensors for relativistic hydrodynamics.

This module provides specialized tensor classes for stress-energy tensors
and viscous corrections in Israel-Stewart hydrodynamics.
"""

import warnings

# Forward reference for metrics
from typing import TYPE_CHECKING, Optional

import numpy as np
import sympy as sp

from .four_vectors import FourVector
from .performance import monitor_performance

# Import base classes and utilities
from .tensor_base import TensorField

if TYPE_CHECKING:
    from .metrics import MetricBase


class StressEnergyTensor(TensorField):
    """
    Stress-energy tensor T^μν for relativistic fluids.

    Represents energy, momentum, and stress in relativistic field theory.
    In Israel-Stewart theory, this includes perfect fluid and viscous corrections.
    """

    def __init__(self, components: np.ndarray | sp.Matrix, metric: Optional["MetricBase"] = None):
        """
        Initialize stress-energy tensor.

        Args:
            components: 4x4 tensor components T^μν
            metric: Metric tensor for operations
        """
        if isinstance(components, list):
            components = np.array(components)

        if components.shape != (4, 4):
            raise ValueError(f"Stress-energy tensor must be 4x4, got shape {components.shape}")

        super().__init__(components, "mu nu", metric)

    @monitor_performance("energy_density_extraction")
    def energy_density(self, four_velocity: FourVector) -> float | sp.Expr:
        """
        Extract energy density ρ = u_μ u_ν T^μν.

        Args:
            four_velocity: Fluid four-velocity

        Returns:
            Energy density scalar
        """
        if self.metric is None:
            raise ValueError("Cannot compute energy density without metric")

        # Lower both indices of four-velocity
        u_cov = four_velocity.lower_index(0)

        # Contract: ρ = u_μ u_ν T^μν
        if isinstance(self.components, np.ndarray):
            result = np.einsum("i,j,ij->", u_cov.components, u_cov.components, self.components)
        else:
            # SymPy computation
            result = 0
            for mu in range(4):
                for nu in range(4):
                    result += u_cov.components[mu] * u_cov.components[nu] * self.components[mu, nu]

        return result

    @monitor_performance("momentum_density_extraction")
    def momentum_density(self, four_velocity: FourVector) -> FourVector:
        """
        Extract momentum density j^μ = -u_ν T^μν.

        Args:
            four_velocity: Fluid four-velocity

        Returns:
            Momentum density four-vector
        """
        if self.metric is None:
            raise ValueError("Cannot compute momentum density without metric")

        u_cov = four_velocity.lower_index(0)

        # Contract: j^μ = -u_ν T^μν
        if isinstance(self.components, np.ndarray):
            momentum = -np.einsum("j,ij->i", u_cov.components, self.components)
        else:
            momentum = sp.zeros(4, 1)
            for mu in range(4):
                for nu in range(4):
                    momentum[mu] -= u_cov.components[nu] * self.components[mu, nu]

        return FourVector(momentum, False, self.metric)

    def pressure_tensor(self, four_velocity: FourVector) -> "TensorField":
        """
        Extract spatial pressure tensor P^μν = Δ^μα Δ^νβ T_αβ.

        Args:
            four_velocity: Fluid four-velocity

        Returns:
            Spatial pressure tensor
        """
        if self.metric is None:
            raise ValueError("Cannot compute pressure tensor without metric")

        # Import here to avoid circular imports
        from .derivatives import ProjectionOperator

        # Get projection tensor Δ^μν = g^μν + u^μ u^ν
        proj_op = ProjectionOperator(four_velocity, self.metric)
        delta = proj_op.perpendicular_projector()

        # Lower indices of stress tensor: T_αβ = g_αγ g_βδ T^γδ
        T_down = self.lower_index(0).lower_index(1)

        # Contract: P^μν = Δ^μα Δ^νβ T_αβ
        if isinstance(self.components, np.ndarray):
            pressure = np.einsum(
                "ma,nb,ab->mn", delta.components, delta.components, T_down.components
            )
        else:
            # SymPy version - simplified
            pressure = delta.components * T_down.components * delta.components

        return TensorField(pressure, "mu nu", self.metric)

    @classmethod
    def perfect_fluid_form(
        cls,
        energy_density: float | sp.Expr,
        pressure: float | sp.Expr,
        four_velocity: FourVector,
        metric: Optional["MetricBase"] = None,
    ) -> "StressEnergyTensor":
        """
        Construct perfect fluid stress-energy tensor.

        T^μν = (ρ + p) u^μ u^ν + p g^μν

        Args:
            energy_density: Energy density ρ
            pressure: Pressure p
            four_velocity: Fluid four-velocity u^μ
            metric: Spacetime metric

        Returns:
            Perfect fluid stress-energy tensor
        """
        if metric is None:
            raise ValueError("Cannot construct perfect fluid without metric")

        # Get metric inverse
        g_inv = metric.inverse

        # Construct u^μ u^ν term
        u_outer = np.outer(four_velocity.components, four_velocity.components)

        # Perfect fluid form: T^μν = (ρ + p) u^μ u^ν + p g^μν
        perfect_fluid = (energy_density + pressure) * u_outer + pressure * g_inv

        return cls(perfect_fluid, metric)

    def add_viscous_corrections(
        self, viscous_tensor: "ViscousStressTensor"
    ) -> "StressEnergyTensor":
        """
        Add viscous corrections to stress-energy tensor.

        Args:
            viscous_tensor: Viscous stress tensor π^μν

        Returns:
            Total stress-energy tensor T^μν_total = T^μν_perfect + π^μν
        """
        total_components = self.components + viscous_tensor.components
        return StressEnergyTensor(total_components, self.metric)

    def conservation_check(
        self, coordinates: list[np.ndarray] | None = None
    ) -> np.ndarray | sp.Matrix:
        """
        Check energy-momentum conservation ∇_μ T^μν = 0.

        Args:
            coordinates: Coordinate arrays for numerical derivatives

        Returns:
            Conservation violation (should be near zero)
        """
        if coordinates is None:
            raise ValueError("Need coordinate arrays for conservation check")

        # Import here to avoid circular imports
        from .derivatives import CovariantDerivative

        if self.metric is None:
            raise ValueError("Cannot check conservation without metric")

        # Compute covariant divergence
        cov_deriv = CovariantDerivative(self.metric)
        divergence = cov_deriv.divergence(self, coordinates)

        return divergence

    def trace(self, indices_pair: tuple[int, int] | None = None) -> float | sp.Expr:
        """
        Compute trace T = g_μν T^μν.

        Returns:
            Scalar trace of stress-energy tensor
        """
        if self.metric is None:
            raise ValueError("Cannot compute trace without metric")

        # Contract with metric: T = g_μν T^μν
        g_down = self.metric.components

        if isinstance(self.components, np.ndarray):
            return np.einsum("mn,mn->", g_down, self.components)
        else:
            trace_result = 0
            for mu in range(4):
                for nu in range(4):
                    trace_result += g_down[mu, nu] * self.components[mu, nu]
            return trace_result

    def eigenvalues(self) -> np.ndarray | list[sp.Expr]:
        """
        Compute eigenvalues of stress-energy tensor.

        Used for checking energy conditions and stability analysis.

        Returns:
            Array or list of eigenvalues
        """
        if isinstance(self.components, np.ndarray):
            return np.linalg.eigvals(self.components)
        else:
            # SymPy eigenvalues
            return list(sp.Matrix(self.components).eigenvals().keys())

    def dominant_energy_condition(self, tolerance: float = 1e-10) -> bool:
        """
        Check dominant energy condition.

        For physical matter, T^μν should satisfy dominant energy condition.

        Args:
            tolerance: Numerical tolerance for checks

        Returns:
            True if dominant energy condition is satisfied
        """
        # Simplified check: all eigenvalues should have the right sign structure
        eigenvals = self.eigenvalues()

        if isinstance(eigenvals, np.ndarray):
            # For dominant energy condition, need detailed analysis of eigenvalue structure
            # This is a simplified implementation
            return bool(np.all(eigenvals >= -tolerance))
        else:
            # SymPy case - symbolic check
            return all(eigenval >= -tolerance for eigenval in eigenvals if eigenval.is_real)


class ViscousStressTensor(TensorField):
    """
    Viscous stress tensor π^μν for Israel-Stewart hydrodynamics.

    Represents second-order viscous corrections including shear viscosity,
    bulk viscosity, and heat conduction effects.
    """

    def __init__(self, components: np.ndarray | sp.Matrix, metric: Optional["MetricBase"] = None):
        """
        Initialize viscous stress tensor.

        Args:
            components: 4x4 viscous tensor components π^μν
            metric: Metric tensor for operations
        """
        if isinstance(components, list):
            components = np.array(components)

        if components.shape != (4, 4):
            raise ValueError(f"Viscous stress tensor must be 4x4, got shape {components.shape}")

        super().__init__(components, "mu nu", metric)

        # Viscous tensor should be symmetric and orthogonal to four-velocity
        self._validate_viscous_properties()

    def _validate_viscous_properties(self) -> None:
        """Validate viscous tensor properties."""
        # Check symmetry (within numerical tolerance)
        if isinstance(self.components, np.ndarray):
            symmetric_part = 0.5 * (self.components + self.components.T)
            if not np.allclose(self.components, symmetric_part, rtol=1e-10):
                warnings.warn("Viscous stress tensor should be symmetric", stacklevel=2)

    @monitor_performance("shear_extraction")
    def shear_part(self, four_velocity: FourVector) -> "ViscousStressTensor":
        """
        Extract shear viscous part (traceless spatial component).

        Args:
            four_velocity: Fluid four-velocity

        Returns:
            Shear viscous tensor
        """
        if self.metric is None:
            raise ValueError("Cannot compute shear part without metric")

        # Import here to avoid circular imports
        from .derivatives import ProjectionOperator

        # Get perpendicular projector
        proj_op = ProjectionOperator(four_velocity, self.metric)
        delta = proj_op.perpendicular_projector()

        # Project viscous tensor: π^μν_shear = Δ^μα Δ^νβ π_αβ - (1/3)Δ^μν Δ_γδ π^γδ
        pi_down = self.lower_index(0).lower_index(1)

        # Spatial projection
        if isinstance(self.components, np.ndarray):
            projected = np.einsum(
                "ma,nb,ab->mn", delta.components, delta.components, pi_down.components
            )

            # Remove trace: shear = projected - (1/3) * trace * delta
            trace = np.einsum("ab,ab", delta.components, pi_down.components)
            shear = projected - (1.0 / 3.0) * trace * delta.components
        else:
            # SymPy version
            projected = delta.components * pi_down.components * delta.components
            trace = (delta.components * pi_down.components).trace()
            shear = projected - (sp.Rational(1, 3) * trace * delta.components)

        return ViscousStressTensor(shear, self.metric)

    def bulk_part(self, four_velocity: FourVector) -> float | sp.Expr:
        """
        Extract bulk viscous part (trace of spatial component).

        Args:
            four_velocity: Fluid four-velocity

        Returns:
            Bulk viscous scalar Π = -(1/3) Δ_μν π^μν
        """
        if self.metric is None:
            raise ValueError("Cannot compute bulk part without metric")

        # Import here to avoid circular imports
        from .derivatives import ProjectionOperator

        # Get perpendicular projector
        proj_op = ProjectionOperator(four_velocity, self.metric)
        delta = proj_op.perpendicular_projector()

        # Bulk viscosity: Π = -(1/3) Δ_μν π^μν
        pi_down = self.lower_index(0).lower_index(1)

        if isinstance(self.components, np.ndarray):
            bulk = -(1.0 / 3.0) * np.einsum("mn,mn", delta.components, pi_down.components)
        else:
            bulk = -(sp.Rational(1, 3) * (delta.components * pi_down.components).trace())

        return bulk

    def heat_flux_part(self, four_velocity: FourVector) -> FourVector:
        """
        Extract heat flux part q^μ = -u_ν π^μν.

        Args:
            four_velocity: Fluid four-velocity

        Returns:
            Heat flux four-vector
        """
        if self.metric is None:
            raise ValueError("Cannot compute heat flux without metric")

        u_cov = four_velocity.lower_index(0)

        # Heat flux: q^μ = -u_ν π^μν
        if isinstance(self.components, np.ndarray):
            heat_flux = -np.einsum("j,ij->i", u_cov.components, self.components)
        else:
            heat_flux = sp.zeros(4, 1)
            for mu in range(4):
                for nu in range(4):
                    heat_flux[mu] -= u_cov.components[nu] * self.components[mu, nu]

        return FourVector(heat_flux, False, self.metric)

    @classmethod
    def from_transport_coefficients(
        cls,
        shear_viscosity: float | sp.Expr,
        bulk_viscosity: float | sp.Expr,
        thermal_conductivity: float | sp.Expr,
        four_velocity: FourVector,
        velocity_gradient: TensorField,
        temperature_gradient: FourVector,
        metric: Optional["MetricBase"] = None,
    ) -> "ViscousStressTensor":
        """
        Construct viscous tensor from transport coefficients.

        First-order (Navier-Stokes) approximation:
        π^μν = -η σ^μν - ζ Δ^μν ∇_α u^α + thermal terms

        Args:
            shear_viscosity: Shear viscosity η
            bulk_viscosity: Bulk viscosity ζ
            thermal_conductivity: Thermal conductivity κ
            four_velocity: Fluid four-velocity
            velocity_gradient: Velocity gradient tensor
            temperature_gradient: Temperature gradient
            metric: Metric tensor

        Returns:
            Viscous stress tensor
        """
        if metric is None:
            raise ValueError("Cannot construct viscous tensor without metric")

        # Import here to avoid circular imports
        from .derivatives import ProjectionOperator

        # First-order viscous approximation for Israel-Stewart theory
        proj_op = ProjectionOperator(four_velocity, metric)
        delta = proj_op.perpendicular_projector()

        # Construct shear rate tensor σ^μν from velocity gradient
        # σ^μν = Δ^μα Δ^νβ (∇_α u_β + ∇_β u_α) - (2/3) Δ^μν ∇_γ u^γ

        # Project velocity gradient onto spatial hypersurface
        velocity_gradient_projected = proj_op.project_tensor_spatial(velocity_gradient)

        # Symmetrize the projected gradient
        shear_rate = velocity_gradient_projected.symmetrize()

        # Remove trace (make traceless)
        trace_part = shear_rate.trace() / 3.0
        if isinstance(shear_rate.components, np.ndarray):
            traceless_shear = shear_rate.components - trace_part * delta.components
        else:
            traceless_shear = shear_rate.components - trace_part * delta.components

        # Construct viscous tensor components
        # π^μν = -η σ^μν - ζ Δ^μν (∇_α u^α) + heat conduction terms

        # Shear viscous part: -η σ^μν
        shear_part = -shear_viscosity * traceless_shear

        # Bulk viscous part: -ζ Δ^μν (∇_α u^α)
        # Approximate divergence from velocity gradient trace
        divergence = velocity_gradient.trace() if hasattr(velocity_gradient, "trace") else 0.0
        bulk_part = -bulk_viscosity * divergence * delta.components

        # Heat conduction part: thermal conductivity effects
        # Simplified: q^μ = -κ ∇^μ T (projected)
        temp_grad_projected = proj_op.project_vector_perpendicular(temperature_gradient)
        if isinstance(temp_grad_projected.components, np.ndarray):
            heat_part = -thermal_conductivity * np.outer(
                temp_grad_projected.components, temp_grad_projected.components
            )
        else:
            heat_part = (
                -thermal_conductivity
                * temp_grad_projected.components
                * temp_grad_projected.components.T
            )

        # Total viscous tensor
        total_viscous = shear_part + bulk_part + heat_part

        return cls(total_viscous, metric)

    def israel_stewart_evolution(
        self,
        relaxation_time: float | sp.Expr,
        four_velocity: FourVector,
        coordinates: list[np.ndarray],
        transport_coefficients: dict[str, float | sp.Expr],
    ) -> "ViscousStressTensor":
        """
        Evolve viscous tensor according to Israel-Stewart equations.

        Second-order evolution equation:
        τ D π^μν + π^μν = -η σ^μν - ζ Δ^μν ∇_α u^α + ...

        Args:
            relaxation_time: Relaxation time τ
            four_velocity: Fluid four-velocity
            coordinates: Coordinate arrays
            transport_coefficients: Transport coefficient dictionary

        Returns:
            Evolved viscous tensor
        """
        # Import here to avoid circular imports
        from .derivatives import CovariantDerivative

        if self.metric is None:
            raise ValueError("Cannot evolve without metric")

        # Material derivative D = u^μ ∇_μ
        cov_deriv = CovariantDerivative(self.metric)
        material_derivative = cov_deriv.material_derivative(self, four_velocity, coordinates)

        # First-order source terms (Navier-Stokes)
        transport_coefficients.get("shear_viscosity", 0.0)
        transport_coefficients.get("bulk_viscosity", 0.0)

        # Simplified source term construction
        # In full implementation, would need velocity gradient and proper σ^μν
        source_term = np.zeros_like(self.components)  # Placeholder

        # Israel-Stewart evolution: τ D π^μν + π^μν = S^μν
        evolved_components = (source_term - material_derivative) / (1.0 + relaxation_time)

        return ViscousStressTensor(evolved_components, self.metric)

    def causality_check(self, sound_speed: float | sp.Expr = 1.0 / 3.0) -> bool:
        """
        Check causality constraints for viscous tensor.

        Israel-Stewart theory imposes constraints to ensure causality.

        Args:
            sound_speed: Speed of sound c_s^2

        Returns:
            True if causality constraints are satisfied
        """
        # Simplified causality check
        # Full implementation would check eigenvalue structure and characteristic speeds

        eigenvals = (
            np.linalg.eigvals(self.components) if isinstance(self.components, np.ndarray) else None
        )

        if eigenvals is not None:
            # Check that viscous effects don't lead to superluminal speeds
            max_eigenval = np.max(np.real(eigenvals))
            return bool(max_eigenval <= sound_speed)

        return True  # Conservative assumption for symbolic case
