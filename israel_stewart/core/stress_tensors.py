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
            # SymPy version - proper tensor contraction: P^μν = Δ^μα Δ^νβ T_αβ
            pressure = sp.zeros(4, 4)
            for mu in range(4):
                for nu in range(4):
                    for alpha in range(4):
                        for beta in range(4):
                            pressure[mu, nu] += (
                                delta.components[mu, alpha]
                                * delta.components[nu, beta]
                                * T_down.components[alpha, beta]
                            )

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

        # Compute covariant divergence ∇_μ T^μν for each ν
        cov_deriv = CovariantDerivative(self.metric)
        divergence = cov_deriv.tensor_covariant_derivative(self, coordinates)

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

    def dominant_energy_condition(
        self, four_velocity: FourVector, tolerance: float = 1e-10
    ) -> bool:
        """
        Check dominant energy condition.

        For physical matter, T^μν should satisfy:
        1. Energy density ρ = u_μ u_ν T^μν ≥ 0
        2. For any timelike or null vector v^μ: v_μ v_ν T^μν ≥ 0

        Args:
            four_velocity: Fluid four-velocity for energy density extraction
            tolerance: Numerical tolerance for checks

        Returns:
            True if dominant energy condition is satisfied
        """
        # Check 1: Energy density must be non-negative
        energy_dens = self.energy_density(four_velocity)
        if isinstance(energy_dens, float | int):
            if energy_dens < -tolerance:
                return False
        elif hasattr(energy_dens, "is_negative"):
            if energy_dens.is_negative:
                return False

        # Check 2: Eigenvalue analysis
        # For the dominant energy condition, we need more sophisticated analysis
        # Here we implement a simplified version checking eigenvalue signs
        eigenvals = self.eigenvalues()

        if isinstance(eigenvals, np.ndarray):
            # Check that there's one negative eigenvalue (timelike direction)
            # and three positive eigenvalues (spacelike directions)
            eigenvals_sorted = np.sort(eigenvals)

            # For physical stress-energy tensor in mostly-minus signature:
            # Should have one large negative eigenvalue and three smaller positive ones
            if len(eigenvals_sorted) == 4:
                # Most negative eigenvalue should dominate
                if eigenvals_sorted[0] >= -tolerance:  # Should be negative
                    return False
                # Check that energy density dominates momentum/stress
                if abs(eigenvals_sorted[0]) < max(abs(eigenvals_sorted[1:])):
                    return False

            return True
        else:
            # SymPy case - basic check for real eigenvalues
            real_eigenvals = [ev for ev in eigenvals if ev.is_real]
            if len(real_eigenvals) < 4:
                return False  # Need all real eigenvalues for physical matter

            return all(abs(ev) > tolerance for ev in real_eigenvals)  # No zero eigenvalues


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

            # Check for NaN or infinite values
            if not np.all(np.isfinite(self.components)):
                raise ValueError("Viscous stress tensor contains NaN or infinite values")

            # Check for reasonable magnitude (basic sanity check)
            max_component = np.max(np.abs(self.components))
            if max_component > 1e10:
                warnings.warn(
                    f"Viscous stress tensor has very large components (max: {max_component:.2e}). "
                    "This may indicate numerical instability.",
                    stacklevel=2,
                )
        else:
            # SymPy case - check for symbolic validity
            try:
                # Attempt to access matrix properties
                symmetric_check = self.components - self.components.T
                if not symmetric_check.equals(sp.zeros(4, 4)):
                    warnings.warn("Viscous stress tensor should be symmetric", stacklevel=2)
            except Exception:
                warnings.warn("Could not validate SymPy viscous tensor properties", stacklevel=2)

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
            # SymPy version - proper tensor contraction
            projected = sp.zeros(4, 4)
            for mu in range(4):
                for nu in range(4):
                    for alpha in range(4):
                        for beta in range(4):
                            projected[mu, nu] += (
                                delta.components[mu, alpha]
                                * delta.components[nu, beta]
                                * pi_down.components[alpha, beta]
                            )

            # Compute trace: Δ_γδ π^γδ
            trace = 0
            for gamma in range(4):
                for delta_idx in range(4):
                    trace += (
                        delta.components[gamma, delta_idx] * pi_down.components[gamma, delta_idx]
                    )

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
            # SymPy version - proper trace computation: Π = -(1/3) Δ_μν π^μν
            trace = 0
            for mu in range(4):
                for nu in range(4):
                    trace += delta.components[mu, nu] * pi_down.components[mu, nu]
            bulk = -sp.Rational(1, 3) * trace

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

        # Validate transport coefficients
        if isinstance(shear_viscosity, float | int) and shear_viscosity < 0:
            raise ValueError(f"Shear viscosity must be non-negative, got {shear_viscosity}")
        if isinstance(bulk_viscosity, float | int) and bulk_viscosity < 0:
            raise ValueError(f"Bulk viscosity must be non-negative, got {bulk_viscosity}")
        if isinstance(thermal_conductivity, float | int) and thermal_conductivity < 0:
            raise ValueError(
                f"Thermal conductivity must be non-negative, got {thermal_conductivity}"
            )

        # Check four-velocity normalization
        if four_velocity.metric is not None:
            norm_squared = four_velocity.dot(four_velocity)
            expected_norm = -1.0  # For mostly-minus signature
            if isinstance(norm_squared, float | int):
                if abs(norm_squared - expected_norm) > 1e-10:
                    warnings.warn(
                        f"Four-velocity not properly normalized: u·u = {norm_squared}, expected {expected_norm}",
                        stacklevel=2,
                    )

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
        # Compute proper four-divergence using covariant derivative
        from .derivatives import CovariantDerivative

        cov_deriv = CovariantDerivative(metric)
        # For proper implementation, we need coordinate arrays to compute ∇_α u^α
        # For now, use a warning and simplified approximation
        warnings.warn(
            "from_transport_coefficients uses simplified four-divergence approximation. "
            "For accurate results, use proper covariant divergence with coordinate arrays.",
            stacklevel=2,
        )

        # Simplified approximation: use trace of velocity gradient as proxy
        # Note: This is not physically correct but allows method to function
        divergence = velocity_gradient.trace() if hasattr(velocity_gradient, "trace") else 0.0
        bulk_part = -bulk_viscosity * divergence * delta.components

        # Heat conduction correction:
        # Heat flux q^μ doesn't directly contribute to viscous stress tensor π^μν
        # In proper Israel-Stewart theory, heat flux is treated separately
        # We set heat_part to zero and add a warning
        warnings.warn(
            "Heat conduction effects in viscous stress tensor are not physically correct "
            "in this simplified implementation. Heat flux should be treated as separate vector quantity.",
            stacklevel=2,
        )

        # Set heat conduction part to zero for physical correctness
        if isinstance(delta.components, np.ndarray):
            heat_part = np.zeros_like(delta.components)
        else:
            heat_part = sp.zeros(4, 4)

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

        # Extract transport coefficients
        eta = transport_coefficients.get("shear_viscosity", 0.0)
        zeta = transport_coefficients.get("bulk_viscosity", 0.0)

        # Second-order coupling coefficients
        lambda_pi_pi = transport_coefficients.get("lambda_pi_pi", 0.0)
        xi_1 = transport_coefficients.get("xi_1", 0.0)

        # Import projection operator
        from .derivatives import ProjectionOperator

        proj_op = ProjectionOperator(four_velocity, self.metric)
        delta = proj_op.perpendicular_projector()

        # Construct Israel-Stewart source terms
        # For full implementation, we need:
        # 1. Shear rate tensor σ^μν from velocity gradients
        # 2. Four-divergence ∇_α u^α from velocity field
        # 3. Second-order coupling terms

        warnings.warn(
            "israel_stewart_evolution uses simplified source terms. "
            "Full implementation requires velocity gradients and proper shear rate tensor construction.",
            stacklevel=2,
        )

        # Simplified source term: S^μν = -η σ^μν - ζ Δ^μν ∇_α u^α + second-order terms
        # Use zero for uncomputed terms
        if isinstance(self.components, np.ndarray):
            # First-order Navier-Stokes source (simplified)
            source_term = np.zeros_like(self.components)

            # Add second-order nonlinear terms (simplified)
            # λ_ππ terms: quadratic in π^μν
            if lambda_pi_pi != 0:
                pi_squared_trace = np.trace(self.components @ self.components)
                source_term += lambda_pi_pi * pi_squared_trace * delta.components
        else:
            # SymPy version
            source_term = sp.zeros(4, 4)

            # Add nonlinear coupling terms
            if lambda_pi_pi != 0:
                pi_squared = self.components * self.components
                pi_squared_trace = pi_squared.trace()
                source_term += lambda_pi_pi * pi_squared_trace * delta.components

        # Israel-Stewart evolution equation: τ D π^μν + π^μν = S^μν
        # Rearrange: D π^μν = (S^μν - π^μν) / τ
        if isinstance(material_derivative, float | int):
            # Handle scalar case
            time_evolution = (source_term - self.components) / relaxation_time
        else:
            # Tensor case
            time_evolution = (source_term - self.components) / relaxation_time

        # For evolution, return time derivative (not updated tensor)
        # In practice, this would be integrated by the solver
        evolved_components = time_evolution

        return ViscousStressTensor(evolved_components, self.metric)

    def causality_check(
        self,
        four_velocity: FourVector,
        energy_density: float | sp.Expr,
        pressure: float | sp.Expr,
        sound_speed: float | sp.Expr = 1.0 / 3.0,
    ) -> dict[str, bool]:
        """
        Check causality constraints for viscous tensor.

        Israel-Stewart theory requires that viscous effects don't violate causality.
        This includes checking characteristic speeds and stability conditions.

        Args:
            four_velocity: Fluid four-velocity
            energy_density: Background energy density ρ
            pressure: Background pressure p
            sound_speed: Speed of sound c_s^2

        Returns:
            Dictionary with causality check results
        """
        results = {
            "eigenvalue_check": True,
            "characteristic_speed_check": True,
            "stability_check": True,
            "overall_causality": True,
        }

        # Check 1: Eigenvalue analysis for characteristic speeds
        eigenvals = (
            np.linalg.eigvals(self.components) if isinstance(self.components, np.ndarray) else None
        )

        if eigenvals is not None:
            # Check that viscous corrections don't dominate background
            max_eigenval = np.max(np.real(eigenvals))

            # Viscous effects should be small compared to background energy scale
            if isinstance(energy_density, float | int) and energy_density > 0:
                viscous_to_energy_ratio = abs(max_eigenval) / energy_density
                if viscous_to_energy_ratio > 1.0:  # Viscous effects too large
                    results["eigenvalue_check"] = False

            # Check characteristic speeds don't exceed light speed
            # For small perturbations: v_char^2 ≈ c_s^2 + (viscous corrections)
            if isinstance(sound_speed, float | int):
                effective_speed_squared = sound_speed + max_eigenval / energy_density
                if effective_speed_squared > 1.0:  # Exceeds speed of light
                    results["characteristic_speed_check"] = False

        # Check 2: Stability condition (simplified)
        # Extract bulk viscous part
        bulk_viscous = self.bulk_part(four_velocity)
        if isinstance(bulk_viscous, float | int):
            # Bulk viscous pressure shouldn't be too large compared to background pressure
            if isinstance(pressure, float | int) and pressure > 0:
                if abs(bulk_viscous) > pressure:  # Unstable regime
                    results["stability_check"] = False

        # Overall causality assessment
        results["overall_causality"] = all(
            [
                results["eigenvalue_check"],
                results["characteristic_speed_check"],
                results["stability_check"],
            ]
        )

        return results
