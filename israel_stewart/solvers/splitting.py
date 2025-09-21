"""
Operator splitting methods for Israel-Stewart hydrodynamics.

This module implements operator splitting techniques to decompose the full
Israel-Stewart system into separate hyperbolic (transport) and parabolic
(relaxation) sub-problems, enabling specialized numerical methods for each.
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Optional, Union

import numpy as np

from ..core.fields import ISFieldConfiguration, TransportCoefficients
from ..core.metrics import MetricBase
from ..core.performance import monitor_performance
from ..core.spacetime_grid import SpacetimeGrid

# Enhanced physics integration
try:
    from ..equations.conservation import ConservationLaws
    from ..equations.relaxation import ISRelaxationEquations

    PHYSICS_AVAILABLE = True
except ImportError:
    PHYSICS_AVAILABLE = False


class OperatorSplittingBase(ABC):
    """
    Abstract base class for operator splitting methods.

    Defines the interface for splitting the Israel-Stewart system into
    sub-problems that can be solved with specialized numerical methods.
    """

    def __init__(
        self,
        grid: SpacetimeGrid,
        metric: MetricBase,
        coefficients: TransportCoefficients,
        hyperbolic_solver: Callable | None = None,
        relaxation_solver: Callable | None = None,
    ):
        """
        Initialize operator splitting scheme.

        Args:
            grid: Spacetime grid
            metric: Spacetime metric
            coefficients: Transport coefficients
            hyperbolic_solver: Solver for hyperbolic sub-problem
            relaxation_solver: Solver for relaxation sub-problem
        """
        self.grid = grid
        self.metric = metric
        self.coefficients = coefficients
        self.hyperbolic_solver = hyperbolic_solver or self._default_hyperbolic_solver
        self.relaxation_solver = relaxation_solver or self._default_relaxation_solver

        # Performance tracking
        self.hyperbolic_times: list[float] = []
        self.relaxation_times: list[float] = []
        self.splitting_errors: list[float] = []

    @abstractmethod
    def advance_timestep(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """
        Advance fields by one timestep using operator splitting.

        Args:
            fields: Current field configuration
            dt: Timestep size

        Returns:
            Updated field configuration
        """
        pass

    @abstractmethod
    def estimate_splitting_error(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> float:
        """
        Estimate the splitting error for adaptive timestep control.

        Args:
            fields: Current field configuration
            dt: Timestep size

        Returns:
            Estimated splitting error
        """
        pass

    def _default_hyperbolic_solver(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """
        Default hyperbolic solver using Lax-Friedrichs finite volume method.

        Solves the conservation law ∂u/∂t + ∇·F(u) = 0 using the divergence
        of the stress-energy tensor from Israel-Stewart theory.

        Args:
            fields: Current field configuration
            dt: Time step size

        Returns:
            Updated field configuration after hyperbolic evolution
        """
        if PHYSICS_AVAILABLE:
            return self._lax_friedrichs_hyperbolic_solver(fields, dt)
        else:
            return self._fallback_hyperbolic_solver(fields, dt)

    def _lax_friedrichs_hyperbolic_solver(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """
        Lax-Friedrichs finite volume solver for conservation laws.

        Uses the stress-energy tensor divergence to evolve conserved quantities
        according to ∂_t T^μν + ∇_μ T^μν = 0.
        """
        result = fields.copy()

        try:
            # Initialize conservation laws
            conservation = ConservationLaws(fields)

            # Compute stress-energy tensor divergence
            div_T = conservation.divergence_T()

            # Apply Lax-Friedrichs update with stability constraint
            max_speed = self._estimate_maximum_characteristic_speed(fields)

            # Compute CFL timestep with robust grid spacing access
            if hasattr(self.grid, "spatial_spacing") and self.grid.spatial_spacing:
                min_dx = min(self.grid.spatial_spacing)
            else:
                # Fallback: compute from grid definition
                min_dx = min(
                    [
                        (r[1] - r[0]) / max(1, n - 1)
                        for r, n in zip(self.grid.spatial_ranges, self.grid.grid_points[1:])
                    ]
                )

            cfl_dt = min(dt, 0.5 * min_dx / max_speed)

            # Update conserved quantities: ∂_t u = -∇·F(u)
            if hasattr(div_T, "shape") and len(div_T.shape) >= 2:
                # Energy-momentum conservation: ∂_t T^0ν = -∇_i T^iν
                if div_T.shape[-1] >= 4:
                    # Energy density evolution (ν=0 component)
                    energy_source = -div_T[..., 0]
                    result.rho += cfl_dt * energy_source

                    # Momentum density evolution (ν=i components)
                    momentum_source = -div_T[..., 1:4]

                    # Apply momentum conservation with proper relativistic treatment
                    # Current momentum density: T^0i = (ρ + p) γ u^i
                    enthalpy = result.rho + result.pressure
                    gamma = result.u_mu[..., 0]

                    # Ensure proper broadcasting for momentum calculation
                    enthalpy_gamma = enthalpy * gamma
                    current_momentum = enthalpy_gamma[..., np.newaxis] * result.u_mu[..., 1:4]

                    # Update momentum density using conservation law
                    # Ensure momentum_source has compatible shape
                    if momentum_source.shape != current_momentum.shape:
                        # Handle potential shape mismatch
                        momentum_source = np.broadcast_to(momentum_source, current_momentum.shape)

                    updated_momentum = current_momentum + cfl_dt * momentum_source

                    # Convert back to four-velocity components
                    # u^i = p^i / ((ρ + p) γ), handle division by zero safely
                    safe_enthalpy_gamma = np.maximum(enthalpy_gamma, 1e-15)
                    result.u_mu[..., 1:4] = updated_momentum / safe_enthalpy_gamma[..., np.newaxis]

                    # Renormalize four-velocity to maintain u^μ u_μ = -1
                    self._renormalize_four_velocity(result)

            # Ensure physical constraints
            result.rho[:] = np.maximum(result.rho, 1e-12)  # Positive energy density
            result.pressure[:] = result.rho / 3.0  # Ideal gas relation

        except Exception as e:
            warnings.warn(f"Physics-based hyperbolic solver failed: {e}", UserWarning, stacklevel=2)
            # Fall back to simple method
            result = self._fallback_hyperbolic_solver(fields, dt)

        return result

    def _fallback_hyperbolic_solver(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """
        Fallback hyperbolic solver when physics modules unavailable.

        Uses simple expansion dynamics as approximation to conservation laws.
        """
        result = fields.copy()

        # Simple Bjorken-like expansion
        expansion_time = 1.0  # Characteristic expansion time
        expansion_rate = -1.0 / expansion_time

        # Adiabatic evolution: ρ ∝ T^4, T ∝ a^-1, a ∝ t^1/3
        time_factor = 1.0 + expansion_rate * dt / 3.0
        result.rho *= time_factor**4
        result.pressure[:] = result.rho / 3.0  # Conformal equation of state

        return result

    def _renormalize_four_velocity(self, fields: ISFieldConfiguration) -> None:
        """
        Renormalize four-velocity to satisfy relativistic constraint u^μ u_μ = -1.

        The four-velocity components are related by u^i = γ v^i where v^i is the
        three-velocity and γ is the Lorentz factor. For Minkowski metric (+,-,-,-):
        u_μ u^μ = (u^0)² - (u^1)² - (u^2)² - (u^3)² = -1

        Args:
            fields: Field configuration with potentially unnormalized four-velocity
        """
        # Extract spatial components of four-velocity u^i = γ v^i
        u_spatial = fields.u_mu[..., 1:4]

        # Compute three-velocity magnitude squared |v|² from u^i = γ v^i
        # We need to solve: u^i = γ v^i and γ = √(1 + |v|²)
        # This gives: |u^i|² = γ² |v|² = (1 + |v|²) |v|²
        u_spatial_sq = np.sum(u_spatial**2, axis=-1)

        # Solve for |v|² from |u^i|² = γ² |v|² where γ² = 1 + |v|²
        # |u^i|² = (1 + |v|²) |v|² → |v|² = |u^i|² / (1 + |v|²)
        # Rearranging: |v|²(1 + |v|²) = |u^i|² → |v|² + |v|⁴ = |u^i|²
        # Quadratic in |v|²: |v|⁴ + |v|² - |u^i|² = 0
        # Solution: |v|² = (-1 + √(1 + 4|u^i|²)) / 2

        # For small velocities, use approximation to avoid numerical issues
        v_sq = np.where(
            u_spatial_sq < 0.01,  # For small u^i, use direct approximation
            u_spatial_sq,  # |v|² ≈ |u^i|² when γ ≈ 1
            (-1.0 + np.sqrt(1.0 + 4.0 * u_spatial_sq)) / 2.0,  # Exact solution
        )

        # Ensure physical bounds: 0 ≤ |v|² < 1
        v_sq = np.clip(v_sq, 0.0, 0.999999)

        # Compute Lorentz factor γ = √(1 + |v|²)
        gamma = np.sqrt(1.0 + v_sq)

        # Compute three-velocity v^i = u^i / γ
        gamma_safe = np.maximum(gamma, 1e-15)  # Avoid division by zero
        v_spatial = u_spatial / gamma_safe[..., np.newaxis]

        # Renormalize four-velocity: u^0 = γ, u^i = γ v^i
        fields.u_mu[..., 0] = gamma
        fields.u_mu[..., 1:4] = gamma_safe[..., np.newaxis] * v_spatial

    def _estimate_maximum_characteristic_speed(self, fields: ISFieldConfiguration) -> float:
        """
        Estimate maximum characteristic speed for CFL condition.

        Returns the sound speed for relativistic hydrodynamics.
        """
        # Speed of sound for conformal fluid
        sound_speed = 1.0 / np.sqrt(3.0)

        # In relativistic case, maximum speed is bounded by speed of light
        max_speed = min(sound_speed, 1.0)  # c = 1 in natural units

        return float(max_speed)

    def _default_relaxation_solver(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """
        Default relaxation solver using Israel-Stewart relaxation equations.

        Integrates the complete relaxation dynamics including all coupling terms
        and nonlinear effects from the ISRelaxationEquations module.

        Args:
            fields: Current field configuration
            dt: Time step size

        Returns:
            Updated field configuration after relaxation evolution
        """
        if PHYSICS_AVAILABLE:
            return self._physics_based_relaxation_solver(fields, dt)
        else:
            return self._exponential_relaxation_solver(fields, dt)

    def _physics_based_relaxation_solver(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """
        Physics-based relaxation solver using ISRelaxationEquations.

        Solves the complete Israel-Stewart relaxation equations with all
        coupling terms and second-order corrections.
        """
        result = fields.copy()

        try:
            # Initialize relaxation equations
            relaxation = ISRelaxationEquations(self.grid, self.metric, self.coefficients)

            # Use implicit method for stiff relaxation equations
            relaxation.evolve_relaxation(result, dt, method="implicit")

        except Exception as e:
            warnings.warn(f"Physics-based relaxation solver failed: {e}", UserWarning, stacklevel=2)
            # Fall back to exponential integrator
            result = self._exponential_relaxation_solver(fields, dt)

        return result

    def _exponential_relaxation_solver(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """
        Exponential integrator for linear relaxation terms.

        Solves ∂u/∂t = -u/τ exactly for the linear relaxation part.
        """
        result = fields.copy()

        # Apply exponential relaxation to dissipative fluxes
        if hasattr(result, "Pi") and result.Pi is not None:
            tau_Pi = self.coefficients.bulk_relaxation_time or 0.1
            relaxation_factor = np.exp(-dt / tau_Pi)
            result.Pi *= relaxation_factor

        if hasattr(result, "pi_munu") and result.pi_munu is not None:
            tau_pi = self.coefficients.shear_relaxation_time or 0.1
            relaxation_factor = np.exp(-dt / tau_pi)
            result.pi_munu *= relaxation_factor

        if hasattr(result, "q_mu") and result.q_mu is not None:
            tau_q = getattr(self.coefficients, "heat_relaxation_time", 0.1) or 0.1
            relaxation_factor = np.exp(-dt / tau_q)
            result.q_mu *= relaxation_factor

        # Add viscous source terms for first-order contributions
        self._add_viscous_sources(result, dt)

        return result

    def _add_viscous_sources(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> None:
        """
        Add viscous source terms to relaxation evolution.

        Computes proper expansion scalar and shear tensor from velocity field
        to calculate realistic viscous sources: -ζ*θ and -η*σ^μν.
        """
        # Bulk viscosity source: -ζ * θ where θ is expansion scalar
        if hasattr(fields, "Pi") and fields.Pi is not None:
            # Compute actual expansion scalar θ = ∇_μ u^μ
            expansion_scalar = self._compute_expansion_scalar(fields)
            bulk_source = -self.coefficients.bulk_viscosity * expansion_scalar
            fields.Pi += dt * bulk_source

        # Shear viscosity source: -η * σ^μν where σ^μν is shear tensor magnitude
        if hasattr(fields, "pi_munu") and fields.pi_munu is not None:
            # Compute shear tensor magnitude |σ^μν|
            shear_magnitude = self._compute_shear_magnitude(fields)
            shear_source = -self.coefficients.shear_viscosity * shear_magnitude
            fields.pi_munu += dt * shear_source

    def _compute_expansion_scalar(self, fields: ISFieldConfiguration) -> float:
        """
        Compute expansion scalar θ = ∇_μ u^μ from velocity field.

        For a Minkowski metric in Cartesian coordinates:
        θ = ∂_t u^0 + ∂_i u^i (using Einstein summation)
        """
        # Simple finite difference approximation for ∇_μ u^μ
        # For uniform grid, use central differences
        try:
            if hasattr(self.grid, "spatial_spacing") and self.grid.spatial_spacing:
                dx = min(self.grid.spatial_spacing)
            else:
                # Fallback: estimate spacing from grid ranges
                dx = min(
                    [
                        (r[1] - r[0]) / max(1, n - 1)
                        for r, n in zip(self.grid.spatial_ranges, self.grid.grid_points[1:])
                    ]
                )

            # Compute spatial divergence ∂_i u^i using finite differences
            u_spatial = fields.u_mu[..., 1:4]
            divergence = 0.0

            for i in range(3):  # Sum over spatial dimensions
                # Central differences: ∂_i u^i ≈ (u^i_{+1} - u^i_{-1}) / (2*dx)
                u_component = u_spatial[..., i]
                if u_component.ndim >= 3:  # 3D grid
                    # Roll to compute differences (periodic boundary approximation)
                    u_plus = np.roll(u_component, -1, axis=i)
                    u_minus = np.roll(u_component, 1, axis=i)
                    divergence += float(np.mean((u_plus - u_minus) / (2.0 * dx)))

            return float(divergence)

        except Exception:
            # Fallback: simple approximation based on velocity magnitude
            u_magnitude = np.sqrt(np.sum(fields.u_mu[..., 1:4] ** 2, axis=-1))
            return float(np.mean(u_magnitude) / 10.0)  # Rough estimate

    def _compute_shear_magnitude(self, fields: ISFieldConfiguration) -> float:
        """
        Compute characteristic shear tensor magnitude |σ^μν|.

        The shear tensor is σ^μν = ∇^⟨μ u^ν⟩ (symmetric traceless part).
        For simplicity, estimate from velocity gradients.
        """
        try:
            if hasattr(self.grid, "spatial_spacing") and self.grid.spatial_spacing:
                dx = min(self.grid.spatial_spacing)
            else:
                dx = min(
                    [
                        (r[1] - r[0]) / max(1, n - 1)
                        for r, n in zip(self.grid.spatial_ranges, self.grid.grid_points[1:])
                    ]
                )

            # Compute velocity gradients and estimate shear
            u_spatial = fields.u_mu[..., 1:4]
            shear_components = []

            for i in range(3):
                for j in range(3):
                    if i <= j:  # Only compute upper triangle (symmetric)
                        u_i = u_spatial[..., i]
                        if u_i.ndim >= 3:
                            # Compute ∂_j u^i
                            u_plus = np.roll(u_i, -1, axis=j)
                            u_minus = np.roll(u_i, 1, axis=j)
                            grad_ij = (u_plus - u_minus) / (2.0 * dx)

                            # Add to shear (symmetric part minus trace)
                            if i == j:
                                grad_ij -= np.mean(grad_ij)  # Remove trace
                            shear_components.append(grad_ij)

            if shear_components:
                # Return characteristic magnitude
                shear_tensor_sq = sum(np.mean(comp**2) for comp in shear_components)
                return float(np.sqrt(shear_tensor_sq))
            else:
                return 0.1  # Fallback

        except Exception:
            # Fallback: estimate from velocity gradients
            u_spatial = fields.u_mu[..., 1:4]
            velocity_scale = np.sqrt(np.mean(u_spatial**2))
            return float(velocity_scale * 0.1)  # Rough shear estimate

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for the splitting method."""
        if not self.hyperbolic_times:
            return {"status": "no_timesteps_taken"}

        return {
            "total_timesteps": len(self.hyperbolic_times),
            "avg_hyperbolic_time": np.mean(self.hyperbolic_times),
            "avg_relaxation_time": np.mean(self.relaxation_times),
            "avg_splitting_error": np.mean(self.splitting_errors) if self.splitting_errors else 0.0,
            "max_splitting_error": np.max(self.splitting_errors) if self.splitting_errors else 0.0,
        }

    @abstractmethod
    def advance_timestep_no_error(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """Advance timestep without error estimation (to avoid recursion)."""
        pass


class StrangSplitting(OperatorSplittingBase):
    """
    Strang (symmetric) operator splitting.

    Second-order accurate splitting scheme:
    u^{n+1} = S_{R}(dt/2) ∘ S_{H}(dt) ∘ S_{R}(dt/2) u^n

    where S_H is the hyperbolic operator and S_R is the relaxation operator.
    """

    def __init__(
        self,
        grid: SpacetimeGrid,
        metric: MetricBase,
        coefficients: TransportCoefficients,
        hyperbolic_solver: Callable | None = None,
        relaxation_solver: Callable | None = None,
        error_estimation: bool = True,
    ):
        """
        Initialize Strang splitting scheme.

        Args:
            grid: Spacetime grid
            metric: Spacetime metric
            coefficients: Transport coefficients
            hyperbolic_solver: Solver for hyperbolic sub-problem
            relaxation_solver: Solver for relaxation sub-problem
            error_estimation: Whether to estimate splitting errors
        """
        super().__init__(grid, metric, coefficients, hyperbolic_solver, relaxation_solver)
        self.error_estimation = error_estimation

    @monitor_performance("strang_splitting_timestep")
    def advance_timestep(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """
        Advance one timestep using Strang splitting.

        The Strang splitting applies operators in the sequence:
        1. Relaxation for dt/2
        2. Hyperbolic for dt
        3. Relaxation for dt/2
        """
        import time

        # Step 1: Relaxation for dt/2
        start_time = time.time()
        intermediate1 = self.relaxation_solver(fields, dt / 2)
        self.relaxation_times.append(time.time() - start_time)

        # Step 2: Hyperbolic for dt
        start_time = time.time()
        intermediate2 = self.hyperbolic_solver(intermediate1, dt)
        self.hyperbolic_times.append(time.time() - start_time)

        # Step 3: Relaxation for dt/2
        start_time = time.time()
        result = self.relaxation_solver(intermediate2, dt / 2)
        self.relaxation_times.append(time.time() - start_time)

        # Estimate splitting error if requested
        if self.error_estimation:
            error = self.estimate_splitting_error(fields, dt)
            self.splitting_errors.append(error)

        return result

    def estimate_splitting_error(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> float:
        """
        Estimate splitting error using Lie-Trotter vs Strang comparison.

        Compares second-order Strang splitting with first-order Lie-Trotter
        to estimate the leading-order splitting error. More efficient than
        Richardson extrapolation.

        Args:
            fields: Current field configuration
            dt: Time step size

        Returns:
            Estimated splitting error magnitude
        """
        # Use efficient Lie-Trotter vs Strang comparison
        return self._estimate_error_strang_vs_lietrotter(fields, dt)

    def _estimate_error_strang_vs_lietrotter(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> float:
        """
        Estimate error by comparing Strang and Lie-Trotter methods.

        The difference between second-order Strang and first-order Lie-Trotter
        provides an estimate of the leading-order splitting error.
        """
        # Strang splitting result (this method)
        strang_result = self.advance_timestep_no_error(fields, dt)

        # Lie-Trotter splitting result
        lietrotter_result = self._lietrotter_step(fields, dt)

        # Compute error in primary fields
        error_rho = np.max(np.abs(strang_result.rho - lietrotter_result.rho))

        # Include dissipative flux errors
        error_dissipative = 0.0
        if hasattr(strang_result, "Pi") and strang_result.Pi is not None:
            error_dissipative += np.max(np.abs(strang_result.Pi - lietrotter_result.Pi))

        if hasattr(strang_result, "pi_munu") and strang_result.pi_munu is not None:
            error_dissipative += np.max(np.abs(strang_result.pi_munu - lietrotter_result.pi_munu))

        # Total error estimate
        total_error = error_rho + error_dissipative

        return float(total_error)

    def _lietrotter_step(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """
        Perform single Lie-Trotter step for error estimation.

        Uses H-R ordering (hyperbolic then relaxation) for first-order method.
        """
        # Hyperbolic step
        intermediate = self.hyperbolic_solver(fields, dt)

        # Relaxation step
        result = self.relaxation_solver(intermediate, dt)

        return result

    def advance_timestep_no_error(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """Advance timestep without error estimation (to avoid recursion)."""
        # Relaxation for dt/2
        intermediate1 = self.relaxation_solver(fields, dt / 2)

        # Hyperbolic for dt
        intermediate2 = self.hyperbolic_solver(intermediate1, dt)

        # Relaxation for dt/2
        result = self.relaxation_solver(intermediate2, dt / 2)

        return result


class LieTrotterSplitting(OperatorSplittingBase):
    """
    Lie-Trotter (first-order) operator splitting.

    First-order accurate splitting scheme:
    u^{n+1} = S_{R}(dt) ∘ S_{H}(dt) u^n

    Simpler but less accurate than Strang splitting.
    """

    def __init__(
        self,
        grid: SpacetimeGrid,
        metric: MetricBase,
        coefficients: TransportCoefficients,
        hyperbolic_solver: Callable | None = None,
        relaxation_solver: Callable | None = None,
        order: str = "HR",  # "HR" = Hyperbolic then Relaxation, "RH" = Relaxation then Hyperbolic
    ):
        """
        Initialize Lie-Trotter splitting scheme.

        Args:
            grid: Spacetime grid
            metric: Spacetime metric
            coefficients: Transport coefficients
            hyperbolic_solver: Solver for hyperbolic sub-problem
            relaxation_solver: Solver for relaxation sub-problem
            order: Order of operations ("HR" or "RH")
        """
        super().__init__(grid, metric, coefficients, hyperbolic_solver, relaxation_solver)
        self.order = order

    @monitor_performance("lietrotter_splitting_timestep")
    def advance_timestep(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """Advance one timestep using Lie-Trotter splitting."""
        import time

        if self.order == "HR":
            # Hyperbolic first, then relaxation
            start_time = time.time()
            intermediate = self.hyperbolic_solver(fields, dt)
            self.hyperbolic_times.append(time.time() - start_time)

            start_time = time.time()
            result = self.relaxation_solver(intermediate, dt)
            self.relaxation_times.append(time.time() - start_time)

        elif self.order == "RH":
            # Relaxation first, then hyperbolic
            start_time = time.time()
            intermediate = self.relaxation_solver(fields, dt)
            self.relaxation_times.append(time.time() - start_time)

            start_time = time.time()
            result = self.hyperbolic_solver(intermediate, dt)
            self.hyperbolic_times.append(time.time() - start_time)

        else:
            raise ValueError(f"Unknown order: {self.order}. Use 'HR' or 'RH'.")

        return result

    def advance_timestep_no_error(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """Advance timestep without error estimation (to avoid recursion)."""
        # Use the same logic as advance_timestep but skip error tracking
        intermediate = fields

        if self.order == "HR":
            # Hyperbolic step
            intermediate = self.hyperbolic_solver(intermediate, dt)
            # Relaxation step
            result = self.relaxation_solver(intermediate, dt)
        elif self.order == "RH":
            # Relaxation step
            intermediate = self.relaxation_solver(intermediate, dt)
            # Hyperbolic step
            result = self.hyperbolic_solver(intermediate, dt)
        else:
            raise ValueError(f"Unknown order: {self.order}. Use 'HR' or 'RH'.")

        return result

    def estimate_splitting_error(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> float:
        """
        Estimate splitting error for Lie-Trotter method.

        Compare HR vs RH orderings to estimate commutator error.
        """
        # Solve with HR ordering
        current_order = self.order
        self.order = "HR"
        solution_hr = self.advance_timestep(fields, dt)

        # Solve with RH ordering
        self.order = "RH"
        solution_rh = self.advance_timestep(fields, dt)

        # Restore original ordering
        self.order = current_order

        # Estimate error from commutator [H, R]
        error = np.max(np.abs(solution_hr.rho - solution_rh.rho))

        if hasattr(solution_hr, "Pi") and solution_hr.Pi is not None:
            error += np.max(np.abs(solution_hr.Pi - solution_rh.Pi))

        return float(error)


class AdaptiveSplitting(OperatorSplittingBase):
    """
    Adaptive operator splitting with automatic timestep control.

    Adjusts timestep based on splitting error estimates and
    switches between different splitting methods based on problem characteristics.
    """

    def __init__(
        self,
        grid: SpacetimeGrid,
        metric: MetricBase,
        coefficients: TransportCoefficients,
        hyperbolic_solver: Callable | None = None,
        relaxation_solver: Callable | None = None,
        tolerance: float = 1e-6,
        max_timestep: float = 1e-2,
        min_timestep: float = 1e-8,
        safety_factor: float = 0.8,
    ):
        """
        Initialize adaptive splitting scheme.

        Args:
            grid: Spacetime grid
            metric: Spacetime metric
            coefficients: Transport coefficients
            hyperbolic_solver: Solver for hyperbolic sub-problem
            relaxation_solver: Solver for relaxation sub-problem
            tolerance: Error tolerance for timestep control
            max_timestep: Maximum allowed timestep
            min_timestep: Minimum allowed timestep
            safety_factor: Safety factor for timestep adjustment
        """
        super().__init__(grid, metric, coefficients, hyperbolic_solver, relaxation_solver)

        self.tolerance = tolerance
        self.max_timestep = max_timestep
        self.min_timestep = min_timestep
        self.safety_factor = safety_factor

        # Initialize with Strang splitting
        self.strang_splitter = StrangSplitting(
            grid, metric, coefficients, hyperbolic_solver, relaxation_solver
        )
        self.lietrotter_splitter = LieTrotterSplitting(
            grid, metric, coefficients, hyperbolic_solver, relaxation_solver
        )

        # Adaptive control variables
        self.current_timestep = max_timestep / 10
        self.accepted_steps = 0
        self.rejected_steps = 0

    @monitor_performance("adaptive_splitting_timestep")
    def advance_timestep(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """
        Advance timestep with adaptive control.

        Automatically adjusts timestep based on error estimates.
        """
        current_dt = min(dt, self.current_timestep)
        time_remaining = dt
        result = fields.copy()

        while time_remaining > 1e-12:
            # Attempt step with current timestep
            step_dt = min(current_dt, time_remaining)

            # Choose splitting method based on problem characteristics
            splitter = self._choose_splitting_method(result, step_dt)

            # Estimate error
            error = splitter.estimate_splitting_error(result, step_dt)

            if error < self.tolerance:
                # Accept step
                result = splitter.advance_timestep_no_error(result, step_dt)
                time_remaining -= step_dt
                self.accepted_steps += 1

                # Update timestep for next iteration
                if error > 0:
                    factor = self.safety_factor * (self.tolerance / error) ** 0.2
                    current_dt = min(self.max_timestep, factor * current_dt)
                else:
                    current_dt = min(self.max_timestep, 1.1 * current_dt)

            else:
                # Reject step and reduce timestep
                self.rejected_steps += 1
                factor = self.safety_factor * (self.tolerance / error) ** 0.25
                current_dt = max(self.min_timestep, factor * current_dt)

                if current_dt < self.min_timestep:
                    warnings.warn(
                        f"Timestep reduced below minimum: {current_dt}. "
                        f"Error: {error}, Tolerance: {self.tolerance}",
                        UserWarning,
                        stacklevel=2,
                    )
                    break

        # Update stored timestep
        self.current_timestep = current_dt

        return result

    def advance_timestep_no_error(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """Advance timestep without error estimation (to avoid recursion)."""
        # Choose method and advance timestep
        method = self._choose_splitting_method(fields, dt)
        return method.advance_timestep_no_error(fields, dt)

    def _choose_splitting_method(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> OperatorSplittingBase:
        """
        Choose splitting method based on problem stiffness.

        Strategy:
        - Non-stiff (dt >> τ): Use Strang splitting for higher accuracy
        - Stiff (dt ~ τ): Use Lie-Trotter for better stability and robustness

        This follows standard practice where simple, robust methods are preferred
        for stiff problems, while accurate methods are used for smooth problems.
        """
        # Estimate stiffness from relaxation timescales
        tau_pi = self.coefficients.shear_relaxation_time or 0.1
        tau_Pi = self.coefficients.bulk_relaxation_time or 0.1
        tau_q = getattr(self.coefficients, "heat_relaxation_time", 0.1) or 0.1

        # Find the fastest (most restrictive) timescale
        min_relaxation_time = min(tau_pi, tau_Pi, tau_q)
        stiffness_ratio = dt / min_relaxation_time

        # Additional stiffness indicators from field gradients
        try:
            # Check for rapid field variations that indicate stiffness
            rho_variation = (
                np.std(fields.rho) / np.mean(fields.rho) if np.mean(fields.rho) > 0 else 0
            )
            u_variation = np.std(fields.u_mu) / (np.mean(np.abs(fields.u_mu)) + 1e-15)

            # If fields vary rapidly, problem is likely stiff
            variation_stiffness = max(rho_variation, u_variation)

            # Combined stiffness measure
            is_stiff = (stiffness_ratio <= 0.1) or (variation_stiffness > 1.0)

        except Exception:
            # Fallback to simple timescale-based detection
            is_stiff = stiffness_ratio <= 0.1

        if is_stiff:
            # Use robust Lie-Trotter for stiff problems
            return self.lietrotter_splitter
        else:
            # Use accurate Strang for non-stiff problems
            return self.strang_splitter

    def estimate_splitting_error(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> float:
        """Estimate splitting error using the chosen method."""
        splitter = self._choose_splitting_method(fields, dt)
        return splitter.estimate_splitting_error(fields, dt)

    def get_adaptive_stats(self) -> dict[str, Any]:
        """Get statistics for adaptive timestep control."""
        total_steps = self.accepted_steps + self.rejected_steps
        acceptance_rate = self.accepted_steps / total_steps if total_steps > 0 else 0.0

        return {
            "current_timestep": self.current_timestep,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "acceptance_rate": acceptance_rate,
            "total_attempts": total_steps,
        }


class PhysicsBasedSplitting(OperatorSplittingBase):
    """
    Physics-based operator splitting for Israel-Stewart equations.

    Splits the system according to physical timescales:
    - Fast: Relaxation of dissipative fluxes
    - Medium: Hydrodynamic transport
    - Slow: Thermodynamic evolution

    This allows optimal treatment of multi-scale physics.
    """

    def __init__(
        self,
        grid: SpacetimeGrid,
        metric: MetricBase,
        coefficients: TransportCoefficients,
        hyperbolic_solver: Callable | None = None,
        relaxation_solver: Callable | None = None,
        thermodynamic_solver: Callable | None = None,
        scale_separation_threshold: float = 10.0,
    ):
        """
        Initialize physics-based splitting.

        Args:
            grid: Spacetime grid
            metric: Spacetime metric
            coefficients: Transport coefficients
            hyperbolic_solver: Solver for hyperbolic transport
            relaxation_solver: Solver for dissipative relaxation
            thermodynamic_solver: Solver for thermodynamic evolution
            scale_separation_threshold: Minimum ratio for scale separation
        """
        super().__init__(grid, metric, coefficients, hyperbolic_solver, relaxation_solver)

        self.thermodynamic_solver = thermodynamic_solver or self._default_thermodynamic_solver
        self.scale_threshold = scale_separation_threshold

        # Timescale analysis
        self.timescales = self._analyze_timescales()

    def _analyze_timescales(self) -> dict[str, float]:
        """Analyze characteristic timescales in the problem."""
        # Relaxation timescales
        tau_pi = self.coefficients.shear_relaxation_time or 0.1
        tau_Pi = self.coefficients.bulk_relaxation_time or 0.1
        tau_q = getattr(self.coefficients, "heat_relaxation_time", 0.1) or 0.1

        # Hydrodynamic timescale (sound crossing time)
        if hasattr(self.grid, "spatial_spacing") and self.grid.spatial_spacing:
            L_char = min(self.grid.spatial_spacing)  # Characteristic length scale
        else:
            # Fallback: estimate from grid ranges and points
            L_char = min(
                [
                    (r[1] - r[0]) / max(1, n - 1)
                    for r, n in zip(self.grid.spatial_ranges, self.grid.grid_points[1:])
                ]
            )

        cs = 1.0 / np.sqrt(3.0)  # Speed of sound
        tau_hydro = L_char / cs

        # Thermodynamic timescale (expansion time)
        tau_thermo = 1.0  # Approximate expansion timescale

        return {
            "relaxation_min": min(tau_pi, tau_Pi, tau_q),
            "relaxation_max": max(tau_pi, tau_Pi, tau_q),
            "hydrodynamic": tau_hydro,
            "thermodynamic": tau_thermo,
        }

    def _default_thermodynamic_solver(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """
        Thermodynamic evolution solver using conservation laws.

        Solves the energy-momentum conservation equations with proper
        thermodynamic consistency and equation of state coupling.

        Args:
            fields: Current field configuration
            dt: Time step size

        Returns:
            Updated field configuration after thermodynamic evolution
        """
        if PHYSICS_AVAILABLE:
            return self._conservation_based_thermodynamic_solver(fields, dt)
        else:
            return self._expansion_cooling_solver(fields, dt)

    def _conservation_based_thermodynamic_solver(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """
        Thermodynamic solver based on energy-momentum conservation.

        Uses the conservation laws module to ensure proper thermodynamic
        evolution with equation of state consistency.
        """
        result = fields.copy()

        try:
            # Initialize conservation laws
            conservation = ConservationLaws(fields)

            # Validate conservation before evolution
            conservation_check = conservation.validate_conservation()

            # Apply thermodynamic consistency
            self._enforce_thermodynamic_consistency(result)

            # Compute expansion cooling from metric
            expansion_rate = self._compute_expansion_rate(result)

            # Adiabatic evolution for ideal gas: T ∝ ρ^(γ-1)/γ
            gamma = 4.0 / 3.0  # Relativistic ideal gas
            temperature_evolution = -(gamma - 1) * expansion_rate / gamma

            # Update thermodynamic variables
            if hasattr(result, "temperature"):
                result.temperature *= np.exp(temperature_evolution * dt)

            # Energy density evolution: ρ ∝ T^4
            result.rho *= np.exp(4 * temperature_evolution * dt)

            # Pressure from equation of state
            result.pressure[:] = result.rho / 3.0  # Conformal equation of state

            # Ensure physical bounds
            result.rho[:] = np.maximum(result.rho, 1e-12)
            result.pressure[:] = np.maximum(result.pressure, 0.0)

        except Exception as e:
            warnings.warn(
                f"Conservation-based thermodynamic solver failed: {e}", UserWarning, stacklevel=2
            )
            result = self._expansion_cooling_solver(fields, dt)

        return result

    def _expansion_cooling_solver(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """
        Fallback thermodynamic solver using expansion cooling.

        Computes expansion rate from actual velocity field rather than
        using hardcoded values. Implements proper relativistic cooling.
        """
        result = fields.copy()

        # Compute actual expansion rate from velocity field
        expansion_rate = self._compute_expansion_rate(fields)

        # Relativistic adiabatic evolution: T ∝ τ^(-1/3) for conformal fluid
        # where τ is proper time and expansion drives cooling
        cooling_rate = expansion_rate / 3.0  # 1/3 from conformal equation of state

        # Temperature evolution: T ∝ exp(-∫ θ/3 dt) where θ is expansion scalar
        if hasattr(result, "temperature"):
            result.temperature *= np.exp(-cooling_rate * dt)

        # Energy density: ρ ∝ T^4 for massless particles
        result.rho *= np.exp(-4.0 * cooling_rate * dt)

        # Pressure: p = ρ/3 for conformal equation of state
        result.pressure[:] = result.rho / 3.0

        return result

    def _enforce_thermodynamic_consistency(self, fields: ISFieldConfiguration) -> None:
        """
        Enforce thermodynamic consistency conditions.

        Ensures that energy density, pressure, and temperature satisfy
        the equation of state and thermodynamic relations.
        """
        # Ideal gas relation: p = ρ/3 for massless particles
        fields.pressure[:] = fields.rho / 3.0

        # Stefan-Boltzmann relation: ρ = aT^4 for radiation
        if hasattr(fields, "temperature"):
            stefan_boltzmann_constant = 1.0  # Normalized units

            # If temperature is not initialized (all zeros), compute it from energy density
            if np.allclose(fields.temperature, 0.0):
                # T = (ρ/a)^(1/4) from Stefan-Boltzmann law
                fields.temperature[:] = np.power(fields.rho / stefan_boltzmann_constant, 0.25)
            else:
                # If temperature is set, enforce ρ = aT^4
                fields.rho[:] = stefan_boltzmann_constant * fields.temperature**4

            fields.pressure[:] = fields.rho / 3.0

    def _compute_expansion_rate(self, fields: ISFieldConfiguration) -> float:
        """
        Compute expansion rate from velocity field.

        Returns the expansion scalar θ = ∇_μ u^μ, which characterizes
        the rate of volume expansion of the fluid.
        """
        # Use the same computation as in _compute_expansion_scalar
        return self._compute_expansion_scalar(fields)

    @monitor_performance("physics_based_splitting")
    def advance_timestep(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """
        Advance timestep using physics-based multi-rate splitting.

        Implements proper nested splitting where fast processes are integrated
        within each step of slower processes: S_slow(dt/2) ∘ [nested fast/medium] ∘ S_slow(dt/2)
        """
        # Determine sub-stepping based on timescale ratios
        tau_fast = self.timescales["relaxation_min"]
        tau_medium = self.timescales["hydrodynamic"]
        tau_slow = self.timescales["thermodynamic"]

        # Determine number of substeps for each timescale
        n_medium = max(1, int(dt / (0.5 * tau_medium)))  # Medium substeps
        n_fast = max(1, int((dt / n_medium) / (0.1 * tau_fast)))  # Fast substeps per medium step

        result = fields.copy()

        # Strang-like splitting for slow process: S_slow(dt/2)
        result = self.thermodynamic_solver(result, dt / 2.0)

        # Nested medium-fast splitting
        dt_medium = dt / n_medium
        for _ in range(n_medium):
            # Strang-like splitting for medium process: S_medium(dt_m/2)
            result = self.hyperbolic_solver(result, dt_medium / 2.0)

            # Fast process integration within medium timestep
            dt_fast = dt_medium / n_fast
            for _ in range(n_fast):
                result = self.relaxation_solver(result, dt_fast)

            # Complete medium step: S_medium(dt_m/2)
            result = self.hyperbolic_solver(result, dt_medium / 2.0)

        # Complete slow step: S_slow(dt/2)
        result = self.thermodynamic_solver(result, dt / 2.0)

        return result

    def advance_timestep_no_error(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """Advance timestep without error estimation using nested splitting."""
        # Use the same nested splitting logic as advance_timestep
        tau_fast = self.timescales["relaxation_min"]
        tau_medium = self.timescales["hydrodynamic"]

        # Determine number of substeps for each timescale
        n_medium = max(1, int(dt / (0.5 * tau_medium)))
        n_fast = max(1, int((dt / n_medium) / (0.1 * tau_fast)))

        result = fields.copy()

        # Nested multi-rate splitting
        result = self.thermodynamic_solver(result, dt / 2.0)

        dt_medium = dt / n_medium
        for _ in range(n_medium):
            result = self.hyperbolic_solver(result, dt_medium / 2.0)

            dt_fast = dt_medium / n_fast
            for _ in range(n_fast):
                result = self.relaxation_solver(result, dt_fast)

            result = self.hyperbolic_solver(result, dt_medium / 2.0)

        result = self.thermodynamic_solver(result, dt / 2.0)

        return result

    def estimate_splitting_error(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> float:
        """Estimate error from physics-based splitting."""
        # Compare with uniform time stepping
        uniform_result = self._uniform_stepping(fields, dt)
        physics_result = self.advance_timestep(fields, dt)

        # Compute difference
        error_rho = np.max(np.abs(uniform_result.rho - physics_result.rho))
        error_temp = np.max(np.abs(uniform_result.temperature - physics_result.temperature))

        return float(error_rho + error_temp)

    def _uniform_stepping(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """Reference solution with uniform time stepping."""
        # Apply all operators with same timestep
        result = self.relaxation_solver(fields, dt)
        result = self.hyperbolic_solver(result, dt)
        result = self.thermodynamic_solver(result, dt)

        return result


# Factory function for creating splitting schemes
def create_splitting_solver(
    splitting_type: str,
    grid: SpacetimeGrid,
    metric: MetricBase,
    coefficients: TransportCoefficients,
    **kwargs: Any,
) -> OperatorSplittingBase:
    """
    Factory function to create operator splitting schemes.

    Args:
        splitting_type: Type of splitting ('strang', 'lietrotter', 'adaptive', 'physics')
        grid: Spacetime grid
        metric: Spacetime metric
        coefficients: Transport coefficients
        **kwargs: Additional splitting-specific parameters

    Returns:
        Configured operator splitting scheme

    Raises:
        ValueError: If splitting_type is not recognized
    """
    if splitting_type.lower() == "strang":
        return StrangSplitting(grid, metric, coefficients, **kwargs)
    elif splitting_type.lower() == "lietrotter":
        return LieTrotterSplitting(grid, metric, coefficients, **kwargs)
    elif splitting_type.lower() == "adaptive":
        return AdaptiveSplitting(grid, metric, coefficients, **kwargs)
    elif splitting_type.lower() == "physics":
        return PhysicsBasedSplitting(grid, metric, coefficients, **kwargs)
    else:
        raise ValueError(
            f"Unknown splitting type: {splitting_type}. "
            f"Available types: 'strang', 'lietrotter', 'adaptive', 'physics'"
        )


# Utility functions for common splitting patterns
def solve_hyperbolic_conservative(
    fields: ISFieldConfiguration,
    dt: float,
    finite_difference_scheme: Any,
) -> ISFieldConfiguration:
    """
    Solve hyperbolic conservation laws using finite difference methods.

    Args:
        fields: Current field configuration
        dt: Timestep
        finite_difference_scheme: Spatial discretization scheme

    Returns:
        Updated fields after hyperbolic evolution
    """
    # Placeholder implementation
    # In practice, this would implement the full conservation law evolution
    result = fields.copy()

    # Simple expansion evolution
    expansion_rate = -1.0 / max(fields.time if hasattr(fields, "time") else 1.0, 0.1)

    # Apply conservation law updates
    result.rho *= 1.0 + expansion_rate * dt / 3.0  # Adiabatic cooling
    result.pressure[:] = result.rho / 3.0  # Ideal gas relation

    return result


def solve_relaxation_exponential(
    fields: ISFieldConfiguration,
    dt: float,
    coefficients: TransportCoefficients,
) -> ISFieldConfiguration:
    """
    Solve relaxation equations using exponential time differencing.

    Args:
        fields: Current field configuration
        dt: Timestep
        coefficients: Transport coefficients

    Returns:
        Updated fields after relaxation evolution
    """
    result = fields.copy()

    # Exponential relaxation for dissipative quantities
    if hasattr(result, "Pi") and result.Pi is not None:
        tau_Pi = coefficients.bulk_relaxation_time or 0.1
        result.Pi *= np.exp(-dt / tau_Pi)

    if hasattr(result, "pi_munu") and result.pi_munu is not None:
        tau_pi = coefficients.shear_relaxation_time or 0.1
        result.pi_munu *= np.exp(-dt / tau_pi)

    if hasattr(result, "q_mu") and result.q_mu is not None:
        tau_q = getattr(coefficients, "heat_relaxation_time", None) or 0.1
        result.q_mu *= np.exp(-dt / tau_q)

    return result
