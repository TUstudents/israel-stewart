"""
Operator splitting methods for Israel-Stewart hydrodynamics.

This module implements operator splitting techniques to decompose the full
Israel-Stewart system into separate hyperbolic (transport) and parabolic
(relaxation) sub-problems, enabling specialized numerical methods for each.
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.fields import ISFieldConfiguration, TransportCoefficients
from ..core.metrics import MetricBase
from ..core.performance import monitor_performance
from ..core.spacetime_grid import SpacetimeGrid


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
        Default hyperbolic solver (explicit Euler for demonstration).

        Solves: ∂u/∂t + ∇·F(u) = 0
        """
        # This is a placeholder - in practice, you'd use a proper hyperbolic solver
        result = fields.copy()

        # Simple explicit update (for demonstration)
        # In reality, this would use finite difference or finite volume methods
        expansion_rate = -1.0 / (fields.time if hasattr(fields, "time") else 1.0)

        # Update conserved quantities based on expansion
        result.rho *= 1.0 + expansion_rate * dt
        result.pressure *= 1.0 + expansion_rate * dt

        return result

    def _default_relaxation_solver(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """
        Default relaxation solver (exponential integrator).

        Solves: ∂u/∂t = -u/τ + S(u)
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
            tau_q = getattr(self.coefficients, "heat_relaxation_time", 0.1)
            relaxation_factor = np.exp(-dt / tau_q)
            result.q_mu *= relaxation_factor

        return result

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
        Estimate splitting error using Richardson extrapolation.

        Compare solution with dt vs two steps with dt/2.
        """
        # Full step
        full_step = self.advance_timestep_no_error(fields, dt)

        # Two half steps
        half_step1 = self.advance_timestep_no_error(fields, dt / 2)
        half_step2 = self.advance_timestep_no_error(half_step1, dt / 2)

        # Estimate error in energy density
        error_rho = np.max(np.abs(full_step.rho - half_step2.rho))
        error_Pi = 0.0
        if hasattr(full_step, "Pi") and full_step.Pi is not None:
            error_Pi = np.max(np.abs(full_step.Pi - half_step2.Pi))

        return float(error_rho + error_Pi)

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
                        f"Error: {error}, Tolerance: {self.tolerance}"
                    )
                    break

        # Update stored timestep
        self.current_timestep = current_dt

        return result

    def _choose_splitting_method(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> OperatorSplittingBase:
        """
        Choose splitting method based on problem characteristics.

        Uses Strang splitting for accurate evolution, falls back to
        Lie-Trotter for very stiff problems.
        """
        # Estimate stiffness from relaxation timescales
        tau_pi = self.coefficients.shear_relaxation_time or 0.1
        tau_Pi = self.coefficients.bulk_relaxation_time or 0.1

        min_relaxation_time = min(tau_pi, tau_Pi)
        stiffness_ratio = dt / min_relaxation_time

        if stiffness_ratio > 0.1:
            # Use Strang splitting for non-stiff problems
            return self.strang_splitter
        else:
            # Use Lie-Trotter for very stiff problems
            return self.lietrotter_splitter

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
        tau_q = getattr(self.coefficients, "heat_relaxation_time", 0.1)

        # Hydrodynamic timescale (sound crossing time)
        L_char = min(self.grid.spatial_spacing)  # Characteristic length scale
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
        """Default thermodynamic evolution solver."""
        # Placeholder for thermodynamic evolution
        # In practice, this would solve energy-momentum conservation
        result = fields.copy()

        # Simple cooling due to expansion
        cooling_rate = 0.01  # 1% per unit time
        result.temperature *= 1.0 - cooling_rate * dt
        result.rho = (result.temperature / fields.temperature) ** 4 * fields.rho
        result.pressure = result.rho / 3.0

        return result

    @monitor_performance("physics_based_splitting")
    def advance_timestep(
        self,
        fields: ISFieldConfiguration,
        dt: float,
    ) -> ISFieldConfiguration:
        """
        Advance timestep using physics-based splitting.

        Applies sub-stepping for fast processes and coarse stepping for slow processes.
        """
        # Determine sub-stepping based on timescale ratios
        tau_fast = self.timescales["relaxation_min"]
        tau_medium = self.timescales["hydrodynamic"]
        tau_slow = self.timescales["thermodynamic"]

        # Number of sub-steps for each process
        n_relax = max(1, int(dt / (0.1 * tau_fast)))
        n_hydro = max(1, int(dt / (0.5 * tau_medium)))
        n_thermo = 1  # Slow process, single step

        result = fields.copy()

        # Sub-step relaxation (fastest process)
        dt_relax = dt / n_relax
        for _ in range(n_relax):
            result = self.relaxation_solver(result, dt_relax)

        # Sub-step hydrodynamics (medium process)
        dt_hydro = dt / n_hydro
        for _ in range(n_hydro):
            result = self.hyperbolic_solver(result, dt_hydro)

        # Single step thermodynamics (slowest process)
        result = self.thermodynamic_solver(result, dt)

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
    result.pressure = result.rho / 3.0  # Ideal gas relation

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
        tau_q = getattr(coefficients, "heat_relaxation_time", 0.1)
        result.q_mu *= np.exp(-dt / tau_q)

    return result
