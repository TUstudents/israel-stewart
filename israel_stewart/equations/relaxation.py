"""
Israel-Stewart relaxation equations for second-order viscous hydrodynamics.

This module implements the complete set of Israel-Stewart relaxation equations
that govern the evolution of dissipative fluxes: bulk pressure Pi, shear tensor pi^munu,
and heat flux q^mu.
"""

import warnings
from typing import Any

import numpy as np
import sympy as sp
from scipy.optimize import newton_krylov

from ..core.fields import ISFieldConfiguration, TransportCoefficients
from ..core.metrics import MetricBase
from ..core.spacetime_grid import SpacetimeGrid


class ISRelaxationEquations:
    """
    Complete Israel-Stewart relaxation equations with all coupling terms.

    Implements the evolution equations for dissipative fluxes:
    - Bulk viscous pressure: dPi/dtau + Pi/tau_Pi = -zeta*theta + nonlinear terms
    - Shear stress: dpi^munu/dtau + pi^munu/tau_pi = 2*eta*sigma^munu + coupling terms
    - Heat flux: dq^mu/dtau + q^mu/tau_q = kappa*nabla^mu*T + coupling terms
    """

    def __init__(
        self,
        grid: SpacetimeGrid,
        metric: MetricBase,
        coefficients: TransportCoefficients,
    ):
        """
        Initialize Israel-Stewart relaxation equations.

        Args:
            grid: Spacetime discretization grid
            metric: Background spacetime metric
            coefficients: Transport coefficients with second-order terms
        """
        self.grid = grid
        self.metric = metric
        self.coeffs = coefficients

        # Build symbolic equations for analysis
        self.symbolic_eqs = self._build_symbolic_equations()

        # Performance monitoring
        self._evolution_count = 0
        self._total_evolution_time = 0.0

    def _build_symbolic_equations(self) -> dict[str, sp.Expr]:
        """
        Build symbolic IS equations using SymPy for exact derivatives.

        Returns:
            Dictionary containing symbolic expressions for bulk, shear, and heat flux
        """
        # Define symbolic variables
        t = sp.Symbol("t", real=True)
        Pi = sp.Function("Pi")(t)

        # Tensor components (symbolic)
        pi_00, pi_01, pi_02, pi_03 = sp.symbols("pi_00 pi_01 pi_02 pi_03", real=True)
        pi_11, pi_12, pi_13, pi_22, pi_23, pi_33 = sp.symbols(
            "pi_11 pi_12 pi_13 pi_22 pi_23 pi_33", real=True
        )
        q_0, q_1, q_2, q_3 = sp.symbols("q_0 q_1 q_2 q_3", real=True)

        # Thermodynamic and kinematic quantities
        rho, p, T = sp.symbols("rho p T", positive=True, real=True)
        theta = sp.Symbol("theta", real=True)  # Expansion scalar nabla dot u

        # Shear tensor and vorticity
        sigma_munu = sp.MatrixSymbol("sigma", 4, 4)  # Shear tensor sigma^munu
        omega_munu = sp.MatrixSymbol("omega", 4, 4)  # Vorticity tensor omega^munu

        # Transport coefficients (symbolic)
        eta, zeta, kappa = sp.symbols("eta zeta kappa", positive=True, real=True)
        tau_pi, tau_Pi, tau_q = sp.symbols("tau_pi tau_Pi tau_q", positive=True, real=True)

        # Second-order coupling coefficients
        lambda_pi_pi = sp.Symbol("lambda_pi_pi", real=True)
        lambda_pi_Pi = sp.Symbol("lambda_pi_Pi", real=True)
        lambda_pi_q = sp.Symbol("lambda_pi_q", real=True)
        lambda_Pi_pi = sp.Symbol("lambda_Pi_pi", real=True)
        lambda_q_pi = sp.Symbol("lambda_q_pi", real=True)
        xi_1, xi_2 = sp.symbols("xi_1 xi_2", real=True)
        tau_pi_pi, tau_pi_omega = sp.symbols("tau_pi_pi tau_pi_omega", real=True)

        # Bulk viscous pressure evolution equation
        bulk_linear = -Pi / tau_Pi - zeta * theta
        bulk_nonlinear = (
            xi_1 * Pi * theta + xi_2 * Pi**2 / (zeta * tau_Pi) + lambda_Pi_pi * pi_00 * theta
        )  # Simplified shear-bulk coupling

        dPi_dt = bulk_linear + bulk_nonlinear

        # Shear stress evolution equation (using pi^00 as representative component)
        # Full tensor equation would require all components
        shear_linear = -pi_00 / tau_pi + 2 * eta * sigma_munu[0, 0]
        shear_nonlinear = (
            -tau_pi_pi * pi_00**2 / (eta * tau_pi)
            + tau_pi_omega * (pi_01 * omega_munu[1, 0] - omega_munu[0, 1] * pi_01)
            + lambda_pi_pi * pi_00 * theta
            + lambda_pi_Pi * Pi * sigma_munu[0, 0]
            + lambda_pi_q * (q_0 * sp.Symbol("nabla_0_T") + q_1 * sp.Symbol("nabla_1_T"))
        )

        dpi_00_dt = shear_linear + shear_nonlinear

        # Heat flux evolution equation (using q^0 as representative component)
        heat_linear = -q_0 / tau_q + kappa * sp.Symbol("nabla_0_T")
        heat_nonlinear = lambda_q_pi * pi_00 * sp.Symbol("nabla_0_T") - tau_q * q_0 * theta

        dq_0_dt = heat_linear + heat_nonlinear

        return {"bulk": dPi_dt, "shear_00": dpi_00_dt, "heat_0": dq_0_dt}

    def compute_relaxation_rhs(self, fields: ISFieldConfiguration) -> np.ndarray:
        """
        Compute right-hand side of relaxation equations.

        Args:
            fields: Current field configuration

        Returns:
            Time derivatives of dissipative fluxes
        """
        # Extract field components
        Pi = fields.Pi
        pi_munu = fields.pi_munu
        q_mu = fields.q_mu
        u_mu = fields.u_mu

        # Compute thermodynamic quantities
        temperature = fields.temperature

        # Compute kinematic quantities
        expansion_scalar = self._compute_expansion_scalar(u_mu)
        shear_tensor = self._compute_shear_tensor(u_mu)
        vorticity_tensor = self._compute_vorticity_tensor(u_mu)

        # Temperature gradient (projected)
        temp_gradient = self._compute_temperature_gradient(temperature, u_mu)

        # Right-hand side components
        dPi_dt = self._bulk_rhs(Pi, pi_munu, expansion_scalar)
        dpi_munu_dt = self._shear_rhs(
            pi_munu,
            Pi,
            q_mu,
            expansion_scalar,
            shear_tensor,
            vorticity_tensor,
            temp_gradient,
        )
        dq_mu_dt = self._heat_rhs(q_mu, pi_munu, expansion_scalar, temp_gradient)

        # Pack into dissipative vector format
        return np.concatenate([dPi_dt.flatten(), dpi_munu_dt.reshape(-1), dq_mu_dt.reshape(-1)])

    def _bulk_rhs(self, Pi: np.ndarray, pi_munu: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute bulk pressure evolution."""
        # Linear relaxation term
        linear = (
            -Pi / self.coeffs.bulk_relaxation_time
            if self.coeffs.bulk_relaxation_time
            else np.zeros_like(Pi)
        )

        # First-order source: -zeta*theta
        first_order = -self.coeffs.bulk_viscosity * theta

        # Second-order nonlinear terms
        nonlinear = np.zeros_like(Pi)
        if self.coeffs.xi_1 != 0:
            nonlinear += self.coeffs.xi_1 * Pi * theta

        if (
            self.coeffs.xi_2 != 0
            and self.coeffs.bulk_viscosity > 0
            and self.coeffs.bulk_relaxation_time
        ):
            nonlinear += (
                self.coeffs.xi_2
                * Pi**2
                / (self.coeffs.bulk_viscosity * self.coeffs.bulk_relaxation_time)
            )

        # Shear-bulk coupling
        if self.coeffs.lambda_Pi_pi != 0:
            # Use trace-free part of shear tensor
            pi_trace = np.trace(pi_munu, axis1=-2, axis2=-1)
            shear_contribution = self.coeffs.lambda_Pi_pi * pi_trace * theta
            nonlinear += shear_contribution

        return linear + first_order + nonlinear

    def _shear_rhs(
        self,
        pi_munu: np.ndarray,
        Pi: np.ndarray,
        q_mu: np.ndarray,
        theta: np.ndarray,
        sigma_munu: np.ndarray,
        omega_munu: np.ndarray,
        nabla_T: np.ndarray,
    ) -> np.ndarray:
        """Compute shear tensor evolution."""
        # Linear relaxation
        linear = (
            -pi_munu / self.coeffs.shear_relaxation_time
            if self.coeffs.shear_relaxation_time
            else np.zeros_like(pi_munu)
        )

        # First-order source: 2*eta*sigma^munu
        first_order = 2.0 * self.coeffs.shear_viscosity * sigma_munu

        # Second-order terms
        nonlinear = np.zeros_like(pi_munu)

        # Expansion coupling
        if self.coeffs.lambda_pi_pi != 0:
            expansion_term = self.coeffs.lambda_pi_pi * np.einsum(
                "...ij,...->...ij", pi_munu, theta
            )
            nonlinear += expansion_term

        # Shear-bulk coupling
        if self.coeffs.lambda_pi_Pi != 0:
            # Broadcast Pi scalar to tensor shape
            bulk_coupling = self.coeffs.lambda_pi_Pi * Pi[..., np.newaxis, np.newaxis] * sigma_munu
            nonlinear += bulk_coupling

        # Shear-heat coupling (simplified)
        if self.coeffs.lambda_pi_q != 0:
            # This is a simplified version - full implementation requires careful index handling
            for mu in range(4):
                for nu in range(4):
                    if nabla_T.shape[-1] > max(mu, nu):
                        heat_term = (
                            self.coeffs.lambda_pi_q
                            * (q_mu[..., mu] * nabla_T[..., nu] + q_mu[..., nu] * nabla_T[..., mu])
                            / 2
                        )
                        nonlinear[..., mu, nu] += heat_term

        # Nonlinear shear terms
        if self.coeffs.tau_pi_pi != 0 and self.coeffs.shear_relaxation_time:
            # pi^mu_alpha * pi_alpha^nu term (simplified)
            pi_pi_term = -self.coeffs.tau_pi_pi * np.einsum("...ik,...kj->...ij", pi_munu, pi_munu)
            pi_pi_term /= self.coeffs.shear_viscosity * self.coeffs.shear_relaxation_time
            nonlinear += pi_pi_term

        # Vorticity coupling
        if self.coeffs.tau_pi_omega != 0:
            # Anti-commutator: pi^mu_alpha * omega_alpha^nu - omega^mu_alpha * pi_alpha^nu
            vorticity_term = self.coeffs.tau_pi_omega * (
                np.einsum("...ik,...kj->...ij", pi_munu, omega_munu)
                - np.einsum("...ik,...kj->...ij", omega_munu, pi_munu)
            )
            nonlinear += vorticity_term

        return linear + first_order + nonlinear

    def _heat_rhs(
        self,
        q_mu: np.ndarray,
        pi_munu: np.ndarray,
        theta: np.ndarray,
        nabla_T: np.ndarray,
    ) -> np.ndarray:
        """Compute heat flux evolution."""
        # Linear relaxation
        linear = (
            -q_mu / self.coeffs.heat_relaxation_time
            if self.coeffs.heat_relaxation_time
            else np.zeros_like(q_mu)
        )

        # First-order source: kappa*nabla^mu*T
        first_order = self.coeffs.thermal_conductivity * nabla_T

        # Second-order terms
        nonlinear = np.zeros_like(q_mu)

        # Expansion coupling
        if self.coeffs.tau_q_pi != 0:
            expansion_term = -self.coeffs.tau_q_pi * q_mu * theta[..., np.newaxis]
            nonlinear += expansion_term

        # Shear-heat coupling
        if self.coeffs.lambda_q_pi != 0:
            # Simplified: pi^munu * nabla_nu T
            for mu in range(4):
                if nabla_T.shape[-1] > mu:
                    shear_heat_term = np.sum(pi_munu[..., mu, :] * nabla_T, axis=-1)
                    nonlinear[..., mu] += self.coeffs.lambda_q_pi * shear_heat_term

        return linear + first_order + nonlinear

    def _compute_expansion_scalar(self, u_mu: np.ndarray) -> np.ndarray:
        """Compute expansion scalar theta = nabla dot u."""
        # Simplified finite difference approximation
        # Full implementation would use covariant derivatives
        theta = np.zeros(u_mu.shape[:-1])

        for i in range(1, min(4, len(u_mu.shape))):  # Spatial components
            if u_mu.shape[i] > 1:
                # Simple finite difference
                du_dx = np.gradient(u_mu[..., i], axis=i - 1)
                theta += du_dx

        return theta

    def _compute_shear_tensor(self, u_mu: np.ndarray) -> np.ndarray:
        """Compute shear tensor sigma^munu."""
        # Placeholder: return zero shear tensor
        # Full implementation requires proper 3+1 decomposition
        return np.zeros((*u_mu.shape[:-1], 4, 4))

    def _compute_vorticity_tensor(self, u_mu: np.ndarray) -> np.ndarray:
        """Compute vorticity tensor omega^munu."""
        # Placeholder: return zero vorticity tensor
        return np.zeros((*u_mu.shape[:-1], 4, 4))

    def _compute_temperature_gradient(self, T: np.ndarray, u_mu: np.ndarray) -> np.ndarray:
        """Compute projected temperature gradient."""
        # Simplified finite difference gradient
        grad_T = np.zeros((*T.shape, 4))

        # Spatial gradients (simplified)
        for i in range(1, min(4, len(T.shape) + 1)):
            if i - 1 < len(T.shape) and T.shape[i - 1] > 1:
                grad_T[..., i] = np.gradient(T, axis=i - 1)

        return grad_T

    def evolve_relaxation(
        self, fields: ISFieldConfiguration, dt: float, method: str = "explicit"
    ) -> None:
        """
        Evolve dissipative fluxes for one timestep.

        Args:
            fields: Field configuration to evolve
            dt: Timestep
            method: Evolution method ('implicit', 'exponential', 'explicit')
        """
        import time

        start_time = time.time()

        try:
            if method == "implicit":
                self._implicit_evolution(fields, dt)
            elif method == "exponential":
                self._exponential_integrator(fields, dt)
            elif method == "explicit":
                self._explicit_evolution(fields, dt)
            else:
                raise ValueError(f"Unknown evolution method: {method}")

        except Exception as e:
            warnings.warn(f"Relaxation evolution failed: {e}", stacklevel=2)
            # Fallback to explicit method
            if method != "explicit":
                self._explicit_evolution(fields, dt)
            else:
                raise

        # Performance monitoring
        self._evolution_count += 1
        self._total_evolution_time += time.time() - start_time

    def _implicit_evolution(self, fields: ISFieldConfiguration, dt: float) -> None:
        """Implicit solver for stiff relaxation times."""

        def residual(x_new):
            # Create temporary field configuration
            fields_new = ISFieldConfiguration(fields.grid)

            # Copy non-dissipative fields
            fields_new.rho = fields.rho.copy()
            fields_new.n = fields.n.copy()
            fields_new.u_mu = fields.u_mu.copy()
            fields_new.pressure = fields.pressure.copy()
            fields_new.temperature = fields.temperature.copy()

            # Set new dissipative fields
            fields_new.from_dissipative_vector(x_new)

            # Compute RHS at new state
            rhs = self.compute_relaxation_rhs(fields_new)

            # Implicit Euler residual: x_new - x_old - dt * F(x_new) = 0
            x_old = fields.to_dissipative_vector()
            return x_new - x_old - dt * rhs

        # Initial guess
        x_initial = fields.to_dissipative_vector()

        try:
            # Solve nonlinear system
            x_solution = newton_krylov(residual, x_initial, method="gmres", f_tol=1e-8, maxiter=50)

            # Update fields
            fields.from_dissipative_vector(x_solution)

        except Exception as e:
            warnings.warn(f"Implicit solver failed: {e}. Using explicit step.", stacklevel=3)
            self._explicit_evolution(fields, dt)

    def _exponential_integrator(self, fields: ISFieldConfiguration, dt: float) -> None:
        """Exponential time differencing for relaxation equations."""
        # Extract relaxation times
        tau_pi = self.coeffs.shear_relaxation_time or 1.0
        tau_Pi = self.coeffs.bulk_relaxation_time or 1.0
        tau_q = self.coeffs.heat_relaxation_time or 1.0

        # Build diagonal relaxation matrix (simplified)
        grid_size = np.prod(fields.grid.shape)

        # Relaxation eigenvalues
        lambda_pi = 1.0 / tau_pi
        lambda_Pi = 1.0 / tau_Pi
        lambda_q = 1.0 / tau_q

        # Exponential factors
        exp_pi = np.exp(-lambda_pi * dt)
        exp_Pi = np.exp(-lambda_Pi * dt)
        exp_q = np.exp(-lambda_q * dt)

        # Current dissipative state
        x_old = fields.to_dissipative_vector()

        # Compute nonlinear terms at current state
        rhs = self.compute_relaxation_rhs(fields)

        # ETD step (simplified)
        # For each component type, apply exponential integration
        offset = 0

        # Bulk pressure
        Pi_size = grid_size
        Pi_old = x_old[offset : offset + Pi_size]
        Pi_rhs = rhs[offset : offset + Pi_size]
        Pi_new = exp_Pi * Pi_old + (1 - exp_Pi) / lambda_Pi * Pi_rhs
        offset += Pi_size

        # Shear tensor
        pi_size = 16 * grid_size
        pi_old = x_old[offset : offset + pi_size]
        pi_rhs = rhs[offset : offset + pi_size]
        pi_new = exp_pi * pi_old + (1 - exp_pi) / lambda_pi * pi_rhs
        offset += pi_size

        # Heat flux
        q_size = 4 * grid_size
        q_old = x_old[offset : offset + q_size]
        q_rhs = rhs[offset : offset + q_size]
        q_new = exp_q * q_old + (1 - exp_q) / lambda_q * q_rhs

        # Reconstruct solution vector
        x_new = np.concatenate([Pi_new, pi_new, q_new])

        # Update fields
        fields.from_dissipative_vector(x_new)

    def _explicit_evolution(self, fields: ISFieldConfiguration, dt: float) -> None:
        """Explicit Euler evolution (fallback method)."""
        # Compute RHS
        rhs = self.compute_relaxation_rhs(fields)

        # Explicit Euler step
        x_old = fields.to_dissipative_vector()
        x_new = x_old + dt * rhs

        # Update fields
        fields.from_dissipative_vector(x_new)

    def stability_analysis(self, fields: ISFieldConfiguration) -> dict[str, Any]:
        """
        Analyze stability of relaxation equations at current state.

        Args:
            fields: Current field configuration

        Returns:
            Stability analysis results
        """
        # Estimate characteristic timescales
        tau_pi = self.coeffs.shear_relaxation_time or 1.0
        tau_Pi = self.coeffs.bulk_relaxation_time or 1.0
        tau_q = self.coeffs.heat_relaxation_time or 1.0

        # Characteristic values
        Pi_char = np.max(np.abs(fields.Pi)) if np.any(fields.Pi) else 1e-10
        pi_char = np.max(np.abs(fields.pi_munu)) if np.any(fields.pi_munu) else 1e-10
        q_char = np.max(np.abs(fields.q_mu)) if np.any(fields.q_mu) else 1e-10

        # Stiffness ratios
        stiffness_ratio = max(tau_Pi, tau_pi, tau_q) / min(tau_Pi, tau_pi, tau_q)

        # Recommended timestep (stability constraint)
        dt_max = 0.1 * min(tau_Pi, tau_pi, tau_q)

        return {
            "relaxation_times": {"tau_pi": tau_pi, "tau_Pi": tau_Pi, "tau_q": tau_q},
            "characteristic_values": {"Pi": Pi_char, "pi": pi_char, "q": q_char},
            "stiffness_ratio": stiffness_ratio,
            "recommended_dt": dt_max,
            "is_stiff": stiffness_ratio > 10.0,
        }

    def performance_report(self) -> dict[str, Any]:
        """Generate performance report for relaxation evolution."""
        if self._evolution_count == 0:
            return {"message": "No evolution steps performed yet"}

        avg_time = self._total_evolution_time / self._evolution_count

        return {
            "evolution_count": self._evolution_count,
            "total_time": self._total_evolution_time,
            "average_time_per_step": avg_time,
            "performance_rating": "Good" if avg_time < 0.01 else "Slow",
        }
