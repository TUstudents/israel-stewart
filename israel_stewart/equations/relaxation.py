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

        result: np.ndarray = linear + first_order + nonlinear
        return result

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
            from ..core.tensor_utils import optimized_einsum
            expansion_term = self.coeffs.lambda_pi_pi * optimized_einsum(
                "...ij,...->...ij", pi_munu, theta
            )
            nonlinear += expansion_term

        # Shear-bulk coupling
        if self.coeffs.lambda_pi_Pi != 0:
            # Broadcast Pi scalar to tensor shape
            bulk_coupling = self.coeffs.lambda_pi_Pi * Pi[..., np.newaxis, np.newaxis] * sigma_munu
            nonlinear += bulk_coupling

        # Shear-heat coupling
        if self.coeffs.lambda_pi_q != 0:
            # Vectorized implementation of the shear-heat coupling term
            # Term is lambda_pi_q * (q^mu * nabla^nu T + q^nu * nabla^mu T) / 2
            from ..core.tensor_utils import optimized_einsum
            outer_product = optimized_einsum("...i,...j->...ij", q_mu, nabla_T)
            heat_term = (
                self.coeffs.lambda_pi_q * 0.5 * (outer_product + np.swapaxes(outer_product, -1, -2))
            )
            nonlinear += heat_term

        # Nonlinear shear terms
        if self.coeffs.tau_pi_pi != 0 and self.coeffs.shear_relaxation_time:
            # pi^mu_alpha * pi_alpha^nu term (simplified)
            from ..core.tensor_utils import optimized_einsum
            pi_pi_term = -self.coeffs.tau_pi_pi * optimized_einsum("...ik,...kj->...ij", pi_munu, pi_munu)
            pi_pi_term /= self.coeffs.shear_viscosity * self.coeffs.shear_relaxation_time
            nonlinear += pi_pi_term

        # Vorticity coupling
        if self.coeffs.tau_pi_omega != 0:
            # Anti-commutator: pi^mu_alpha * omega_alpha^nu - omega^mu_alpha * pi_alpha^nu
            from ..core.tensor_utils import optimized_einsum
            vorticity_term = self.coeffs.tau_pi_omega * (
                optimized_einsum("...ik,...kj->...ij", pi_munu, omega_munu)
                - optimized_einsum("...ik,...kj->...ij", omega_munu, pi_munu)
            )
            nonlinear += vorticity_term

        result: np.ndarray = linear + first_order + nonlinear
        return result

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
            # Vectorized implementation of the shear-heat coupling term
            # Term is lambda_q_pi * pi^munu * nabla_nu T
            from ..core.tensor_utils import optimized_einsum
            shear_heat_term = self.coeffs.lambda_q_pi * optimized_einsum(
                "...ij,...j->...i", pi_munu, nabla_T
            )
            nonlinear += shear_heat_term

        result: np.ndarray = linear + first_order + nonlinear
        return result

    def _compute_expansion_scalar(self, u_mu: np.ndarray) -> np.ndarray:
        """
        Compute expansion scalar θ = ∇_μ u^μ using proper covariant derivatives.

        Uses a field-aware wrapper to interface with the corrected vector_divergence method.
        """
        from ..core.derivatives import CovariantDerivative

        # Compute divergence using manual gradients and Christoffel symbols
        # ∇_μ u^μ = ∂_μ u^μ + Γ^μ_μν u^ν

        # Initialize covariant derivative operator for Christoffel symbols
        cov_deriv = CovariantDerivative(self.metric)

        # Get Christoffel symbols - handle both numerical and symbolic cases
        christoffel = cov_deriv.christoffel_symbols

        # Check if christoffel is symbolic or contains symbolic elements
        is_symbolic = (
            hasattr(christoffel, "dtype") and christoffel.dtype == "O"
        ) or not isinstance(christoffel, np.ndarray)

        if is_symbolic:
            # This is symbolic - for now use flat space approximation
            import warnings

            warnings.warn(
                "Using flat space approximation for symbolic metric", UserWarning, stacklevel=2
            )
            christoffel = np.zeros((4, 4, 4))

        # Compute partial derivatives ∂_μ u^μ
        partial_div = np.zeros(u_mu.shape[:-1])
        for mu in range(4):
            partial_div += np.gradient(u_mu[..., mu], axis=mu, edge_order=1)

        # Add Christoffel term: Γ^μ_μν u^ν = Γ^μ_νμ u^ν (using symmetry)
        christoffel_term = np.zeros(u_mu.shape[:-1])
        for mu in range(4):
            for nu in range(4):
                christoffel_term += christoffel[mu, mu, nu] * u_mu[..., nu]

        theta = partial_div + christoffel_term

        return theta

    def _compute_shear_tensor(self, u_mu: np.ndarray) -> np.ndarray:
        """
        Compute shear tensor σ^μν using manual gradients and Christoffel symbols.

        Formula: σ^μν = ∇^(μ u^ν) + a^(μ u^ν) - (1/3)Δ^μν θ
        """
        from ..core.derivatives import CovariantDerivative
        from ..core.tensor_utils import optimized_einsum

        # Initialize covariant derivative operator for Christoffel symbols
        cov_deriv = CovariantDerivative(self.metric)

        # Get Christoffel symbols - handle both numerical and symbolic cases
        christoffel = cov_deriv.christoffel_symbols

        # Check if christoffel is symbolic or contains symbolic elements
        is_symbolic = (
            hasattr(christoffel, "dtype") and christoffel.dtype == "O"
        ) or not isinstance(christoffel, np.ndarray)

        if is_symbolic:
            # This is symbolic - for now use flat space approximation
            import warnings

            warnings.warn(
                "Using flat space approximation for symbolic metric", UserWarning, stacklevel=2
            )
            christoffel = np.zeros((4, 4, 4))

        # Compute velocity gradients using finite differences
        nabla_u_partial = np.zeros(u_mu.shape[:-1] + (4, 4))
        for mu in range(4):
            for nu in range(4):
                nabla_u_partial[..., mu, nu] = np.gradient(u_mu[..., nu], axis=mu, edge_order=1)

        # Add Christoffel correction: ∇_μ u_ν = ∂_μ u_ν - Γ^ρ_{μν} u_ρ
        nabla_u = nabla_u_partial.copy()
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    nabla_u[..., mu, nu] -= christoffel[rho, mu, nu] * u_mu[..., rho]

        # Get metric tensors
        g_inv = self.metric.inverse
        g = self.metric.components
        if isinstance(g_inv, np.ndarray) and g_inv.ndim == 2:
            g_inv = np.broadcast_to(g_inv, u_mu.shape[:-1] + (4, 4))
            g = np.broadcast_to(g, u_mu.shape[:-1] + (4, 4))

        # Raise indices: ∇^μ u^ν = g^μρ g^νσ ∇_ρ u_σ
        nabla_u_up = optimized_einsum("...ac,...bd,...cd->...ab", g_inv, g_inv, nabla_u)

        # Symmetrize: ∇^(μ u^ν) = (1/2)(∇^μ u^ν + ∇^ν u^μ)
        symmetric_grad = 0.5 * (nabla_u_up + np.swapaxes(nabla_u_up, -2, -1))

        # Compute four-acceleration: a^μ = u^ρ ∇_ρ u^μ
        u_lower = optimized_einsum("...ab,...b->...a", g, u_mu)
        acceleration = optimized_einsum("...a,...ab->...b", u_lower, nabla_u)
        a_up = optimized_einsum("...ab,...b->...a", g_inv, acceleration)

        # Symmetrized acceleration outer product: a^(μ u^ν)
        accel_outer = 0.5 * (
            optimized_einsum("...a,...b->...ab", a_up, u_mu)
            + optimized_einsum("...a,...b->...ba", a_up, u_mu)
        )

        # Get expansion scalar for trace removal
        theta = self._compute_expansion_scalar(u_mu)

        # Compute perpendicular projector: Δ^μν = g^μν + u^μ u^ν
        delta = g_inv + optimized_einsum("...a,...b->...ab", u_mu, u_mu)

        # Project to spatial hypersurface: Δ^μρ Δ^νσ (∇^(ρ u^σ) + a^(ρ u^σ))
        full_tensor = symmetric_grad + accel_outer
        projected = optimized_einsum("...ac,...bd,...cd->...ab", delta, delta, full_tensor)

        # Remove trace: σ^μν = projected - (1/3)Δ^μν θ
        trace_part = (1.0 / 3.0) * optimized_einsum("...,...ab->...ab", theta, delta)
        sigma_munu = projected - trace_part

        return sigma_munu

    def _compute_vorticity_tensor(self, u_mu: np.ndarray) -> np.ndarray:
        """
        Compute vorticity tensor ω^μν using manual gradients and Christoffel symbols.

        Formula: ω^μν = ∇^[μ u^ν] + a^[μ u^ν]
        """
        from ..core.derivatives import CovariantDerivative
        from ..core.tensor_utils import optimized_einsum

        # Initialize covariant derivative operator for Christoffel symbols
        cov_deriv = CovariantDerivative(self.metric)

        # Get Christoffel symbols - handle both numerical and symbolic cases
        christoffel = cov_deriv.christoffel_symbols

        # Check if christoffel is symbolic or contains symbolic elements
        is_symbolic = (
            hasattr(christoffel, "dtype") and christoffel.dtype == "O"
        ) or not isinstance(christoffel, np.ndarray)

        if is_symbolic:
            # This is symbolic - for now use flat space approximation
            import warnings

            warnings.warn(
                "Using flat space approximation for symbolic metric", UserWarning, stacklevel=2
            )
            christoffel = np.zeros((4, 4, 4))

        # Compute velocity gradients using finite differences
        nabla_u_partial = np.zeros(u_mu.shape[:-1] + (4, 4))
        for mu in range(4):
            for nu in range(4):
                nabla_u_partial[..., mu, nu] = np.gradient(u_mu[..., nu], axis=mu, edge_order=1)

        # Add Christoffel correction: ∇_μ u_ν = ∂_μ u_ν - Γ^ρ_{μν} u_ρ
        nabla_u = nabla_u_partial.copy()
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    nabla_u[..., mu, nu] -= christoffel[rho, mu, nu] * u_mu[..., rho]

        # Get metric tensors
        g_inv = self.metric.inverse
        g = self.metric.components
        if isinstance(g_inv, np.ndarray) and g_inv.ndim == 2:
            g_inv = np.broadcast_to(g_inv, u_mu.shape[:-1] + (4, 4))
            g = np.broadcast_to(g, u_mu.shape[:-1] + (4, 4))

        # Raise indices: ∇^μ u^ν = g^μρ g^νσ ∇_ρ u_σ
        nabla_u_up = optimized_einsum("...ac,...bd,...cd->...ab", g_inv, g_inv, nabla_u)

        # Antisymmetrize: ∇^[μ u^ν] = (1/2)(∇^μ u^ν - ∇^ν u^μ)
        antisymmetric_grad = 0.5 * (nabla_u_up - np.swapaxes(nabla_u_up, -2, -1))

        # Compute four-acceleration: a^μ = u^ρ ∇_ρ u^μ
        u_lower = optimized_einsum("...ab,...b->...a", g, u_mu)
        acceleration = optimized_einsum("...a,...ab->...b", u_lower, nabla_u)
        a_up = optimized_einsum("...ab,...b->...a", g_inv, acceleration)

        # Antisymmetrized acceleration outer product: a^[μ u^ν]
        accel_antisymmetric = 0.5 * (
            optimized_einsum("...a,...b->...ab", a_up, u_mu)
            - optimized_einsum("...a,...b->...ba", a_up, u_mu)
        )

        # Compute perpendicular projector: Δ^μν = g^μν + u^μ u^ν
        delta = g_inv + optimized_einsum("...a,...b->...ab", u_mu, u_mu)

        # Project to spatial hypersurface: Δ^μρ Δ^νσ (∇^[ρ u^σ] + a^[ρ u^σ])
        full_antisymmetric = antisymmetric_grad + accel_antisymmetric
        projected = optimized_einsum("...ac,...bd,...cd->...ab", delta, delta, full_antisymmetric)

        # Ensure exact antisymmetry: ω^μν = (1/2)(projected - projected^T)
        omega_munu = 0.5 * (projected - np.swapaxes(projected, -2, -1))

        return np.asarray(omega_munu)

    def _compute_temperature_gradient(self, T: np.ndarray, u_mu: np.ndarray) -> np.ndarray:
        """
        Compute projected temperature gradient ∇^μ T = Δ^μν ∇_ν T using vectorized operations.

        This gives the spatial gradient of temperature orthogonal to the fluid flow.
        """
        from ..core.derivatives import CovariantDerivative
        from ..core.tensor_utils import optimized_einsum

        # Initialize covariant derivative operator
        cov_deriv = CovariantDerivative(self.metric)
        grid_coords = [
            self.grid.coordinates.get("t"),
            self.grid.coordinates.get("x"),
            self.grid.coordinates.get("y"),
            self.grid.coordinates.get("z"),
        ]

        # Compute gradient of temperature using finite differences: ∂_μ T
        grad_T_lower = np.zeros(T.shape + (4,))
        for mu in range(4):
            grad_T_lower[..., mu] = np.gradient(T, axis=mu, edge_order=1)

        # Get metric inverse for raising indices
        g_inv = self.metric.inverse
        if isinstance(g_inv, np.ndarray) and g_inv.ndim == 2:
            # Broadcast metric to match field dimensions
            g_inv = np.broadcast_to(g_inv, T.shape + (4, 4))

        # Raise gradient indices: ∇^μ T = g^μν ∇_ν T
        grad_T_up = optimized_einsum("...ab,...b->...a", g_inv, grad_T_lower)

        # Compute perpendicular projector: Δ^μν = g^μν + u^μ u^ν
        # Need to align u_mu with T dimensions
        if u_mu.shape[:-1] != T.shape:
            # Assume T has same spatial dimensions as u_mu
            u_aligned = u_mu[: T.shape[0], : T.shape[1], : T.shape[2], : T.shape[3]]
        else:
            u_aligned = u_mu

        delta = g_inv + optimized_einsum("...a,...b->...ab", u_aligned, u_aligned)

        # Project gradient to spatial hypersurface: ∇^μ T = Δ^μν ∇_ν T
        nabla_T = optimized_einsum("...ab,...b->...a", delta, grad_T_lower)

        return nabla_T

    def validate_kinematic_quantities(self, fields: ISFieldConfiguration) -> dict[str, bool]:
        """
        Validate kinematic quantities satisfy required physical constraints.

        Tests:
        - σ^μν u_ν = 0 (shear orthogonality to velocity)
        - σ^μ_μ = 0 (traceless condition)
        - ω^μν = -ω^νμ (antisymmetry)
        - ω^μν u_ν = 0 (vorticity orthogonality to velocity)
        """
        from ..core.four_vectors import FourVector
        from ..core.tensor_utils import optimized_einsum

        u_mu = fields.u_mu

        # Compute kinematic quantities
        theta = self._compute_expansion_scalar(u_mu)
        sigma_munu = self._compute_shear_tensor(u_mu)
        omega_munu = self._compute_vorticity_tensor(u_mu)

        validation = {}

        # Test 1: Shear tensor orthogonality σ^μν u_ν = 0
        shear_u_contraction = optimized_einsum("...ij,...j->...i", sigma_munu, u_mu)
        validation["shear_orthogonal_to_velocity"] = np.allclose(
            shear_u_contraction, 0.0, atol=1e-10
        )

        # Test 2: Shear tensor traceless σ^μ_μ = 0
        shear_trace = np.trace(sigma_munu, axis1=-2, axis2=-1)
        validation["shear_tensor_traceless"] = np.allclose(shear_trace, 0.0, atol=1e-12)

        # Test 3: Vorticity antisymmetry ω^μν = -ω^νμ
        omega_transpose = np.transpose(
            omega_munu, axes=list(range(len(omega_munu.shape) - 2)) + [-1, -2]
        )
        validation["vorticity_antisymmetric"] = np.allclose(
            omega_munu + omega_transpose, 0.0, atol=1e-12
        )

        # Test 4: Vorticity orthogonality ω^μν u_ν = 0
        vorticity_u_contraction = optimized_einsum("...ij,...j->...i", omega_munu, u_mu)
        validation["vorticity_orthogonal_to_velocity"] = np.allclose(
            vorticity_u_contraction, 0.0, atol=1e-10
        )

        # Test 5: Expansion scalar dimensionality (should be scalar)
        validation["expansion_scalar_shape"] = theta.shape == u_mu.shape[:-1]

        # Overall validation
        validation["all_kinematic_constraints_satisfied"] = all(validation.values())

        return validation

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

        def residual(x_new: np.ndarray) -> np.ndarray:
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
            rhs: np.ndarray = self.compute_relaxation_rhs(fields_new)

            # Implicit Euler residual: x_new - x_old - dt * F(x_new) = 0
            x_old = fields.to_dissipative_vector()
            result: np.ndarray = x_new - x_old - dt * rhs
            return result

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
        grid_size = int(np.prod(fields.grid.shape))

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
