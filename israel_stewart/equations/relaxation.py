"""
Israel-Stewart relaxation equations for second-order viscous hydrodynamics (Landau frame).

This module implements the complete set of Israel-Stewart relaxation equations
that govern the evolution of dissipative fluxes in the Landau frame:
    1. Bulk pressure Π (scalar)
    2. Shear stress π^μν (traceless, spatial tensor)
    3. Particle diffusion current V^μ (four-vector, orthogonal to u^μ)

Landau Frame Definition:
------------------------
In Landau frame, the frame is defined by zero energy flux:
    T^μν u_ν = ε u^μ  (no heat flux q^μ = 0)

The dissipative fluxes evolve according to:
    - Bulk: dΠ/dτ + Π/τ_Π = -ζθ + nonlinear terms
    - Shear: dπ^μν/dτ + π^μν/τ_π = 2η σ^μν + coupling terms
    - Diffusion: dV^μ/dτ + V^μ/τ_V = -D ∇^μ(μ_B/T) + coupling terms

where D is the diffusion coefficient and μ_B/T is the baryon chemical potential over temperature.

See Wagner & Gavassino (2024) IReD paper, docs/LANDAU_FRAME_FORMULATION.md,
and Israel & Stewart (1979) for the theoretical foundation.
"""

import warnings
from typing import Any

import numpy as np
import sympy as sp
from scipy.optimize import newton_krylov

from ..core.fields import ISFieldConfiguration, TransportCoefficients
from ..core.metrics import MetricBase
from ..core.spacegrid import SpaceGrid
from ..core.spacetime_grid import SpacetimeGrid


class ISRelaxationEquations:
    """
    Complete Israel-Stewart relaxation equations with all coupling terms (Landau frame).

    Implements the evolution equations for dissipative fluxes in Landau frame:
    - Bulk viscous pressure: dΠ/dτ + Π/τ_Π = -ζ θ + nonlinear terms
    - Shear stress: dπ^μν/dτ + π^μν/τ_π = 2η σ^μν + coupling terms
    - Particle diffusion: dV^μ/dτ + V^μ/τ_V = D ∇^μ(μ_B/T) + coupling terms
    """

    def __init__(
        self,
        grid: SpaceGrid | SpacetimeGrid,
        metric: MetricBase,
        coefficients: TransportCoefficients,
        spectral_solver: Any | None = None,
    ):
        """
        Initialize Israel-Stewart relaxation equations.

        Args:
            grid: Spatial grid (SpaceGrid for pure 3D or SpacetimeGrid for 4D)
            metric: Background spacetime metric
            coefficients: Transport coefficients with second-order terms
            spectral_solver: Optional spectral solver for high-accuracy derivatives
                           If provided, uses spectral derivatives instead of finite differences
        """
        self.grid = grid
        self.metric = metric
        self.coeffs = coefficients
        self.spectral_solver = spectral_solver

        # Cache Christoffel symbols and flat-space status for performance
        # (Avoids recomputing on every call to expansion_scalar/shear_tensor)
        self._is_flat = self.metric.is_flat()
        if not self._is_flat:
            from ..core.derivatives import CovariantDerivative

            cov_deriv = CovariantDerivative(self.metric)
            christoffel = cov_deriv.christoffel_symbols

            # Check if symbolic or numerical
            is_symbolic = (
                hasattr(christoffel, "dtype") and christoffel.dtype == "O"
            ) or not isinstance(christoffel, np.ndarray)

            if is_symbolic:
                # Symbolic metric - use flat space approximation
                warnings.warn(
                    "Using flat space approximation for symbolic metric",
                    UserWarning,
                    stacklevel=2,
                )
                self._christoffel = np.zeros((4, 4, 4))
            else:
                self._christoffel = christoffel
        else:
            # Flat space: Christoffel symbols are zero, never needed
            self._christoffel = None

        # Build symbolic equations for analysis
        self.symbolic_eqs = self._build_symbolic_equations()

        # Performance monitoring
        self._evolution_count = 0
        self._total_evolution_time = 0.0

    def _build_symbolic_equations(self) -> dict[str, sp.Expr]:
        """
        Build symbolic IS equations using SymPy for exact derivatives (Landau frame).

        Returns:
            Dictionary containing symbolic expressions for bulk, shear, and diffusion flux
        """
        # Define symbolic variables
        t = sp.Symbol("t", real=True)
        Pi = sp.Function("Pi")(t)

        # Tensor components (symbolic)
        pi_00, pi_01, pi_02, pi_03 = sp.symbols("pi_00 pi_01 pi_02 pi_03", real=True)
        pi_11, pi_12, pi_13, pi_22, pi_23, pi_33 = sp.symbols(
            "pi_11 pi_12 pi_13 pi_22 pi_23 pi_33", real=True
        )
        V_0, V_1, V_2, V_3 = sp.symbols(
            "V_0 V_1 V_2 V_3", real=True
        )  # Particle diffusion (Landau frame)

        # Thermodynamic and kinematic quantities
        rho, p, T = sp.symbols("rho p T", positive=True, real=True)
        mu_over_T = sp.Symbol("mu_over_T", real=True)  # Chemical potential over temperature
        theta = sp.Symbol("theta", real=True)  # Expansion scalar nabla dot u

        # Shear tensor and vorticity
        sigma_munu = sp.MatrixSymbol("sigma", 4, 4)  # Shear tensor sigma^munu
        omega_munu = sp.MatrixSymbol("omega", 4, 4)  # Vorticity tensor omega^munu

        # Transport coefficients (symbolic)
        eta, zeta, D = sp.symbols(
            "eta zeta D", positive=True, real=True
        )  # D is diffusion coefficient
        tau_pi, tau_Pi, tau_V = sp.symbols("tau_pi tau_Pi tau_V", positive=True, real=True)

        # Second-order coupling coefficients (Landau frame)
        lambda_pi_pi = sp.Symbol("lambda_pi_pi", real=True)
        lambda_pi_Pi = sp.Symbol("lambda_pi_Pi", real=True)
        lambda_pi_V = sp.Symbol("lambda_pi_V", real=True)  # Shear-diffusion coupling
        lambda_Pi_pi = sp.Symbol("lambda_Pi_pi", real=True)
        lambda_V_pi = sp.Symbol("lambda_V_pi", real=True)  # Diffusion-shear coupling
        delta_V_V = sp.Symbol("delta_V_V", real=True)  # Diffusion expansion coupling
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
            + lambda_pi_V
            * (V_0 * sp.Symbol("nabla_0_mu_over_T") + V_1 * sp.Symbol("nabla_1_mu_over_T"))
        )

        dpi_00_dt = shear_linear + shear_nonlinear

        # Particle diffusion evolution equation (Landau frame - using V^0 as representative component)
        diffusion_linear = -V_0 / tau_V - D * sp.Symbol("nabla_0_mu_over_T")
        diffusion_nonlinear = (
            lambda_V_pi * pi_00 * sp.Symbol("nabla_0_mu_over_T") - delta_V_V * V_0 * theta
        )

        dV_0_dt = diffusion_linear + diffusion_nonlinear

        return {"bulk": dPi_dt, "shear_00": dpi_00_dt, "diffusion_0": dV_0_dt}

    def compute_relaxation_rhs(self, fields: ISFieldConfiguration) -> np.ndarray:
        """
        Compute right-hand side of relaxation equations (Landau frame).

        Args:
            fields: Current field configuration

        Returns:
            Time derivatives of dissipative fluxes [dΠ/dτ, dπ^μν/dτ, dV^μ/dτ]
        """
        # Extract field components
        Pi = fields.Pi
        pi_munu = fields.pi_munu
        V_mu = fields.V_mu  # Particle diffusion current (Landau frame)
        u_mu = fields.u_mu

        # Compute kinematic quantities
        expansion_scalar = self._compute_expansion_scalar(u_mu)
        shear_tensor = self._compute_shear_tensor(u_mu)
        vorticity_tensor = self._compute_vorticity_tensor(u_mu)

        # Chemical potential gradient (projected) - Landau frame driving force
        mu_over_T_gradient = self._compute_chemical_potential_gradient(fields, u_mu)

        # Right-hand side components
        dPi_dt = self._bulk_rhs(Pi, pi_munu, expansion_scalar)
        dpi_munu_dt = self._shear_rhs(
            pi_munu,
            Pi,
            V_mu,  # Diffusion current (Landau frame)
            expansion_scalar,
            shear_tensor,
            vorticity_tensor,
            mu_over_T_gradient,  # Chemical potential gradient
            fields.temperature,  # Temperature for dimensional scaling
        )
        dV_mu_dt = self._diffusion_rhs(
            V_mu, pi_munu, expansion_scalar, mu_over_T_gradient, fields.temperature
        )

        # Pack into dissipative vector format
        return np.concatenate([dPi_dt.flatten(), dpi_munu_dt.reshape(-1), dV_mu_dt.reshape(-1)])

    def _bulk_rhs(self, Pi: np.ndarray, pi_munu: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute bulk pressure evolution RHS.

        Israel-Stewart equation: dΠ/dt = -Π/τ_Π - ζθ

        IMPORTANT: For IMEX time integration, the linear term -Π/τ_Π is handled
        implicitly, so the spectral solver adds it back to get the explicit part.
        This function computes the FULL RHS including both linear and source terms.
        The IMEX splitting is done in spectral.py, not here.

        Returns:
            Full RHS: -Π/τ_Π - ζθ + (second-order terms)
        """
        # Linear relaxation term: -Π/τ_Π
        linear = (
            -Pi / self.coeffs.bulk_relaxation_time
            if self.coeffs.bulk_relaxation_time
            else np.zeros_like(Pi)
        )

        # First-order source: -ζ*θ (Form B - standard IReD formulation)
        # NOTE: This is the correct Israel-Stewart/IReD relaxation equation form.
        # See docs/IRED_THEORY.md and Wagner, Palermo, Ambrus (2022), arXiv:2203.12608.
        # The apparent "paradox" with dispersion relations was resolved—this form
        # correctly implements operator splitting, not algebraic solution.
        if self.coeffs.bulk_viscosity:
            first_order = -self.coeffs.bulk_viscosity * theta
        else:
            first_order = np.zeros_like(Pi)

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
        # NOTE: The shear tensor π^μν is traceless by definition (g_μν π^μν = 0),
        # so this coupling term is identically zero. The standard IReD formulation
        # does not include a λ_Ππ coefficient for bulk-shear coupling.
        # See docs/IRED_THEORY.md Section 2 for proper coupling structure.
        if self.coeffs.lambda_Pi_pi != 0:
            warnings.warn(
                "lambda_Pi_pi coefficient is non-zero, but shear tensor is traceless. "
                "This coupling term has no effect. Check IReD formulation.",
                UserWarning,
                stacklevel=3,
            )
        result: np.ndarray = linear + first_order + nonlinear
        return result

    def _shear_rhs(
        self,
        pi_munu: np.ndarray,
        Pi: np.ndarray,
        V_mu: np.ndarray,
        theta: np.ndarray,
        sigma_munu: np.ndarray,
        omega_munu: np.ndarray,
        nabla_mu_over_T: np.ndarray,
        temperature: np.ndarray,
    ) -> np.ndarray:
        """
        Compute shear stress tensor evolution RHS (Landau frame).

        Israel-Stewart equation: dπ^μν/dτ = -π^μν/τ_π + 2η σ^μν + coupling terms

        IMPORTANT: For IMEX time integration, the linear term -π/τ_π is handled
        implicitly, so the spectral solver adds it back to get the explicit part.
        This function computes the FULL RHS including both linear and source terms.
        The IMEX splitting is done in spectral.py, not here.

        Args:
            pi_munu: Shear stress tensor π^μν
            Pi: Bulk viscous pressure Π
            V_mu: Particle diffusion current V^μ (Landau frame)
            theta: Expansion scalar θ = ∇_μ u^μ
            sigma_munu: Shear tensor σ^μν
            omega_munu: Vorticity tensor ω^μν
            nabla_mu_over_T: Chemical potential gradient ∇^μ(μ_B/T) (Landau frame)
            temperature: Temperature field T (GeV) for dimensional scaling

        Returns:
            Full RHS: -π^μν/τ_π + 2η σ^μν + (second-order coupling terms)
        """
        # Linear relaxation: -π^μν/τ_π
        linear = (
            -pi_munu / self.coeffs.shear_relaxation_time
            if self.coeffs.shear_relaxation_time
            else np.zeros_like(pi_munu)
        )

        # First-order source: 2η σ^μν (Form B - standard IReD formulation)
        # NOTE: This is the correct Israel-Stewart/IReD relaxation equation form.
        # See docs/LANDAU_FRAME_FORMULATION.md and Wagner, Palermo, Ambrus (2022), arXiv:2203.12608.
        if self.coeffs.shear_viscosity:
            first_order = 2.0 * self.coeffs.shear_viscosity * sigma_munu
        else:
            first_order = np.zeros_like(pi_munu)

        # Second-order terms
        nonlinear = np.zeros_like(pi_munu)

        # Expansion coupling: λ_ππ π^μν θ
        if self.coeffs.lambda_pi_pi != 0:
            from ..core.tensor_utils import optimized_einsum

            expansion_term = self.coeffs.lambda_pi_pi * optimized_einsum(
                "...ij,...->...ij", pi_munu, theta
            )
            nonlinear += expansion_term

        # Shear-bulk coupling: λ_πΠ Π σ^μν
        if self.coeffs.lambda_pi_Pi != 0:
            # Broadcast Pi scalar to tensor shape
            bulk_coupling = self.coeffs.lambda_pi_Pi * Pi[..., np.newaxis, np.newaxis] * sigma_munu
            nonlinear += bulk_coupling

        # Shear-particle diffusion coupling (Landau frame)
        # Term: λ_πV * (V^μ ∇^ν(μ_B/T) + V^ν ∇^μ(μ_B/T)) / 2
        # This couples shear stress to particle diffusion gradients
        # NOTE: λ_πV from IReD is ALREADY DIMENSIONLESS (Table III: 0.20890 τ_π/β)
        # DO NOT multiply by T - it has correct dimensions as-is!
        if self.coeffs.lambda_pi_V != 0:
            from ..core.tensor_utils import optimized_einsum

            # Outer product: V^μ ∇^ν(μ_B/T)
            outer_product = optimized_einsum("...i,...j->...ij", V_mu, nabla_mu_over_T)
            # Symmetrize: (V^μ ∇^ν + V^ν ∇^μ) / 2
            diffusion_term = (
                self.coeffs.lambda_pi_V
                * 0.5
                * (outer_product + np.swapaxes(outer_product, -1, -2))
            )
            nonlinear += diffusion_term

        # Nonlinear shear self-coupling terms
        # NOTE: This is an O(Re⁻²) R term, not an O(Re⁻¹Kn) J term.
        # In the DNMR/IReD classification:
        #   - J terms are O(Re⁻¹Kn): first-order gradients times dissipative fluxes
        #   - R terms are O(Re⁻²): quadratic in dissipative fluxes
        # This π^μ_ρ π^ρ_ν term is a higher-order correction beyond standard IS theory.
        # It appears in some extended formulations but should be justified/documented.
        # See Denicol et al. (2012) PRD 85:114047 for full R term structure.
        if self.coeffs.tau_pi_pi != 0 and self.coeffs.shear_relaxation_time:
            from ..core.tensor_utils import optimized_einsum

            # Term: -τ_ππ/(η·τ_π) · π^μ_ρ π^ρ_ν
            pi_pi_term = -self.coeffs.tau_pi_pi * optimized_einsum(
                "...ik,...kj->...ij", pi_munu, pi_munu
            )
            pi_pi_term /= self.coeffs.shear_viscosity * self.coeffs.shear_relaxation_time
            nonlinear += pi_pi_term

        # Vorticity coupling: τ_πω (π^μ_α ω^α_ν - ω^μ_α π^α_ν)
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

    def _diffusion_rhs(
        self,
        V_mu: np.ndarray,
        pi_munu: np.ndarray,
        theta: np.ndarray,
        nabla_mu_over_T: np.ndarray,
        temperature: np.ndarray,
    ) -> np.ndarray:
        """
        Compute particle diffusion current evolution (Landau frame).

        In Landau frame, the particle diffusion current V^μ is the dissipative flux
        that appears in the particle current: J^μ = n u^μ + V^μ.
        It satisfies the orthogonality condition V^μ u_μ = 0.

        Evolution equation:
            dV^μ/dτ + V^μ/τ_V = -D ∇^μ(μ_B/T) + coupling terms

        Fick's law: V^μ = -D ∇^μ(μ_B/T) (particles flow down chemical potential gradient)

        where:
            - D is the diffusion coefficient (not thermal conductivity!)
            - μ_B/T is the baryon chemical potential over temperature
            - τ_V is the diffusion relaxation time
            - Negative sign ensures particles flow from high μ to low μ

        Args:
            V_mu: Current particle diffusion current V^μ (Landau frame)
            pi_munu: Shear stress tensor π^μν
            theta: Expansion scalar θ = ∇_μ u^μ
            nabla_mu_over_T: Projected gradient of chemical potential ∇^μ(μ_B/T)
            temperature: Temperature field T (GeV) for dimensional scaling

        Returns:
            Time derivative dV^μ/dτ

        See:
            Wagner & Gavassino (2024) IReD paper, Section 3.2
            docs/LANDAU_FRAME_FORMULATION.md for derivation
        """
        # Linear relaxation term: -V^μ/τ_V
        linear = (
            -V_mu / self.coeffs.diffusion_relaxation_time
            if self.coeffs.diffusion_relaxation_time
            else np.zeros_like(V_mu)
        )

        # First-order source: -D ∇^μ(μ_B/T) (Fick's law)
        # Negative sign: particles flow DOWN chemical potential gradient (from high μ to low μ)
        first_order = -self.coeffs.diffusion_coefficient * nabla_mu_over_T

        # Second-order nonlinear terms
        nonlinear = np.zeros_like(V_mu)

        # Expansion coupling: -δ_VV V^μ θ
        # This is the CORRECT coefficient from IReD Eq. (29b): J^μ = −δₙₙ n^μ θ
        # δ_VV is dimensionless (= 1 for hard sphere gas, IReD Table III)
        # Expansion of fluid suppresses diffusion current
        if self.coeffs.delta_V_V != 0:
            expansion_term = -self.coeffs.delta_V_V * V_mu * theta[..., np.newaxis]
            nonlinear += expansion_term

        # Shear-diffusion coupling: λ_Vπ * T² * π^μν ∇_ν(μ_B/T)
        # Shear flow couples to diffusion gradients
        # NOTE: λ_Vπ from IReD has units GeV⁻² (= 0.069240 β τ_V)
        # Multiply by T² for dimensional consistency: [λ_Vπ × T²] = dimensionless
        if self.coeffs.lambda_V_pi != 0:
            from ..core.tensor_utils import optimized_einsum

            # Term: λ_Vπ * T² * π^μν ∇_ν(μ_B/T)
            # Scale by T² for dimensional consistency
            shear_diffusion_term = (
                self.coeffs.lambda_V_pi
                * (temperature[..., np.newaxis] ** 2)
                * optimized_einsum("...ij,...j->...i", pi_munu, nabla_mu_over_T)
            )
            nonlinear += shear_diffusion_term

        result: np.ndarray = linear + first_order + nonlinear
        return result

    def _compute_expansion_scalar(self, u_mu: np.ndarray) -> np.ndarray:
        """
        Compute expansion scalar θ = ∇_μ u^μ using proper covariant derivatives.

        Uses a field-aware wrapper to interface with the corrected vector_divergence method.
        """
        # Compute divergence using manual gradients and Christoffel symbols
        # ∇_μ u^μ = ∂_μ u^μ + Γ^μ_μν u^ν

        # Use cached Christoffel symbols (computed once at initialization)
        christoffel = self._christoffel

        # Compute partial derivatives ∂_μ u^μ
        # Detect grid type: SpaceGrid (3D) vs SpacetimeGrid (4D)
        is_spacegrid = u_mu.ndim == 4 and u_mu.shape[-1] == 4  # (nx,ny,nz,4)

        partial_div = np.zeros(u_mu.shape[:-1])
        if is_spacegrid:
            # Pure 3D: only spatial derivatives (mu=1,2,3 → axes 0,1,2)
            # Time derivative (mu=0) handled by time evolution, not spatial gradients

            # Use spectral derivatives if available (machine precision accuracy)
            if self.spectral_solver is not None and hasattr(
                self.spectral_solver, "spatial_derivative"
            ):
                for mu in range(1, 4):
                    spatial_axis = mu - 1  # Map mu=1,2,3 → axis=0,1,2
                    partial_div += self.spectral_solver.spatial_derivative(
                        u_mu[..., mu], direction=spatial_axis
                    )
            # Fall back to grid.gradient() for proper boundary condition handling
            elif hasattr(self.grid, "gradient"):
                for mu in range(1, 4):
                    spatial_axis = mu - 1  # Map mu=1,2,3 → axis=0,1,2
                    partial_div += self.grid.gradient(u_mu[..., mu], axis=spatial_axis, order=2)
            else:
                for mu in range(1, 4):
                    spatial_axis = mu - 1  # Map mu=1,2,3 → axis=0,1,2
                    partial_div += np.gradient(u_mu[..., mu], axis=spatial_axis, edge_order=1)
        else:
            # 4D spacetime: all derivatives including time (legacy SpacetimeGrid)
            for mu in range(4):
                partial_div += np.gradient(u_mu[..., mu], axis=mu, edge_order=1)

        # Add Christoffel term if metric is not flat (use cached flag)
        if not self._is_flat:
            christoffel_term = np.zeros(u_mu.shape[:-1])
            for mu in range(4):
                for nu in range(4):
                    christoffel_term += christoffel[mu, mu, nu] * u_mu[..., nu]
            theta = partial_div + christoffel_term
        else:
            # Flat space: Christoffel contribution is zero
            theta = partial_div

        return theta

    def _compute_shear_tensor(self, u_mu: np.ndarray) -> np.ndarray:
        """
        Compute shear tensor σ^μν using manual gradients and Christoffel symbols.

        Formula: σ^μν = ∇^(μ u^ν) + a^(μ u^ν) - (1/3)Δ^μν θ
        """
        from ..core.tensor_utils import optimized_einsum

        # Use cached Christoffel symbols (computed once at initialization)
        christoffel = self._christoffel

        # Detect grid type: SpaceGrid (3D) vs SpacetimeGrid (4D)
        is_spacegrid = u_mu.ndim == 4 and u_mu.shape[-1] == 4  # (nx,ny,nz,4)

        # Compute velocity gradients using spectral or finite differences
        nabla_u_partial = np.zeros(u_mu.shape[:-1] + (4, 4))
        if is_spacegrid:
            # Pure 3D: only spatial derivatives (mu=1,2,3 → axes 0,1,2)

            # Use spectral derivatives if available (machine precision accuracy)
            if self.spectral_solver is not None and hasattr(
                self.spectral_solver, "spatial_derivative"
            ):
                for mu in range(1, 4):
                    spatial_axis = mu - 1  # Map mu=1,2,3 → axis=0,1,2
                    for nu in range(4):
                        nabla_u_partial[..., mu, nu] = self.spectral_solver.spatial_derivative(
                            u_mu[..., nu], direction=spatial_axis
                        )
            # Fall back to grid.gradient() for proper boundary condition handling
            elif hasattr(self.grid, "gradient"):
                for mu in range(1, 4):
                    spatial_axis = mu - 1  # Map mu=1,2,3 → axis=0,1,2
                    for nu in range(4):
                        nabla_u_partial[..., mu, nu] = self.grid.gradient(
                            u_mu[..., nu], axis=spatial_axis, order=2
                        )
            else:
                for mu in range(1, 4):
                    spatial_axis = mu - 1  # Map mu=1,2,3 → axis=0,1,2
                    for nu in range(4):
                        nabla_u_partial[..., mu, nu] = np.gradient(
                            u_mu[..., nu], axis=spatial_axis, edge_order=1
                        )
        else:
            # 4D spacetime: all derivatives (legacy SpacetimeGrid)
            for mu in range(4):
                for nu in range(4):
                    nabla_u_partial[..., mu, nu] = np.gradient(u_mu[..., nu], axis=mu, edge_order=1)

        # Add Christoffel correction if metric is not flat (use cached flag)
        if not self._is_flat:
            nabla_u = nabla_u_partial.copy()
            for mu in range(4):
                for nu in range(4):
                    for rho in range(4):
                        nabla_u[..., mu, nu] -= christoffel[rho, mu, nu] * u_mu[..., rho]
        else:
            # Flat space: no Christoffel correction needed
            nabla_u = nabla_u_partial

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
        from ..core.tensor_utils import optimized_einsum

        # Use cached Christoffel symbols (computed once at initialization)
        christoffel = self._christoffel

        # Detect grid type: SpaceGrid (3D) vs SpacetimeGrid (4D)
        is_spacegrid = u_mu.ndim == 4 and u_mu.shape[-1] == 4  # (nx,ny,nz,4)

        # Compute velocity gradients using spectral or finite differences
        nabla_u_partial = np.zeros(u_mu.shape[:-1] + (4, 4))
        if is_spacegrid:
            # Pure 3D: only spatial derivatives (mu=1,2,3 → axes 0,1,2)

            # Use spectral derivatives if available (machine precision accuracy)
            if self.spectral_solver is not None and hasattr(
                self.spectral_solver, "spatial_derivative"
            ):
                for mu in range(1, 4):
                    spatial_axis = mu - 1  # Map mu=1,2,3 → axis=0,1,2
                    for nu in range(4):
                        nabla_u_partial[..., mu, nu] = self.spectral_solver.spatial_derivative(
                            u_mu[..., nu], direction=spatial_axis
                        )
            # Fall back to grid.gradient() for proper boundary condition handling
            elif hasattr(self.grid, "gradient"):
                for mu in range(1, 4):
                    spatial_axis = mu - 1  # Map mu=1,2,3 → axis=0,1,2
                    for nu in range(4):
                        nabla_u_partial[..., mu, nu] = self.grid.gradient(
                            u_mu[..., nu], axis=spatial_axis, order=2
                        )
            else:
                for mu in range(1, 4):
                    spatial_axis = mu - 1  # Map mu=1,2,3 → axis=0,1,2
                    for nu in range(4):
                        nabla_u_partial[..., mu, nu] = np.gradient(
                            u_mu[..., nu], axis=spatial_axis, edge_order=1
                        )
        else:
            # 4D spacetime: all derivatives (legacy SpacetimeGrid)
            for mu in range(4):
                for nu in range(4):
                    nabla_u_partial[..., mu, nu] = np.gradient(u_mu[..., nu], axis=mu, edge_order=1)

        # Add Christoffel correction if metric is not flat (use cached flag)
        if not self._is_flat:
            nabla_u = nabla_u_partial.copy()
            for mu in range(4):
                for nu in range(4):
                    for rho in range(4):
                        nabla_u[..., mu, nu] -= christoffel[rho, mu, nu] * u_mu[..., rho]
        else:
            # Flat space: no Christoffel correction needed
            nabla_u = nabla_u_partial

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
        # Detect grid type: SpaceGrid (3D) vs SpacetimeGrid (4D)
        is_spacegrid = T.ndim == 3  # (nx,ny,nz) for SpaceGrid

        grad_T_lower = np.zeros(T.shape + (4,))
        if is_spacegrid:
            # Pure 3D: only spatial derivatives (mu=1,2,3 → axes 0,1,2)
            # Use grid.gradient() for proper boundary condition handling
            if hasattr(self.grid, "gradient"):
                for mu in range(1, 4):
                    spatial_axis = mu - 1  # Map mu=1,2,3 → axis=0,1,2
                    grad_T_lower[..., mu] = self.grid.gradient(T, axis=spatial_axis, order=2)
            else:
                for mu in range(1, 4):
                    spatial_axis = mu - 1  # Map mu=1,2,3 → axis=0,1,2
                    grad_T_lower[..., mu] = np.gradient(T, axis=spatial_axis, edge_order=1)
        else:
            # 4D spacetime: all derivatives (legacy SpacetimeGrid)
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

    def _compute_chemical_potential_gradient(
        self, fields: ISFieldConfiguration, u_mu: np.ndarray
    ) -> np.ndarray:
        """
        Compute projected chemical potential gradient ∇^μ(μ_B/T) (Landau frame).

        This is the driving force for particle diffusion in Landau frame:
            dV^μ/dτ = -V^μ/τ_V + D ∇^μ(μ_B/T) + ...

        Uses the chemical potential computation from ISFieldConfiguration:
            μ_B/T = ln(n/n_eq(T)) for radiation fluid

        Args:
            fields: Current field configuration
            u_mu: Four-velocity field

        Returns:
            Projected gradient ∇^μ(μ_B/T) with shape (..., 4)
        """
        from ..core.tensor_utils import optimized_einsum

        # Compute chemical potential over temperature field
        mu_over_T = fields.compute_chemical_potential_over_temperature(eos_type="radiation")

        # Detect grid type: SpaceGrid (3D) vs SpacetimeGrid (4D)
        is_spacegrid = mu_over_T.ndim == 3  # (nx,ny,nz) for SpaceGrid

        # Compute gradient of μ_B/T using finite differences: ∂_μ(μ_B/T)
        grad_mu_lower = np.zeros(mu_over_T.shape + (4,))
        if is_spacegrid:
            # Pure 3D: only spatial derivatives (mu=1,2,3 → axes 0,1,2)
            # Use spectral derivatives if available for high accuracy
            if self.spectral_solver is not None and hasattr(
                self.spectral_solver, "spatial_derivative"
            ):
                for mu in range(1, 4):
                    spatial_axis = mu - 1  # Map mu=1,2,3 → axis=0,1,2
                    grad_mu_lower[..., mu] = self.spectral_solver.spatial_derivative(
                        mu_over_T, direction=spatial_axis
                    )
            # Fall back to grid.gradient() for proper boundary condition handling
            elif hasattr(self.grid, "gradient"):
                for mu in range(1, 4):
                    spatial_axis = mu - 1  # Map mu=1,2,3 → axis=0,1,2
                    grad_mu_lower[..., mu] = self.grid.gradient(
                        mu_over_T, axis=spatial_axis, order=2
                    )
            else:
                for mu in range(1, 4):
                    spatial_axis = mu - 1  # Map mu=1,2,3 → axis=0,1,2
                    grad_mu_lower[..., mu] = np.gradient(mu_over_T, axis=spatial_axis, edge_order=1)
        else:
            # 4D spacetime: all derivatives (legacy SpacetimeGrid)
            for mu in range(4):
                grad_mu_lower[..., mu] = np.gradient(mu_over_T, axis=mu, edge_order=1)

        # Get metric inverse for raising indices
        g_inv = self.metric.inverse
        if isinstance(g_inv, np.ndarray) and g_inv.ndim == 2:
            # Broadcast metric to match field dimensions
            g_inv = np.broadcast_to(g_inv, mu_over_T.shape + (4, 4))

        # Raise gradient indices: ∇^μ(μ_B/T) = g^μν ∇_ν(μ_B/T)
        grad_mu_up = optimized_einsum("...ab,...b->...a", g_inv, grad_mu_lower)

        # Compute perpendicular projector: Δ^μν = g^μν + u^μ u^ν
        # Need to align u_mu with mu_over_T dimensions
        if u_mu.shape[:-1] != mu_over_T.shape:
            # Assume mu_over_T has same spatial dimensions as u_mu
            u_aligned = u_mu[: mu_over_T.shape[0], : mu_over_T.shape[1], : mu_over_T.shape[2], :]
        else:
            u_aligned = u_mu

        delta = g_inv + optimized_einsum("...a,...b->...ab", u_aligned, u_aligned)

        # Project gradient to spatial hypersurface: ∇^μ(μ_B/T) = Δ^μν ∇_ν(μ_B/T)
        nabla_mu_over_T = optimized_einsum("...ab,...b->...a", delta, grad_mu_lower)

        return nabla_mu_over_T

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
        """Exponential time differencing for relaxation equations (Landau frame)."""
        # Extract relaxation times
        tau_pi = self.coeffs.shear_relaxation_time or 1.0
        tau_Pi = self.coeffs.bulk_relaxation_time or 1.0
        tau_V = self.coeffs.diffusion_relaxation_time or 1.0

        # Build diagonal relaxation matrix (simplified)
        grid_size = int(np.prod(fields.grid.shape))

        # Relaxation eigenvalues
        lambda_pi = 1.0 / tau_pi
        lambda_Pi = 1.0 / tau_Pi
        lambda_V = 1.0 / tau_V

        # Exponential factors
        exp_pi = np.exp(-lambda_pi * dt)
        exp_Pi = np.exp(-lambda_Pi * dt)
        exp_V = np.exp(-lambda_V * dt)

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

        # Particle diffusion current (Landau frame)
        V_size = 4 * grid_size
        V_old = x_old[offset : offset + V_size]
        V_rhs = rhs[offset : offset + V_size]
        V_new = exp_V * V_old + (1 - exp_V) / lambda_V * V_rhs

        # Reconstruct solution vector
        x_new = np.concatenate([Pi_new, pi_new, V_new])

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
        Analyze stability of relaxation equations at current state (Landau frame).

        Args:
            fields: Current field configuration

        Returns:
            Stability analysis results
        """
        # Estimate characteristic timescales
        tau_pi = self.coeffs.shear_relaxation_time or 1.0
        tau_Pi = self.coeffs.bulk_relaxation_time or 1.0
        tau_V = self.coeffs.diffusion_relaxation_time or 1.0

        # Characteristic values
        Pi_char = np.max(np.abs(fields.Pi)) if np.any(fields.Pi) else 1e-10
        pi_char = np.max(np.abs(fields.pi_munu)) if np.any(fields.pi_munu) else 1e-10
        V_char = np.max(np.abs(fields.V_mu)) if np.any(fields.V_mu) else 1e-10

        # Stiffness ratios
        stiffness_ratio = max(tau_Pi, tau_pi, tau_V) / min(tau_Pi, tau_pi, tau_V)

        # Recommended timestep (stability constraint)
        dt_max = 0.1 * min(tau_Pi, tau_pi, tau_V)

        return {
            "relaxation_times": {"tau_pi": tau_pi, "tau_Pi": tau_Pi, "tau_V": tau_V},
            "characteristic_values": {"Pi": Pi_char, "pi": pi_char, "V": V_char},
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
