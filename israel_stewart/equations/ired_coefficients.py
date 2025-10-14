"""
IReD Transport Coefficients

This module implements the full Inverse-Reynolds-Dominance (IReD) formulation
for computing transport coefficients from kinetic theory, following:

Wagner, Palermo, Ambrus (2022), "Inverse-Reynolds-Dominance approach to
transient fluid dynamics", arXiv:2203.12608v2 [nucl-th]

The IReD approach computes all transport coefficients (first-order and
second-order) from the linearized Boltzmann equation, eliminating parabolic
K terms by construction.

Architecture:
- EquationOfState: Thermodynamic state (P, ε, n as functions of T, μ)
- ThermodynamicIntegrals: Momentum integrals (I_rq, J_rq, D_rq)
- CollisionMatrix: Linearized collision operator A^(ℓ)_{rn}
- FirstOrderCoefficients: η, ζ, κ (Landau frame)
- CoefficientRatios: C^(ℓ)_r = (coeff_r)/(coeff_0)
- IReD RelaxationTimes: τ_Π, τ_π, τ_V (weighted averages)
- SecondOrderCoefficients: All J term couplings (Appendix B)
- IReD TransportCoefficients: Main interface

Key Equations:
- IReD eq. (19): First-order coefficients (η, ζ, κ)
- IReD eq. (38): Relaxation times (weighted averages)
- IReD Appendix B: Second-order coefficients (20+ terms)

Usage:
    >>> from israel_stewart.equations.ired_coefficients import HardSphereGas
    >>> model = HardSphereGas(temperature=0.4, sigma=1.0)
    >>> print(f"η/s = {model.shear_viscosity() / model.entropy_density():.4f}")
    >>> print(f"τ_π = {model.tau_pi():.4f} fm/c")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import integrate, special
from scipy.linalg import inv, pinv


# ============================================================================
# Part 1: Equation of State
# ============================================================================


class EquationOfState(ABC):
    """
    Abstract base class for equation of state.

    An EOS provides thermodynamic quantities (P, ε, n) as functions of
    temperature T and chemical potential μ.

    Attributes:
        temperature: Temperature in natural units (GeV)
        chemical_potential: Chemical potential μ in natural units (GeV)
        mass: Particle mass m in natural units (GeV)
    """

    def __init__(self, temperature: float, chemical_potential: float = 0.0, mass: float = 0.0):
        """
        Initialize equation of state.

        Args:
            temperature: Temperature T (GeV)
            chemical_potential: Chemical potential μ (GeV)
            mass: Particle mass m (GeV)
        """
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

        self.temperature = temperature
        self.chemical_potential = chemical_potential
        self.mass = mass

        # Derived quantities
        self.beta = 1.0 / temperature  # Inverse temperature
        self.alpha = chemical_potential / temperature  # Dimensionless chemical potential

    @abstractmethod
    def pressure(self) -> float:
        """Compute pressure P."""
        pass

    @abstractmethod
    def energy_density(self) -> float:
        """Compute energy density ε."""
        pass

    @abstractmethod
    def particle_density(self) -> float:
        """Compute particle number density n."""
        pass

    def enthalpy_density(self) -> float:
        """Compute enthalpy density h = ε + P."""
        return self.energy_density() + self.pressure()

    def specific_enthalpy(self) -> float:
        """Compute specific enthalpy h = (ε+P)/n."""
        n = self.particle_density()
        if n == 0:
            raise ValueError("Particle density is zero, specific enthalpy undefined")
        return self.enthalpy_density() / n

    def entropy_density(self) -> float:
        """Compute entropy density s = (ε + P - μn)/T."""
        return (self.energy_density() + self.pressure()
                - self.chemical_potential * self.particle_density()) / self.temperature

    def sound_speed_squared(self) -> float:
        """
        Compute speed of sound squared c_s² = dP/dε.

        Default implementation uses finite differences. Override for analytic result.
        """
        dT = 1e-6 * self.temperature

        # Save original state
        T_orig = self.temperature

        # Compute P(T + dT) and ε(T + dT)
        self.temperature = T_orig + dT
        P_plus = self.pressure()
        eps_plus = self.energy_density()

        # Compute P(T - dT) and ε(T - dT)
        self.temperature = T_orig - dT
        P_minus = self.pressure()
        eps_minus = self.energy_density()

        # Restore original state
        self.temperature = T_orig

        # Compute derivative
        dP_dT = (P_plus - P_minus) / (2 * dT)
        deps_dT = (eps_plus - eps_minus) / (2 * dT)

        if deps_dT == 0:
            raise ValueError("Energy density derivative is zero")

        return dP_dT / deps_dT


class IdealGasEOS(EquationOfState):
    """
    Ideal relativistic gas equation of state.

    For quantum statistics:
    - Fermions: Fermi-Dirac distribution
    - Bosons: Bose-Einstein distribution
    - Classical: Boltzmann distribution (high T limit)

    Uses exact integrals of the distribution function.
    """

    def __init__(
        self,
        temperature: float,
        chemical_potential: float = 0.0,
        mass: float = 0.0,
        degeneracy: float = 1.0,
        statistics: str = "classical",
    ):
        """
        Initialize ideal gas EOS.

        Args:
            temperature: Temperature T (GeV)
            chemical_potential: Chemical potential μ (GeV)
            mass: Particle mass m (GeV)
            degeneracy: Spin degeneracy g (e.g., g=2 for spin-1/2)
            statistics: 'classical', 'fermi', or 'bose'
        """
        super().__init__(temperature, chemical_potential, mass)
        self.degeneracy = degeneracy
        self.statistics = statistics.lower()

        if self.statistics not in ["classical", "fermi", "bose"]:
            raise ValueError(f"Unknown statistics: {statistics}")

    def _distribution_sign(self) -> int:
        """Get sign for quantum statistics: +1 (Fermi), -1 (Bose), 0 (Classical)."""
        if self.statistics == "fermi":
            return 1
        elif self.statistics == "bose":
            return -1
        else:
            return 0

    def _integral_I(self, n: int) -> float:
        """
        Compute thermodynamic integral:
        I_n = ∫ d³p/(2π)³ E^n f_0(E)

        where E = sqrt(p² + m²) and f_0 is the equilibrium distribution.
        """
        sign = self._distribution_sign()

        def integrand(p):
            E = np.sqrt(p**2 + self.mass**2)
            exp_arg = self.beta * (E - self.chemical_potential)

            # Avoid overflow
            if exp_arg > 100:
                return 0.0

            if sign == 0:  # Classical
                f = np.exp(-exp_arg)
            else:  # Quantum
                f = 1.0 / (np.exp(exp_arg) + sign)

            return (self.degeneracy / (2 * np.pi**2)) * p**2 * E**n * f

        # Integrate from 0 to infinity
        result, _ = integrate.quad(integrand, 0, np.inf, limit=100)
        return result

    def pressure(self) -> float:
        """
        Compute pressure P = (1/3) ∫ d³p/(2π)³ (p²/E) f_0(E).

        This is I_0 with weight p²/E instead of E^0.
        """
        sign = self._distribution_sign()

        def integrand(p):
            E = np.sqrt(p**2 + self.mass**2)
            exp_arg = self.beta * (E - self.chemical_potential)

            if exp_arg > 100:
                return 0.0

            if sign == 0:  # Classical
                f = np.exp(-exp_arg)
            else:  # Quantum
                f = 1.0 / (np.exp(exp_arg) + sign)

            return (self.degeneracy / (2 * np.pi**2)) * (p**4 / (3 * E)) * f

        result, _ = integrate.quad(integrand, 0, np.inf, limit=100)
        return result

    def energy_density(self) -> float:
        """Compute energy density ε = ∫ d³p/(2π)³ E f_0(E)."""
        return self._integral_I(1)

    def particle_density(self) -> float:
        """Compute particle density n = ∫ d³p/(2π)³ f_0(E)."""
        return self._integral_I(0)


class UltrarelativisticIdealGas(EquationOfState):
    """
    Ultrarelativistic ideal gas (m → 0 limit).

    For massless particles, thermodynamic integrals can be computed analytically
    using polylogarithm functions.

    This provides exact results and is much faster than numerical integration.
    """

    def __init__(
        self,
        temperature: float,
        chemical_potential: float = 0.0,
        degeneracy: float = 1.0,
        statistics: str = "classical",
    ):
        """Initialize ultrarelativistic ideal gas (m=0)."""
        super().__init__(temperature, chemical_potential, mass=0.0)
        self.degeneracy = degeneracy
        self.statistics = statistics.lower()

    def _polylog_integral(self, n: int, sign: int) -> float:
        """
        Compute polylogarithm integral for massless particles.

        I_n = (g T^{n+1})/(2π²) ∫₀^∞ dx x^{n+2} / (e^{x-α} ± 1)
            = (g T^{n+1})/(2π²) Γ(n+3) [Li_{n+3}(±e^α) for Bose/Fermi
                                        or e^α for classical]
        """
        if sign == 0:  # Classical (Boltzmann)
            # Analytic result: (g T^{n+1})/(2π²) Γ(n+3) e^α
            return (self.degeneracy * self.temperature**(n + 1) / (2 * np.pi**2)
                    * special.gamma(n + 3) * np.exp(self.alpha))
        else:
            # Quantum: use polylogarithm Li_{n+3}(z) with z = ±exp(α)
            z = sign * np.exp(self.alpha)
            polylog = special.zeta(n + 3, 1) if abs(z) < 1e-10 else self._polylog(n + 3, z)
            return (self.degeneracy * self.temperature**(n + 1) / (2 * np.pi**2)
                    * special.gamma(n + 3) * polylog)

    def _polylog(self, s: int, z: float) -> float:
        """
        Compute polylogarithm Li_s(z) = Σ_{k=1}^∞ z^k / k^s.

        For now, use series expansion (works well for |z| < 1).
        For better accuracy, could use mpmath.polylog.
        """
        if abs(z) < 1e-10:
            return special.zeta(s, 1)

        # Series expansion (converges for |z| < 1)
        max_terms = 100
        result = 0.0
        for k in range(1, max_terms + 1):
            term = z**k / k**s
            result += term
            if abs(term) < 1e-15:
                break

        return result

    def pressure(self) -> float:
        """Pressure for massless gas: P = ε/3."""
        return self.energy_density() / 3.0

    def energy_density(self) -> float:
        """Energy density ε for massless particles."""
        sign = 1 if self.statistics == "fermi" else (-1 if self.statistics == "bose" else 0)
        return self._polylog_integral(1, sign)

    def particle_density(self) -> float:
        """Particle density n for massless particles."""
        sign = 1 if self.statistics == "fermi" else (-1 if self.statistics == "bose" else 0)
        return self._polylog_integral(0, sign)

    def sound_speed_squared(self) -> float:
        """Speed of sound squared for ultrarelativistic gas: c_s² = 1/3."""
        return 1.0 / 3.0


# ============================================================================
# Part 2: Thermodynamic Integrals
# ============================================================================


@dataclass
class ThermodynamicIntegrals:
    """
    Thermodynamic integrals for IReD transport coefficient calculation.

    These integrals appear in the moment expansion of the Boltzmann equation.
    Following IReD eq. (9):

    I_{rq} = ∫ dK E_k^r k^q δf_k       (energy-weighted)
    J_{rq} = ∫ dK E_k^r k^q (∂f₀/∂E) (particle-weighted derivative)
    D_{rq} = ∫ dK E_k^r k^q (∂f₀/∂α) (chemical potential derivative)

    where:
    - dK = g d³k/[(2π)³E_k] is the Lorentz-invariant measure
    - E_k = sqrt(k² + m²) is the on-shell energy
    - f₀ is the local equilibrium distribution

    These integrals depend on the equation of state and are needed for computing
    transport coefficient ratios C^(ℓ)_r.
    """

    eos: EquationOfState
    max_r: int = 10  # Maximum r index to compute
    max_q: int = 10  # Maximum q index to compute

    # Cached values
    _I_cache: Optional[dict] = None
    _J_cache: Optional[dict] = None
    _D_cache: Optional[dict] = None

    def __post_init__(self):
        """Initialize caches."""
        self._I_cache = {}
        self._J_cache = {}
        self._D_cache = {}

    def I_rq(self, r: int, q: int) -> float:
        """
        Compute energy-weighted integral I_{rq}.

        I_{rq} = ∫ dK E^r k^q δf

        Args:
            r: Energy power
            q: Momentum power

        Returns:
            Value of integral
        """
        key = (r, q)
        if key in self._I_cache:
            return self._I_cache[key]

        # Compute integral
        result = self._compute_I_integral(r, q)
        self._I_cache[key] = result
        return result

    def _compute_I_integral(self, r: int, q: int) -> float:
        """Compute I_{rq} integral numerically."""
        def integrand(k):
            E = np.sqrt(k**2 + self.eos.mass**2)
            exp_arg = self.eos.beta * (E - self.eos.chemical_potential)

            # Avoid overflow
            if exp_arg > 100:
                return 0.0

            # Distribution function (assuming classical for now)
            f0 = np.exp(-exp_arg)

            # I_rq uses equilibrium distribution
            return (self.eos.degeneracy / (2 * np.pi**2)) * k**2 * E**r * k**q * f0

        result, _ = integrate.quad(integrand, 0, np.inf, limit=100)
        return result

    def J_rq(self, r: int, q: int) -> float:
        """
        Compute J_{rq} = ∫ dK E^r k^q (∂f₀/∂E).

        This integral involves the derivative of the distribution with respect to energy.
        """
        key = (r, q)
        if key in self._J_cache:
            return self._J_cache[key]

        result = self._compute_J_integral(r, q)
        self._J_cache[key] = result
        return result

    def _compute_J_integral(self, r: int, q: int) -> float:
        """Compute J_{rq} integral numerically."""
        def integrand(k):
            E = np.sqrt(k**2 + self.eos.mass**2)
            exp_arg = self.eos.beta * (E - self.eos.chemical_potential)

            if exp_arg > 100:
                return 0.0

            # ∂f₀/∂E = -β f₀ for classical distribution
            f0 = np.exp(-exp_arg)
            df_dE = -self.eos.beta * f0

            return (self.eos.degeneracy / (2 * np.pi**2)) * k**2 * E**r * k**q * df_dE

        result, _ = integrate.quad(integrand, 0, np.inf, limit=100)
        return result

    def D_rq(self, r: int, q: int) -> float:
        """
        Compute D_{rq} = ∫ dK E^r k^q (∂f₀/∂α).

        This integral involves the derivative with respect to chemical potential.
        """
        key = (r, q)
        if key in self._D_cache:
            return self._D_cache[key]

        result = self._compute_D_integral(r, q)
        self._D_cache[key] = result
        return result

    def _compute_D_integral(self, r: int, q: int) -> float:
        """Compute D_{rq} integral numerically."""
        def integrand(k):
            E = np.sqrt(k**2 + self.eos.mass**2)
            exp_arg = self.eos.beta * (E - self.eos.chemical_potential)

            if exp_arg > 100:
                return 0.0

            # ∂f₀/∂α = T ∂f₀/∂μ = f₀ for classical distribution
            f0 = np.exp(-exp_arg)
            df_dalpha = f0

            return (self.eos.degeneracy / (2 * np.pi**2)) * k**2 * E**r * k**q * df_dalpha

        result, _ = integrate.quad(integrand, 0, np.inf, limit=100)
        return result

    def compute_H_derivatives(self) -> tuple[float, float]:
        """
        Compute thermodynamic derivatives H and H̄ from IReD eq. (A2b).

        H = [J_{20}(ε+P) - J_{30}n] / D_{20}
        H̄ = [J_{10}(ε+P) - J_{20}n] / D_{20}

        These appear in second-order transport coefficients.

        Returns:
            (H, H_bar) tuple
        """
        J20 = self.J_rq(2, 0)
        J30 = self.J_rq(3, 0)
        J10 = self.J_rq(1, 0)
        D20 = self.D_rq(2, 0)

        eps = self.eos.energy_density()
        P = self.eos.pressure()
        n = self.eos.particle_density()

        H = (J20 * (eps + P) - J30 * n) / D20
        H_bar = (J10 * (eps + P) - J20 * n) / D20

        return H, H_bar


# Module constants and helper functions will be added in next part...
