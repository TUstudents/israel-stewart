"""
Transport coefficient calculation for Israel-Stewart hydrodynamics.

This module implements physically realistic transport coefficients with proper
temperature and density dependencies based on kinetic theory and QCD-inspired models.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import sympy as sp

from ..core.constants import HBAR, KBOLTZ, MPROTON
from ..core.fields import TransportCoefficients
from ..core.performance import monitor_performance


class TransportCoefficientModel(ABC):
    """
    Abstract base class for transport coefficient models.

    Defines the interface for computing temperature and density-dependent
    transport coefficients in relativistic hydrodynamics.
    """

    @abstractmethod
    def shear_viscosity(self, temperature: float, density: float) -> float:
        """Compute shear viscosity η(T, ρ)."""
        pass

    @abstractmethod
    def bulk_viscosity(self, temperature: float, density: float) -> float:
        """Compute bulk viscosity ζ(T, ρ)."""
        pass

    @abstractmethod
    def thermal_conductivity(self, temperature: float, density: float) -> float:
        """Compute thermal conductivity κ(T, ρ)."""
        pass

    @abstractmethod
    def shear_relaxation_time(self, temperature: float, density: float) -> float:
        """Compute shear relaxation time τ_π(T, ρ)."""
        pass

    @abstractmethod
    def bulk_relaxation_time(self, temperature: float, density: float) -> float:
        """Compute bulk relaxation time τ_Π(T, ρ)."""
        pass

    @abstractmethod
    def heat_relaxation_time(self, temperature: float, density: float) -> float:
        """Compute heat relaxation time τ_q(T, ρ)."""
        pass


class KineticTheoryModel(TransportCoefficientModel):
    """
    Kinetic theory model for transport coefficients.

    Based on the Chapman-Enskog expansion for dilute gases with
    realistic cross-sections and molecular properties.
    """

    def __init__(
        self,
        particle_mass: float = MPROTON,
        cross_section: float = 1.0,  # fm^2 or cm^2 (auto-detected)
        degrees_of_freedom: float = 3.0,  # Translational DOF
    ):
        """
        Initialize kinetic theory model.

        Args:
            particle_mass: Particle mass in GeV
            cross_section: Collision cross-section (fm² if > 1e-20, cm² if < 1e-20)
            degrees_of_freedom: Effective degrees of freedom
        """
        self.particle_mass = particle_mass

        # Auto-detect cross section units and convert to fm²
        if cross_section < 1e-20:
            # Assume cm² input, convert to fm²
            # 1 cm² = (10^13 fm)² = 10^26 fm²
            self.cross_section = cross_section * 1e26  # cm² to fm²
        else:
            # Assume fm² input
            self.cross_section = cross_section

        self.dof = degrees_of_freedom

    @monitor_performance("kinetic_shear_viscosity")
    def shear_viscosity(self, temperature: float, density: float) -> float:
        """
        Kinetic theory shear viscosity.

        η ≈ (ρ * v_th) / (n * σ) where v_th is thermal velocity

        Args:
            temperature: Temperature in GeV
            density: Mass density in GeV/fm^3
        """
        if temperature <= 0 or density <= 0:
            return 0.0

        # Number density (particles per fm^3)
        number_density = density / self.particle_mass

        # Thermal velocity (natural units, c=1)
        thermal_velocity = np.sqrt(temperature / self.particle_mass)

        # Mean free path: λ = 1 / (n * σ)
        if number_density * self.cross_section == 0:
            return float("inf")

        mean_free_path = 1.0 / (number_density * self.cross_section)

        # Phenomenological kinetic theory with density corrections
        # Base kinetic theory: η₀ = (5/16) * sqrt(mkT/π) / σ
        thermal_velocity = np.sqrt(temperature / self.particle_mass)
        eta_0 = (
            (5.0 / 16.0) * np.sqrt(self.particle_mass * temperature / np.pi) / self.cross_section
        )

        # Include density dependence from excluded volume effects and correlation functions
        # η = η₀ * f(n) where f(n) accounts for finite-size effects
        # For hard spheres: f(n) ≈ (1 - b*n)^(-1) where b is excluded volume
        # At higher densities, particles have less space to move → reduced transport

        # Simplified phenomenological form: η = η₀ / (1 + α * n * σ)
        # where α ≈ 2-5 captures excluded volume and correlation effects
        alpha = 3.0  # Phenomenological parameter
        density_factor = 1.0 / (1.0 + alpha * number_density * self.cross_section)

        eta = eta_0 * density_factor

        return float(eta)

    @monitor_performance("kinetic_bulk_viscosity")
    def bulk_viscosity(self, temperature: float, density: float) -> float:
        """
        Kinetic theory bulk viscosity.

        For monatomic ideal gas: ζ = 0 (no internal degrees of freedom)
        For molecular gas: ζ ∝ (γ - 1) where γ is heat capacity ratio
        """
        # For ideal monatomic gas, bulk viscosity is exactly zero
        gamma = (self.dof + 2) / self.dof  # Heat capacity ratio

        if abs(gamma - 5.0 / 3.0) < 1e-10:  # γ = 5/3 for monatomic
            return 0.0

        # Approximate bulk viscosity for non-monatomic systems
        eta = self.shear_viscosity(temperature, density)
        zeta = (2.0 / 3.0) * eta * abs(gamma - 5.0 / 3.0)

        return float(zeta)

    def thermal_conductivity(self, temperature: float, density: float) -> float:
        """
        Kinetic theory thermal conductivity.

        κ = (15/4) * (k_B / m) * η

        From Chapman-Enskog theory for monatomic gas.
        """
        eta = self.shear_viscosity(temperature, density)
        kappa = (15.0 / 4.0) * (KBOLTZ / self.particle_mass) * eta
        return float(kappa)

    def shear_relaxation_time(self, temperature: float, density: float) -> float:
        """
        Shear relaxation time from kinetic theory.

        τ_π ≈ η / (β * P) where β is a coefficient O(1) and P is pressure.
        """
        if temperature <= 0 or density <= 0:
            return 0.0

        # Pressure for ideal gas
        number_density = density / self.particle_mass
        pressure = number_density * KBOLTZ * temperature

        eta = self.shear_viscosity(temperature, density)

        if pressure <= 0:
            return 0.0

        # Coefficient from kinetic theory (typically β ≈ 1-2)
        beta = 1.5
        tau_pi = eta / (beta * pressure)

        return float(tau_pi)

    def bulk_relaxation_time(self, temperature: float, density: float) -> float:
        """
        Bulk relaxation time from kinetic theory.

        τ_Π ≈ ζ / (β * P) similar to shear case.
        For monatomic gas with ζ = 0, returns small positive value for numerical stability.
        """
        if temperature <= 0 or density <= 0:
            return 0.0

        number_density = density / self.particle_mass
        pressure = number_density * KBOLTZ * temperature

        zeta = self.bulk_viscosity(temperature, density)

        if pressure <= 0:
            return 0.0

        # For monatomic gas, ζ = 0 but we need finite relaxation time for numerical stability
        if zeta <= 1e-15:  # Effectively zero bulk viscosity
            # Use typical microscopic time scale τ ≈ 1/T
            return 1.0 / temperature

        beta = 1.0  # Different coefficient for bulk
        tau_Pi = zeta / (beta * pressure)

        return float(tau_Pi)

    def heat_relaxation_time(self, temperature: float, density: float) -> float:
        """
        Heat relaxation time from kinetic theory.

        τ_q ≈ κ * T / (β * P) where κ is thermal conductivity.
        """
        if temperature <= 0 or density <= 0:
            return 0.0

        number_density = density / self.particle_mass
        pressure = number_density * KBOLTZ * temperature

        kappa = self.thermal_conductivity(temperature, density)

        if pressure <= 0:
            return 0.0

        beta = 1.2  # Heat-specific coefficient
        tau_q = kappa * temperature / (beta * pressure)

        return tau_q


class QCDInspiredModel(TransportCoefficientModel):
    """
    QCD-inspired model for transport coefficients.

    Based on phenomenological models relevant for the quark-gluon plasma
    and dense hadronic matter near the QCD phase transition.
    """

    def __init__(
        self,
        critical_temperature: float = 0.170,  # GeV (≈ 170 MeV)
        eta_over_s_minimum: float = 0.08,  # η/s at minimum (near Tc)
        zeta_over_s_peak: float = 0.10,  # ζ/s peak value near Tc
        crossover_width: float = 0.020,  # GeV (transition width)
    ):
        """
        Initialize QCD-inspired model.

        Args:
            critical_temperature: Critical temperature for QCD transition
            eta_over_s_minimum: Minimum value of η/s (KSS bound ≈ 1/4π ≈ 0.08)
            zeta_over_s_peak: Peak value of ζ/s near phase transition
            crossover_width: Width of crossover region
        """
        self.Tc = critical_temperature
        self.eta_s_min = eta_over_s_minimum
        self.zeta_s_peak = zeta_over_s_peak
        self.width = crossover_width

    @monitor_performance("qcd_shear_viscosity")
    def shear_viscosity(self, temperature: float, density: float) -> float:
        """
        QCD-inspired shear viscosity.

        Based on AdS/CFT minimum η/s ≈ 1/(4π) with temperature dependence
        and corrections near the phase transition.
        """
        if temperature <= 0:
            return 0.0

        # Entropy density for ideal gas (approximate)
        # s ≈ (π²/90) * g_eff * T³ where g_eff is effective DOF
        g_eff = 37.5  # QGP degrees of freedom
        entropy_density = (np.pi**2 / 90) * g_eff * temperature**3

        # η/s with temperature dependence
        # Minimum near Tc, increases away from transition
        x = (temperature - self.Tc) / self.width
        eta_over_s = self.eta_s_min * (1.0 + 0.5 * x**2 + 0.1 * x**4)

        # Ensure physical minimum (KSS bound)
        eta_over_s = max(eta_over_s, self.eta_s_min)

        eta = eta_over_s * entropy_density
        return eta

    @monitor_performance("qcd_bulk_viscosity")
    def bulk_viscosity(self, temperature: float, density: float) -> float:
        """
        QCD-inspired bulk viscosity.

        Peaks near the phase transition due to conformal symmetry breaking
        and vanishes for conformal systems (high T QGP).
        """
        if temperature <= 0:
            return 0.0

        g_eff = 37.5
        entropy_density = (np.pi**2 / 90) * g_eff * temperature**3

        # Gaussian peak near critical temperature
        x = (temperature - self.Tc) / self.width
        zeta_over_s = self.zeta_s_peak * np.exp(-0.5 * x**2)

        # High temperature suppression (conformal limit)
        if temperature > 2 * self.Tc:
            suppression = (self.Tc / temperature) ** 3
            zeta_over_s *= suppression

        zeta = zeta_over_s * entropy_density
        return float(zeta)

    def thermal_conductivity(self, temperature: float, density: float) -> float:
        """
        QCD thermal conductivity using Wiedemann-Franz law analog.

        κ ∝ T * σ_electric where σ_electric is electrical conductivity.
        """
        if temperature <= 0:
            return 0.0

        # Electrical conductivity estimate for QGP
        # σ ≈ C * T where C is a constant
        C = 0.4  # Estimated from lattice QCD
        sigma_electric = C * temperature

        # Wiedemann-Franz relation: κ/T ∝ σ_electric/T
        kappa = 2.0 * sigma_electric  # Factor from kinetic theory

        return kappa

    def shear_relaxation_time(self, temperature: float, density: float) -> float:
        """
        QCD shear relaxation time.

        τ_π ≈ (2-3) * η / (s * T) from kinetic theory and holographic models.
        """
        if temperature <= 0:
            return 0.0

        eta = self.shear_viscosity(temperature, density)
        g_eff = 37.5
        entropy_density = (np.pi**2 / 90) * g_eff * temperature**3

        if entropy_density <= 0:
            return 0.0

        # Coefficient from various theoretical estimates
        C = 2.5
        tau_pi = C * eta / (entropy_density * temperature)

        return float(tau_pi)

    def bulk_relaxation_time(self, temperature: float, density: float) -> float:
        """
        QCD bulk relaxation time.

        τ_Π ≈ ζ / (s * T) with similar scaling as shear case.
        """
        if temperature <= 0:
            return 0.0

        zeta = self.bulk_viscosity(temperature, density)
        g_eff = 37.5
        entropy_density = (np.pi**2 / 90) * g_eff * temperature**3

        if entropy_density <= 0:
            return 0.0

        C = 1.0  # Typically smaller than shear coefficient
        tau_Pi = C * zeta / (entropy_density * temperature)

        return float(tau_Pi)

    def heat_relaxation_time(self, temperature: float, density: float) -> float:
        """
        QCD heat relaxation time.

        τ_q related to thermal diffusion timescale.
        """
        if temperature <= 0:
            return 0.0

        kappa = self.thermal_conductivity(temperature, density)
        g_eff = 37.5
        entropy_density = (np.pi**2 / 90) * g_eff * temperature**3

        if entropy_density <= 0:
            return 0.0

        # Heat capacity at constant volume
        cv = (np.pi**2 / 30) * g_eff * temperature**3

        if cv <= 0:
            return 0.0

        # Thermal diffusion time
        tau_q = cv / (entropy_density * temperature) * (kappa / cv)

        return tau_q


class TransportCoefficientCalculator:
    """
    Main calculator for temperature and density-dependent transport coefficients.

    Manages different physical models and provides a unified interface for
    computing all transport coefficients needed in Israel-Stewart hydrodynamics.
    """

    def __init__(
        self,
        model: TransportCoefficientModel | None = None,
        enable_second_order: bool = True,
    ):
        """
        Initialize transport coefficient calculator.

        Args:
            model: Physical model for coefficient computation
            enable_second_order: Whether to compute second-order coefficients
        """
        self.model = model or KineticTheoryModel()
        self.enable_second_order = enable_second_order

        # Cache for expensive computations
        self._cache: dict[tuple[float, float], dict[str, float]] = {}
        self._cache_tolerance = 1e-10

    @monitor_performance("transport_coefficient_computation")
    def compute_coefficients(
        self, temperature: float, density: float, validate: bool = True
    ) -> TransportCoefficients:
        """
        Compute complete set of transport coefficients.

        Args:
            temperature: Temperature in natural units
            density: Mass density in natural units
            validate: Whether to validate coefficient values

        Returns:
            TransportCoefficients object with all computed values
        """
        # Check cache first
        cache_key = (temperature, density)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return self._create_transport_coefficients(cached)

        # Compute first-order coefficients
        eta = self.model.shear_viscosity(temperature, density)
        zeta = self.model.bulk_viscosity(temperature, density)
        kappa = self.model.thermal_conductivity(temperature, density)

        tau_pi = self.model.shear_relaxation_time(temperature, density)
        tau_Pi = self.model.bulk_relaxation_time(temperature, density)
        tau_q = self.model.heat_relaxation_time(temperature, density)

        # Store in cache
        coeffs = {
            "eta": eta,
            "zeta": zeta,
            "kappa": kappa,
            "tau_pi": tau_pi,
            "tau_Pi": tau_Pi,
            "tau_q": tau_q,
        }

        # Compute second-order coefficients if enabled
        if self.enable_second_order:
            second_order = self._compute_second_order_coefficients(
                temperature, density, eta, zeta, kappa, tau_pi, tau_Pi, tau_q
            )
            coeffs.update(second_order)

        # Validation
        if validate:
            self._validate_coefficients(coeffs, temperature, density)

        # Cache results
        if len(self._cache) < 1000:  # Limit cache size
            self._cache[cache_key] = coeffs

        return self._create_transport_coefficients(coeffs)

    def _compute_second_order_coefficients(
        self,
        temperature: float,
        density: float,
        eta: float,
        zeta: float,
        kappa: float,
        tau_pi: float,
        tau_Pi: float,
        tau_q: float,
    ) -> dict[str, float]:
        """
        Compute second-order Israel-Stewart coefficients.

        Based on kinetic theory relations and phenomenological estimates.
        """
        # Second-order coefficients from kinetic theory and phenomenology
        # These are typically O(1) relative to first-order coefficients

        # Shear-shear couplings
        lambda_pi_pi = 0.8 * tau_pi  # Self-interaction strength

        # Shear-bulk couplings
        lambda_pi_Pi = 0.6 * np.sqrt(tau_pi * tau_Pi)
        lambda_Pi_pi = 0.4 * np.sqrt(tau_pi * tau_Pi)

        # Heat flux couplings
        lambda_pi_q = 0.5 * np.sqrt(tau_pi * tau_q)
        lambda_q_pi = 0.3 * np.sqrt(tau_pi * tau_q)

        # Bulk nonlinear coefficients
        if zeta > 0 and tau_Pi > 0:
            xi_1 = 0.5 * tau_Pi / (density + 3 * temperature)  # Temperature dependence
            xi_2 = 0.2 * tau_Pi
        else:
            xi_1 = xi_2 = 0.0

        # Vorticity coupling (typically small)
        tau_pi_omega = 0.1 * tau_pi

        return {
            "lambda_pi_pi": lambda_pi_pi,
            "lambda_pi_Pi": lambda_pi_Pi,
            "lambda_pi_q": lambda_pi_q,
            "lambda_Pi_pi": lambda_Pi_pi,
            "lambda_q_pi": lambda_q_pi,
            "xi_1": xi_1,
            "xi_2": xi_2,
            "tau_pi_omega": tau_pi_omega,
        }

    def _validate_coefficients(
        self, coeffs: dict[str, float], temperature: float, density: float
    ) -> None:
        """Validate computed coefficients for physical consistency."""
        # Check for negative values (unphysical)
        for name, value in coeffs.items():
            if value < 0 and name not in [
                "lambda_pi_Pi",
                "lambda_Pi_pi",
            ]:  # Some couplings can be negative
                warnings.warn(
                    f"Negative coefficient {name} = {value} at T={temperature}, ρ={density}",
                    stacklevel=3,
                )

        # Check causality constraints (relaxation times > 0)
        for tau_name in ["tau_pi", "tau_Pi", "tau_q"]:
            if tau_name in coeffs and coeffs[tau_name] <= 0:
                warnings.warn(
                    f"Non-positive relaxation time {tau_name} = {coeffs[tau_name]}", stacklevel=3
                )

        # Check thermodynamic constraints
        eta, zeta = coeffs.get("eta", 0), coeffs.get("zeta", 0)
        if eta > 0 and zeta > 0:
            # Second viscosity constraint: ζ + (2/3)η ≥ 0
            if zeta + (2.0 / 3.0) * eta < 0:
                warnings.warn(
                    f"Thermodynamic constraint violated: ζ + (2/3)η = {zeta + (2.0 / 3.0) * eta} < 0",
                    stacklevel=3,
                )

    def _create_transport_coefficients(self, coeffs: dict[str, float]) -> TransportCoefficients:
        """Create TransportCoefficients object from computed values."""
        return TransportCoefficients(
            shear_viscosity=coeffs.get("eta", 0.0),
            bulk_viscosity=coeffs.get("zeta", 0.0),
            thermal_conductivity=coeffs.get("kappa", 0.0),
            shear_relaxation_time=coeffs.get("tau_pi"),
            bulk_relaxation_time=coeffs.get("tau_Pi"),
            heat_relaxation_time=coeffs.get("tau_q"),
            # Second-order coefficients
            lambda_pi_pi=coeffs.get("lambda_pi_pi", 0.0),
            lambda_pi_Pi=coeffs.get("lambda_pi_Pi", 0.0),
            lambda_pi_q=coeffs.get("lambda_pi_q", 0.0),
            lambda_Pi_pi=coeffs.get("lambda_Pi_pi", 0.0),
            lambda_q_pi=coeffs.get("lambda_q_pi", 0.0),
            xi_1=coeffs.get("xi_1", 0.0),
            xi_2=coeffs.get("xi_2", 0.0),
            tau_pi_omega=coeffs.get("tau_pi_omega", 0.0),
        )

    def clear_cache(self) -> None:
        """Clear the coefficient computation cache."""
        self._cache.clear()

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about the coefficient cache."""
        return {
            "cache_size": len(self._cache),
            "cache_tolerance": self._cache_tolerance,
        }
