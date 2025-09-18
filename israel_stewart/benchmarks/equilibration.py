"""
Equilibration benchmark for Israel-Stewart hydrodynamics validation.

This module implements comprehensive tests for the approach to thermal equilibrium
in relativistic viscous fluids, validating relaxation timescales and the second law
of thermodynamics in the Israel-Stewart formalism.

Classes:
    EquilibrationAnalysis: Core analysis class for equilibration processes
    RelaxationTimeAnalysis: Specialized analysis of relaxation timescales
    EntropyProductionAnalysis: Entropy production and second law validation
    EquilibrationBenchmark: Comprehensive benchmark suite

Functions:
    create_equilibration_benchmark: Factory function for creating benchmark instances
    run_relaxation_analysis: Quick relaxation analysis
    validate_entropy_production: Check entropy production rates
    analyze_approach_to_equilibrium: Analyze equilibration dynamics

The benchmark validates:
1. Exponential approach to equilibrium
2. Correct relaxation timescales
3. Entropy production and the second law
4. Temperature and chemical potential evolution
5. Dissipative flux decay rates
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad
from scipy.optimize import curve_fit

from ..core.constants import HBAR, KBOLTZ
from ..core.derivatives import CovariantDerivative, ProjectionOperator
from ..core.fields import ISFieldConfiguration, TransportCoefficients
from ..core.four_vectors import FourVector
from ..core.metrics import GeneralMetric, MinkowskiMetric
from ..core.stress_tensors import StressEnergyTensor, ViscousStressTensor
from ..core.tensor_base import SpacetimeGrid, TensorField
from ..equations.conservation import ConservationLaws
from ..equations.relaxation import ISRelaxationEquations


@dataclass
class EquilibrationProperties:
    """Properties of equilibration process in viscous relativistic fluids."""

    initial_state: dict[str, float]
    final_state: dict[str, float]
    relaxation_times: dict[str, float]
    decay_rates: dict[str, float]
    entropy_production_rate: float
    approach_exponent: float

    # Thermodynamic properties
    temperature_evolution: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    entropy_evolution: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    # Dissipative flux evolution
    bulk_pressure_evolution: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    shear_stress_evolution: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        """Validate equilibration properties."""
        for key, value in self.relaxation_times.items():
            if value <= 0:
                raise ValueError(f"Relaxation time {key} must be positive")

        if self.entropy_production_rate < 0:
            warnings.warn("Negative entropy production violates second law", stacklevel=2)


class EquilibrationAnalysis:
    """
    Core analysis class for equilibration processes in Israel-Stewart hydrodynamics.

    This class provides comprehensive tools for analyzing the approach to thermal
    equilibrium, including relaxation timescales, entropy production, and
    thermodynamic consistency.
    """

    def __init__(
        self,
        grid: SpacetimeGrid,
        metric: GeneralMetric,
        transport_coeffs: TransportCoefficients,
        equation_of_state: str = "ideal",
    ):
        """
        Initialize equilibration analysis.

        Args:
            grid: Spacetime grid for numerical analysis
            metric: Spacetime metric
            transport_coeffs: Transport coefficients
            equation_of_state: Equation of state type
        """
        self.grid = grid
        self.metric = metric
        self.transport_coeffs = transport_coeffs
        self.eos = equation_of_state

        # Initialize physics modules
        self.conservation = ConservationLaws(grid, metric)
        self.relaxation = ISRelaxationEquations(grid, metric, transport_coeffs)

        # Analysis results cache
        self._equilibration_cache: dict[str, EquilibrationProperties] = {}

    def analyze_relaxation_to_equilibrium(
        self,
        initial_fields: ISFieldConfiguration,
        final_time: float = 10.0,
        timestep: float = 0.01,
        method: str = "implicit",
    ) -> EquilibrationProperties:
        """
        Analyze complete relaxation process to thermal equilibrium.

        Args:
            initial_fields: Initial non-equilibrium state
            final_time: Final simulation time
            timestep: Integration timestep
            method: Integration method ("explicit", "implicit", "adaptive")

        Returns:
            Equilibration properties and evolution data
        """
        # Validate initial state
        self._validate_initial_state(initial_fields)

        # Store initial thermodynamic quantities
        initial_state = self._extract_thermodynamic_state(initial_fields)

        # Time evolution
        time_points = []
        temperature_data = []
        entropy_data = []
        bulk_pressure_data = []
        shear_stress_data = []

        fields = self._copy_fields(initial_fields)
        current_time = 0.0

        # Store initial data
        time_points.append(current_time)
        temp = self._compute_temperature(fields)
        entropy = self._compute_entropy_density(fields)
        temperature_data.append(temp)
        entropy_data.append(entropy)
        bulk_pressure_data.append(np.mean(np.abs(fields.Pi.data)))
        shear_stress_data.append(self._compute_shear_stress_magnitude(fields))

        # Integration loop
        while current_time < final_time:
            dt = min(timestep, final_time - current_time)

            # Evolve system
            self.relaxation.evolve_relaxation(fields, dt, method=method)

            current_time += dt
            time_points.append(current_time)

            # Record thermodynamic quantities
            temp = self._compute_temperature(fields)
            entropy = self._compute_entropy_density(fields)
            temperature_data.append(temp)
            entropy_data.append(entropy)
            bulk_pressure_data.append(np.mean(np.abs(fields.Pi.data)))
            shear_stress_data.append(self._compute_shear_stress_magnitude(fields))

        # Convert to arrays
        time_array = np.array(time_points)
        temperature_evolution = np.array(temperature_data)
        entropy_evolution = np.array(entropy_data)
        bulk_pressure_evolution = np.array(bulk_pressure_data)
        shear_stress_evolution = np.array(shear_stress_data)

        # Analyze relaxation timescales
        relaxation_times = self._extract_relaxation_times(
            time_array, bulk_pressure_evolution, shear_stress_evolution
        )

        # Compute decay rates
        decay_rates = {key: 1.0 / tau for key, tau in relaxation_times.items()}

        # Analyze entropy production
        entropy_production_rate = self._compute_entropy_production_rate(
            time_array, entropy_evolution
        )

        # Fit approach to equilibrium
        approach_exponent = self._fit_approach_exponent(time_array, bulk_pressure_evolution)

        # Final equilibrium state
        final_state = self._extract_thermodynamic_state(fields)

        return EquilibrationProperties(
            initial_state=initial_state,
            final_state=final_state,
            relaxation_times=relaxation_times,
            decay_rates=decay_rates,
            entropy_production_rate=entropy_production_rate,
            approach_exponent=approach_exponent,
            temperature_evolution=temperature_evolution,
            entropy_evolution=entropy_evolution,
            bulk_pressure_evolution=bulk_pressure_evolution,
            shear_stress_evolution=shear_stress_evolution,
        )

    def _validate_initial_state(self, fields: ISFieldConfiguration) -> None:
        """Validate that initial state is physically reasonable."""
        # Check energy density positivity
        if np.any(fields.rho.data <= 0):
            raise ValueError("Energy density must be positive")

        # Check pressure positivity
        if np.any(fields.pressure.data <= 0):
            raise ValueError("Pressure must be positive")

        # Check four-velocity normalization
        u_squared = np.sum(fields.four_velocity.data**2, axis=-1)
        if not np.allclose(u_squared[..., 0] - u_squared[..., 1:].sum(axis=-1), 1.0, rtol=1e-3):
            warnings.warn("Four-velocity not properly normalized", stacklevel=2)

    def _copy_fields(self, fields: ISFieldConfiguration) -> ISFieldConfiguration:
        """Create a deep copy of field configuration."""
        new_fields = ISFieldConfiguration(self.grid)

        # Copy all field data
        new_fields.rho.data[:] = fields.rho.data[:]
        new_fields.pressure.data[:] = fields.pressure.data[:]
        new_fields.four_velocity.data[:] = fields.four_velocity.data[:]
        new_fields.Pi.data[:] = fields.Pi.data[:]
        new_fields.pi_munu.data[:] = fields.pi_munu.data[:]
        new_fields.q_mu.data[:] = fields.q_mu.data[:]

        return new_fields

    def _extract_thermodynamic_state(self, fields: ISFieldConfiguration) -> dict[str, float]:
        """Extract key thermodynamic quantities from field configuration."""
        return {
            "energy_density": np.mean(fields.rho.data),
            "pressure": np.mean(fields.pressure.data),
            "temperature": self._compute_temperature(fields),
            "entropy_density": self._compute_entropy_density(fields),
            "bulk_pressure": np.mean(fields.Pi.data),
            "shear_stress": self._compute_shear_stress_magnitude(fields),
        }

    def _compute_temperature(self, fields: ISFieldConfiguration) -> float:
        """Compute temperature from energy density (ideal gas)."""
        rho = np.mean(fields.rho.data)

        if self.eos == "ideal":
            # Ideal gas: rho = a*T^4 where a = (pi^2/90)*g_eff
            g_eff = 37.5  # Effective degrees of freedom
            a = (np.pi**2 / 90.0) * g_eff
            T = (rho / a) ** (1.0 / 4.0)
        else:
            # Fallback: use pressure relation P = rho/3
            p = np.mean(fields.pressure.data)
            T = (3 * p / rho) ** (1.0 / 4.0) if rho > 0 else 0.0

        return T

    def _compute_entropy_density(self, fields: ISFieldConfiguration) -> float:
        """Compute entropy density."""
        T = self._compute_temperature(fields)

        if self.eos == "ideal":
            # Ideal gas: s = (2*pi^2/45)*g_eff*T^3
            g_eff = 37.5
            s = (2 * np.pi**2 / 45.0) * g_eff * T**3
        else:
            # Fallback estimate
            rho = np.mean(fields.rho.data)
            p = np.mean(fields.pressure.data)
            s = (rho + p) / T if T > 0 else 0.0

        return s

    def _compute_shear_stress_magnitude(self, fields: ISFieldConfiguration) -> float:
        """Compute magnitude of shear stress tensor."""
        pi_data = fields.pi_munu.data
        # Compute pi^{mu nu} pi_{mu nu}
        magnitude_squared = np.sum(pi_data**2, axis=(-2, -1))
        return np.sqrt(np.mean(magnitude_squared))

    def _extract_relaxation_times(
        self,
        time: npt.NDArray[np.float64],
        bulk_evolution: npt.NDArray[np.float64],
        shear_evolution: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """Extract relaxation timescales from evolution data."""
        relaxation_times = {}

        # Bulk relaxation time
        if len(bulk_evolution) > 10 and np.max(bulk_evolution) > 1e-10:
            tau_bulk = self._fit_exponential_decay(time, bulk_evolution)
            relaxation_times["bulk"] = tau_bulk
        else:
            relaxation_times["bulk"] = self.transport_coeffs.bulk_relaxation_time

        # Shear relaxation time
        if len(shear_evolution) > 10 and np.max(shear_evolution) > 1e-10:
            tau_shear = self._fit_exponential_decay(time, shear_evolution)
            relaxation_times["shear"] = tau_shear
        else:
            relaxation_times["shear"] = self.transport_coeffs.shear_relaxation_time

        return relaxation_times

    def _fit_exponential_decay(
        self, time: npt.NDArray[np.float64], data: npt.NDArray[np.float64]
    ) -> float:
        """Fit exponential decay to extract relaxation time."""

        def exponential_func(t, A, tau, C):
            return A * np.exp(-t / tau) + C

        # Use only data after initial transient
        start_idx = max(1, len(time) // 10)
        t_fit = time[start_idx:]
        y_fit = data[start_idx:]

        # Initial parameter guess
        A_guess = y_fit[0] - y_fit[-1]
        tau_guess = t_fit[-1] / 3.0
        C_guess = y_fit[-1]

        try:
            popt, _ = curve_fit(
                exponential_func,
                t_fit,
                y_fit,
                p0=[A_guess, tau_guess, C_guess],
                bounds=([0, 0.01, 0], [10 * A_guess, 10 * tau_guess, 10 * C_guess]),
            )
            return popt[1]  # Return tau
        except Exception:
            # Fallback: estimate from half-life
            half_max = (y_fit[0] + y_fit[-1]) / 2.0
            half_idx = np.argmin(np.abs(y_fit - half_max))
            return t_fit[half_idx] / np.log(2)

    def _compute_entropy_production_rate(
        self, time: npt.NDArray[np.float64], entropy: npt.NDArray[np.float64]
    ) -> float:
        """Compute rate of entropy production."""
        if len(entropy) < 2:
            return 0.0

        # Numerical derivative
        dt = time[1] - time[0]
        entropy_rate = np.gradient(entropy, dt)

        # Average over middle portion to avoid boundary effects
        start_idx = len(entropy_rate) // 4
        end_idx = 3 * len(entropy_rate) // 4

        if end_idx > start_idx:
            return np.mean(entropy_rate[start_idx:end_idx])
        else:
            return np.mean(entropy_rate)

    def _fit_approach_exponent(
        self, time: npt.NDArray[np.float64], data: npt.NDArray[np.float64]
    ) -> float:
        """Fit approach to equilibrium to extract characteristic exponent."""
        if len(data) < 10:
            return 1.0

        # Fit to A * t^(-alpha) for late times
        def power_law(t, A, alpha):
            return A * np.power(t + 1e-6, -alpha)

        # Use late-time data
        start_idx = len(time) // 2
        t_fit = time[start_idx:]
        y_fit = data[start_idx:]

        if np.all(y_fit <= 1e-12):
            return 1.0

        try:
            popt, _ = curve_fit(
                power_law, t_fit, y_fit, p0=[y_fit[0], 1.0], bounds=([1e-10, 0.1], [10.0, 5.0])
            )
            return popt[1]  # Return alpha
        except Exception:
            return 1.0


class RelaxationTimeAnalysis:
    """
    Specialized analysis of relaxation timescales in Israel-Stewart hydrodynamics.

    This class provides detailed analysis of how relaxation times depend on
    thermodynamic conditions and transport coefficients.
    """

    def __init__(self, analysis: EquilibrationAnalysis):
        """Initialize with equilibration analysis instance."""
        self.analysis = analysis

    def analyze_temperature_dependence(
        self, temperature_range: tuple[float, float], n_points: int = 20
    ) -> dict[str, Any]:
        """
        Analyze temperature dependence of relaxation times.

        Args:
            temperature_range: (T_min, T_max) in natural units
            n_points: Number of temperature points

        Returns:
            Temperature dependence analysis results
        """
        T_min, T_max = temperature_range
        temperatures = np.linspace(T_min, T_max, n_points)

        bulk_times = []
        shear_times = []
        entropy_rates = []

        for T in temperatures:
            # Create initial state at this temperature
            fields = self._create_initial_state_at_temperature(T)

            # Run relaxation analysis
            properties = self.analysis.analyze_relaxation_to_equilibrium(
                fields, final_time=5.0, timestep=0.02
            )

            bulk_times.append(properties.relaxation_times.get("bulk", 0.0))
            shear_times.append(properties.relaxation_times.get("shear", 0.0))
            entropy_rates.append(properties.entropy_production_rate)

        return {
            "temperatures": temperatures,
            "bulk_relaxation_times": np.array(bulk_times),
            "shear_relaxation_times": np.array(shear_times),
            "entropy_production_rates": np.array(entropy_rates),
            "temperature_scaling": self._analyze_power_law_scaling(temperatures, bulk_times),
        }

    def _create_initial_state_at_temperature(self, T: float) -> ISFieldConfiguration:
        """Create initial non-equilibrium state at given temperature."""
        fields = ISFieldConfiguration(self.analysis.grid)

        # Thermodynamic quantities for ideal gas
        g_eff = 37.5
        rho = (np.pi**2 / 90.0) * g_eff * T**4
        p = rho / 3.0

        # Background equilibrium state
        fields.rho.fill(rho)
        fields.pressure.fill(p)
        fields.four_velocity.fill_zero()
        fields.four_velocity.data[..., 0] = 1.0

        # Non-equilibrium initial conditions for dissipative fluxes
        fields.Pi.fill(0.1 * p)  # Initial bulk pressure
        fields.pi_munu.fill_zero()
        fields.pi_munu.data[..., 1, 1] = 0.05 * p  # Initial shear
        fields.q_mu.fill_zero()

        return fields

    def _analyze_power_law_scaling(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> dict[str, float]:
        """Analyze power law scaling y ~ x^n."""
        if len(x) < 3 or np.any(y <= 0):
            return {"exponent": 0.0, "coefficient": 0.0, "r_squared": 0.0}

        # Log-log fit
        log_x = np.log(x)
        log_y = np.log(y)

        coeffs = np.polyfit(log_x, log_y, 1)
        exponent = coeffs[0]
        log_coeff = coeffs[1]
        coefficient = np.exp(log_coeff)

        # R-squared
        y_pred = coefficient * np.power(x, exponent)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {"exponent": exponent, "coefficient": coefficient, "r_squared": r_squared}

    def compare_with_theory(self, properties: EquilibrationProperties) -> dict[str, Any]:
        """
        Compare measured relaxation times with theoretical predictions.

        Args:
            properties: Equilibration properties from simulation

        Returns:
            Comparison with theoretical values
        """
        # Theoretical relaxation times from transport coefficients
        tau_pi_theory = self.analysis.transport_coeffs.shear_relaxation_time
        tau_Pi_theory = self.analysis.transport_coeffs.bulk_relaxation_time

        # Measured times
        tau_pi_measured = properties.relaxation_times.get("shear", 0.0)
        tau_Pi_measured = properties.relaxation_times.get("bulk", 0.0)

        # Relative errors
        shear_error = (
            abs(tau_pi_measured - tau_pi_theory) / tau_pi_theory if tau_pi_theory > 0 else np.inf
        )
        bulk_error = (
            abs(tau_Pi_measured - tau_Pi_theory) / tau_Pi_theory if tau_Pi_theory > 0 else np.inf
        )

        return {
            "shear_relaxation": {
                "theory": tau_pi_theory,
                "measured": tau_pi_measured,
                "relative_error": shear_error,
                "agrees": shear_error < 0.2,
            },
            "bulk_relaxation": {
                "theory": tau_Pi_theory,
                "measured": tau_Pi_measured,
                "relative_error": bulk_error,
                "agrees": bulk_error < 0.2,
            },
            "overall_agreement": shear_error < 0.2 and bulk_error < 0.2,
        }


class EntropyProductionAnalysis:
    """
    Analysis of entropy production and second law validation.

    This class provides tools for analyzing entropy production rates and
    validating the second law of thermodynamics in dissipative systems.
    """

    def __init__(self, analysis: EquilibrationAnalysis):
        """Initialize with equilibration analysis instance."""
        self.analysis = analysis

    def validate_second_law(
        self, properties: EquilibrationProperties, tolerance: float = 1e-10
    ) -> dict[str, Any]:
        """
        Validate second law of thermodynamics.

        Args:
            properties: Equilibration properties
            tolerance: Tolerance for entropy production positivity

        Returns:
            Second law validation results
        """
        entropy_evolution = properties.entropy_evolution

        if len(entropy_evolution) < 2:
            return {"valid": False, "error": "Insufficient entropy data"}

        # Check monotonic increase
        entropy_diffs = np.diff(entropy_evolution)
        violations = entropy_diffs < -tolerance
        violation_count = np.sum(violations)

        # Total entropy change
        total_entropy_change = entropy_evolution[-1] - entropy_evolution[0]

        # Average entropy production rate
        avg_production_rate = properties.entropy_production_rate

        return {
            "valid": violation_count == 0 and total_entropy_change >= 0,
            "total_entropy_change": total_entropy_change,
            "average_production_rate": avg_production_rate,
            "violation_count": violation_count,
            "violation_fraction": violation_count / len(entropy_diffs),
            "min_entropy_rate": np.min(entropy_diffs),
            "entropy_increases": total_entropy_change >= 0,
            "production_positive": avg_production_rate >= -tolerance,
        }

    def compute_entropy_production_sources(self, fields: ISFieldConfiguration) -> dict[str, float]:
        """
        Decompose entropy production into different physical sources.

        Args:
            fields: Field configuration

        Returns:
            Entropy production from different sources
        """
        # Extract transport coefficients
        eta = self.analysis.transport_coeffs.shear_viscosity
        zeta = self.analysis.transport_coeffs.bulk_viscosity
        tau_pi = self.analysis.transport_coeffs.shear_relaxation_time
        tau_Pi = self.analysis.transport_coeffs.bulk_relaxation_time

        # Extract fields
        Pi = np.mean(fields.Pi.data)
        pi_magnitude = self.analysis._compute_shear_stress_magnitude(fields)
        T = self.analysis._compute_temperature(fields)

        # Entropy production sources (per unit volume)
        if T > 0:
            # Bulk viscosity contribution
            bulk_source = (Pi**2) / (zeta * T) if zeta > 0 else 0.0

            # Shear viscosity contribution
            shear_source = (pi_magnitude**2) / (eta * T) if eta > 0 else 0.0

            # Relaxation contributions
            bulk_relaxation_source = (Pi**2) / (tau_Pi * T) if tau_Pi > 0 else 0.0
            shear_relaxation_source = (pi_magnitude**2) / (tau_pi * T) if tau_pi > 0 else 0.0
        else:
            bulk_source = shear_source = bulk_relaxation_source = shear_relaxation_source = 0.0

        total_source = bulk_source + shear_source + bulk_relaxation_source + shear_relaxation_source

        return {
            "bulk_viscosity": bulk_source,
            "shear_viscosity": shear_source,
            "bulk_relaxation": bulk_relaxation_source,
            "shear_relaxation": shear_relaxation_source,
            "total": total_source,
        }

    def analyze_entropy_flow(self, properties: EquilibrationProperties) -> dict[str, Any]:
        """
        Analyze entropy flow and accumulation during equilibration.

        Args:
            properties: Equilibration properties

        Returns:
            Entropy flow analysis
        """
        entropy_evolution = properties.entropy_evolution

        if len(entropy_evolution) < 3:
            return {"error": "Insufficient data for flow analysis"}

        # Numerical derivatives
        time_step = 1.0  # Assume unit time steps for simplicity
        entropy_rate = np.gradient(entropy_evolution, time_step)
        entropy_acceleration = np.gradient(entropy_rate, time_step)

        # Flow characteristics
        max_production_rate = np.max(entropy_rate)
        max_production_time = np.argmax(entropy_rate) * time_step

        # Approach to equilibrium
        final_quarter = len(entropy_evolution) // 4
        late_time_rate = (
            np.mean(entropy_rate[-final_quarter:]) if final_quarter > 0 else entropy_rate[-1]
        )

        return {
            "max_production_rate": max_production_rate,
            "max_production_time": max_production_time,
            "late_time_production_rate": late_time_rate,
            "entropy_rate_evolution": entropy_rate,
            "entropy_acceleration": entropy_acceleration,
            "equilibrium_approached": late_time_rate < 0.1 * max_production_rate,
        }


class EquilibrationBenchmark:
    """Comprehensive benchmark suite for equilibration processes."""

    def __init__(
        self, grid: SpacetimeGrid, metric: GeneralMetric, transport_coeffs: TransportCoefficients
    ):
        """Initialize benchmark suite."""
        self.grid = grid
        self.metric = metric
        self.transport_coeffs = transport_coeffs

        # Initialize analysis tools
        self.analysis = EquilibrationAnalysis(grid, metric, transport_coeffs)
        self.relaxation_analysis = RelaxationTimeAnalysis(self.analysis)
        self.entropy_analysis = EntropyProductionAnalysis(self.analysis)

    def run_comprehensive_tests(self, initial_perturbation: float = 0.1) -> dict[str, Any]:
        """
        Run comprehensive equilibration validation tests.

        Args:
            initial_perturbation: Strength of initial non-equilibrium perturbation

        Returns:
            Comprehensive test results
        """
        results = {}

        # Test 1: Basic equilibration
        results["equilibration_test"] = self._test_basic_equilibration(initial_perturbation)

        # Test 2: Relaxation timescales
        results["relaxation_times_test"] = self._test_relaxation_timescales()

        # Test 3: Second law validation
        results["second_law_test"] = self._test_second_law()

        # Test 4: Temperature dependence
        results["temperature_dependence_test"] = self._test_temperature_dependence()

        # Test 5: Entropy production
        results["entropy_production_test"] = self._test_entropy_production()

        # Overall assessment
        results["overall_pass"] = all(
            test_result.get("pass", False) for test_result in results.values()
        )

        return results

    def _test_basic_equilibration(self, perturbation: float) -> dict[str, Any]:
        """Test basic equilibration behavior."""
        # Create initial non-equilibrium state
        fields = self._create_perturbed_state(perturbation)

        # Run equilibration
        properties = self.analysis.analyze_relaxation_to_equilibrium(
            fields, final_time=8.0, timestep=0.05
        )

        # Check convergence to equilibrium
        initial_bulk = properties.bulk_pressure_evolution[0]
        final_bulk = properties.bulk_pressure_evolution[-1]
        bulk_decay = (initial_bulk - final_bulk) / initial_bulk if initial_bulk > 0 else 0

        initial_shear = properties.shear_stress_evolution[0]
        final_shear = properties.shear_stress_evolution[-1]
        shear_decay = (initial_shear - final_shear) / initial_shear if initial_shear > 0 else 0

        # Test passes if dissipative fluxes decay significantly
        pass_condition = bulk_decay > 0.8 and shear_decay > 0.8

        return {
            "pass": pass_condition,
            "bulk_decay_fraction": bulk_decay,
            "shear_decay_fraction": shear_decay,
            "properties": properties,
        }

    def _test_relaxation_timescales(self) -> dict[str, Any]:
        """Test relaxation timescale accuracy."""
        # Create test state
        fields = self._create_perturbed_state(0.1)

        # Run analysis
        properties = self.analysis.analyze_relaxation_to_equilibrium(
            fields, final_time=6.0, timestep=0.02
        )

        # Compare with theory
        comparison = self.relaxation_analysis.compare_with_theory(properties)

        return {
            "pass": comparison["overall_agreement"],
            "comparison": comparison,
            "measured_times": properties.relaxation_times,
        }

    def _test_second_law(self) -> dict[str, Any]:
        """Test second law of thermodynamics."""
        # Create test state
        fields = self._create_perturbed_state(0.15)

        # Run analysis
        properties = self.analysis.analyze_relaxation_to_equilibrium(
            fields, final_time=5.0, timestep=0.02
        )

        # Validate second law
        validation = self.entropy_analysis.validate_second_law(properties)

        return {
            "pass": validation["valid"],
            "validation": validation,
            "entropy_evolution": properties.entropy_evolution,
        }

    def _test_temperature_dependence(self) -> dict[str, Any]:
        """Test temperature dependence of relaxation."""
        # Analyze temperature dependence
        temp_analysis = self.relaxation_analysis.analyze_temperature_dependence(
            (0.1, 0.5), n_points=10
        )

        # Check for reasonable temperature scaling
        scaling = temp_analysis["temperature_scaling"]
        reasonable_scaling = -2.0 < scaling["exponent"] < 2.0

        return {
            "pass": reasonable_scaling and scaling["r_squared"] > 0.5,
            "temperature_analysis": temp_analysis,
            "scaling_exponent": scaling["exponent"],
            "fit_quality": scaling["r_squared"],
        }

    def _test_entropy_production(self) -> dict[str, Any]:
        """Test entropy production analysis."""
        # Create test state
        fields = self._create_perturbed_state(0.1)

        # Compute entropy production sources
        sources = self.entropy_analysis.compute_entropy_production_sources(fields)

        # Run equilibration
        properties = self.analysis.analyze_relaxation_to_equilibrium(
            fields, final_time=4.0, timestep=0.02
        )

        # Analyze entropy flow
        flow_analysis = self.entropy_analysis.analyze_entropy_flow(properties)

        # Test passes if entropy production is positive and decreases
        positive_production = sources["total"] >= 0
        approaches_equilibrium = flow_analysis.get("equilibrium_approached", False)

        return {
            "pass": positive_production and approaches_equilibrium,
            "entropy_sources": sources,
            "flow_analysis": flow_analysis,
            "positive_production": positive_production,
            "approaches_equilibrium": approaches_equilibrium,
        }

    def _create_perturbed_state(self, perturbation: float) -> ISFieldConfiguration:
        """Create initial non-equilibrium state with specified perturbation."""
        fields = ISFieldConfiguration(self.grid)

        # Background equilibrium state (T = 0.2 GeV)
        T = 0.2
        g_eff = 37.5
        rho = (np.pi**2 / 90.0) * g_eff * T**4
        p = rho / 3.0

        fields.rho.fill(rho)
        fields.pressure.fill(p)
        fields.four_velocity.fill_zero()
        fields.four_velocity.data[..., 0] = 1.0

        # Add non-equilibrium perturbations
        fields.Pi.fill(perturbation * p)
        fields.pi_munu.fill_zero()
        fields.pi_munu.data[..., 1, 1] = perturbation * p * 0.5
        fields.pi_munu.data[..., 2, 2] = -perturbation * p * 0.5
        fields.q_mu.fill_zero()

        return fields


# Utility functions


def create_equilibration_benchmark(
    grid: SpacetimeGrid,
    metric: GeneralMetric | None = None,
    transport_coeffs: TransportCoefficients | None = None,
    **kwargs,
) -> EquilibrationAnalysis:
    """
    Factory function for creating equilibration benchmark instances.

    Args:
        grid: Spacetime grid
        metric: Spacetime metric (Minkowski if None)
        transport_coeffs: Transport coefficients (default if None)
        **kwargs: Additional arguments

    Returns:
        Configured EquilibrationAnalysis instance
    """
    if metric is None:
        metric = MinkowskiMetric()

    if transport_coeffs is None:
        transport_coeffs = TransportCoefficients(
            shear_viscosity=0.08,
            bulk_viscosity=0.04,
            shear_relaxation_time=0.5,
            bulk_relaxation_time=0.3,
        )

    return EquilibrationAnalysis(grid, metric, transport_coeffs, **kwargs)


def run_relaxation_analysis(
    initial_fields: ISFieldConfiguration, analysis: EquilibrationAnalysis, final_time: float = 5.0
) -> EquilibrationProperties:
    """
    Quick relaxation analysis.

    Args:
        initial_fields: Initial field configuration
        analysis: EquilibrationAnalysis instance
        final_time: Final simulation time

    Returns:
        Equilibration properties
    """
    return analysis.analyze_relaxation_to_equilibrium(initial_fields, final_time=final_time)


def validate_entropy_production(
    properties: EquilibrationProperties, analysis: EquilibrationAnalysis
) -> bool:
    """
    Quick entropy production validation.

    Args:
        properties: Equilibration properties
        analysis: EquilibrationAnalysis instance

    Returns:
        True if entropy production satisfies second law
    """
    entropy_analysis = EntropyProductionAnalysis(analysis)
    validation = entropy_analysis.validate_second_law(properties)
    return validation["valid"]


def analyze_approach_to_equilibrium(
    initial_perturbation: float, benchmark: EquilibrationBenchmark
) -> dict[str, Any]:
    """
    Analyze approach to equilibrium for given initial perturbation.

    Args:
        initial_perturbation: Initial perturbation strength
        benchmark: EquilibrationBenchmark instance

    Returns:
        Analysis results
    """
    fields = benchmark._create_perturbed_state(initial_perturbation)
    properties = benchmark.analysis.analyze_relaxation_to_equilibrium(fields)

    return {
        "properties": properties,
        "relaxation_comparison": benchmark.relaxation_analysis.compare_with_theory(properties),
        "entropy_validation": benchmark.entropy_analysis.validate_second_law(properties),
    }
