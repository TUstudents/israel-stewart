"""
Sound wave propagation benchmark for Israel-Stewart hydrodynamics.

This module implements comprehensive tests for linear wave propagation in relativistic
viscous fluids, validating dispersion relations and stability properties of the
Israel-Stewart equations.

Classes:
    SoundWaveAnalysis: Core analysis class for sound wave properties
    DispersionRelation: Dispersion relation solver and analyzer
    LinearStabilityAnalysis: Linear stability analysis tools
    WaveTestSuite: Comprehensive test suite for wave propagation

Functions:
    create_sound_wave_benchmark: Factory function for creating benchmark instances
    run_dispersion_analysis: Quick dispersion relation analysis
    validate_causality: Check causality constraints
    analyze_wave_modes: Analyze normal modes of the system

The benchmark validates:
1. Sound speed recovery in the ideal limit
2. Viscous damping rates and dispersion
3. Second-order corrections to dispersion relations
4. Causality and stability constraints
5. Mode coupling in the full Israel-Stewart system
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from ..core.derivatives import CovariantDerivative, ProjectionOperator
from ..core.fields import ISFieldConfiguration, TransportCoefficients
from ..core.four_vectors import FourVector
from ..core.metrics import GeneralMetric, MinkowskiMetric
from ..core.stress_tensors import StressEnergyTensor, ViscousStressTensor
from ..core.tensor_base import SpacetimeGrid, TensorField
from ..equations.conservation import ConservationLaws
from ..equations.relaxation import ISRelaxationEquations


@dataclass
class WaveProperties:
    """Properties of sound waves in viscous relativistic fluids."""

    frequency: float
    wave_vector: npt.NDArray[np.float64]
    sound_speed: float
    attenuation: float
    dispersion: float
    group_velocity: npt.NDArray[np.float64]
    phase_velocity: float

    # Israel-Stewart specific properties
    bulk_viscous_correction: float = 0.0
    shear_viscous_correction: float = 0.0
    second_order_correction: float = 0.0

    def __post_init__(self):
        """Validate wave properties."""
        if self.frequency < 0:
            raise ValueError("Frequency must be non-negative")
        if self.sound_speed < 0 or self.sound_speed > 1:
            warnings.warn("Sound speed outside physical range [0,1]")
        if self.attenuation < 0:
            warnings.warn("Negative attenuation indicates instability")


class SoundWaveAnalysis:
    """
    Core analysis class for sound wave properties in Israel-Stewart hydrodynamics.

    This class provides comprehensive tools for analyzing linear wave propagation
    in relativistic viscous fluids, including dispersion relations, damping rates,
    and stability analysis.
    """

    def __init__(
        self,
        grid: SpacetimeGrid,
        metric: GeneralMetric,
        transport_coeffs: TransportCoefficients,
        background_fields: ISFieldConfiguration | None = None,
    ):
        """
        Initialize sound wave analysis.

        Args:
            grid: Spacetime grid for numerical analysis
            metric: Spacetime metric
            transport_coeffs: Transport coefficients
            background_fields: Background field configuration
        """
        self.grid = grid
        self.metric = metric
        self.transport_coeffs = transport_coeffs
        self.background_fields = background_fields or self._default_background()

        # Initialize physics modules
        self.conservation = ConservationLaws(grid, metric)
        self.relaxation = ISRelaxationEquations(grid, metric, transport_coeffs)

        # Analysis results cache
        self._dispersion_cache: dict[tuple[float, ...], WaveProperties] = {}
        self._stability_cache: dict[str, dict[str, Any]] = {}

    def _default_background(self) -> ISFieldConfiguration:
        """Create default equilibrium background state."""
        fields = ISFieldConfiguration(self.grid)

        # Equilibrium state
        fields.rho.fill(1.0)  # Energy density
        fields.pressure.fill(1.0 / 3.0)  # Radiation pressure
        fields.four_velocity.fill_zero()
        fields.four_velocity.data[..., 0] = 1.0  # u^t = 1 in rest frame

        # Zero dissipative fluxes
        fields.Pi.fill(0.0)
        fields.pi_munu.fill_zero()
        fields.q_mu.fill_zero()

        return fields

    def analyze_dispersion_relation(
        self,
        wave_vector: npt.NDArray[np.float64],
        frequencies: npt.NDArray[np.float64] | None = None,
    ) -> list[WaveProperties]:
        """
        Analyze dispersion relation for given wave vector.

        Args:
            wave_vector: Spatial wave vector components
            frequencies: Frequency range to analyze (auto-generated if None)

        Returns:
            List of wave mode properties
        """
        k_magnitude = np.linalg.norm(wave_vector)

        # Check cache
        cache_key = tuple(wave_vector)
        if cache_key in self._dispersion_cache:
            return [self._dispersion_cache[cache_key]]

        if frequencies is None:
            # Auto-generate frequency range around expected sound speed
            cs_estimate = self._estimate_sound_speed()
            omega_max = 2.0 * k_magnitude * cs_estimate
            frequencies = np.linspace(0, omega_max, 100)

        # Solve dispersion relation
        wave_modes = []
        for omega in frequencies:
            try:
                properties = self._solve_single_mode(omega, wave_vector)
                if properties is not None:
                    wave_modes.append(properties)
            except Exception as e:
                warnings.warn(f"Failed to solve for omega={omega}: {e}")

        # Find physical modes (positive group velocity)
        physical_modes = [mode for mode in wave_modes if self._is_physical_mode(mode)]

        if physical_modes:
            # Cache the most physical mode
            best_mode = min(physical_modes, key=lambda m: abs(m.attenuation))
            self._dispersion_cache[cache_key] = best_mode

        return physical_modes

    def _estimate_sound_speed(self) -> float:
        """Estimate sound speed from thermodynamic properties."""
        # For radiation: c_s^2 = 1/3
        # Include viscous corrections
        rho = np.mean(self.background_fields.rho.data)
        p = np.mean(self.background_fields.pressure.data)

        if rho <= 0:
            return 1.0 / np.sqrt(3.0)

        # Ideal gas sound speed
        cs_squared = p / (rho + p)

        # Viscous corrections (first-order estimate)
        if hasattr(self.transport_coeffs, "bulk_viscosity"):
            zeta = self.transport_coeffs.bulk_viscosity
            cs_squared *= 1.0 - zeta / (rho + p)

        return np.sqrt(max(0.0, min(1.0, cs_squared)))

    def _solve_single_mode(
        self, omega: float, wave_vector: npt.NDArray[np.float64]
    ) -> WaveProperties | None:
        """Solve for a single wave mode."""
        k = np.linalg.norm(wave_vector)
        if k == 0:
            return None

        # Construct linearized equations matrix
        matrix = self._build_dispersion_matrix(omega, wave_vector)

        # Find determinant zeros (dispersion relation)
        det = np.linalg.det(matrix)

        if abs(det) > 1e-6:  # Not a solution
            return None

        # Calculate wave properties
        sound_speed = omega / k if k > 0 else 0
        attenuation = self._calculate_attenuation(omega, wave_vector)
        dispersion = self._calculate_dispersion(omega, wave_vector)

        # Group velocity from dispersion relation
        group_velocity = self._calculate_group_velocity(omega, wave_vector)

        return WaveProperties(
            frequency=omega,
            wave_vector=wave_vector.copy(),
            sound_speed=sound_speed,
            attenuation=attenuation,
            dispersion=dispersion,
            group_velocity=group_velocity,
            phase_velocity=sound_speed,
            bulk_viscous_correction=self._bulk_viscous_correction(omega, k),
            shear_viscous_correction=self._shear_viscous_correction(omega, k),
            second_order_correction=self._second_order_correction(omega, k),
        )

    def _build_dispersion_matrix(
        self, omega: float, wave_vector: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.complex128]:
        """Build the linearized equations matrix."""
        k = np.linalg.norm(wave_vector)

        # Background state
        rho0 = np.mean(self.background_fields.rho.data)
        p0 = np.mean(self.background_fields.pressure.data)

        # Transport coefficients
        eta = getattr(self.transport_coeffs, "shear_viscosity", 0.0)
        zeta = getattr(self.transport_coeffs, "bulk_viscosity", 0.0)
        tau_pi = getattr(self.transport_coeffs, "shear_relaxation_time", 0.1)
        tau_Pi = getattr(self.transport_coeffs, "bulk_relaxation_time", 0.1)

        # Matrix size: [drho, dp, dPi, dpi]
        matrix = np.zeros((4, 4), dtype=np.complex128)

        # Conservation equations
        # Continuity: i*omega*drho + i*k*d(rho*v) = 0
        matrix[0, 0] = 1j * omega  # drho term
        matrix[0, 1] = 1j * k * rho0 / (rho0 + p0)  # pressure gradient term

        # Euler equation: i*omega*d(rho*v) + i*k*dp + i*k*dPi + viscous = 0
        matrix[1, 0] = 1j * omega * rho0 / (rho0 + p0)
        matrix[1, 1] = 1j * k
        matrix[1, 2] = 1j * k  # bulk viscous stress
        matrix[1, 3] = 1j * k  # shear viscous stress (simplified)

        # Israel-Stewart equations
        # Bulk: (1 + i*omega*tau_Pi)*dPi + zeta*i*k*dv = 0
        matrix[2, 1] = zeta * 1j * k / (rho0 + p0)
        matrix[2, 2] = 1.0 + 1j * omega * tau_Pi

        # Shear: (1 + i*omega*tau_pi)*dpi + eta*i*k*dv = 0
        matrix[3, 1] = eta * 1j * k / (rho0 + p0)
        matrix[3, 3] = 1.0 + 1j * omega * tau_pi

        return matrix

    def _calculate_attenuation(self, omega: float, wave_vector: npt.NDArray[np.float64]) -> float:
        """Calculate wave attenuation due to viscosity."""
        k = np.linalg.norm(wave_vector)
        if k == 0:
            return 0.0

        # Viscous attenuation
        eta = getattr(self.transport_coeffs, "shear_viscosity", 0.0)
        zeta = getattr(self.transport_coeffs, "bulk_viscosity", 0.0)

        rho0 = np.mean(self.background_fields.rho.data)
        p0 = np.mean(self.background_fields.pressure.data)
        enthalpy = rho0 + p0

        # First-order viscous damping
        gamma_bulk = zeta * k**2 / enthalpy
        gamma_shear = (4.0 / 3.0) * eta * k**2 / enthalpy

        return gamma_bulk + gamma_shear

    def _calculate_dispersion(self, omega: float, wave_vector: npt.NDArray[np.float64]) -> float:
        """Calculate wave dispersion (deviation from linear relation)."""
        k = np.linalg.norm(wave_vector)
        if k == 0:
            return 0.0

        # Linear dispersion: omega = c_s * k
        cs = self._estimate_sound_speed()
        linear_omega = cs * k

        return abs(omega - linear_omega) / linear_omega if linear_omega > 0 else 0.0

    def _calculate_group_velocity(
        self, omega: float, wave_vector: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate group velocity vector."""
        k = np.linalg.norm(wave_vector)
        if k == 0:
            return np.zeros(3)

        # For isotropic medium, group velocity is parallel to wave vector
        cs = self._estimate_sound_speed()
        return cs * wave_vector / k

    def _bulk_viscous_correction(self, omega: float, k: float) -> float:
        """Calculate bulk viscosity correction to dispersion."""
        if k == 0:
            return 0.0

        zeta = getattr(self.transport_coeffs, "bulk_viscosity", 0.0)
        tau_Pi = getattr(self.transport_coeffs, "bulk_relaxation_time", 0.1)

        return -zeta * k**2 / (1.0 + (omega * tau_Pi) ** 2)

    def _shear_viscous_correction(self, omega: float, k: float) -> float:
        """Calculate shear viscosity correction to dispersion."""
        if k == 0:
            return 0.0

        eta = getattr(self.transport_coeffs, "shear_viscosity", 0.0)
        tau_pi = getattr(self.transport_coeffs, "shear_relaxation_time", 0.1)

        return -(4.0 / 3.0) * eta * k**2 / (1.0 + (omega * tau_pi) ** 2)

    def _second_order_correction(self, omega: float, k: float) -> float:
        """Calculate second-order corrections to dispersion."""
        if k == 0:
            return 0.0

        # Second-order transport coefficients
        lambda_pi_pi = getattr(self.transport_coeffs, "lambda_pi_pi", 0.0)
        xi_1 = getattr(self.transport_coeffs, "xi_1", 0.0)

        return (lambda_pi_pi + xi_1) * k**4  # k^4 correction

    def _is_physical_mode(self, properties: WaveProperties) -> bool:
        """Check if wave mode is physical (causal and stable)."""
        # Causality: phase velocity <= 1
        if properties.phase_velocity > 1.0:
            return False

        # Stability: attenuation >= 0
        if properties.attenuation < -1e-10:
            return False

        # Group velocity causality
        if np.linalg.norm(properties.group_velocity) > 1.0:
            return False

        return True


class DispersionRelation:
    """
    Dispersion relation solver and analyzer for Israel-Stewart hydrodynamics.

    This class provides specialized tools for solving and analyzing dispersion
    relations in relativistic viscous fluids.
    """

    def __init__(self, analysis: SoundWaveAnalysis):
        """Initialize with sound wave analysis instance."""
        self.analysis = analysis
        self._polynomial_cache: dict[tuple[float, ...], npt.NDArray[np.complex128]] = {}

    def solve_exact_dispersion(self, k: float, max_order: int = 4) -> list[complex]:
        """
        Solve exact dispersion relation polynomial.

        Args:
            k: Wave number magnitude
            max_order: Maximum polynomial order to consider

        Returns:
            List of complex frequency solutions
        """
        # Build characteristic polynomial
        coeffs = self._build_characteristic_polynomial(k, max_order)

        # Solve polynomial
        roots = np.roots(coeffs)

        # Filter physical solutions
        physical_roots = []
        for root in roots:
            omega = root
            if np.isreal(omega) and np.real(omega) >= 0:
                physical_roots.append(omega)
            elif np.imag(omega) < 0:  # Stable growing mode
                physical_roots.append(omega)

        return physical_roots

    def _build_characteristic_polynomial(
        self, k: float, max_order: int
    ) -> npt.NDArray[np.complex128]:
        """Build characteristic polynomial for dispersion relation."""
        # Cache check
        cache_key = (k, max_order)
        if cache_key in self._polynomial_cache:
            return self._polynomial_cache[cache_key]

        # Transport coefficients
        eta = getattr(self.analysis.transport_coeffs, "shear_viscosity", 0.0)
        zeta = getattr(self.analysis.transport_coeffs, "bulk_viscosity", 0.0)
        tau_pi = getattr(self.analysis.transport_coeffs, "shear_relaxation_time", 0.1)
        tau_Pi = getattr(self.analysis.transport_coeffs, "bulk_relaxation_time", 0.1)

        # Background state
        rho0 = np.mean(self.analysis.background_fields.rho.data)
        p0 = np.mean(self.analysis.background_fields.pressure.data)
        cs_squared = p0 / (rho0 + p0)

        # Build polynomial coefficients
        # General form: sum_n a_n * omega^n = 0
        coeffs = np.zeros(max_order + 1, dtype=np.complex128)

        # Leading order (ideal fluid)
        coeffs[2] = 1.0  # omega^2 term
        coeffs[0] = -cs_squared * k**2  # constant term

        # First-order viscous corrections
        if max_order >= 3:
            coeffs[3] = tau_pi + tau_Pi  # omega^3 term
            coeffs[1] = -(eta + zeta) * k**2 / (rho0 + p0)  # omega term

        # Second-order corrections
        if max_order >= 4:
            lambda_pi_pi = getattr(self.analysis.transport_coeffs, "lambda_pi_pi", 0.0)
            xi_1 = getattr(self.analysis.transport_coeffs, "xi_1", 0.0)

            coeffs[4] = tau_pi * tau_Pi  # omega^4 term
            coeffs[2] += (lambda_pi_pi + xi_1) * k**2  # omega^2 correction

        # Cache result
        self._polynomial_cache[cache_key] = coeffs

        return coeffs

    def analyze_dispersion_curve(
        self, k_range: npt.NDArray[np.float64], mode_type: str = "sound"
    ) -> dict[str, npt.NDArray[np.float64]]:
        """
        Analyze complete dispersion curve omega(k).

        Args:
            k_range: Array of wave numbers to analyze
            mode_type: Type of mode to track ("sound", "diffusive", "all")

        Returns:
            Dictionary with dispersion curve data
        """
        frequencies = []
        attenuations = []
        phase_velocities = []
        group_velocities = []

        for k in k_range:
            if k == 0:
                frequencies.append(0.0)
                attenuations.append(0.0)
                phase_velocities.append(0.0)
                group_velocities.append(0.0)
                continue

            # Solve for this k
            roots = self.solve_exact_dispersion(k)

            if not roots:
                frequencies.append(np.nan)
                attenuations.append(np.nan)
                phase_velocities.append(np.nan)
                group_velocities.append(np.nan)
                continue

            # Select appropriate mode
            if mode_type == "sound":
                # Take mode closest to sound speed
                cs = self.analysis._estimate_sound_speed()
                target_omega = cs * k
                best_root = min(roots, key=lambda r: abs(np.real(r) - target_omega))
            else:
                # Take most physical mode
                best_root = max(roots, key=lambda r: np.real(r))

            omega = best_root
            frequencies.append(np.real(omega))
            attenuations.append(-np.imag(omega))
            phase_velocities.append(np.real(omega) / k if k > 0 else 0)

            # Group velocity (numerical derivative)
            dk = 0.01 * k
            roots_plus = self.solve_exact_dispersion(k + dk)
            if roots_plus:
                omega_plus = min(roots_plus, key=lambda r: abs(r - omega))
                group_vel = np.real(omega_plus - omega) / dk
            else:
                group_vel = np.real(omega) / k
            group_velocities.append(group_vel)

        return {
            "k": k_range,
            "omega": np.array(frequencies),
            "attenuation": np.array(attenuations),
            "phase_velocity": np.array(phase_velocities),
            "group_velocity": np.array(group_velocities),
        }


class LinearStabilityAnalysis:
    """Linear stability analysis tools for Israel-Stewart hydrodynamics."""

    def __init__(self, analysis: SoundWaveAnalysis):
        """Initialize with sound wave analysis instance."""
        self.analysis = analysis

    def analyze_stability_matrix(self, k: float = 0.0) -> dict[str, Any]:
        """
        Analyze stability of the linearized system.

        Args:
            k: Wave number for stability analysis

        Returns:
            Stability analysis results
        """
        # Build linearization matrix
        matrix = self._build_linearization_matrix(k)

        # Eigenvalue analysis
        eigenvalues, eigenvectors = np.linalg.eig(matrix)

        # Stability assessment
        max_real_part = np.max(np.real(eigenvalues))
        is_stable = max_real_part <= 0

        # Growth rates
        growth_rates = np.real(eigenvalues)
        oscillation_frequencies = np.imag(eigenvalues)

        return {
            "matrix": matrix,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "is_stable": is_stable,
            "max_growth_rate": max_real_part,
            "growth_rates": growth_rates,
            "oscillation_frequencies": oscillation_frequencies,
            "spectral_radius": np.max(np.abs(eigenvalues)),
            "condition_number": np.linalg.cond(matrix),
        }

    def _build_linearization_matrix(self, k: float) -> npt.NDArray[np.complex128]:
        """Build linearization matrix for stability analysis."""
        # This is similar to dispersion matrix but for stability analysis
        wave_vector = np.array([k, 0.0, 0.0])
        return self.analysis._build_dispersion_matrix(0.0, wave_vector)

    def causality_analysis(self, k_max: float = 10.0, n_points: int = 100) -> dict[str, Any]:
        """
        Analyze causality constraints for the Israel-Stewart system.

        Args:
            k_max: Maximum wave number to analyze
            n_points: Number of points in analysis

        Returns:
            Causality analysis results
        """
        k_range = np.linspace(0.1, k_max, n_points)

        max_phase_velocities = []
        max_group_velocities = []
        causality_violations = []

        for k in k_range:
            # Analyze dispersion
            wave_vector = np.array([k, 0.0, 0.0])
            modes = self.analysis.analyze_dispersion_relation(wave_vector)

            if not modes:
                max_phase_velocities.append(np.nan)
                max_group_velocities.append(np.nan)
                causality_violations.append(True)
                continue

            # Find maximum velocities
            phase_vels = [mode.phase_velocity for mode in modes]
            group_vels = [np.linalg.norm(mode.group_velocity) for mode in modes]

            max_phase_vel = max(phase_vels) if phase_vels else 0
            max_group_vel = max(group_vels) if group_vels else 0

            max_phase_velocities.append(max_phase_vel)
            max_group_velocities.append(max_group_vel)

            # Check causality violation (v > c = 1)
            violation = max_phase_vel > 1.0 or max_group_vel > 1.0
            causality_violations.append(violation)

        return {
            "k_range": k_range,
            "max_phase_velocities": np.array(max_phase_velocities),
            "max_group_velocities": np.array(max_group_velocities),
            "causality_violations": np.array(causality_violations),
            "is_causal": not np.any(causality_violations),
            "violation_threshold": k_range[np.argmax(causality_violations)]
            if np.any(causality_violations)
            else np.inf,
        }


class WaveTestSuite:
    """Comprehensive test suite for wave propagation in Israel-Stewart hydrodynamics."""

    def __init__(
        self, grid: SpacetimeGrid, metric: GeneralMetric, transport_coeffs: TransportCoefficients
    ):
        """Initialize test suite."""
        self.grid = grid
        self.metric = metric
        self.transport_coeffs = transport_coeffs

        # Initialize analysis tools
        self.analysis = SoundWaveAnalysis(grid, metric, transport_coeffs)
        self.dispersion = DispersionRelation(self.analysis)
        self.stability = LinearStabilityAnalysis(self.analysis)

    def run_comprehensive_tests(self) -> dict[str, Any]:
        """Run comprehensive wave propagation tests."""
        results = {}

        # Test 1: Sound speed recovery
        results["sound_speed_test"] = self._test_sound_speed_recovery()

        # Test 2: Viscous damping
        results["damping_test"] = self._test_viscous_damping()

        # Test 3: Dispersion relation
        results["dispersion_test"] = self._test_dispersion_relation()

        # Test 4: Stability analysis
        results["stability_test"] = self._test_linear_stability()

        # Test 5: Causality constraints
        results["causality_test"] = self._test_causality_constraints()

        # Test 6: Second-order corrections
        results["second_order_test"] = self._test_second_order_corrections()

        # Overall assessment
        results["overall_pass"] = all(
            test_result.get("pass", False) for test_result in results.values()
        )

        return results

    def _test_sound_speed_recovery(self) -> dict[str, Any]:
        """Test that sound speed is recovered correctly in ideal limit."""
        # Create ideal transport coefficients
        ideal_coeffs = TransportCoefficients(
            shear_viscosity=0.0,
            bulk_viscosity=0.0,
            shear_relaxation_time=0.1,
            bulk_relaxation_time=0.1,
        )

        ideal_analysis = SoundWaveAnalysis(self.grid, self.metric, ideal_coeffs)

        # Test dispersion for small k
        k_test = 0.1
        wave_vector = np.array([k_test, 0.0, 0.0])
        modes = ideal_analysis.analyze_dispersion_relation(wave_vector)

        if not modes:
            return {"pass": False, "error": "No modes found"}

        # Check sound speed
        mode = modes[0]
        expected_cs = 1.0 / np.sqrt(3.0)  # Radiation
        measured_cs = mode.sound_speed

        error = abs(measured_cs - expected_cs) / expected_cs
        tolerance = 0.05  # 5% tolerance

        return {
            "pass": error < tolerance,
            "expected_sound_speed": expected_cs,
            "measured_sound_speed": measured_cs,
            "relative_error": error,
            "tolerance": tolerance,
        }

    def _test_viscous_damping(self) -> dict[str, Any]:
        """Test viscous damping rates."""
        # Test with significant viscosity
        k_test = 1.0
        wave_vector = np.array([k_test, 0.0, 0.0])
        modes = self.analysis.analyze_dispersion_relation(wave_vector)

        if not modes:
            return {"pass": False, "error": "No modes found"}

        mode = modes[0]
        attenuation = mode.attenuation

        # Check that viscosity produces damping
        has_damping = attenuation > 0

        # Check scaling with k^2 (for small k)
        k_test2 = 0.5
        wave_vector2 = np.array([k_test2, 0.0, 0.0])
        modes2 = self.analysis.analyze_dispersion_relation(wave_vector2)

        if modes2:
            attenuation2 = modes2[0].attenuation
            # Should scale as k^2
            expected_ratio = (k_test / k_test2) ** 2
            actual_ratio = attenuation / attenuation2 if attenuation2 > 0 else np.inf
            scaling_error = abs(actual_ratio - expected_ratio) / expected_ratio
        else:
            scaling_error = np.inf

        return {
            "pass": has_damping and scaling_error < 0.3,
            "attenuation": attenuation,
            "has_damping": has_damping,
            "scaling_error": scaling_error,
        }

    def _test_dispersion_relation(self) -> dict[str, Any]:
        """Test dispersion relation properties."""
        k_range = np.linspace(0.1, 2.0, 20)
        dispersion_data = self.dispersion.analyze_dispersion_curve(k_range)

        # Check for valid dispersion curve
        valid_points = ~np.isnan(dispersion_data["omega"])
        fraction_valid = np.sum(valid_points) / len(k_range)

        # Check monotonicity of phase velocity
        phase_vels = dispersion_data["phase_velocity"][valid_points]
        is_monotonic = np.all(np.diff(phase_vels) <= 0.1)  # Allow some noise

        return {
            "pass": fraction_valid > 0.8 and is_monotonic,
            "fraction_valid_points": fraction_valid,
            "is_monotonic": is_monotonic,
            "dispersion_data": dispersion_data,
        }

    def _test_linear_stability(self) -> dict[str, Any]:
        """Test linear stability of the system."""
        stability_results = self.stability.analyze_stability_matrix(k=1.0)

        is_stable = stability_results["is_stable"]
        max_growth_rate = stability_results["max_growth_rate"]

        return {
            "pass": is_stable,
            "is_stable": is_stable,
            "max_growth_rate": max_growth_rate,
            "eigenvalues": stability_results["eigenvalues"],
        }

    def _test_causality_constraints(self) -> dict[str, Any]:
        """Test causality constraints."""
        causality_results = self.stability.causality_analysis(k_max=5.0, n_points=50)

        is_causal = causality_results["is_causal"]
        violation_threshold = causality_results["violation_threshold"]

        return {
            "pass": is_causal or violation_threshold > 2.0,  # Allow violations at high k
            "is_causal": is_causal,
            "violation_threshold": violation_threshold,
            "causality_data": causality_results,
        }

    def _test_second_order_corrections(self) -> dict[str, Any]:
        """Test second-order transport coefficient effects."""
        # Compare with and without second-order coefficients
        base_coeffs = self.transport_coeffs

        # Enhanced coefficients with second-order terms
        enhanced_coeffs = TransportCoefficients(
            shear_viscosity=base_coeffs.shear_viscosity,
            bulk_viscosity=base_coeffs.bulk_viscosity,
            shear_relaxation_time=base_coeffs.shear_relaxation_time,
            bulk_relaxation_time=base_coeffs.bulk_relaxation_time,
            lambda_pi_pi=0.1,
            xi_1=0.1,
        )

        enhanced_analysis = SoundWaveAnalysis(self.grid, self.metric, enhanced_coeffs)

        # Compare dispersion relations
        k_test = 1.0
        wave_vector = np.array([k_test, 0.0, 0.0])

        base_modes = self.analysis.analyze_dispersion_relation(wave_vector)
        enhanced_modes = enhanced_analysis.analyze_dispersion_relation(wave_vector)

        if not (base_modes and enhanced_modes):
            return {"pass": False, "error": "Could not compute modes"}

        # Check for measurable difference
        base_freq = base_modes[0].frequency
        enhanced_freq = enhanced_modes[0].frequency

        relative_change = abs(enhanced_freq - base_freq) / base_freq
        has_second_order_effect = relative_change > 0.01  # 1% change

        return {
            "pass": has_second_order_effect,
            "base_frequency": base_freq,
            "enhanced_frequency": enhanced_freq,
            "relative_change": relative_change,
            "has_second_order_effect": has_second_order_effect,
        }


# Utility functions


def create_sound_wave_benchmark(
    grid: SpacetimeGrid,
    metric: GeneralMetric | None = None,
    transport_coeffs: TransportCoefficients | None = None,
    **kwargs,
) -> SoundWaveAnalysis:
    """
    Factory function for creating sound wave benchmark instances.

    Args:
        grid: Spacetime grid
        metric: Spacetime metric (Minkowski if None)
        transport_coeffs: Transport coefficients (default if None)
        **kwargs: Additional arguments for SoundWaveAnalysis

    Returns:
        Configured SoundWaveAnalysis instance
    """
    if metric is None:
        metric = MinkowskiMetric()

    if transport_coeffs is None:
        transport_coeffs = TransportCoefficients(
            shear_viscosity=0.1,
            bulk_viscosity=0.05,
            shear_relaxation_time=0.5,
            bulk_relaxation_time=0.3,
        )

    return SoundWaveAnalysis(grid, metric, transport_coeffs, **kwargs)


def run_dispersion_analysis(
    k_range: npt.NDArray[np.float64], analysis: SoundWaveAnalysis
) -> dict[str, npt.NDArray[np.float64]]:
    """
    Quick dispersion relation analysis.

    Args:
        k_range: Array of wave numbers
        analysis: SoundWaveAnalysis instance

    Returns:
        Dispersion curve data
    """
    dispersion = DispersionRelation(analysis)
    return dispersion.analyze_dispersion_curve(k_range)


def validate_causality(analysis: SoundWaveAnalysis, k_max: float = 10.0) -> bool:
    """
    Quick causality validation.

    Args:
        analysis: SoundWaveAnalysis instance
        k_max: Maximum wave number to check

    Returns:
        True if system is causal
    """
    stability = LinearStabilityAnalysis(analysis)
    results = stability.causality_analysis(k_max=k_max)
    return results["is_causal"]


def analyze_wave_modes(
    wave_vector: npt.NDArray[np.float64], analysis: SoundWaveAnalysis
) -> list[WaveProperties]:
    """
    Analyze wave modes for a specific wave vector.

    Args:
        wave_vector: Wave vector components
        analysis: SoundWaveAnalysis instance

    Returns:
        List of wave mode properties
    """
    return analysis.analyze_dispersion_relation(wave_vector)
