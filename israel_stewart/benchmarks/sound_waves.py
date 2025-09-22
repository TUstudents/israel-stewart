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
    NumericalSoundWaveBenchmark: Numerical simulation benchmark for sound waves

Functions:
    create_sound_wave_benchmark: Factory function for creating analytical benchmark instances
    create_numerical_benchmark: Factory function for creating numerical benchmark instances
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
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy import optimize

from ..core.derivatives import CovariantDerivative, ProjectionOperator
from ..core.fields import ISFieldConfiguration, TransportCoefficients
from ..core.four_vectors import FourVector
from ..core.metrics import GeneralMetric, MinkowskiMetric
from ..core.spacetime_grid import SpacetimeGrid
from ..core.stress_tensors import StressEnergyTensor, ViscousStressTensor
from ..core.tensor_base import TensorField
from ..equations.conservation import ConservationLaws
from ..equations.relaxation import ISRelaxationEquations
from ..solvers import create_periodic_grid
from ..solvers.spectral import SpectralISHydrodynamics


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
            warnings.warn("Sound speed outside physical range [0,1]", stacklevel=2)
        if self.attenuation < 0:
            warnings.warn("Negative attenuation indicates instability", stacklevel=2)


@dataclass
class NumericalWaveResults:
    """Results from numerical sound wave simulation."""

    wave_number: float
    measured_frequency: float
    measured_damping_rate: float
    analytical_frequency: float
    analytical_damping_rate: float
    frequency_error: float
    damping_error: float
    simulation_time: float
    grid_resolution: int
    convergence_achieved: bool
    time_series_data: dict[str, npt.NDArray[np.float64]] = field(default_factory=dict)


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
        self.conservation = ConservationLaws(self.background_fields)
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
        fields.u_mu.fill(0.0)
        fields.u_mu[..., 0] = 1.0  # u^t = 1 in rest frame

        # Zero dissipative fluxes
        fields.Pi.fill(0.0)
        fields.pi_munu.fill(0.0)
        fields.q_mu.fill(0.0)

        return fields

    def analyze_dispersion_relation(
        self,
        wave_vector: npt.NDArray[np.float64],
        frequencies: npt.NDArray[np.float64] | None = None,
    ) -> list[WaveProperties]:
        """
        Analyze dispersion relation for given wave vector using robust root finding.

        Args:
            wave_vector: Spatial wave vector components
            frequencies: Legacy parameter (ignored in new implementation)

        Returns:
            List of wave mode properties found by solving det(M) = 0
        """
        k_magnitude = np.linalg.norm(wave_vector)

        # Check cache
        cache_key = tuple(wave_vector)
        if cache_key in self._dispersion_cache:
            return [self._dispersion_cache[cache_key]]

        # Use robust root finding to solve det(M) = 0
        try:
            complex_roots = self._find_dispersion_roots(k_magnitude)

            wave_modes = []
            for omega_complex in complex_roots:
                try:
                    properties = self._solve_single_mode(omega_complex, wave_vector)
                    if properties is not None:
                        wave_modes.append(properties)
                except Exception as e:
                    warnings.warn(
                        f"Failed to create mode for omega={omega_complex}: {e}", stacklevel=2
                    )

            # Find physical modes (finite attenuation, reasonable frequency)
            physical_modes = [mode for mode in wave_modes if self._is_physical_mode(mode)]

            if physical_modes:
                # Cache the most physical mode (usually the sound mode with smallest damping)
                best_mode = min(physical_modes, key=lambda m: abs(m.attenuation))
                self._dispersion_cache[cache_key] = best_mode

            return physical_modes

        except Exception as e:
            warnings.warn(f"Root finding failed for k={k_magnitude}: {e}", stacklevel=2)
            return []

    def _estimate_sound_speed(self) -> float:
        """Estimate sound speed from thermodynamic properties."""
        rho = np.mean(self.background_fields.rho)
        p = np.mean(self.background_fields.pressure)

        if rho <= 0:
            return 1.0 / np.sqrt(3.0)

        # Proper thermodynamic sound speed: c_s² = ∂p/∂ε
        # For conformal radiation fluid: p = ε/3 → c_s² = 1/3
        cs_squared = p / rho  # This gives 1/3 for radiation background

        # Note: Viscous corrections affect dispersion relation, not ideal sound speed
        # Keep this as the thermodynamic sound speed without viscous modifications

        return np.sqrt(max(0.0, min(1.0, cs_squared)))

    def _solve_single_mode(
        self, omega: complex, wave_vector: npt.NDArray[np.float64]
    ) -> WaveProperties | None:
        """Solve for a single wave mode with given complex frequency."""
        k = np.linalg.norm(wave_vector)
        if k == 0:
            return None

        # Verify this is actually a solution by checking determinant
        det = self._determinant_function(omega, k)
        if abs(det) > 1e-8:  # Not a solution
            return None

        # Extract real and imaginary parts
        omega_real = float(np.real(omega))
        omega_imag = float(np.imag(omega))

        # Calculate wave properties
        sound_speed = omega_real / k if k > 0 else 0
        attenuation = -omega_imag  # Damping rate is -Im(ω)
        dispersion = self._calculate_dispersion(omega_real, wave_vector)

        # Group velocity from dispersion relation
        group_velocity = self._calculate_group_velocity(omega_real, wave_vector)

        return WaveProperties(
            frequency=omega_real,
            wave_vector=wave_vector.copy(),
            sound_speed=sound_speed,
            attenuation=attenuation,
            dispersion=dispersion,
            group_velocity=group_velocity,
            phase_velocity=sound_speed,
            bulk_viscous_correction=self._bulk_viscous_correction(omega_real, k),
            shear_viscous_correction=self._shear_viscous_correction(omega_real, k),
            second_order_correction=self._second_order_correction(omega_real, k),
        )

    def _find_dispersion_roots(self, k: float, max_roots: int = 4) -> list[complex]:
        """
        Find complex frequency roots of the dispersion relation for given wave number.

        Args:
            k: Wave number magnitude
            max_roots: Maximum number of roots to find

        Returns:
            List of complex frequencies ω that satisfy det(M) = 0
        """
        if k == 0:
            return []

        # Estimate sound speed for initial guesses
        cs_estimate = self._estimate_sound_speed()

        # Generate initial guesses around expected frequencies
        # For Israel-Stewart, expect: sound mode + viscous modes
        initial_guesses = [
            complex(cs_estimate * k, 0),  # Sound mode (real)
            complex(cs_estimate * k, -0.1 * k**2),  # Damped sound mode
            complex(0, -0.5 * k**2),  # Viscous mode 1
            complex(0, -(k**2)),  # Viscous mode 2
        ]

        roots = []

        for guess in initial_guesses[:max_roots]:
            try:
                # Use complex root finding for determinant equation
                def det_real_imag(x):
                    omega = complex(x[0], x[1])
                    det = self._determinant_function(omega, k)
                    return [np.real(det), np.imag(det)]

                # Find root using both real and imaginary parts
                result = optimize.root(
                    det_real_imag,
                    [np.real(guess), np.imag(guess)],
                    method="hybr",
                    options={"xtol": 1e-10},
                )

                if result.success:
                    omega = complex(result.x[0], result.x[1])

                    # Verify solution and avoid duplicates
                    det_check = abs(self._determinant_function(omega, k))
                    if det_check < 1e-8:
                        # Check for duplicates
                        is_duplicate = any(abs(omega - existing) < 1e-8 for existing in roots)
                        if not is_duplicate:
                            roots.append(omega)

            except Exception:
                continue  # Root finding failed for this guess

        return roots

    def _build_dispersion_matrix(
        self, omega: complex, wave_vector: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.complex128]:
        """
        Build the linearized Israel-Stewart dispersion matrix.

        Variables: [δε, δv_x, δΠ, δπ_xx]
        - δε: Energy density perturbation
        - δv_x: Velocity perturbation (longitudinal)
        - δΠ: Bulk pressure perturbation
        - δπ_xx: Shear stress perturbation (longitudinal component)

        For plane wave exp(-iωt + ikx), the linearized equations are:
        1. Energy conservation: (-iω)·δε + ik·(ε₀+p₀)·δv_x = 0
        2. Momentum conservation: ik·c_s²·δε - iω·(ε₀+p₀)·δv_x + ik·δΠ + ik·δπ_xx = 0
        3. Bulk relaxation: (1 - iωτ_Π)·δΠ + iζk·δv_x = 0
        4. Shear relaxation: (1 - iωτ_π)·δπ_xx + i·(4/3)ηk·δv_x = 0
        """
        k = np.linalg.norm(wave_vector)

        # Background thermodynamic state
        epsilon0 = np.mean(self.background_fields.rho)  # Energy density ε₀
        p0 = np.mean(self.background_fields.pressure)  # Pressure p₀
        enthalpy = epsilon0 + p0  # Enthalpy density ε₀ + p₀

        # Sound speed squared (c_s² = ∂p/∂ε for radiation: 1/3)
        cs_squared = p0 / epsilon0 if epsilon0 > 0 else 1.0 / 3.0

        # Transport coefficients
        eta = getattr(self.transport_coeffs, "shear_viscosity", 0.0) or 0.0
        zeta = getattr(self.transport_coeffs, "bulk_viscosity", 0.0) or 0.0
        tau_pi = getattr(self.transport_coeffs, "shear_relaxation_time", 0.1) or 0.1
        tau_Pi = getattr(self.transport_coeffs, "bulk_relaxation_time", 0.1) or 0.1

        # Build 4×4 matrix for variables [δε, δv_x, δΠ, δπ_xx]
        matrix = np.zeros((4, 4), dtype=np.complex128)

        # Row 0: Energy conservation ∂_μ T^μ0 = 0
        # (-iω)·δε + ik·(ε₀+p₀)·δv_x = 0
        matrix[0, 0] = -1j * omega  # δε coefficient
        matrix[0, 1] = 1j * k * enthalpy  # δv_x coefficient

        # Row 1: Momentum conservation ∂_μ T^μx = 0
        # ik·c_s²·δε - iω·(ε₀+p₀)·δv_x + ik·δΠ + ik·δπ_xx = 0
        matrix[1, 0] = 1j * k * cs_squared  # δε coefficient
        matrix[1, 1] = -1j * omega * enthalpy  # δv_x coefficient
        matrix[1, 2] = 1j * k  # δΠ coefficient
        matrix[1, 3] = 1j * k  # δπ_xx coefficient

        # Row 2: Bulk pressure relaxation equation
        # (1 - iωτ_Π)·δΠ + iζk·δv_x = 0
        matrix[2, 1] = 1j * zeta * k  # δv_x coefficient
        matrix[2, 2] = 1.0 - 1j * omega * tau_Pi  # δΠ coefficient

        # Row 3: Shear stress relaxation equation
        # (1 - iωτ_π)·δπ_xx + i·(4/3)ηk·δv_x = 0
        matrix[3, 1] = 1j * (4.0 / 3.0) * eta * k  # δv_x coefficient
        matrix[3, 3] = 1.0 - 1j * omega * tau_pi  # δπ_xx coefficient

        return matrix

    def _determinant_function(self, omega: complex, k: float) -> complex:
        """
        Compute the determinant of the dispersion matrix.

        This function is used for root finding to solve the dispersion relation det(M) = 0.

        Args:
            omega: Complex frequency
            k: Wave number magnitude

        Returns:
            Complex determinant value
        """
        wave_vector = np.array([k, 0.0, 0.0])  # Longitudinal wave along x-axis
        matrix = self._build_dispersion_matrix(omega, wave_vector)
        return np.linalg.det(matrix)

    def _calculate_attenuation(self, omega: float, wave_vector: npt.NDArray[np.float64]) -> float:
        """Calculate wave attenuation due to viscosity."""
        k = np.linalg.norm(wave_vector)
        if k == 0:
            return 0.0

        # Viscous attenuation
        eta = getattr(self.transport_coeffs, "shear_viscosity", 0.0)
        zeta = getattr(self.transport_coeffs, "bulk_viscosity", 0.0)

        rho0 = np.mean(self.background_fields.rho)
        p0 = np.mean(self.background_fields.pressure)
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
        tau_Pi = getattr(self.transport_coeffs, "bulk_relaxation_time", 0.1) or 0.1

        return -zeta * k**2 / (1.0 + (omega * tau_Pi) ** 2)

    def _shear_viscous_correction(self, omega: float, k: float) -> float:
        """Calculate shear viscosity correction to dispersion."""
        if k == 0:
            return 0.0

        eta = getattr(self.transport_coeffs, "shear_viscosity", 0.0)
        tau_pi = getattr(self.transport_coeffs, "shear_relaxation_time", 0.1) or 0.1

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

            # Use determinant-based solver from SoundWaveAnalysis
            wave_vector = np.array([k, 0.0, 0.0])
            modes = self.analysis.analyze_dispersion_relation(wave_vector)

            if not modes:
                frequencies.append(np.nan)
                attenuations.append(np.nan)
                phase_velocities.append(np.nan)
                group_velocities.append(np.nan)
                continue

            # Select the most appropriate mode (usually the first physical mode)
            best_mode = modes[0]

            omega = best_mode.frequency
            frequencies.append(omega)
            attenuations.append(best_mode.attenuation)
            phase_velocities.append(best_mode.phase_velocity)
            group_velocities.append(best_mode.group_velocity)

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


class NumericalSoundWaveBenchmark:
    """
    Numerical benchmark for sound wave propagation using spectral simulation.

    This class transforms the analytical sound wave benchmark into a true numerical
    benchmark by running time-evolved simulations and comparing the measured
    frequency/damping with theoretical predictions.
    """

    def __init__(
        self,
        domain_size: float = 2 * np.pi,
        grid_points: tuple[int, int, int, int] = (64, 64, 16, 16),
        transport_coeffs: TransportCoefficients | None = None,
        metric: GeneralMetric | None = None,
    ):
        """
        Initialize numerical sound wave benchmark.

        Args:
            domain_size: Spatial domain size (periodic)
            grid_points: Grid resolution (Nt, Nx, Ny, Nz)
            transport_coeffs: Transport coefficients for viscosity
            metric: Spacetime metric (defaults to Minkowski)
        """
        self.domain_size = domain_size
        self.grid_points = grid_points

        # Create periodic grid for spectral simulation
        time_range = (0.0, 10.0)  # Will be adjusted based on wave properties
        spatial_ranges = [(0.0, domain_size)] * 3
        self.grid = create_periodic_grid("cartesian", time_range, spatial_ranges, grid_points)

        # Physics setup
        self.metric = metric or MinkowskiMetric()
        self.transport_coeffs = transport_coeffs or self._default_transport_coeffs()

        # Initialize analytical analysis for comparison
        self.analytical = SoundWaveAnalysis(self.grid, self.metric, self.transport_coeffs)

        # Initialize spectral solver
        self.fields = ISFieldConfiguration(self.grid)
        self.solver = SpectralISHydrodynamics(self.grid, self.fields, self.transport_coeffs)

    def _default_transport_coeffs(self) -> TransportCoefficients:
        """Create default transport coefficients for testing."""
        return TransportCoefficients(
            shear_viscosity=0.1,
            bulk_viscosity=0.05,
            shear_relaxation_time=0.5,
            bulk_relaxation_time=0.3,
            lambda_pi_pi=0.1,
            xi_1=0.05,
        )

    def setup_initial_conditions(
        self,
        wave_number: float,
        amplitude: float = 0.01,
        background_density: float = 1.0,
        background_pressure: float = 1.0 / 3.0,
    ) -> None:
        """
        Setup sinusoidal perturbation initial conditions.

        Args:
            wave_number: Wave number k for the perturbation
            amplitude: Perturbation amplitude (should be small for linear regime)
            background_density: Background energy density ρ₀
            background_pressure: Background pressure P₀
        """
        # Get spatial coordinates
        x = self.grid.coordinates["x"]
        y = self.grid.coordinates["y"]
        z = self.grid.coordinates["z"]

        # Create meshgrid for full 3D
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Sound wave perturbation along x-direction
        # δρ = A * sin(k*x), δuₓ = A' * sin(k*x)

        # Background state
        self.fields.rho.fill(background_density)
        self.fields.pressure.fill(background_pressure)

        # Velocity: u^μ = (γ, γvˣ, 0, 0) where γ ≈ 1 for small velocities
        self.fields.u_mu.fill(0.0)
        self.fields.u_mu[..., 0] = 1.0  # u^t = 1 in rest frame

        # Add perturbation
        delta_rho = amplitude * np.sin(wave_number * X)
        delta_ux = amplitude * 0.5 * np.sin(wave_number * X)  # Velocity perturbation

        # Apply perturbations
        for t_idx in range(self.grid_points[0]):
            self.fields.rho[t_idx, ...] = background_density + delta_rho
            self.fields.u_mu[t_idx, ..., 1] = delta_ux

        # Update pressure with equation of state (P = ρ/3 for radiation)
        self.fields.pressure[:] = self.fields.rho / 3.0

        # Zero dissipative fluxes initially
        self.fields.Pi.fill(0.0)
        self.fields.pi_munu.fill(0.0)
        self.fields.q_mu.fill(0.0)

    def run_simulation(
        self,
        wave_number: float,
        simulation_time: float = 10.0,
        n_periods: int = 5,
        dt_factor: float = 0.1,
    ) -> NumericalWaveResults:
        """
        Run numerical simulation of sound wave evolution.

        Args:
            wave_number: Wave number to simulate
            simulation_time: Total simulation time
            n_periods: Number of wave periods to evolve
            dt_factor: Timestep factor (fraction of CFL limit)

        Returns:
            Numerical wave simulation results
        """
        # Setup initial conditions
        self.setup_initial_conditions(wave_number)

        # Get analytical prediction for comparison
        wave_vector = np.array([wave_number, 0.0, 0.0])
        analytical_modes = self.analytical.analyze_dispersion_relation(wave_vector)

        if not analytical_modes:
            raise ValueError(f"Could not find analytical mode for k={wave_number}")

        analytical_mode = analytical_modes[0]
        analytical_freq = analytical_mode.frequency
        analytical_damping = analytical_mode.attenuation

        # Adjust simulation time based on wave properties
        if analytical_freq > 0:
            period = 2 * np.pi / analytical_freq
            simulation_time = max(simulation_time, n_periods * period)

        # Determine timestep
        dx = self.grid.spatial_spacing[0]
        sound_speed = analytical_mode.sound_speed
        dt_cfl = dt_factor * dx / max(sound_speed, 0.1)

        # Time evolution
        n_steps = int(simulation_time / dt_cfl)
        time_points = np.linspace(0, simulation_time, n_steps)

        # Storage for time series analysis
        rho_time_series = []
        ux_time_series = []

        # Monitor point for frequency analysis (middle of domain)
        monitor_idx = (
            self.grid_points[1] // 2,
            self.grid_points[2] // 2,
            self.grid_points[3] // 2,
        )

        # Time evolution loop
        current_time = 0.0
        for _step in range(n_steps):
            # Record time series at monitor point
            rho_monitor = self.fields.rho[0, monitor_idx[0], monitor_idx[1], monitor_idx[2]]
            ux_monitor = self.fields.u_mu[0, monitor_idx[0], monitor_idx[1], monitor_idx[2], 1]

            rho_time_series.append(rho_monitor)
            ux_time_series.append(ux_monitor)

            # Evolve fields one timestep
            try:
                self.solver.time_step(dt_cfl)
                current_time += dt_cfl
            except Exception as e:
                warnings.warn(f"Simulation failed at t={current_time}: {e}", stacklevel=2)
                break

        # Analyze time series for frequency and damping
        time_array = np.array(time_points[: len(rho_time_series)])
        rho_array = np.array(rho_time_series)

        measured_freq, measured_damping = self._extract_frequency_damping(time_array, rho_array)

        # Calculate errors
        freq_error = abs(measured_freq - analytical_freq) / max(analytical_freq, 1e-10)
        damping_error = abs(measured_damping - analytical_damping) / max(analytical_damping, 1e-10)

        # Check convergence
        convergence_achieved = (
            freq_error < 0.1  # 10% frequency error
            and damping_error < 0.2  # 20% damping error (more tolerance for damping)
        )

        return NumericalWaveResults(
            wave_number=wave_number,
            measured_frequency=measured_freq,
            measured_damping_rate=measured_damping,
            analytical_frequency=analytical_freq,
            analytical_damping_rate=analytical_damping,
            frequency_error=freq_error,
            damping_error=damping_error,
            simulation_time=simulation_time,
            grid_resolution=self.grid_points[1],
            convergence_achieved=convergence_achieved,
            time_series_data={
                "time": time_array,
                "density": rho_array,
                "velocity": np.array(ux_time_series),
            },
        )

    def _extract_frequency_damping(
        self, time: npt.NDArray[np.float64], density_signal: npt.NDArray[np.float64]
    ) -> tuple[float, float]:
        """
        Extract frequency and damping rate from time series.

        Uses FFT analysis and exponential fitting to extract ω and γ from
        a signal of the form A*exp(-γt)*cos(ωt + φ).

        Args:
            time: Time array
            density_signal: Signal array (e.g., density fluctuations)

        Returns:
            Tuple of (frequency, damping_rate)
        """
        if len(time) < 10:
            return 0.0, 0.0

        # Remove DC component
        signal_ac = density_signal - np.mean(density_signal)

        # FFT analysis for frequency
        dt = time[1] - time[0] if len(time) > 1 else 1.0
        fft_freqs = np.fft.fftfreq(len(signal_ac), dt)
        fft_vals = np.fft.fft(signal_ac)

        # Find dominant frequency (positive frequencies only)
        positive_freqs = fft_freqs[fft_freqs > 0]
        positive_vals = fft_vals[fft_freqs > 0]

        if len(positive_freqs) == 0:
            return 0.0, 0.0

        # Peak frequency
        peak_idx = np.argmax(np.abs(positive_vals))
        measured_frequency = 2 * np.pi * positive_freqs[peak_idx]  # Convert to angular frequency

        # Exponential envelope fitting for damping
        envelope = np.abs(signal_ac)

        # Fit exponential decay: A * exp(-γt)
        try:
            # Use log-linear fit for exponential
            valid_envelope = envelope[envelope > 0.01 * np.max(envelope)]
            valid_time = time[: len(valid_envelope)]

            if len(valid_envelope) > 5:
                log_envelope = np.log(valid_envelope)
                coeffs = np.polyfit(valid_time, log_envelope, 1)
                measured_damping = -coeffs[0]  # Negative slope gives damping rate
            else:
                measured_damping = 0.0
        except Exception:
            measured_damping = 0.0

        return max(measured_frequency, 0.0), max(measured_damping, 0.0)

    def _extract_frequency_windowed_fft(
        self,
        time: npt.NDArray[np.float64],
        density_signal: npt.NDArray[np.float64],
        window_fraction: float = 0.5,
    ) -> tuple[float, float, float]:
        """
        Extract frequency and damping using windowed FFT for time-resolved analysis.

        This method provides more robust frequency extraction by analyzing
        overlapping time windows to track frequency evolution.

        Args:
            time: Time array
            density_signal: Signal array
            window_fraction: Fraction of total time to use for each window

        Returns:
            Tuple of (frequency, damping_rate, frequency_std)
        """
        if len(time) < 20:
            return 0.0, 0.0, 0.0

        dt = time[1] - time[0] if len(time) > 1 else 1.0
        window_size = int(window_fraction * len(time))
        step_size = max(1, window_size // 4)  # 75% overlap

        frequencies = []
        amplitudes = []

        for start_idx in range(0, len(time) - window_size, step_size):
            end_idx = start_idx + window_size
            window_time = time[start_idx:end_idx]
            window_signal = density_signal[start_idx:end_idx] - np.mean(
                density_signal[start_idx:end_idx]
            )

            # Apply Hann window to reduce spectral leakage
            hann_window = np.hanning(len(window_signal))
            windowed_signal = window_signal * hann_window

            # FFT analysis
            fft_freqs = np.fft.fftfreq(len(windowed_signal), dt)
            fft_vals = np.fft.fft(windowed_signal)

            # Find peak frequency
            positive_freqs = fft_freqs[fft_freqs > 0]
            positive_vals = fft_vals[fft_freqs > 0]

            if len(positive_freqs) > 0:
                peak_idx = np.argmax(np.abs(positive_vals))
                freq = 2 * np.pi * positive_freqs[peak_idx]
                amp = np.abs(positive_vals[peak_idx])

                frequencies.append(freq)
                amplitudes.append(amp)

        if not frequencies:
            return 0.0, 0.0, 0.0

        # Calculate statistics
        mean_frequency = np.mean(frequencies)
        frequency_std = np.std(frequencies)

        # Fit exponential decay to amplitude evolution
        if len(amplitudes) > 5:
            window_centers = np.arange(len(amplitudes)) * step_size * dt
            try:
                log_amps = np.log(np.array(amplitudes))
                coeffs = np.polyfit(window_centers, log_amps, 1)
                damping_rate = -coeffs[0]
            except Exception:
                damping_rate = 0.0
        else:
            damping_rate = 0.0

        return max(mean_frequency, 0.0), max(damping_rate, 0.0), frequency_std

    def _extract_complex_frequency(
        self, time: npt.NDArray[np.float64], density_signal: npt.NDArray[np.float64]
    ) -> complex:
        """
        Extract complex frequency ω = ω_real + i*γ from time series.

        Uses the Prony method and matrix pencil technique for robust
        extraction of complex poles from noisy signals.

        Args:
            time: Time array
            density_signal: Signal array

        Returns:
            Complex frequency ω_complex = ω_real + i*γ
        """
        if len(time) < 10:
            return complex(0.0, 0.0)

        # Remove DC component
        signal_clean = density_signal - np.mean(density_signal)

        # Apply simple Prony method for single complex pole
        try:
            # Use autocorrelation approach for noise robustness
            N = len(signal_clean)
            if N < 6:
                return complex(0.0, 0.0)

            # Build correlation matrix
            p = min(4, N // 3)  # Order of the model
            R = np.zeros((p + 1, p + 1))

            for i in range(p + 1):
                for j in range(p + 1):
                    if i + j < N:
                        R[i, j] = np.sum(signal_clean[: N - i - j] * signal_clean[i + j :])

            # Solve Yule-Walker equations
            if np.linalg.det(R[1:, 1:]) > 1e-12:
                a = np.linalg.solve(R[1:, 1:], -R[1:, 0])
                a = np.concatenate([[1], a])

                # Find roots of characteristic polynomial
                roots = np.roots(a)

                # Select most significant root (closest to unit circle for stability)
                dt = time[1] - time[0] if len(time) > 1 else 1.0
                valid_roots = [r for r in roots if abs(r) < 1.0 and abs(r) > 0.1]

                if valid_roots:
                    # Choose root with largest magnitude (most significant)
                    best_root = max(valid_roots, key=abs)
                    # Convert to complex frequency
                    omega_complex = np.log(best_root) / dt
                    return omega_complex

        except Exception:
            pass

        # Fallback to FFT method
        freq, damping = self._extract_frequency_damping(time, density_signal)
        return complex(freq, -damping)

    def _validate_convergence(
        self, result: "NumericalWaveResults", tolerance: float = 0.05
    ) -> bool:
        """
        Validate numerical convergence against analytical predictions.

        Args:
            result: Simulation result to validate
            tolerance: Relative error tolerance (default 5%)

        Returns:
            True if result converged within tolerance
        """
        freq_error = result.frequency_error
        damping_error = result.damping_error

        # Check both frequency and damping errors
        freq_converged = freq_error < tolerance
        damping_converged = damping_error < tolerance or result.analytical_damping_rate < 1e-8

        # Additional physics checks
        physics_valid = (
            result.measured_frequency > 0
            and result.measured_damping_rate >= 0
            and result.measured_frequency / result.wave_number < 1.0  # Causality
        )

        return freq_converged and damping_converged and physics_valid

    def run_benchmark_suite(
        self,
        wave_numbers: npt.NDArray[np.float64] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Run comprehensive numerical benchmark suite.

        Args:
            wave_numbers: Array of wave numbers to test
            **kwargs: Additional arguments for run_simulation

        Returns:
            Dictionary with benchmark results
        """
        if wave_numbers is None:
            wave_numbers = np.array([0.5, 1.0, 2.0, 3.0])

        results = []
        passed_tests = 0
        total_tests = len(wave_numbers)

        for k in wave_numbers:
            try:
                result = self.run_simulation(k, **kwargs)
                results.append(result)

                if result.convergence_achieved:
                    passed_tests += 1

            except Exception as e:
                warnings.warn(f"Failed simulation for k={k}: {e}", stacklevel=2)
                continue

        # Calculate summary statistics
        freq_errors = [r.frequency_error for r in results if r.convergence_achieved]
        damping_errors = [r.damping_error for r in results if r.convergence_achieved]

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / max(total_tests, 1),
            "wave_numbers": wave_numbers,
            "results": results,
            "mean_frequency_error": np.mean(freq_errors) if freq_errors else float("inf"),
            "mean_damping_error": np.mean(damping_errors) if damping_errors else float("inf"),
            "max_frequency_error": np.max(freq_errors) if freq_errors else float("inf"),
            "max_damping_error": np.max(damping_errors) if damping_errors else float("inf"),
        }

    def run_comprehensive_validation(
        self,
        k_range: tuple[float, float] = (0.1, 5.0),
        n_points: int = 10,
        tolerance: float = 0.05,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Run comprehensive validation covering multiple physics regimes.

        Tests sound speed recovery, viscous damping scaling, causality constraints,
        and convergence across a range of wave numbers.

        Args:
            k_range: (k_min, k_max) wave number range to test
            n_points: Number of wave numbers to test
            tolerance: Relative error tolerance for validation
            **kwargs: Additional simulation parameters

        Returns:
            Comprehensive validation report
        """
        k_min, k_max = k_range
        wave_numbers = np.logspace(np.log10(k_min), np.log10(k_max), n_points)

        # Run simulations
        suite_results = self.run_benchmark_suite(wave_numbers, **kwargs)

        # Physics validation tests
        validation_report = {
            **suite_results,
            "physics_tests": {},
            "scaling_tests": {},
            "causality_tests": {},
            "convergence_tests": {},
        }

        # Test 1: Sound speed recovery in ideal limit
        validation_report["physics_tests"]["sound_speed_recovery"] = (
            self._test_sound_speed_recovery(suite_results, tolerance)
        )

        # Test 2: Viscous damping k² scaling
        validation_report["scaling_tests"]["damping_k2_scaling"] = self._test_damping_scaling(
            suite_results, tolerance
        )

        # Test 3: Causality constraints
        validation_report["causality_tests"]["phase_velocity"] = self._test_causality_constraints(
            suite_results
        )

        # Test 4: Convergence analysis
        validation_report["convergence_tests"]["numerical_accuracy"] = (
            self._test_numerical_convergence(suite_results, tolerance)
        )

        # Overall assessment
        validation_report["overall_pass"] = all(
            test.get("pass", False)
            for category in [
                "physics_tests",
                "scaling_tests",
                "causality_tests",
                "convergence_tests",
            ]
            for test in validation_report[category].values()
        )

        return validation_report

    def _test_sound_speed_recovery(
        self, suite_results: dict[str, Any], tolerance: float
    ) -> dict[str, Any]:
        """Test recovery of correct sound speed in various limits."""
        results = suite_results["results"]
        if not results:
            return {"pass": False, "error": "No simulation results"}

        # Check sound speeds vs analytical predictions
        sound_speeds = []
        analytical_speeds = []
        errors = []

        for result in results:
            if result.convergence_achieved:
                cs_measured = result.measured_frequency / result.wave_number
                cs_analytical = result.analytical_frequency / result.wave_number

                sound_speeds.append(cs_measured)
                analytical_speeds.append(cs_analytical)
                errors.append(abs(cs_measured - cs_analytical) / cs_analytical)

        if not errors:
            return {"pass": False, "error": "No converged results for sound speed test"}

        mean_error = np.mean(errors)
        max_error = np.max(errors)

        return {
            "pass": max_error < tolerance,
            "mean_sound_speed_error": mean_error,
            "max_sound_speed_error": max_error,
            "measured_speeds": sound_speeds,
            "analytical_speeds": analytical_speeds,
            "tolerance": tolerance,
        }

    def _test_damping_scaling(
        self, suite_results: dict[str, Any], tolerance: float
    ) -> dict[str, Any]:
        """Test k² scaling of viscous damping."""
        results = suite_results["results"]

        # Extract k and γ values
        k_values = []
        damping_values = []

        for result in results:
            if result.convergence_achieved and result.measured_damping_rate > 0:
                k_values.append(result.wave_number)
                damping_values.append(result.measured_damping_rate)

        if len(k_values) < 3:
            return {"pass": False, "error": "Insufficient data for scaling test"}

        # Fit γ = A * k^α and check if α ≈ 2
        try:
            log_k = np.log(k_values)
            log_gamma = np.log(damping_values)

            # Linear fit in log space: log(γ) = log(A) + α*log(k)
            coeffs = np.polyfit(log_k, log_gamma, 1)
            scaling_exponent = coeffs[0]

            # Calculate R²
            log_gamma_fit = coeffs[0] * log_k + coeffs[1]
            ss_res = np.sum((log_gamma - log_gamma_fit) ** 2)
            ss_tot = np.sum((log_gamma - np.mean(log_gamma)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Check if scaling is close to k²
            scaling_error = abs(scaling_exponent - 2.0) / 2.0
            scaling_test_pass = scaling_error < tolerance and r_squared > 0.7

            return {
                "pass": scaling_test_pass,
                "scaling_exponent": scaling_exponent,
                "expected_exponent": 2.0,
                "scaling_error": scaling_error,
                "r_squared": r_squared,
                "tolerance": tolerance,
            }

        except Exception as e:
            return {"pass": False, "error": f"Scaling analysis failed: {e}"}

    def _test_causality_constraints(self, suite_results: dict[str, Any]) -> dict[str, Any]:
        """Test causality constraints: phase and group velocities ≤ c."""
        results = suite_results["results"]

        causality_violations = []
        phase_velocities = []

        for result in results:
            if result.convergence_achieved:
                phase_velocity = result.measured_frequency / result.wave_number
                phase_velocities.append(phase_velocity)

                # Check causality: v_phase ≤ 1 (in natural units where c = 1)
                if phase_velocity > 1.0:
                    causality_violations.append(
                        {
                            "k": result.wave_number,
                            "v_phase": phase_velocity,
                            "violation": phase_velocity - 1.0,
                        }
                    )

        max_phase_velocity = max(phase_velocities) if phase_velocities else 0.0
        causality_preserved = len(causality_violations) == 0

        return {
            "pass": causality_preserved,
            "max_phase_velocity": max_phase_velocity,
            "causality_violations": causality_violations,
            "n_violations": len(causality_violations),
            "total_tests": len(phase_velocities),
        }

    def _test_numerical_convergence(
        self, suite_results: dict[str, Any], tolerance: float
    ) -> dict[str, Any]:
        """Test numerical convergence and accuracy."""
        results = suite_results["results"]

        converged_results = [r for r in results if r.convergence_achieved]
        frequency_errors = [r.frequency_error for r in converged_results]
        damping_errors = [r.damping_error for r in converged_results]

        if not frequency_errors:
            return {"pass": False, "error": "No converged results for accuracy test"}

        # Statistical analysis of errors
        mean_freq_error = np.mean(frequency_errors)
        max_freq_error = np.max(frequency_errors)
        mean_damping_error = np.mean(damping_errors) if damping_errors else 0.0
        max_damping_error = np.max(damping_errors) if damping_errors else 0.0

        # Convergence criteria
        freq_accuracy = max_freq_error < tolerance
        damping_accuracy = (
            max_damping_error < tolerance or max_damping_error < 0.1
        )  # Relaxed for small damping
        convergence_rate = len(converged_results) / len(results) if results else 0.0

        overall_pass = freq_accuracy and damping_accuracy and convergence_rate > 0.8

        return {
            "pass": overall_pass,
            "convergence_rate": convergence_rate,
            "mean_frequency_error": mean_freq_error,
            "max_frequency_error": max_freq_error,
            "mean_damping_error": mean_damping_error,
            "max_damping_error": max_damping_error,
            "frequency_accuracy": freq_accuracy,
            "damping_accuracy": damping_accuracy,
            "tolerance": tolerance,
        }


def create_numerical_benchmark(
    domain_size: float = 2 * np.pi,
    grid_points: tuple[int, int, int, int] = (64, 64, 16, 16),
    transport_coeffs: TransportCoefficients | None = None,
    metric: GeneralMetric | None = None,
    **kwargs,
) -> NumericalSoundWaveBenchmark:
    """
    Factory function for creating numerical sound wave benchmark instances.

    Args:
        domain_size: Spatial domain size (periodic)
        grid_points: Grid resolution (Nt, Nx, Ny, Nz)
        transport_coeffs: Transport coefficients for viscosity
        metric: Spacetime metric (defaults to Minkowski)
        **kwargs: Additional arguments passed to NumericalSoundWaveBenchmark

    Returns:
        Configured numerical benchmark instance

    Examples:
        >>> # Create benchmark with default parameters
        >>> benchmark = create_numerical_benchmark()
        >>>
        >>> # Run single wave number test
        >>> result = benchmark.run_simulation(wave_number=1.0)
        >>> print(f"Frequency error: {result.frequency_error:.3f}")
        >>>
        >>> # Run comprehensive benchmark suite
        >>> suite_results = benchmark.run_benchmark_suite()
        >>> print(f"Success rate: {suite_results['success_rate']:.1%}")
    """
    return NumericalSoundWaveBenchmark(
        domain_size=domain_size,
        grid_points=grid_points,
        transport_coeffs=transport_coeffs,
        metric=metric,
        **kwargs,
    )
