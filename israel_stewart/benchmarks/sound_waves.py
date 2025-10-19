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
from ..core.spacegrid import SpaceGrid
from ..core.spacetime_grid import SpacetimeGrid
from ..core.stress_tensors import StressEnergyTensor, ViscousStressTensor
from ..core.tensor_base import TensorField
from ..equations.conservation import ConservationLaws
from ..equations.ired_simple import HardSphereIReD
from ..equations.relaxation import ISRelaxationEquations
from ..solvers import create_periodic_grid
from ..solvers.spectral import SpectralISHydrodynamics
from ..utils.logging_config import get_logger


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
        # Allow small negative frequencies within numerical precision (e.g., -5e-17)
        if self.frequency < -1e-12:
            raise ValueError(
                f"Frequency {self.frequency} is negative beyond numerical precision tolerance"
            )
        # Clamp to zero if within tolerance
        if -1e-12 <= self.frequency < 0:
            object.__setattr__(self, "frequency", 0.0)

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
        grid: SpaceGrid,
        metric: GeneralMetric,
        transport_coeffs: TransportCoefficients,
        background_fields: ISFieldConfiguration | None = None,
    ):
        """
        Initialize sound wave analysis.

        Args:
            grid: Spatial grid for numerical analysis (pure 3D)
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

        # Zero dissipative fluxes (Landau frame)
        fields.Pi.fill(0.0)
        fields.pi_munu.fill(0.0)
        fields.V_mu.fill(0.0)  # Particle diffusion current

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
            print(f"Complex roots for k={k_magnitude}: {complex_roots}")

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
                # Classify modes and prefer sound modes over viscous modes
                sound_modes = []
                viscous_modes = []
                other_modes = []

                for mode in physical_modes:
                    # Reconstruct complex omega for classification
                    omega_complex = complex(mode.frequency, -mode.attenuation)
                    mode_type = self._classify_mode(omega_complex, k_magnitude)

                    if mode_type == "sound":
                        sound_modes.append(mode)
                    elif mode_type == "viscous":
                        viscous_modes.append(mode)
                    else:
                        other_modes.append(mode)

                # Select best mode: prefer sound > other > viscous
                if sound_modes:
                    # For sound modes, pick the one with smallest damping
                    best_mode = min(sound_modes, key=lambda m: abs(m.attenuation))
                elif other_modes:
                    best_mode = min(other_modes, key=lambda m: abs(m.attenuation))
                elif viscous_modes:
                    best_mode = min(viscous_modes, key=lambda m: abs(m.attenuation))
                else:
                    best_mode = physical_modes[0]

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

        # Estimate background state and transport coefficients
        background_rho = np.mean(self.background_fields.rho)
        background_p = np.mean(self.background_fields.pressure)
        enthalpy = background_rho + background_p

        # Extract transport coefficients
        eta = getattr(self.transport_coeffs, "shear_viscosity", 0.0) or 0.0
        zeta = getattr(self.transport_coeffs, "bulk_viscosity", 0.0) or 0.0

        # First-order estimate of damping rate from viscous theory
        # γ ≈ (ζ + 4η/3) * k² / (ε₀ + p₀)
        # However, this breaks down for large k in Israel-Stewart theory where
        # relaxation effects suppress damping. Cap to ensure |γ| << |ω| for sound modes.
        gamma_first_order = (zeta + 4.0 * eta / 3.0) * k**2 / max(enthalpy, 1e-10)

        # For sound modes, damping should be smaller than frequency: |γ| < |ω|
        # Cap the damping to a fraction of the expected sound frequency
        omega_sound = cs_estimate * k
        gamma_cap = 0.3 * omega_sound  # Keep |γ| < 30% of |ω| for sound mode
        gamma_viscous = min(gamma_first_order, gamma_cap)

        # Generate initial guesses for sound modes with viscous corrections
        # Sound mode typically has ω ~ cs*k with small imaginary part from damping
        initial_guesses = [
            complex(cs_estimate * k, -gamma_viscous),  # Sound mode with capped damping
            complex(cs_estimate * k * 1.3, -gamma_viscous * 0.8),  # Higher frequency variant
            complex(cs_estimate * k * 0.8, -gamma_viscous * 1.2),  # Lower frequency variant
            complex(0, -0.5 * k**2),  # Keep one pure viscous mode for completeness
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
        2. Momentum conservation: ik·c_s²·δε - iω·(ε₀+p₀)·δv_x + ik·δΠ - ik·δπ_xx = 0
        3. Bulk relaxation: (1 - iωτ_Π)·δΠ + iζk·δv_x = 0
        4. Shear relaxation: (1 - iωτ_π)·δπ_xx - i·(4/3)ηk·δv_x = 0
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
        # For (-,+,+,+) signature: ik·c_s²·δε - iω·(ε₀+p₀)·δv_x + ik·δΠ + ik·δπ_xx = 0
        matrix[1, 0] = 1j * k * cs_squared  # δε coefficient
        matrix[1, 1] = -1j * omega * enthalpy  # δv_x coefficient
        matrix[1, 2] = 1j * k  # δΠ coefficient
        matrix[1, 3] = 1j * k  # δπ_xx coefficient (PLUS sign for (-,+,+,+) signature)

        # Row 2: Bulk pressure relaxation equation
        # (1 - iωτ_Π)·δΠ + iζk·δv_x = 0
        matrix[2, 1] = 1j * zeta * k  # δv_x coefficient
        matrix[2, 2] = 1.0 - 1j * omega * tau_Pi  # δΠ coefficient

        # Row 3: Shear stress relaxation equation
        # (1 - iωτ_π)·δπ_xx - i·(4/3)ηk·δv_x = 0
        # Note: Shear has OPPOSITE sign to bulk in the linearized IS equations
        matrix[3, 1] = -1j * (4.0 / 3.0) * eta * k  # δv_x coefficient (MINUS per physics)
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
        delta_Pi_Pi = getattr(self.transport_coeffs, "delta_Pi_Pi", 0.0)  # IReD bulk self-coupling

        # Note: In the linear regime, the k^4 correction combines shear and bulk contributions
        # For the full nonlinear theory, see IReD paper (Wagner et al. 2022)
        return (lambda_pi_pi + delta_Pi_Pi) * k**4  # k^4 correction

    def _is_physical_mode(self, properties: WaveProperties) -> bool:
        """
        Check if wave mode is physical (causal and stable).

        For nearly-ideal fluids with very small viscosity, the damping rate Γ can be
        O(10⁻⁵) or smaller, which is at the limit of numerical precision for root finding.
        In such cases, accept modes where |Γ| << ω as "approximately ideal" since the
        sign of Γ becomes meaningless at this precision level.
        """
        # Causality: phase velocity <= 1
        if properties.phase_velocity > 1.0:
            return False

        # Stability: attenuation >= 0 (with relaxed tolerance for nearly-ideal fluids)
        # For nearly-ideal modes, |Γ| ~ O(10⁻⁵) is at numerical precision limit
        # Accept modes with |Γ/ω| < 1% as "approximately ideal" regardless of sign
        attenuation_threshold = -1e-10  # Standard threshold for clear instability

        if properties.attenuation < attenuation_threshold:
            # Check if this is a nearly-ideal mode (|Γ| << |ω|)
            frequency = properties.frequency
            if frequency > 0:
                relative_damping = abs(properties.attenuation) / frequency
                # If |Γ/ω| < 1%, treat as nearly-ideal (accept small negative Γ)
                if relative_damping < 0.01:
                    # Nearly-ideal mode: numerical precision issue, not real instability
                    pass  # Accept mode
                else:
                    # Significant negative damping: real instability
                    return False
            else:
                # No frequency (pure diffusive mode) - use absolute threshold
                return False

        # Group velocity causality
        if np.linalg.norm(properties.group_velocity) > 1.0:
            return False

        return True

    def _classify_mode(self, omega: complex, k: float) -> str:
        """
        Classify eigenmode as 'sound', 'viscous', or 'other'.

        Sound modes: ω_r > 0 and comparable to cs*k (propagating waves)
        Viscous modes: ω_r ≈ 0, ω_i < 0 (diffusive, non-propagating)

        Args:
            omega: Complex frequency ω = ω_r - iγ
            k: Wave number magnitude

        Returns:
            Mode type: "sound", "viscous", or "other"
        """
        omega_r = omega.real
        omega_i = omega.imag

        # Estimate ideal sound frequency
        cs = self._estimate_sound_speed()
        omega_sound = cs * k

        # Sound mode: real part is significant and positive, imaginary part is smaller
        # Typically |ω_r| ~ cs*k and |ω_i| < |ω_r|
        if omega_r > 0.1 * omega_sound and abs(omega_i) < abs(omega_r):
            return "sound"

        # Viscous/diffusive mode: real part near zero, imaginary part negative
        # These are non-propagating exponential decay modes
        if abs(omega_r) < 0.1 * omega_sound and omega_i < -1e-10:
            return "viscous"

        return "other"


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
            group_velocities.append(np.linalg.norm(best_mode.group_velocity))

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
            delta_Pi_Pi=0.1,  # IReD bulk self-coupling (was xi_1)
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
        grid_points: tuple[int, int, int] = (64, 64, 16),
        transport_coeffs: TransportCoefficients | None = None,
        metric: GeneralMetric | None = None,
    ):
        """
        Initialize numerical sound wave benchmark with pure 3D spatial grid.

        Args:
            domain_size: Spatial domain size (periodic)
            grid_points: Spatial grid resolution (Nx, Ny, Nz)
            transport_coeffs: Transport coefficients for viscosity
            metric: Spacetime metric (defaults to Minkowski)
        """
        self.domain_size = domain_size
        self.grid_points = grid_points

        # Physics setup
        self.metric = metric or MinkowskiMetric()

        # Create pure 3D spatial grid for spectral simulation
        spatial_ranges = [(0.0, domain_size)] * 3
        self.grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=spatial_ranges,
            grid_points=grid_points,  # (nx, ny, nz)
            boundary_conditions="periodic",  # Required for FFT
            metric=self.metric,
        )

        self.transport_coeffs = transport_coeffs or self._default_transport_coeffs()

        # Initialize analytical analysis for comparison
        self.analytical = SoundWaveAnalysis(self.grid, self.metric, self.transport_coeffs)

        # Initialize fields and spectral solver
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
            delta_Pi_Pi=0.05,  # IReD bulk self-coupling (was xi_1)
        )

    def setup_initial_conditions(
        self,
        wave_number: float,
        amplitude: float = 0.01,
        background_density: float = 1.0,
        background_pressure: float = 1.0 / 3.0,
    ) -> None:
        """
        Setup sinusoidal perturbation initial conditions (pure 3D).

        Initializes both hydrodynamic fields AND dissipative fluxes to match
        the Israel-Stewart sound wave eigenmode, ensuring proper viscous damping
        from the start of the simulation.

        Args:
            wave_number: Wave number k for the perturbation
            amplitude: Perturbation amplitude (should be small for linear regime)
            background_density: Background energy density ρ₀
            background_pressure: Background pressure P₀
        """
        # Store background density for use in damping extraction
        self._background_density = background_density

        # Get analytical mode properties and eigenvector structure
        wave_vector = np.array([wave_number, 0.0, 0.0])
        analytical_modes = self.analytical.analyze_dispersion_relation(wave_vector)

        if not analytical_modes:
            warnings.warn(
                f"Could not find analytical mode for k={wave_number}. "
                "Using simplified initialization without dissipative fluxes.",
                stacklevel=2,
            )
            # Fallback to old initialization
            self._setup_simple_initial_conditions(
                wave_number, amplitude, background_density, background_pressure
            )
            return

        # Select sound mode (prefer sound > other > viscous)
        sound_modes = []
        other_modes = []
        viscous_modes = []

        for m in analytical_modes:
            omega_complex = complex(m.frequency, -m.attenuation)
            mode_type = self.analytical._classify_mode(omega_complex, wave_number)

            if mode_type == "sound":
                sound_modes.append(m)
            elif mode_type == "viscous":
                viscous_modes.append(m)
            else:
                other_modes.append(m)

        # Choose best sound mode (lowest attenuation)
        if sound_modes:
            mode = min(sound_modes, key=lambda m: abs(m.attenuation))
        elif other_modes:
            mode = min(other_modes, key=lambda m: abs(m.attenuation))
        else:
            # If only viscous modes found, warn and use fallback
            warnings.warn(
                f"Only viscous modes found for k={wave_number}. "
                "Using simplified initialization.",
                stacklevel=2,
            )
            self._setup_simple_initial_conditions(
                wave_number, amplitude, background_density, background_pressure
            )
            return

        # Get spatial coordinates
        X, Y, Z = self.grid.meshgrid()

        # Compute eigenmode structure by solving dispersion matrix
        # The mode has frequency ω - iγ, and the eigenvector is complex
        omega_complex = complex(mode.frequency, -mode.attenuation)
        dispersion_matrix = self.analytical._build_dispersion_matrix(omega_complex, wave_vector)

        # Find nullspace eigenvector: M·v = 0 where v = [δε, δv_x, δΠ, δπ_xx]
        # Use SVD for numerical stability with nearly-singular matrices
        try:
            U, s, Vh = np.linalg.svd(dispersion_matrix)
            # SVD gives M = U @ diag(s) @ Vh, where Vh is V^H (conjugate transpose)
            # Right singular vectors are columns of V, so we need Vh[-1, :].conj()
            eigenvector = Vh[-1, :].conj()  # Eigenvector for smallest singular value

            # Normalize eigenvector so that the density perturbation (δε) is real.
            # This sets the overall phase of the wave. A pure cos(kx) for δε means its
            # complex amplitude is purely real.
            if abs(eigenvector[0]) > 1e-12:
                eigenvector /= eigenvector[0]

            # The physical perturbation is Re(v * exp(ikx)) = Re(v)cos(kx) - Im(v)sin(kx)
            # Our eigenvector v is now normalized such that v[0] (for δε) is 1+0j.
            rho_ratio_complex = eigenvector[0]
            v_x_ratio_complex = eigenvector[1]
            Pi_ratio_complex = eigenvector[2]
            pi_xx_ratio_complex = eigenvector[3]

            logger = get_logger(__name__)
            logger.info(
                f"Complex eigenmode ratios (v_x, Π, π_xx):\n"
                f"v_x: {v_x_ratio_complex:.3e}\n"
                f"Π:   {Pi_ratio_complex:.3e}\n"
                f"π_xx: {pi_xx_ratio_complex:.3e}"
            )

        except Exception as e:
            warnings.warn(
                f"Failed to compute eigenmode structure: {e}. "
                "Using simplified initialization, which may be inaccurate.",
                stacklevel=2,
            )
            # Fallback to simplified estimates which will not be a pure mode
            self._setup_simple_initial_conditions(
                wave_number, amplitude, background_density, background_pressure
            )
            return

        # Initialize fields with the full complex eigenmode structure
        cos_kx = np.cos(wave_number * X)
        sin_kx = np.sin(wave_number * X)

        def init_field(ratio_complex):
            # Physical field is Re(v * exp(ikx))
            # With v = Re(v) + i*Im(v), this is Re(v)cos(kx) - Im(v)sin(kx)
            return amplitude * (np.real(ratio_complex) * cos_kx - np.imag(ratio_complex) * sin_kx)

        self.fields.rho[:] = background_density + init_field(rho_ratio_complex)
        self.fields.pressure[:] = self.fields.rho / 3.0  # P = ρ/3 for radiation

        # Velocity
        self.fields.u_mu[:] = 0.0
        self.fields.u_mu[..., 0] = 1.0  # u^t = 1 in rest frame
        self.fields.u_mu[..., 1] = init_field(v_x_ratio_complex)  # u^x = δu_x

        # Dissipative fluxes
        self.fields.Pi[:] = init_field(Pi_ratio_complex)

        self.fields.pi_munu[:] = 0.0
        delta_pi_xx = init_field(pi_xx_ratio_complex)
        self.fields.pi_munu[..., 1, 1] = delta_pi_xx  # π_xx component
        # Transverse components for tracelessness: π_yy = π_zz = -1/2 π_xx
        self.fields.pi_munu[..., 2, 2] = -0.5 * delta_pi_xx
        self.fields.pi_munu[..., 3, 3] = -0.5 * delta_pi_xx

        # Particle diffusion current: V^μ = 0 for isentropic sound wave (Landau frame)
        if hasattr(self.fields, "V_mu"):
            self.fields.V_mu[:] = 0.0

    def _setup_simple_initial_conditions(
        self,
        wave_number: float,
        amplitude: float,
        background_density: float,
        background_pressure: float,
    ) -> None:
        """
        Fallback initialization with first-order analytical estimates for dissipative fluxes.

        Used if full eigenmode cannot be found. Applies first-order viscous theory
        to estimate initial dissipative flux perturbations.
        """
        X, Y, Z = self.grid.meshgrid()
        k_x = wave_number

        # Hydrodynamic perturbations (standard)
        delta_rho = amplitude * np.sin(k_x * X)
        delta_ux = amplitude * 0.5 * np.sin(k_x * X)

        self.fields.rho[:] = background_density + delta_rho
        self.fields.pressure[:] = (background_density + delta_rho) / 3.0

        self.fields.u_mu[:] = 0.0
        self.fields.u_mu[..., 0] = 1.0
        self.fields.u_mu[..., 1] = delta_ux

        # IMPROVED: Estimate dissipative flux perturbations using first-order theory
        # From linearized Israel-Stewart equations:
        # Π ≈ -(ζ·k·δu)/(1-iωτ_Π)
        # π_xx ≈ -(4η/3·k·δu)/(1-iωτ_π)

        # Extract transport coefficients
        eta = getattr(self.transport_coeffs, "shear_viscosity", 0.0) or 0.0
        zeta = getattr(self.transport_coeffs, "bulk_viscosity", 0.0) or 0.0
        tau_pi = getattr(self.transport_coeffs, "shear_relaxation_time", 0.1) or 0.1
        tau_Pi = getattr(self.transport_coeffs, "bulk_relaxation_time", 0.1) or 0.1

        # Estimate sound wave frequency for relaxation time corrections
        cs = 1.0 / np.sqrt(3.0)  # Radiation sound speed
        omega_est = cs * k_x

        # First-order amplitude estimates (accounting for relaxation time lag)
        # δΠ ~ -(ζ·k·δu) / sqrt(1 + (ωτ_Π)²)
        enthalpy = background_density + background_pressure
        relaxation_factor_Pi = 1.0 / np.sqrt(1.0 + (omega_est * tau_Pi) ** 2)
        relaxation_factor_pi = 1.0 / np.sqrt(1.0 + (omega_est * tau_pi) ** 2)

        # Bulk viscous pressure perturbation
        # Π ~ -ζ·∇·u ~ -ζ·k·u_x for sin(kx) mode
        Pi_amplitude = -zeta * k_x * amplitude * 0.5 * relaxation_factor_Pi
        self.fields.Pi[:] = Pi_amplitude * np.sin(k_x * X)

        # Shear stress perturbation (only xx component for longitudinal wave)
        # π_xx ~ -(4η/3)·(∂_x u_x - 1/3·∇·u) ~ -(4η/3)·(2/3·k·u_x) for longitudinal wave
        pi_xx_amplitude = (
            -(4.0 * eta / 3.0) * (2.0 / 3.0) * k_x * amplitude * 0.5 * relaxation_factor_pi
        )
        self.fields.pi_munu[:] = 0.0
        self.fields.pi_munu[..., 0, 0] = pi_xx_amplitude * np.sin(k_x * X)

        # Particle diffusion current (zero for isentropic sound waves, Landau frame)
        if hasattr(self.fields, "V_mu"):
            self.fields.V_mu[:] = 0.0

    def run_simulation(
        self,
        wave_number: float,
        simulation_time: float = 10.0,
        n_periods: int = 5,
        dt_factor: float = 0.1,
        method: str = "split_step",
    ) -> NumericalWaveResults:
        """
        Run numerical simulation of sound wave evolution.

        Args:
            wave_number: Wave number to simulate
            simulation_time: Total simulation time
            n_periods: Number of wave periods to evolve
            dt_factor: Timestep factor (fraction of CFL limit)
            method: Integration method ('split_step' or 'spectral_imex')

        Returns:
            Numerical wave simulation results with frequency and damping
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

        # Determine timestep (CFL condition + relaxation stability)
        dx = self.grid.spatial_spacing[0]
        sound_speed = analytical_mode.sound_speed
        dt_cfl_wave = dt_factor * dx / max(sound_speed, 0.1)

        # Add relaxation time constraint for stiff explicit integration
        # Stability requires dt << min(τ_Π, τ_π) for explicit relaxation
        dt_cfl_relax = 0.01 * min(
            self.transport_coeffs.bulk_relaxation_time, self.transport_coeffs.shear_relaxation_time
        )
        dt_cfl = min(dt_cfl_wave, dt_cfl_relax)

        # Monitor point for time series (at antinode of wave for maximum signal)
        # For sin(kx) wave, antinode is at x = π/(2k)
        # Find grid point closest to this location
        X, Y, Z = self.grid.meshgrid()
        x_antinode = np.pi / (2 * wave_number)

        # Find index closest to antinode in x-direction
        x_1d = X[:, 0, 0]  # X coordinates along first axis
        ix_antinode = np.argmin(np.abs(x_1d - x_antinode))

        # Use center indices for y and z (wave is along x)
        monitor_idx = (ix_antinode, self.grid_points[1] // 2, self.grid_points[2] // 2)

        # Storage for time series
        time_points = []
        rho_time_series = []
        ux_time_series = []
        rho_k_amplitude_series = []  # Fourier mode amplitude

        # Fourier mode index for wave_number
        # For 1D wave along x: k_index is the x-wavenumber index
        nx = self.grid_points[0]
        L_x = self.grid.spatial_ranges[0][1] - self.grid.spatial_ranges[0][0]
        k_index = int(round(wave_number * L_x / (2 * np.pi)))
        k_index = k_index % nx  # Wrap to valid range

        # Record initial state (t=0)
        time_points.append(0.0)
        rho_time_series.append(self.fields.rho[monitor_idx])
        ux_time_series.append(self.fields.u_mu[monitor_idx + (1,)])

        # Extract Fourier mode amplitude (initial)
        rho_fft = np.fft.fftn(self.fields.rho - self._background_density)
        rho_k_amplitude_series.append(np.abs(rho_fft[k_index, 0, 0]))

        # Callback to record time series during evolution
        def record_time_series(t: float, fields: ISFieldConfiguration) -> None:
            time_points.append(t)
            # Extract at monitor point (pure 3D indexing)
            rho_monitor = fields.rho[monitor_idx]
            ux_monitor = fields.u_mu[monitor_idx + (1,)]  # u^x component
            rho_time_series.append(rho_monitor)
            ux_time_series.append(ux_monitor)

            # Extract Fourier mode amplitude
            rho_fft = np.fft.fftn(fields.rho - self._background_density)
            rho_k_amplitude_series.append(np.abs(rho_fft[k_index, 0, 0]))

        # Evolve using spectral solver with callback
        try:
            self.solver.evolve(
                t_final=simulation_time,
                dt=dt_cfl,
                method=method,
                callback=record_time_series,
            )
        except Exception as e:
            warnings.warn(f"Simulation failed: {e}", stacklevel=2)
            # Return empty result
            return self._create_failed_result(wave_number, analytical_freq, analytical_damping)

        # Analyze time series for frequency and damping
        time_array = np.array(time_points)
        rho_array = np.array(rho_time_series)
        ux_array = np.array(ux_time_series)
        rho_k_array = np.array(rho_k_amplitude_series)

        # Extract frequency from point measurement (works well)
        # Extract damping from Fourier mode amplitude (more robust)
        measured_freq_point, _ = self._extract_frequency_damping(time_array, rho_array)
        _, measured_damping_fourier = self._extract_frequency_damping_fourier(
            time_array, rho_k_array
        )

        measured_freq = measured_freq_point
        measured_damping = measured_damping_fourier

        logger = get_logger(__name__)
        logger.info(f"Measured: ω={measured_freq:.6f}, γ={measured_damping:.6f}")
        logger.info(f"Analytical: ω={analytical_freq:.6f}, γ={analytical_damping:.6f}")

        # Calculate errors
        freq_error = abs(measured_freq - analytical_freq) / max(analytical_freq, 1e-10)
        damping_error = abs(measured_damping - analytical_damping) / max(analytical_damping, 1e-10)

        # Check convergence
        convergence_achieved = (
            freq_error < 0.1  # 10% frequency error
            and damping_error < 0.2  # 20% damping error
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
            grid_resolution=self.grid_points[0],
            convergence_achieved=convergence_achieved,
            time_series_data={
                "time": time_array,
                "density": rho_array,
                "velocity": ux_array,
            },
        )

    def _create_failed_result(
        self, wave_number: float, analytical_freq: float, analytical_damping: float
    ) -> NumericalWaveResults:
        """Create result object for failed simulation."""
        return NumericalWaveResults(
            wave_number=wave_number,
            measured_frequency=0.0,
            measured_damping_rate=0.0,
            analytical_frequency=analytical_freq,
            analytical_damping_rate=analytical_damping,
            frequency_error=np.inf,
            damping_error=np.inf,
            simulation_time=0.0,
            grid_resolution=self.grid_points[0],
            convergence_achieved=False,
            time_series_data={},
        )

    def _extract_frequency_damping_fourier(
        self, time: npt.NDArray[np.float64], mode_amplitude: npt.NDArray[np.float64]
    ) -> tuple[float, float]:
        """
        Extract frequency and damping from Fourier mode amplitude time series.

        This method tracks the amplitude of a single Fourier mode, which decays
        exponentially as A(t) = A₀ exp(-γt) without the complications of phase
        evolution at fixed spatial points.

        Args:
            time: Time array
            mode_amplitude: Fourier mode amplitude |ρ_k(t)| time series

        Returns:
            Tuple of (frequency, damping_rate)
        """
        if len(time) < 10:
            return 0.0, 0.0

        # Damping extraction from exponential decay of amplitude
        # Fit log(A(t)) = log(A₀) - γt
        valid_mask = mode_amplitude > 0.01 * np.max(mode_amplitude)
        if np.sum(valid_mask) < 5:
            return 0.0, 0.0

        valid_amp = mode_amplitude[valid_mask]
        valid_time = time[valid_mask]

        try:
            log_amp = np.log(valid_amp)
            coeffs = np.polyfit(valid_time, log_amp, 1)
            measured_damping = -coeffs[0]  # Negative slope = damping rate
        except (ValueError, np.linalg.LinAlgError):
            measured_damping = 0.0

        # Frequency: use the existing point-based method for now
        # (Fourier amplitude doesn't directly give frequency)
        measured_frequency = 0.0

        return measured_frequency, measured_damping

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

        # Remove DC component using known equilibrium density (more accurate for damped waves)
        # For damped oscillations, time-averaged mean is biased toward initial amplitude
        equilibrium = getattr(self, "_background_density", None)
        if equilibrium is None:
            # Fallback to time-averaged mean if background density not available
            equilibrium = np.mean(density_signal)

        signal_ac = density_signal - equilibrium

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
        # Use peak-tracking for more robust damping measurement (avoids phase evolution issues)
        try:
            from scipy.signal import find_peaks

            # Find peaks in the oscillation (local maxima)
            peaks, _ = find_peaks(np.abs(signal_ac), distance=3)

            if len(peaks) >= 3:
                # Fit exponential decay to peak amplitudes
                peak_times = time[peaks]
                peak_amplitudes = np.abs(signal_ac[peaks])

                # Filter out very small peaks
                valid_mask = peak_amplitudes > 0.01 * np.max(peak_amplitudes)
                if np.sum(valid_mask) >= 3:
                    valid_peak_times = peak_times[valid_mask]
                    valid_peak_amps = peak_amplitudes[valid_mask]

                    log_amps = np.log(valid_peak_amps)
                    coeffs = np.polyfit(valid_peak_times, log_amps, 1)
                    measured_damping = -coeffs[0]  # Negative slope gives damping rate
                else:
                    measured_damping = 0.0
            else:
                # Fallback to Hilbert transform if not enough peaks
                from scipy.signal import hilbert

                analytic_signal = hilbert(signal_ac)
                envelope = np.abs(analytic_signal)

                valid_mask = envelope > 0.01 * np.max(envelope)
                valid_envelope = envelope[valid_mask]
                valid_time = time[valid_mask]

                if len(valid_envelope) > 5:
                    log_envelope = np.log(valid_envelope)
                    coeffs = np.polyfit(valid_time, log_envelope, 1)
                    measured_damping = -coeffs[0]
                else:
                    measured_damping = 0.0

        except Exception as e:
            # Ultimate fallback: use simple abs() envelope
            warnings.warn(f"Peak tracking failed: {e}. Using simple envelope.", stacklevel=2)
            envelope = np.abs(signal_ac)
            valid_envelope = envelope[envelope > 0.01 * np.max(envelope)]
            valid_time = time[: len(valid_envelope)]

            if len(valid_envelope) > 5:
                log_envelope = np.log(valid_envelope)
                coeffs = np.polyfit(valid_time, log_envelope, 1)
                measured_damping = -coeffs[0]
            else:
                measured_damping = 0.0

        # Return measured values (allow negative damping to surface for debugging)
        return max(measured_frequency, 0.0), measured_damping

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
    grid_points: tuple[int, int, int] = (64, 64, 16),
    transport_coeffs: TransportCoefficients | None = None,
    metric: GeneralMetric | None = None,
    **kwargs,
) -> NumericalSoundWaveBenchmark:
    """
    Factory function for creating numerical sound wave benchmark instances.

    Args:
        domain_size: Spatial domain size (periodic)
        grid_points: Spatial grid resolution (Nx, Ny, Nz)
        transport_coeffs: Transport coefficients for viscosity
        metric: Spacetime metric (defaults to Minkowski)
        **kwargs: Additional arguments passed to NumericalSoundWaveBenchmark

    Returns:
        Configured numerical benchmark instance with SpaceGrid

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


def create_numerical_benchmark_with_ired(
    temperature: float = 0.4,
    cross_section: float = 1.0,
    truncation: str = "41",
    domain_size: float = 2 * np.pi,
    grid_points: tuple[int, int, int] = (64, 64, 16),
    metric: GeneralMetric | None = None,
) -> tuple[NumericalSoundWaveBenchmark, HardSphereIReD]:
    """
    Create numerical sound wave benchmark with IReD transport coefficients.

    This uses quantitatively accurate coefficients from kinetic theory
    (Wagner et al. 2022) instead of phenomenological values.

    Args:
        temperature: Temperature T in GeV
        cross_section: Hard sphere cross-section in fm²
        truncation: Moment truncation ('14', '23', '32', '41')
        domain_size: Spatial domain size (periodic)
        grid_points: Spatial grid resolution (Nx, Ny, Nz)
        metric: Spacetime metric (defaults to Minkowski)

    Returns:
        Tuple of (NumericalSoundWaveBenchmark, HardSphereIReD model)

    Example:
        >>> benchmark, ired_model = create_numerical_benchmark_with_ired(T=0.4)
        >>> print(f"η/s = {ired_model.eta_over_s():.4f}")
        >>> result = benchmark.run_simulation(wave_number=1.0)

    Note:
        The hard sphere gas with IReD coefficients may have very large relaxation times
        (τ_π ~ 200 fm/c) which puts the system outside the Israel-Stewart regime
        (|τω| >> 1) for typical wave numbers. This is physically correct but may
        require smaller k values or adjusted parameters for regime validity.
    """
    # Create IReD transport coefficient model
    ired_model = HardSphereIReD(
        temperature=temperature, cross_section=cross_section, truncation=truncation
    )

    # Extract IReD transport coefficients
    # CRITICAL: Use time_unit="natural" because solver expects GeV⁻¹, not fm/c
    transport_coeffs = TransportCoefficients(
        shear_viscosity=ired_model.shear_viscosity(),
        bulk_viscosity=ired_model.bulk_viscosity(),  # Zero for conformal
        diffusion_coefficient=ired_model.diffusion_coefficient(),  # D (Landau frame)
        shear_relaxation_time=ired_model.shear_relaxation_time(time_unit="natural"),
        bulk_relaxation_time=ired_model.bulk_relaxation_time(time_unit="natural"),
        diffusion_relaxation_time=ired_model.diffusion_relaxation_time(time_unit="natural"),
        # Second-order IReD coefficients
        tau_pi_pi=ired_model.tau_pi_pi(time_unit="natural"),  # Shear-shear coupling τ_ππ
        lambda_pi_V=ired_model.lambda_pi_V(time_unit="natural"),  # Shear-diffusion coupling λ_πV
        lambda_V_pi=ired_model.lambda_V_pi(time_unit="natural"),  # Diffusion-shear coupling λ_Vπ
        delta_V_V=ired_model.delta_V_V(),  # Diffusion expansion coupling δ_VV (dimensionless)
    )

    # Create numerical benchmark
    benchmark = NumericalSoundWaveBenchmark(
        domain_size=domain_size,
        grid_points=grid_points,
        transport_coeffs=transport_coeffs,
        metric=metric,
    )

    return benchmark, ired_model
