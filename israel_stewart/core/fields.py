"""
Field variables and state vectors for relativistic hydrodynamics.

This module defines the fundamental field variables used in Israel-Stewart
hydrodynamics, including thermodynamic state variables and fluid flow fields.
"""


# Forward reference for metric
from typing import TYPE_CHECKING, Optional

import numpy as np

from .constants import (
    C_LIGHT,
    ENERGY_DENSITY_MIN,
    validate_temperature,
    validate_transport_coefficient,
)
from .tensors import FourVector, StressEnergyTensor, TensorField, ViscousStressTensor

if TYPE_CHECKING:
    from .metrics import MetricBase


class FieldValidationError(Exception):
    """Exception for field validation errors."""
    pass


class ThermodynamicState:
    """
    Thermodynamic state variables for relativistic fluid.

    Encapsulates energy density, pressure, temperature, and other
    thermodynamic quantities with consistency checks.
    """

    def __init__(self,
                 energy_density: float,
                 pressure: float,
                 temperature: float | None = None,
                 particle_density: float | None = None,
                 entropy_density: float | None = None):
        """
        Initialize thermodynamic state.

        Args:
            energy_density: Energy density � (� in some notations)
            pressure: Pressure p
            temperature: Temperature T (optional)
            particle_density: Particle number density n (optional)
            entropy_density: Entropy density s (optional)
        """
        self.energy_density = self._validate_energy_density(energy_density)
        self.pressure = self._validate_pressure(pressure)
        self.temperature = temperature
        self.particle_density = particle_density
        self.entropy_density = entropy_density

        if temperature is not None:
            validate_temperature(temperature)

        # Check thermodynamic consistency
        self._validate_thermodynamic_consistency()

    def _validate_energy_density(self, rho: float) -> float:
        """Validate energy density."""
        if rho < ENERGY_DENSITY_MIN:
            raise FieldValidationError(f"Energy density {rho} below minimum {ENERGY_DENSITY_MIN}")
        if not np.isfinite(rho):
            raise FieldValidationError(f"Energy density must be finite, got {rho}")
        return rho

    def _validate_pressure(self, p: float) -> float:
        """Validate pressure."""
        if not np.isfinite(p):
            raise FieldValidationError(f"Pressure must be finite, got {p}")
        # Pressure can be negative for exotic matter, but warn
        if p < 0:
            import warnings
            warnings.warn(f"Negative pressure {p} indicates exotic matter", stacklevel=2)
        return p

    def _validate_thermodynamic_consistency(self) -> None:
        """Check basic thermodynamic consistency conditions."""
        # Speed of sound should be subluminal: cs� = dp/d� d 1
        if hasattr(self, 'sound_speed_squared'):
            if self.sound_speed_squared > C_LIGHT**2:
                raise FieldValidationError(f"Sound speed squared {self.sound_speed_squared} exceeds c�")

    @property
    def enthalpy_density(self) -> float:
        """Enthalpy density w = � + p."""
        return self.energy_density + self.pressure

    @property
    def sound_speed_squared(self) -> float:
        """
        Speed of sound squared cs� = dp/d�.

        For ideal gas: cs� = �p/� where � is adiabatic index.
        Placeholder implementation returns conformal value.
        """
        # Conformal fluid: cs� = 1/3
        return 1.0 / 3.0

    def equation_of_state(self, eos_type: str = "ideal") -> dict[str, float]:
        """
        Apply equation of state to relate thermodynamic quantities.

        Args:
            eos_type: Type of EOS ("ideal", "bag_model", "quark_gluon")

        Returns:
            Dictionary of derived quantities
        """
        if eos_type == "ideal":
            # Ideal gas: p = �/3 (radiation-dominated)
            if abs(self.pressure - self.energy_density / 3.0) > 1e-10:
                import warnings
                warnings.warn("Pressure inconsistent with ideal gas EOS", stacklevel=2)

            return {
                'adiabatic_index': 4.0/3.0,
                'sound_speed_squared': 1.0/3.0,
                'trace_anomaly': 0.0
            }

        elif eos_type == "bag_model":
            # MIT bag model: p = �/3 - B
            bag_constant = 0.2  # Placeholder value
            return {
                'bag_constant': bag_constant,
                'sound_speed_squared': 1.0/3.0,
                'trace_anomaly': -4.0 * bag_constant
            }

        else:
            raise ValueError(f"Unknown equation of state: {eos_type}")

    def __str__(self) -> str:
        return f"ThermodynamicState(�={self.energy_density:.3e}, p={self.pressure:.3e})"

    def __repr__(self) -> str:
        return (f"ThermodynamicState(energy_density={self.energy_density}, "
                f"pressure={self.pressure}, temperature={self.temperature})")


class FluidVelocityField:
    """
    Fluid velocity field for relativistic hydrodynamics.

    Manages four-velocity and three-velocity representations with
    proper normalization and relativistic transformations.
    """

    def __init__(self,
                 four_velocity: FourVector | None = None,
                 three_velocity: np.ndarray | None = None,
                 metric: Optional['MetricBase'] = None):
        """
        Initialize velocity field.

        Args:
            four_velocity: Four-velocity u^� (if provided)
            three_velocity: Three-velocity v^i (alternative input)
            metric: Spacetime metric
        """
        self.metric = metric

        if four_velocity is not None:
            self.four_velocity = four_velocity
            self._validate_four_velocity()
        elif three_velocity is not None:
            self.four_velocity = self._construct_four_velocity_from_three(three_velocity)
        else:
            # Default to rest frame
            self.four_velocity = FourVector([1.0, 0.0, 0.0, 0.0], False, metric)

    def _validate_four_velocity(self) -> None:
        """Validate four-velocity normalization."""
        if self.metric is None:
            return  # Cannot validate without metric

        norm_sq = self.four_velocity.magnitude_squared()
        signature = getattr(self.metric, 'signature', (-1, 1, 1, 1))

        expected_norm = -1.0 if signature[0] < 0 else 1.0
        if abs(norm_sq - expected_norm) > 1e-10:
            raise FieldValidationError(f"Four-velocity not normalized: u�u = {norm_sq}, expected {expected_norm}")

    def _construct_four_velocity_from_three(self, three_velocity: np.ndarray) -> FourVector:
        """Construct four-velocity from three-velocity."""
        from .constants import compute_lorentz_factor, validate_relativistic_velocity

        # Validate subluminal velocity
        validate_relativistic_velocity(three_velocity)

        # Compute Lorentz factor
        gamma = compute_lorentz_factor(three_velocity)

        # Four-velocity: u^� = �(1, v�)
        four_components = np.zeros(4)
        four_components[0] = gamma
        four_components[1:4] = gamma * three_velocity

        return FourVector(four_components, False, self.metric)

    @property
    def three_velocity(self) -> np.ndarray:
        """Extract three-velocity from four-velocity."""
        gamma = self.four_velocity.time_component
        if abs(gamma) < 1e-15:
            raise FieldValidationError("Cannot extract three-velocity from null four-velocity")

        return self.four_velocity.spatial_components / gamma

    @property
    def lorentz_factor(self) -> float:
        """Lorentz factor � = u^0."""
        return abs(self.four_velocity.time_component)

    def boost_to_rest_frame(self) -> 'FluidVelocityField':
        """Return velocity field in rest frame (zero three-velocity)."""
        rest_four_velocity = FourVector([1.0, 0.0, 0.0, 0.0], False, self.metric)
        return FluidVelocityField(rest_four_velocity, metric=self.metric)

    def is_at_rest(self, tolerance: float = 1e-10) -> bool:
        """Check if fluid is at rest."""
        three_vel_squared = np.dot(self.three_velocity, self.three_velocity)
        return three_vel_squared < tolerance**2

    def __str__(self) -> str:
        v = self.three_velocity
        return f"FluidVelocityField(�={self.lorentz_factor:.3f}, v=[{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}])"


class TransportCoefficients:
    """
    Transport coefficients for Israel-Stewart hydrodynamics.

    Manages viscosity coefficients, thermal conductivity, and relaxation times
    with physical constraints and temperature dependence.
    """

    def __init__(self,
                 shear_viscosity: float,
                 bulk_viscosity: float = 0.0,
                 thermal_conductivity: float = 0.0,
                 shear_relaxation_time: float | None = None,
                 bulk_relaxation_time: float | None = None,
                 heat_relaxation_time: float | None = None,
                 # Second-order coupling coefficients
                 lambda_pi_pi: float = 0.0,
                 lambda_pi_Pi: float = 0.0,
                 lambda_pi_q: float = 0.0,
                 lambda_Pi_pi: float = 0.0,
                 lambda_q_pi: float = 0.0,
                 xi_1: float = 0.0,
                 xi_2: float = 0.0,
                 # Nonlinear relaxation parameters
                 tau_pi_pi: float = 0.0,
                 tau_pi_omega: float = 0.0,
                 tau_Pi_pi: float = 0.0,
                 tau_q_pi: float = 0.0):
        """
        Initialize transport coefficients with Israel-Stewart second-order terms.

        Args:
            shear_viscosity: Shear viscosity η
            bulk_viscosity: Bulk viscosity ζ
            thermal_conductivity: Thermal conductivity κ
            shear_relaxation_time: Shear viscosity relaxation time τ_π
            bulk_relaxation_time: Bulk viscosity relaxation time τ_Π
            heat_relaxation_time: Heat conduction relaxation time τ_q
            lambda_pi_pi: Shear-shear coupling coefficient λ_ππ
            lambda_pi_Pi: Shear-bulk coupling coefficient λ_πΠ
            lambda_pi_q: Shear-heat coupling coefficient λ_πq
            lambda_Pi_pi: Bulk-shear coupling coefficient λ_Ππ
            lambda_q_pi: Heat-shear coupling coefficient λ_qπ
            xi_1: Bulk nonlinearity coefficient ξ₁
            xi_2: Bulk nonlinearity coefficient ξ₂
            tau_pi_pi: Shear-shear relaxation coupling τ_ππ
            tau_pi_omega: Shear-vorticity coupling τ_πω
            tau_Pi_pi: Bulk-shear relaxation coupling τ_Ππ
            tau_q_pi: Heat-shear relaxation coupling τ_qπ
        """
        # First-order transport coefficients
        self.shear_viscosity = self._validate_coefficient(shear_viscosity, "shear_viscosity")
        self.bulk_viscosity = self._validate_coefficient(bulk_viscosity, "bulk_viscosity")
        self.thermal_conductivity = self._validate_coefficient(thermal_conductivity, "thermal_conductivity")

        # Relaxation times
        self.shear_relaxation_time = shear_relaxation_time
        self.bulk_relaxation_time = bulk_relaxation_time
        self.heat_relaxation_time = heat_relaxation_time

        if shear_relaxation_time is not None:
            self._validate_coefficient(shear_relaxation_time, "shear_relaxation_time")
        if bulk_relaxation_time is not None:
            self._validate_coefficient(bulk_relaxation_time, "bulk_relaxation_time")
        if heat_relaxation_time is not None:
            self._validate_coefficient(heat_relaxation_time, "heat_relaxation_time")

        # Second-order coupling coefficients
        self.lambda_pi_pi = lambda_pi_pi
        self.lambda_pi_Pi = lambda_pi_Pi
        self.lambda_pi_q = lambda_pi_q
        self.lambda_Pi_pi = lambda_Pi_pi
        self.lambda_q_pi = lambda_q_pi
        self.xi_1 = xi_1
        self.xi_2 = xi_2

        # Nonlinear relaxation parameters
        self.tau_pi_pi = tau_pi_pi
        self.tau_pi_omega = tau_pi_omega
        self.tau_Pi_pi = tau_Pi_pi
        self.tau_q_pi = tau_q_pi

        # Validate thermodynamic stability
        self._validate_stability_constraints()

    def _validate_coefficient(self, coeff: float, name: str) -> float:
        """Validate transport coefficient."""
        validate_transport_coefficient(coeff, name)
        return coeff

    def _validate_stability_constraints(self) -> None:
        """
        Validate Israel-Stewart stability constraints.

        Ensures transport coefficients satisfy thermodynamic stability requirements
        for second-order viscous hydrodynamics.
        """
        # Basic positivity constraints
        if self.shear_viscosity < 0:
            raise ValueError("Shear viscosity must be non-negative")
        if self.bulk_viscosity < 0:
            raise ValueError("Bulk viscosity must be non-negative")
        if self.thermal_conductivity < 0:
            raise ValueError("Thermal conductivity must be non-negative")

        # Relaxation time positivity (when specified)
        relaxation_times = [
            (self.shear_relaxation_time, "shear_relaxation_time"),
            (self.bulk_relaxation_time, "bulk_relaxation_time"),
            (self.heat_relaxation_time, "heat_relaxation_time")
        ]

        for time_val, name in relaxation_times:
            if time_val is not None and time_val <= 0:
                raise ValueError(f"{name} must be positive when specified")

        # Second-order coupling stability (approximate constraints)
        # These ensure the relaxation equations don't develop instabilities
        coupling_magnitudes = [
            abs(self.lambda_pi_pi), abs(self.lambda_pi_Pi), abs(self.lambda_pi_q),
            abs(self.lambda_Pi_pi), abs(self.lambda_q_pi), abs(self.xi_1), abs(self.xi_2)
        ]

        # Heuristic stability bound: coupling coefficients shouldn't be too large
        max_coupling = 10.0  # Typical bound from kinetic theory
        for coupling in coupling_magnitudes:
            if coupling > max_coupling:
                import warnings
                warnings.warn(
                    f"Large coupling coefficient {coupling:.2f} may cause instability",
                    stacklevel=3
                )

    @property
    def viscosity_ratio(self) -> float:
        """Bulk to shear viscosity ratio �/�."""
        if self.shear_viscosity == 0:
            return float('inf') if self.bulk_viscosity > 0 else 0.0
        return self.bulk_viscosity / self.shear_viscosity

    def estimate_relaxation_times(self, thermodynamic_state: ThermodynamicState) -> None:
        """
        Estimate relaxation times from thermodynamic state.

        Uses kinetic theory estimates: � ~ �/(� + p)
        """
        enthalpy = thermodynamic_state.enthalpy_density

        if enthalpy > 0:
            if self.shear_relaxation_time is None:
                self.shear_relaxation_time = self.shear_viscosity / enthalpy

            if self.bulk_relaxation_time is None and self.bulk_viscosity > 0:
                self.bulk_relaxation_time = self.bulk_viscosity / enthalpy

            if self.heat_relaxation_time is None and self.thermal_conductivity > 0:
                # Rough estimate for heat conduction relaxation
                self.heat_relaxation_time = self.thermal_conductivity / enthalpy

    def temperature_dependence(self, temperature: float, model: str = "constant") -> 'TransportCoefficients':
        """
        Apply temperature dependence to transport coefficients.

        Args:
            temperature: Current temperature
            model: Temperature dependence model

        Returns:
            New TransportCoefficients with temperature scaling
        """
        validate_temperature(temperature)

        if model == "constant":
            return self

        elif model == "kinetic_theory":
            # Kinetic theory: �  T^(1/2) for massive particles
            temp_factor = np.sqrt(temperature)

            return TransportCoefficients(
                shear_viscosity=self.shear_viscosity * temp_factor,
                bulk_viscosity=self.bulk_viscosity * temp_factor,
                thermal_conductivity=self.thermal_conductivity * temp_factor,
                shear_relaxation_time=self.shear_relaxation_time,
                bulk_relaxation_time=self.bulk_relaxation_time,
                heat_relaxation_time=self.heat_relaxation_time,
                # Second-order coefficients (typically temperature independent)
                lambda_pi_pi=self.lambda_pi_pi,
                lambda_pi_Pi=self.lambda_pi_Pi,
                lambda_pi_q=self.lambda_pi_q,
                lambda_Pi_pi=self.lambda_Pi_pi,
                lambda_q_pi=self.lambda_q_pi,
                xi_1=self.xi_1,
                xi_2=self.xi_2,
                tau_pi_pi=self.tau_pi_pi,
                tau_pi_omega=self.tau_pi_omega,
                tau_Pi_pi=self.tau_Pi_pi,
                tau_q_pi=self.tau_q_pi
            )

        else:
            raise ValueError(f"Unknown temperature dependence model: {model}")

    def __str__(self) -> str:
        return f"TransportCoefficients(�={self.shear_viscosity:.3e}, �={self.bulk_viscosity:.3e})"


class HydrodynamicState:
    """
    Complete hydrodynamic state for Israel-Stewart theory.

    Combines thermodynamic state, velocity field, and transport coefficients
    with consistency checks and evolution methods.
    """

    def __init__(self,
                 thermodynamic_state: ThermodynamicState,
                 velocity_field: FluidVelocityField,
                 transport_coefficients: TransportCoefficients,
                 viscous_stress: ViscousStressTensor | None = None):
        """
        Initialize complete hydrodynamic state.

        Args:
            thermodynamic_state: Thermodynamic variables
            velocity_field: Fluid velocity field
            transport_coefficients: Transport coefficients
            viscous_stress: Current viscous stress tensor (optional)
        """
        self.thermodynamic = thermodynamic_state
        self.velocity = velocity_field
        self.transport = transport_coefficients
        self.viscous_stress = viscous_stress

        # Validate consistency
        self._validate_state_consistency()

    def _validate_state_consistency(self) -> None:
        """Validate consistency between state components."""
        # Check that velocity field and thermodynamic state are compatible
        if self.velocity.metric != self.velocity.four_velocity.metric:
            import warnings
            warnings.warn("Velocity field and four-velocity have different metrics", stacklevel=2)

        # Estimate relaxation times if not provided
        self.transport.estimate_relaxation_times(self.thermodynamic)

    def stress_energy_tensor(self, metric: 'MetricBase') -> StressEnergyTensor:
        """
        Construct total stress-energy tensor T^��.

        T^�� = (� + p) u^� u^� + p g^�� + �^��

        Args:
            metric: Spacetime metric

        Returns:
            Complete stress-energy tensor
        """
        # Perfect fluid part
        rho = self.thermodynamic.energy_density
        p = self.thermodynamic.pressure
        u = self.velocity.four_velocity

        # u^� u^� term
        u_outer = np.outer(u.components, u.components)

        # Perfect fluid: T^��_perfect = (� + p) u^� u^� + p g^��
        perfect_fluid_components = (rho + p) * u_outer + p * metric.inverse
        perfect_fluid = StressEnergyTensor(perfect_fluid_components, metric)

        # Add viscous corrections if present
        if self.viscous_stress is not None:
            return perfect_fluid.add_viscous_corrections(self.viscous_stress)
        else:
            return perfect_fluid

    def energy_momentum_source(self) -> TensorField:
        """
        Compute energy-momentum conservation source terms.

        _� T^�� = 0 (conservation)

        Returns:
            Source tensor for energy-momentum equations
        """
        # For ideal fluid in curved spacetime, source comes from geometry
        # This would be implemented with specific metric and coordinates

        # Placeholder: return zero source (flat spacetime)
        zero_source = np.zeros((4,))
        return TensorField(zero_source, "nu", self.velocity.metric)

    def relaxation_time_hierarchy(self) -> dict[str, float]:
        """
        Check relaxation time hierarchy for Israel-Stewart stability.

        Stability requires proper ordering of relaxation times.

        Returns:
            Dictionary of relaxation time ratios
        """
        tau_pi = self.transport.shear_relaxation_time or 1.0
        tau_Pi = self.transport.bulk_relaxation_time or 1.0
        tau_q = self.transport.heat_relaxation_time or 1.0

        return {
            'tau_pi': tau_pi,
            'tau_Pi': tau_Pi,
            'tau_q': tau_q,
            'tau_Pi_over_tau_pi': tau_Pi / tau_pi if tau_pi > 0 else float('inf'),
            'tau_q_over_tau_pi': tau_q / tau_pi if tau_pi > 0 else float('inf')
        }

    def __str__(self) -> str:
        return (f"HydrodynamicState(\n"
                f"  {self.thermodynamic}\n"
                f"  {self.velocity}\n"
                f"  {self.transport}\n"
                f")")

    def __repr__(self) -> str:
        return f"HydrodynamicState(thermodynamic={self.thermodynamic!r}, velocity={self.velocity!r})"


class ISFieldConfiguration:
    """
    Complete field configuration for Israel-Stewart hydrodynamics.

    Manages all primary hydrodynamic variables, dissipative fluxes, and
    auxiliary fields for MSRJD stochastic analysis on spacetime grids.
    """

    def __init__(self, grid: 'SpacetimeGrid'):
        """
        Initialize field configuration on spacetime grid.

        Args:
            grid: SpacetimeGrid defining coordinate system and discretization
        """
        from .spacetime_grid import SpacetimeGrid

        if not isinstance(grid, SpacetimeGrid):
            raise TypeError("grid must be a SpacetimeGrid instance")

        self.grid = grid

        # Primary hydrodynamic variables
        self.rho = np.zeros(grid.shape)          # Energy density ρ
        self.n = np.zeros(grid.shape)            # Particle density n
        self.u_mu = np.zeros((*grid.shape, 4))   # Four-velocity u^μ

        # Initialize four-velocity to rest frame
        self.u_mu[..., 0] = 1.0  # u^0 = 1 (rest frame)

        # Dissipative fluxes (Israel-Stewart variables)
        self.Pi = np.zeros(grid.shape)           # Bulk pressure Π
        self.pi_munu = np.zeros((*grid.shape, 4, 4))  # Shear tensor π^μν
        self.q_mu = np.zeros((*grid.shape, 4))   # Heat flux q^μ

        # Auxiliary fields for MSRJD formalism
        self.rho_tilde = np.zeros(grid.shape, dtype=complex)      # Energy density noise
        self.u_mu_tilde = np.zeros((*grid.shape, 4), dtype=complex)  # Velocity noise

        # Thermodynamic variables
        self.pressure = np.zeros(grid.shape)     # Pressure p
        self.temperature = np.zeros(grid.shape)  # Temperature T

        # Transport coefficients (can be spatially dependent)
        self.eta = np.ones(grid.shape)           # Shear viscosity η
        self.zeta = np.zeros(grid.shape)         # Bulk viscosity ζ
        self.kappa = np.zeros(grid.shape)        # Thermal conductivity κ

        # Cache for derived quantities
        self._energy_momentum_tensor = None
        self._total_stress_tensor = None

        # Validation flags
        self._constraints_enforced = False
        self._thermodynamic_consistent = False

    @property
    def total_field_count(self) -> int:
        """Total number of field variables."""
        return (
            2 * np.prod(self.grid.shape) +      # ρ, n
            4 * np.prod(self.grid.shape) +      # u^μ
            1 * np.prod(self.grid.shape) +      # Π
            16 * np.prod(self.grid.shape) +     # π^μν
            4 * np.prod(self.grid.shape)       # q^μ
        )

    def to_state_vector(self) -> np.ndarray:
        """
        Pack all fields into single state vector for evolution.

        Returns:
            Flattened state vector containing all field variables
        """
        return np.concatenate([
            self.rho.flatten(),
            self.n.flatten(),
            self.u_mu.reshape(-1),  # Flatten all components
            self.Pi.flatten(),
            self.pi_munu.reshape(-1),  # Flatten all tensor components
            self.q_mu.reshape(-1)   # Flatten all vector components
        ])

    def from_state_vector(self, state: np.ndarray) -> None:
        """
        Unpack state vector into field components.

        Args:
            state: Flattened state vector from evolution
        """
        expected_size = self.total_field_count
        if len(state) != expected_size:
            raise ValueError(f"State vector size {len(state)} doesn't match expected {expected_size}")

        grid_size = np.prod(self.grid.shape)
        offset = 0

        # Unpack energy density
        self.rho = state[offset:offset + grid_size].reshape(self.grid.shape)
        offset += grid_size

        # Unpack particle density
        self.n = state[offset:offset + grid_size].reshape(self.grid.shape)
        offset += grid_size

        # Unpack four-velocity
        u_size = 4 * grid_size
        self.u_mu = state[offset:offset + u_size].reshape((*self.grid.shape, 4))
        offset += u_size

        # Unpack bulk pressure
        self.Pi = state[offset:offset + grid_size].reshape(self.grid.shape)
        offset += grid_size

        # Unpack shear tensor
        pi_size = 16 * grid_size
        self.pi_munu = state[offset:offset + pi_size].reshape((*self.grid.shape, 4, 4))
        offset += pi_size

        # Unpack heat flux
        q_size = 4 * grid_size
        self.q_mu = state[offset:offset + q_size].reshape((*self.grid.shape, 4))

        # Reset validation flags
        self._constraints_enforced = False
        self._thermodynamic_consistent = False

    def to_dissipative_vector(self) -> np.ndarray:
        """
        Pack dissipative fluxes into single vector for relaxation evolution.

        Returns:
            Flattened vector containing [Π, π^μν, q^μ] components
        """
        return np.concatenate([
            self.Pi.flatten(),                    # Bulk pressure
            self.pi_munu.reshape(-1),            # Shear tensor (all components)
            self.q_mu.reshape(-1)                # Heat flux vector
        ])

    def from_dissipative_vector(self, dissipative_state: np.ndarray) -> None:
        """
        Unpack dissipative vector back to field components.

        Args:
            dissipative_state: Flattened dissipative flux vector
        """
        grid_size = np.prod(self.grid.shape)

        # Expected sizes for each field
        pi_size = grid_size
        pi_munu_size = 16 * grid_size  # 4×4 tensor components
        q_size = 4 * grid_size         # 4-vector components

        expected_size = pi_size + pi_munu_size + q_size
        if len(dissipative_state) != expected_size:
            raise ValueError(
                f"Dissipative vector size {len(dissipative_state)} doesn't match "
                f"expected {expected_size}"
            )

        offset = 0

        # Unpack bulk pressure Π
        self.Pi = dissipative_state[offset:offset + pi_size].reshape(self.grid.shape)
        offset += pi_size

        # Unpack shear tensor π^μν
        self.pi_munu = dissipative_state[offset:offset + pi_munu_size].reshape(
            (*self.grid.shape, 4, 4)
        )
        offset += pi_munu_size

        # Unpack heat flux q^μ
        self.q_mu = dissipative_state[offset:offset + q_size].reshape(
            (*self.grid.shape, 4)
        )

    @property
    def dissipative_field_count(self) -> int:
        """Total number of dissipative field variables."""
        grid_size = np.prod(self.grid.shape)
        return (
            1 * grid_size +      # Π
            16 * grid_size +     # π^μν
            4 * grid_size        # q^μ
        )

    def apply_constraints(self) -> None:
        """
        Enforce physical constraints on field variables.

        Constraints enforced:
        - u^μ u_μ = -c² (four-velocity normalization)
        - π^μν u_μ = 0 (shear tensor orthogonality)
        - π^μ_μ = 0 (shear tensor traceless)
        - q^μ u_μ = 0 (heat flux orthogonality)
        - Thermodynamic positivity conditions
        """
        # Import tensor utilities

        # 1. Normalize four-velocity
        self._normalize_four_velocity()

        # 2. Project shear tensor to be orthogonal and traceless
        self._project_shear_tensor()

        # 3. Project heat flux to be orthogonal to u^μ
        self._project_heat_flux()

        # 4. Apply thermodynamic constraints
        self._enforce_thermodynamic_constraints()

        self._constraints_enforced = True

    def _normalize_four_velocity(self) -> None:
        """Normalize four-velocity to satisfy u^μ u_μ = -c²."""
        from .tensor_utils import optimized_einsum

        if self.grid.metric is None:
            # Minkowski metric normalization
            u_squared = (-self.u_mu[..., 0]**2 +
                        np.sum(self.u_mu[..., 1:4]**2, axis=-1))
        else:
            # General metric normalization
            u_squared = optimized_einsum('...i,...j,ij->...',
                                       self.u_mu, self.u_mu,
                                       self.grid.metric.components)

        # Handle signature conventions
        signature = getattr(self.grid.metric, 'signature', (-1, 1, 1, 1))
        expected_norm = -1.0 if signature[0] < 0 else 1.0

        # Normalize to correct value
        normalization_factor = np.sqrt(np.abs(u_squared / expected_norm))
        normalization_factor = np.where(normalization_factor > 1e-15,
                                      normalization_factor, 1.0)

        self.u_mu /= normalization_factor[..., np.newaxis]

    def _project_shear_tensor(self) -> None:
        """Project shear tensor to be orthogonal to u^μ and traceless."""
        from .derivatives import ProjectionOperator
        from .four_vectors import FourVector
        from .tensor_utils import optimized_einsum

        # For each grid point, apply projection
        for indices in np.ndindex(self.grid.shape):
            u_at_point = FourVector(self.u_mu[indices], False, self.grid.metric)
            projector = ProjectionOperator(u_at_point, self.grid.metric)

            # Get perpendicular projector Δ^μν
            delta = projector.perpendicular_projector()
            pi_at_point = self.pi_munu[indices]

            # Project: π^μν = Δ^μα Δ^νβ π_αβ - (1/3) Δ^μν Δ_γδ π^γδ
            pi_projected = optimized_einsum('ma,nb,ab->mn',
                                           delta.components, delta.components, pi_at_point)

            # Remove trace
            trace = optimized_einsum('mn,mn->', delta.components, pi_at_point)
            pi_traceless = pi_projected - (1.0/3.0) * trace * delta.components

            self.pi_munu[indices] = pi_traceless

    def _project_heat_flux(self) -> None:
        """Project heat flux to be orthogonal to u^μ."""
        from .derivatives import ProjectionOperator
        from .four_vectors import FourVector

        # For each grid point, apply projection
        for indices in np.ndindex(self.grid.shape):
            u_at_point = FourVector(self.u_mu[indices], False, self.grid.metric)
            projector = ProjectionOperator(u_at_point, self.grid.metric)

            q_at_point = FourVector(self.q_mu[indices], False, self.grid.metric)
            q_projected = projector.project_vector_perpendicular(q_at_point)

            self.q_mu[indices] = q_projected.components

    def _enforce_thermodynamic_constraints(self) -> None:
        """Enforce thermodynamic positivity and consistency constraints."""
        # Energy density must be positive
        self.rho = np.maximum(self.rho, 1e-15)

        # Particle density must be non-negative
        self.n = np.maximum(self.n, 0.0)

        # Pressure positivity (can be relaxed for exotic matter)
        self.pressure = np.maximum(self.pressure, -0.1 * self.rho)  # Allow some negativity

        # Temperature must be positive
        if hasattr(self, 'temperature'):
            self.temperature = np.maximum(self.temperature, 1e-10)

        self._thermodynamic_consistent = True

    def compute_stress_energy_tensor(self) -> np.ndarray:
        """
        Compute total stress-energy tensor T^μν = T^μν_perfect + π^μν.

        Returns:
            Stress-energy tensor at each grid point with shape (*grid.shape, 4, 4)
        """
        if not self._constraints_enforced:
            warnings.warn("Computing stress-energy tensor without enforcing constraints")

        T_total = np.zeros((*self.grid.shape, 4, 4))

        # Perfect fluid part: T^μν_perfect = (ρ + p) u^μ u^ν + p g^μν
        enthalpy_density = self.rho + self.pressure

        for indices in np.ndindex(self.grid.shape):
            rho_h = enthalpy_density[indices]
            p = self.pressure[indices]
            u = self.u_mu[indices]

            # u^μ u^ν outer product
            u_outer = np.outer(u, u)

            # Metric tensor
            if self.grid.metric is None:
                g_inv = np.diag([-1, 1, 1, 1])  # Minkowski
            else:
                g_inv = self.grid.metric.inverse

            # Perfect fluid + viscous corrections
            T_total[indices] = rho_h * u_outer + p * g_inv + self.pi_munu[indices]

        self._total_stress_tensor = T_total
        return T_total

    def compute_conserved_charges(self) -> dict[str, np.ndarray]:
        """
        Compute conserved charges (energy, momentum, particle number).

        Returns:
            Dictionary with conserved charge densities
        """
        # Compute volume element
        volume_element = self.grid.volume_element()

        charges = {}

        # Energy density in lab frame: T^00
        T_munu = self.compute_stress_energy_tensor()
        charges['energy_density'] = T_munu[..., 0, 0]

        # Momentum density: T^0i
        charges['momentum_density'] = T_munu[..., 0, 1:4]

        # Particle number current: n^μ = n u^μ
        charges['particle_current'] = self.n[..., np.newaxis] * self.u_mu

        # Total conserved charges (integrated over spatial volume)
        if self.grid.ndim == 4:
            spatial_volume = np.prod([self.grid.spatial_spacing])
            charges['total_energy'] = np.sum(charges['energy_density'] * volume_element) * spatial_volume
            charges['total_momentum'] = np.sum(charges['momentum_density'] * volume_element[..., np.newaxis], axis=(1,2,3)) * spatial_volume
            charges['total_particle_number'] = np.sum(charges['particle_current'][..., 0] * volume_element) * spatial_volume

        return charges

    def validate_field_configuration(self) -> dict[str, bool]:
        """
        Validate physical consistency of field configuration.

        Returns:
            Dictionary of validation results
        """
        from .tensor_utils import optimized_einsum

        validation = {}

        # Check four-velocity normalization
        if self.grid.metric is None:
            u_norm_sq = (-self.u_mu[..., 0]**2 + np.sum(self.u_mu[..., 1:4]**2, axis=-1))
        else:
            u_norm_sq = optimized_einsum('...i,...j,ij->...',
                                       self.u_mu, self.u_mu,
                                       self.grid.metric.components)

        expected_norm = -1.0 if getattr(self.grid.metric, 'signature', (-1, 1, 1, 1))[0] < 0 else 1.0
        validation['four_velocity_normalized'] = np.allclose(u_norm_sq, expected_norm, rtol=1e-10)

        # Check shear tensor properties
        pi_trace = np.trace(self.pi_munu, axis1=-2, axis2=-1)
        validation['shear_tensor_traceless'] = np.allclose(pi_trace, 0.0, atol=1e-12)

        # Check orthogonality constraints
        pi_u_contraction = optimized_einsum('...ij,...i->...j', self.pi_munu, self.u_mu)
        q_u_contraction = optimized_einsum('...i,...i->...', self.q_mu, self.u_mu)

        validation['shear_orthogonal_to_velocity'] = np.allclose(pi_u_contraction, 0.0, atol=1e-12)
        validation['heat_flux_orthogonal_to_velocity'] = np.allclose(q_u_contraction, 0.0, atol=1e-12)

        # Check thermodynamic positivity
        validation['energy_density_positive'] = np.all(self.rho > 0)
        validation['particle_density_non_negative'] = np.all(self.n >= 0)

        validation['overall_valid'] = all(validation.values())

        return validation

    def copy(self) -> 'ISFieldConfiguration':
        """Create deep copy of field configuration."""
        new_config = ISFieldConfiguration(self.grid)

        # Copy all field arrays
        new_config.rho = self.rho.copy()
        new_config.n = self.n.copy()
        new_config.u_mu = self.u_mu.copy()
        new_config.Pi = self.Pi.copy()
        new_config.pi_munu = self.pi_munu.copy()
        new_config.q_mu = self.q_mu.copy()

        # Copy auxiliary fields
        new_config.rho_tilde = self.rho_tilde.copy()
        new_config.u_mu_tilde = self.u_mu_tilde.copy()

        # Copy thermodynamic variables
        new_config.pressure = self.pressure.copy()
        new_config.temperature = self.temperature.copy()
        new_config.eta = self.eta.copy()
        new_config.zeta = self.zeta.copy()
        new_config.kappa = self.kappa.copy()

        # Copy validation state
        new_config._constraints_enforced = self._constraints_enforced
        new_config._thermodynamic_consistent = self._thermodynamic_consistent

        return new_config

    def save_to_hdf5(self, filename: str) -> None:
        """Save field configuration to HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 save functionality")

        with h5py.File(filename, 'w') as f:
            # Save grid information
            grid_group = f.create_group('grid')
            grid_group.attrs['coordinate_system'] = self.grid.coordinate_system
            grid_group.attrs['time_range'] = self.grid.time_range
            grid_group.attrs['grid_points'] = self.grid.grid_points

            # Save field data
            fields_group = f.create_group('fields')
            fields_group.create_dataset('rho', data=self.rho)
            fields_group.create_dataset('n', data=self.n)
            fields_group.create_dataset('u_mu', data=self.u_mu)
            fields_group.create_dataset('Pi', data=self.Pi)
            fields_group.create_dataset('pi_munu', data=self.pi_munu)
            fields_group.create_dataset('q_mu', data=self.q_mu)

            # Save auxiliary fields (complex arrays)
            aux_group = f.create_group('auxiliary')
            aux_group.create_dataset('rho_tilde', data=self.rho_tilde)
            aux_group.create_dataset('u_mu_tilde', data=self.u_mu_tilde)

            # Save metadata
            f.attrs['constraints_enforced'] = self._constraints_enforced
            f.attrs['thermodynamic_consistent'] = self._thermodynamic_consistent

    @classmethod
    def load_from_hdf5(cls, filename: str, grid: 'SpacetimeGrid') -> 'ISFieldConfiguration':
        """Load field configuration from HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 load functionality")

        config = cls(grid)

        with h5py.File(filename, 'r') as f:
            # Load field data
            fields_group = f['fields']
            config.rho = fields_group['rho'][:]
            config.n = fields_group['n'][:]
            config.u_mu = fields_group['u_mu'][:]
            config.Pi = fields_group['Pi'][:]
            config.pi_munu = fields_group['pi_munu'][:]
            config.q_mu = fields_group['q_mu'][:]

            # Load auxiliary fields
            aux_group = f['auxiliary']
            config.rho_tilde = aux_group['rho_tilde'][:]
            config.u_mu_tilde = aux_group['u_mu_tilde'][:]

            # Load metadata
            config._constraints_enforced = f.attrs['constraints_enforced']
            config._thermodynamic_consistent = f.attrs['thermodynamic_consistent']

        return config

    def __str__(self) -> str:
        return (f"ISFieldConfiguration(grid={self.grid}, "
                f"total_fields={self.total_field_count}, "
                f"constraints_enforced={self._constraints_enforced})")

    def __repr__(self) -> str:
        return (f"ISFieldConfiguration(grid={self.grid!r}, "
                f"constraints_enforced={self._constraints_enforced}, "
                f"thermodynamic_consistent={self._thermodynamic_consistent})")
