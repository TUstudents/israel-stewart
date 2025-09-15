"""
Core module for Israel-Stewart relativistic hydrodynamics.

This module provides the fundamental building blocks for relativistic
hydrodynamics simulations including tensor algebra, spacetime metrics,
field variables, and physical constants.
"""

# Import order is important due to dependencies
import numpy as np

from .constants import (
    BOLTZMANN_K,
    # Physical constants
    C_LIGHT,
    # Numerical constants
    CONDITION_NUMBER_WARN,
    CONDUCTIVITY_MIN,
    COORDINATE_SYSTEMS,
    DEFAULT_UNITS,
    DETERMINANT_MIN,
    ENERGY_DENSITY_MIN,
    # Naming conventions
    FOUR_VECTOR_NAMES,
    GAMMA_MAX,
    HBAR,
    METRIC_SIGNATURES,
    NATURAL_UNITS,
    SI_CONSTANTS,
    TEMPERATURE_MIN,
    TENSOR_NAMES,
    # Tolerances
    TOLERANCE_DEFAULT,
    TOLERANCE_LOOSE,
    TOLERANCE_STRICT,
    # Physical limits
    VELOCITY_MAX,
    VISCOSITY_MIN,
    compute_lorentz_factor,
    # Utility functions
    get_physical_constant,
    validate_relativistic_velocity,
    validate_temperature,
    validate_transport_coefficient,
)
from .fields import (
    FieldValidationError,
    FluidVelocityField,
    HydrodynamicState,
    ISFieldConfiguration,
    ThermodynamicState,
    TransportCoefficients,
)
from .metrics import (
    BJorkenMetric,
    FLRWMetric,
    GeneralMetric,
    MetricBase,
    MilneMetric,
    MinkowskiMetric,
    SchwarzschildMetric,
)
from .spacetime_grid import (
    AdaptiveMeshRefinement,
    SpacetimeGrid,
    create_cartesian_grid,
    create_milne_grid,
)
from .tensors import (
    CoordinateTransformation,
    CovariantDerivative,
    FourVector,
    LorentzTransformation,
    PhysicsError,
    ProjectionOperator,
    RelativisticError,
    StressEnergyTensor,
    TensorField,
    TensorValidationError,
    ViscousStressTensor,
)

__all__ = [
    # Constants
    "C_LIGHT",
    "HBAR",
    "BOLTZMANN_K",
    "NATURAL_UNITS",
    "SI_CONSTANTS",
    "DEFAULT_UNITS",
    "TOLERANCE_DEFAULT",
    "TOLERANCE_STRICT",
    "TOLERANCE_LOOSE",
    "VELOCITY_MAX",
    "GAMMA_MAX",
    "FOUR_VECTOR_NAMES",
    "TENSOR_NAMES",
    "COORDINATE_SYSTEMS",
    "METRIC_SIGNATURES",
    "get_physical_constant",
    "validate_relativistic_velocity",
    "compute_lorentz_factor",
    "validate_temperature",
    "validate_transport_coefficient",
    # Metrics
    "MetricBase",
    "MinkowskiMetric",
    "GeneralMetric",
    # Tensors
    "TensorField",
    "FourVector",
    "StressEnergyTensor",
    "ViscousStressTensor",
    "CovariantDerivative",
    "ProjectionOperator",
    "LorentzTransformation",
    "CoordinateTransformation",
    "TensorValidationError",
    "PhysicsError",
    "RelativisticError",
    # Fields
    "ThermodynamicState",
    "FluidVelocityField",
    "TransportCoefficients",
    "HydrodynamicState",
    "FieldValidationError",
    "ISFieldConfiguration",
    # Grids
    "SpacetimeGrid",
    "AdaptiveMeshRefinement",
    "create_cartesian_grid",
    "create_milne_grid",
]

# Version information
__version__ = "0.1.0"
__author__ = "Relativistic Hydrodynamics Team"

# Default metric for convenience
default_metric = MinkowskiMetric()


def create_minkowski_metric(signature: str = "mostly_plus") -> MinkowskiMetric:
    """
    Create Minkowski metric with specified signature.

    Args:
        signature: "mostly_plus" (-,+,+,+) or "mostly_minus" (+,-,-,-)

    Returns:
        MinkowskiMetric instance
    """
    return MinkowskiMetric(signature=signature)


def create_four_velocity(three_velocity: list, metric: MetricBase | None = None) -> FourVector:
    """
    Create normalized four-velocity from three-velocity.

    Args:
        three_velocity: List of 3-velocity components [vx, vy, vz]
        metric: Spacetime metric (default: Minkowski)

    Returns:
        Normalized FourVector
    """
    if metric is None:
        metric = default_metric

    from .fields import FluidVelocityField

    velocity_field = FluidVelocityField(three_velocity=np.array(three_velocity), metric=metric)
    return velocity_field.four_velocity


def create_perfect_fluid_state(
    energy_density: float,
    pressure: float,
    three_velocity: list | None = None,
    metric: MetricBase | None = None,
) -> HydrodynamicState:
    """
    Create perfect fluid hydrodynamic state.

    Args:
        energy_density: Energy density ρ
        pressure: Pressure p
        three_velocity: 3-velocity [vx, vy, vz] (default: at rest)
        metric: Spacetime metric (default: Minkowski)

    Returns:
        HydrodynamicState for perfect fluid
    """
    if metric is None:
        metric = default_metric
    if three_velocity is None:
        three_velocity = [0.0, 0.0, 0.0]

    # Create components
    thermo_state = ThermodynamicState(energy_density, pressure)
    three_vel_array = np.array(three_velocity) if three_velocity is not None else None
    velocity_field = FluidVelocityField(three_velocity=three_vel_array, metric=metric)
    transport_coeffs = TransportCoefficients(shear_viscosity=0.0)  # Perfect fluid

    return HydrodynamicState(thermo_state, velocity_field, transport_coeffs)


# Common physics shortcuts
def lorentz_boost_matrix(velocity: list) -> "np.ndarray":
    """Create Lorentz boost matrix for given velocity."""
    from .tensors import LorentzTransformation

    return LorentzTransformation.boost_matrix(velocity)


def four_vector_from_components(
    components: list, covariant: bool = False, metric: MetricBase | None = None
) -> FourVector:
    """Create four-vector from component list."""
    if metric is None:
        metric = default_metric
    return FourVector(components, covariant, metric)
