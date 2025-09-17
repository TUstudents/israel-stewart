"""
Physical constants and unit systems for relativistic hydrodynamics.

This module provides fundamental constants used in relativistic physics
and Israel-Stewart hydrodynamics, with consistent unit systems.
"""

from typing import cast

import numpy as np

# Speed of light (set to 1 in natural units)
C_LIGHT = 1.0

# Planck’s constant (reduced, ħ = h/2π, set to 1 in natural units)
HBAR = 1.0

# Boltzmann constant (set to 1 in natural units)
BOLTZMANN_K = 1.0

# Natural units system (c = ħ = k_B = 1)
NATURAL_UNITS = {"c": C_LIGHT, "hbar": HBAR, "k_B": BOLTZMANN_K, "system": "natural"}

# SI units (for dimensional analysis and conversion)
SI_CONSTANTS = {
    "c": 2.99792458e8,  # m/s
    "hbar": 1.054571817e-34,  # J·s
    "k_B": 1.380649e-23,  # J/K
    "system": "SI",
}

# Default unit system
DEFAULT_UNITS = NATURAL_UNITS

# Numerical tolerances for relativistic calculations
TOLERANCE_DEFAULT = 1e-10
TOLERANCE_STRICT = 1e-15
TOLERANCE_LOOSE = 1e-8

# Velocity limits (in units of c)
VELOCITY_MAX = 0.999999  # Maximum allowed velocity to avoid singularities
GAMMA_MAX = 1e6  # Maximum Lorentz factor

# Temperature and thermodynamic constants (in natural units)
TEMPERATURE_MIN = 1e-10  # Minimum temperature to avoid singularities
ENERGY_DENSITY_MIN = 1e-15  # Minimum energy density

# Israel-Stewart transport coefficient bounds
VISCOSITY_MIN = 0.0  # Minimum viscosity (entropy production constraint)
CONDUCTIVITY_MIN = 0.0  # Minimum thermal conductivity

# Particle masses (in natural units, GeV)
MPROTON = 0.938272  # Proton mass in GeV
MNEUTRON = 0.939565  # Neutron mass in GeV
MELECTRON = 0.000511  # Electron mass in GeV

# Convenience aliases for natural units
KBOLTZ = BOLTZMANN_K  # Alias for Boltzmann constant

# Numerical stability parameters
CONDITION_NUMBER_WARN = 1e12  # Warn if matrix condition number exceeds this
DETERMINANT_MIN = 1e-15  # Minimum determinant for non-singular matrices

# String representations for common physics quantities
FOUR_VECTOR_NAMES = {
    0: "t",  # Time component
    1: "x",  # X spatial component
    2: "y",  # Y spatial component
    3: "z",  # Z spatial component
}

TENSOR_NAMES = {
    "stress_energy": "T",
    "viscous_stress": "pi",
    "four_velocity": "u",
    "four_acceleration": "a",
    "metric": "g",
}

# Common coordinate systems
COORDINATE_SYSTEMS = {
    "cartesian": ["t", "x", "y", "z"],
    "spherical": ["t", "r", "theta", "phi"],
    "cylindrical": ["t", "rho", "phi", "z"],
    "lightcone": ["u", "v", "x", "y"],
    "milne": ["tau", "eta", "x", "y"],
}

# Metric signatures
METRIC_SIGNATURES = {
    "mostly_plus": (-1, 1, 1, 1),  # Particle physics convention
    "mostly_minus": (1, -1, -1, -1),  # General relativity convention
}


def get_physical_constant(name: str, unit_system: str = "natural") -> float:
    """
    Get physical constant in specified unit system.

    Args:
        name: Constant name ('c', 'hbar', 'k_B')
        unit_system: Unit system ('natural' or 'SI')

    Returns:
        Constant value
    """
    if unit_system == "natural":
        constants = NATURAL_UNITS
    elif unit_system == "SI":
        constants = SI_CONSTANTS
    else:
        raise ValueError(f"Unknown unit system: {unit_system}")

    if name not in constants:
        raise ValueError(f"Unknown constant: {name}")

    return cast(float, constants[name])


def validate_relativistic_velocity(
    velocity: np.ndarray, tolerance: float = TOLERANCE_DEFAULT
) -> bool:
    """
    Validate that velocity is subluminal.

    Args:
        velocity: 3-velocity array
        tolerance: Tolerance for velocity limit check

    Returns:
        True if velocity is valid

    Raises:
        ValueError: If velocity exceeds speed of light
    """
    v_squared = np.dot(velocity, velocity)
    if v_squared >= (C_LIGHT**2 - tolerance):
        raise ValueError(f"Velocity magnitude {np.sqrt(v_squared):.6f} exceeds speed of light")
    return True


def compute_lorentz_factor(velocity: np.ndarray) -> float:
    """
    Compute Lorentz factor γ = 1/√(1 − v²/c²).

    Args:
        velocity: 3-velocity array

    Returns:
        Lorentz factor
    """
    validate_relativistic_velocity(velocity)
    v_squared = np.dot(velocity, velocity)
    return float(1.0 / np.sqrt(1.0 - v_squared / C_LIGHT**2))


def validate_temperature(temperature: float) -> bool:
    """
    Validate temperature for thermodynamic calculations.

    Args:
        temperature: Temperature value

    Returns:
        True if valid

    Raises:
        ValueError: If temperature is invalid
    """
    if temperature < TEMPERATURE_MIN:
        raise ValueError(f"Temperature {temperature} below minimum {TEMPERATURE_MIN}")
    if not np.isfinite(temperature):
        raise ValueError(f"Temperature must be finite, got {temperature}")
    return True


def validate_transport_coefficient(coefficient: float, name: str) -> bool:
    """
    Validate transport coefficient (viscosity, conductivity, etc.).

    Args:
        coefficient: Coefficient value
        name: Coefficient name for error messages

    Returns:
        True if valid

    Raises:
        ValueError: If coefficient is invalid
    """
    if coefficient < 0.0:
        raise ValueError(f"{name} must be non-negative, got {coefficient}")
    if not np.isfinite(coefficient):
        raise ValueError(f"{name} must be finite, got {coefficient}")
    return True


# Common unit conversions (from natural units)
def natural_to_si_energy(energy_natural: float) -> float:
    """Convert energy from natural units to SI (Joules)."""
    return (
        energy_natural * cast(float, SI_CONSTANTS["hbar"]) * cast(float, SI_CONSTANTS["c"]) / 1.0
    )  # c


def natural_to_si_temperature(temp_natural: float) -> float:
    """Convert temperature from natural units to SI (Kelvin)."""
    return temp_natural / cast(float, SI_CONSTANTS["k_B"])


def natural_to_si_time(time_natural: float) -> float:
    """Convert time from natural units to SI (seconds)."""
    return time_natural * cast(float, SI_CONSTANTS["hbar"]) / natural_to_si_energy(1.0)


def natural_to_si_length(length_natural: float) -> float:
    """Convert length from natural units to SI (meters)."""
    return (
        length_natural
        * cast(float, SI_CONSTANTS["hbar"])
        * cast(float, SI_CONSTANTS["c"])
        / natural_to_si_energy(1.0)
    )


# Dimensionless combinations for Israel-Stewart hydrodynamics
def reynolds_number_estimate(
    viscosity: float, density: float, velocity: float, length_scale: float
) -> float:
    """
    Estimate Reynolds number Re = ρ v L / η.

    Args:
        viscosity: Shear viscosity
        density: Fluid density
        velocity: Characteristic velocity
        length_scale: Characteristic length

    Returns:
        Reynolds number estimate
    """
    if viscosity <= 0:
        return float("inf")
    return density * velocity * length_scale / viscosity


def knudsen_number_estimate(mean_free_path: float, length_scale: float) -> float:
    """
    Estimate Knudsen number Kn = λ/L.

    Args:
        mean_free_path: Particle mean free path
        length_scale: Characteristic system length

    Returns:
        Knudsen number
    """
    return mean_free_path / length_scale
