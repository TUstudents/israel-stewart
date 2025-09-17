"""
Physical constants and unit systems for relativistic hydrodynamics.

This module provides fundamental constants used in relativistic physics
and Israel-Stewart hydrodynamics, with consistent unit systems.
"""

from typing import TypedDict

import numpy as np

# Speed of light (set to 1 in natural units)
C_LIGHT = 1.0

# Planck’s constant (reduced, ħ = h/2π, set to 1 in natural units)
HBAR = 1.0

# Boltzmann constant (set to 1 in natural units)
BOLTZMANN_K = 1.0

# Unit conversion constants
GEV_TO_JOULE = 1.602176634e-10  # Conversion factor: 1 GeV = 1.602176634e-10 J
JOULE_TO_GEV = 1.0 / GEV_TO_JOULE  # Inverse conversion


class PhysicalConstants(TypedDict):
    """Type definition for physical constants dictionaries."""
    c: float
    hbar: float
    k_B: float
    system: str


# Natural units system (c = ħ = k_B = 1)
NATURAL_UNITS: PhysicalConstants = {
    "c": C_LIGHT,
    "hbar": HBAR,
    "k_B": BOLTZMANN_K,
    "system": "natural"
}

# SI units (for dimensional analysis and conversion)
SI_CONSTANTS: PhysicalConstants = {
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

# Additional conversion utilities
def si_to_natural_energy(energy_si: float) -> float:
    """Convert energy from SI (Joules) to natural units (GeV)."""
    return energy_si * JOULE_TO_GEV


def si_to_natural_temperature(temp_si: float) -> float:
    """Convert temperature from SI (Kelvin) to natural units (GeV)."""
    temp_joules = temp_si * SI_CONSTANTS["k_B"]
    return temp_joules * JOULE_TO_GEV


def si_to_natural_time(time_si: float) -> float:
    """Convert time from SI (seconds) to natural units (GeV^-1)."""
    return time_si * GEV_TO_JOULE / SI_CONSTANTS["hbar"]


def si_to_natural_length(length_si: float) -> float:
    """Convert length from SI (meters) to natural units (GeV^-1)."""
    hbar_c = SI_CONSTANTS["hbar"] * SI_CONSTANTS["c"]
    return length_si * GEV_TO_JOULE / hbar_c

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
    unit_systems: dict[str, PhysicalConstants] = {"natural": NATURAL_UNITS, "SI": SI_CONSTANTS}

    if unit_system not in unit_systems:
        raise ValueError(f"Unknown unit system: {unit_system}")

    constants = unit_systems[unit_system]
    if name not in constants:
        raise ValueError(f"Unknown constant: {name}")

    # Cast needed for TypedDict access with variable key
    return float(constants[name])  # type: ignore[literal-required]


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
# In natural units: c = ħ = k_B = 1
# Energy scale: 1 GeV = 1.602176634e-10 J
# Length scale: 1 GeV^-1 = ħc/GeV ≈ 1.973e-16 m
# Time scale: 1 GeV^-1 = ħ/GeV ≈ 6.582e-25 s

def natural_to_si_energy(energy_natural: float) -> float:
    """Convert energy from natural units (GeV) to SI (Joules).

    In natural units, energies are measured in GeV.
    Conversion: E[J] = E[GeV] × (1 GeV in Joules)
    """
    return energy_natural * GEV_TO_JOULE


def natural_to_si_temperature(temp_natural: float) -> float:
    """Convert temperature from natural units (GeV) to SI (Kelvin).

    In natural units with k_B = 1, temperature has units of energy (GeV).
    Conversion: T[K] = T[GeV] × (1 GeV in Joules) / k_B[J/K]
    """
    temp_joules = temp_natural * GEV_TO_JOULE
    return temp_joules / SI_CONSTANTS["k_B"]


def natural_to_si_time(time_natural: float) -> float:
    """Convert time from natural units (GeV^-1) to SI (seconds).

    In natural units with ħ = 1, time has units GeV^-1.
    Conversion: t[s] = t[GeV^-1] × ħ[J·s] / (1 GeV in Joules)
    """
    return time_natural * SI_CONSTANTS["hbar"] / GEV_TO_JOULE


def natural_to_si_length(length_natural: float) -> float:
    """Convert length from natural units (GeV^-1) to SI (meters).

    In natural units with ħ = c = 1, length has units GeV^-1.
    Conversion: L[m] = L[GeV^-1] × ħc[J·m] / (1 GeV in Joules)
    """
    hbar_c = SI_CONSTANTS["hbar"] * SI_CONSTANTS["c"]
    return length_natural * hbar_c / GEV_TO_JOULE


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
