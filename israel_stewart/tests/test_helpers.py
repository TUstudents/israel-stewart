"""
Test helper functions for IReD validation.

Provides utilities for regime checking, analytical comparisons, and test setup.
"""

import numpy as np
import pytest

from israel_stewart.core.fields import TransportCoefficients
from israel_stewart.core.spacegrid import SpaceGrid


def compute_regime_parameter(
    grid: SpaceGrid,
    transport_coeffs: TransportCoefficients,
) -> float:
    """
    Compute Israel-Stewart regime parameter |τω|.

    The Israel-Stewart formalism is valid when |τω| ≲ 1, where:
    - τ is the maximum relaxation time (shear, bulk, or diffusion)
    - ω is the characteristic frequency (estimated as k_max × c_s)

    For radiation fluid: c_s = 1/√3 ≈ 0.577

    Reference: Wagner & Gavassino (2024), arXiv:2309.14828v2

    Args:
        grid: Spatial grid (determines k_max)
        transport_coeffs: Transport coefficients (determines τ)

    Returns:
        float: Regime parameter |τω|
    """
    # Maximum wavenumber from grid
    k_max = compute_k_max(grid)

    # Speed of sound for radiation fluid (conformal EOS: p = ρ/3)
    c_s = 1.0 / np.sqrt(3.0)

    # Characteristic frequency
    omega_max = k_max * c_s

    # Maximum relaxation time
    tau_max = max(
        transport_coeffs.shear_relaxation_time,
        transport_coeffs.bulk_relaxation_time,
        getattr(transport_coeffs, "diffusion_relaxation_time", 0.0),
    )

    # Regime parameter
    regime_param = abs(tau_max * omega_max)

    return regime_param


def compute_k_max(grid: SpaceGrid) -> float:
    """
    Compute maximum wavenumber from spatial grid.

    For periodic boundary conditions with N points over domain L:
    k_max = π·N / L

    Args:
        grid: Spatial grid

    Returns:
        float: Maximum wavenumber (natural units, GeV for c=ℏ=1)
    """
    # Get grid dimensions
    nx, ny, nz = grid.shape

    # Get domain sizes (assume cubic for simplicity)
    # grid.spatial_ranges is [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
    Lx = grid.spatial_ranges[0][1] - grid.spatial_ranges[0][0]
    Ly = grid.spatial_ranges[1][1] - grid.spatial_ranges[1][0]
    Lz = grid.spatial_ranges[2][1] - grid.spatial_ranges[2][0]

    # Maximum wavenumber (Nyquist frequency)
    kx_max = np.pi * nx / Lx
    ky_max = np.pi * ny / Ly
    kz_max = np.pi * nz / Lz

    # Conservative estimate: use maximum of all directions
    k_max = max(kx_max, ky_max, kz_max)

    return k_max


def check_regime_validity(
    grid: SpaceGrid,
    transport_coeffs: TransportCoefficients,
    max_allowed: float = 1.0,
) -> float:
    """
    Check if parameters are within Israel-Stewart regime.

    If regime parameter |τω| > max_allowed, skip test with informative message.

    Args:
        grid: Spatial grid
        transport_coeffs: Transport coefficients
        max_allowed: Maximum allowed regime parameter (default: 1.0)

    Returns:
        float: Regime parameter |τω|

    Raises:
        pytest.skip: If outside valid regime
    """
    regime_param = compute_regime_parameter(grid, transport_coeffs)

    if regime_param > max_allowed:
        pytest.skip(
            f"Outside Israel-Stewart regime: |τω| = {regime_param:.2f} > {max_allowed:.2f}. "
            f"Test requires regime-valid parameters. See Wagner & Gavassino (2024)."
        )

    return regime_param


def fail_if_outside_regime(
    grid: SpaceGrid,
    transport_coeffs: TransportCoefficients,
    max_allowed: float = 1.0,
) -> float:
    """
    Fail test if parameters are outside Israel-Stewart regime.

    Similar to check_regime_validity(), but fails instead of skipping.
    Use when regime validity is a correctness requirement, not a test limitation.

    Args:
        grid: Spatial grid
        transport_coeffs: Transport coefficients
        max_allowed: Maximum allowed regime parameter (default: 1.0)

    Returns:
        float: Regime parameter |τω|

    Raises:
        AssertionError: If outside valid regime
    """
    regime_param = compute_regime_parameter(grid, transport_coeffs)

    assert regime_param <= max_allowed, (
        f"Regime violation: |τω| = {regime_param:.2f} > {max_allowed:.2f}. "
        f"Israel-Stewart formalism invalid. See Wagner & Gavassino (2024)."
    )

    return regime_param
