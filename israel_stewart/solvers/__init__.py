"""
Numerical solvers for Israel-Stewart hydrodynamics equations.

This module provides various numerical methods for solving relativistic
hydrodynamics with second-order viscous corrections.
"""

from .spectral import SpectralISHydrodynamics, SpectralISolver

__all__ = [
    "SpectralISolver",
    "SpectralISHydrodynamics",
]
