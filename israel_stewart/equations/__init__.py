"""
Physics equations for Israel-Stewart relativistic hydrodynamics.

This module contains the fundamental physics equations:
- Conservation laws (energy-momentum and particle number)
- Relaxation equations (Israel-Stewart second-order evolution)
- Transport coefficients (viscosity and thermal conductivity)
- Thermodynamic constraints and consistency conditions
"""

from .conservation import ConservationLaws

__all__ = [
    'ConservationLaws',
]
