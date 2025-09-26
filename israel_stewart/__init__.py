"""
Israel-Stewart Relativistic Hydrodynamics Package

A sophisticated Python implementation of relativistic hydrodynamics
using the Israel-Stewart formalism with second-order viscous corrections.
"""

# Initialize logging system early
from .utils.logging_config import setup_from_environment

setup_from_environment()

__version__ = "0.1.0"
__author__ = "Relativistic Hydrodynamics Team"

from . import (
    benchmarks,
    core,
    equations,
    linearization,
    rg_analysis,
    solvers,
    stochastic,
    utils,
)
