"""
Numerical solvers for Israel-Stewart hydrodynamics equations.

This module provides a comprehensive suite of numerical methods for solving
relativistic hydrodynamics with second-order viscous corrections, including:

- **Finite Difference Methods**: Conservative, upwind, and WENO schemes for spatial discretization
- **Implicit Time Integration**: Backward Euler, IMEX Runge-Kutta, and exponential integrators
- **Operator Splitting**: Strang, Lie-Trotter, adaptive, and physics-based splitting methods
- **Spectral Methods**: FFT-based high-accuracy solvers for periodic problems

## Quick Start Examples

### Basic Solver Creation
```python
from israel_stewart.solvers import create_solver
from israel_stewart.core import SpacetimeGrid, ISFieldConfiguration, TransportCoefficients

# Setup problem
grid = SpacetimeGrid(...)
fields = ISFieldConfiguration(grid)
coeffs = TransportCoefficients(...)

# Create different solver types
fd_solver = create_solver("finite_difference", "conservative", grid, coeffs)
implicit_solver = create_solver("implicit", "backward_euler", grid, coeffs)
spectral_solver = create_solver("spectral", grid, fields, coeffs)
splitting_solver = create_solver("splitting", "strang", grid, coeffs)
```

### Specialized Factory Functions
```python
from israel_stewart.solvers import (
    create_finite_difference_solver,
    create_implicit_solver,
    create_splitting_solver,
)

# High-order finite difference
weno_solver = create_finite_difference_solver("weno", grid, metric, order=5)

# Adaptive implicit integration
adaptive_implicit = create_implicit_solver("imex_rk", grid, metric, coeffs, order=3)

# Physics-aware splitting
physics_split = create_splitting_solver("physics", grid, metric, coeffs)
```

## Performance Guidelines

- **Spectral methods**: Best for smooth, periodic problems (exponential convergence)
- **WENO finite difference**: Excellent for shock-capturing and discontinuities
- **Implicit solvers**: Essential for stiff relaxation equations (large τ/dt ratios)
- **Operator splitting**: Optimal for multi-scale physics with disparate timescales

## Solver Type Reference

### Finite Difference Schemes
- `ConservativeFiniteDifference`: Mass-conserving, stable for long-time evolution
- `UpwindFiniteDifference`: Characteristic-based, prevents numerical diffusion
- `WENOFiniteDifference`: High-order shock-capturing with adaptive stencils

### Implicit Time Integrators
- `BackwardEulerSolver`: Robust first-order, excellent stability properties
- `IMEXRungeKuttaSolver`: Higher-order with explicit hyperbolic/implicit relaxation
- `ExponentialIntegrator`: Specialized for linear relaxation operators

### Operator Splitting Methods
- `StrangSplitting`: Second-order symmetric splitting, good accuracy/cost balance
- `LieTrotterSplitting`: First-order, simple and robust for exploratory work
- `AdaptiveSplitting`: Automatic timestep control with error estimation
- `PhysicsBasedSplitting`: Multi-scale aware, optimized for IS equations

### Spectral Methods
- `SpectralISolver`: Core FFT-based spatial differentiation and operators
- `SpectralISHydrodynamics`: Complete spectral hydro solver with conservation laws
"""

# Import finite difference methods
# Additional imports for master factory
from typing import Any, Optional, Union

from ..core.fields import ISFieldConfiguration, TransportCoefficients
from ..core.metrics import MetricBase
from ..core.spacetime_grid import SpacetimeGrid
from .finite_difference import (
    ConservativeFiniteDifference,
    FiniteDifferenceScheme,
    UpwindFiniteDifference,
    WENOFiniteDifference,
    create_finite_difference_solver,
)

# Import implicit time integration methods
from .implicit import (
    BackwardEulerSolver,
    ExponentialIntegrator,
    IMEXRungeKuttaSolver,
    ImplicitSolverBase,
    create_implicit_solver,
)

# Import spectral methods
from .spectral import SpectralISHydrodynamics, SpectralISolver

# Import operator splitting methods
from .splitting import (
    AdaptiveSplitting,
    LieTrotterSplitting,
    OperatorSplittingBase,
    PhysicsBasedSplitting,
    StrangSplitting,
    create_splitting_solver,
    solve_hyperbolic_conservative,
    solve_relaxation_exponential,
)


def create_solver(
    solver_type: str,
    solver_subtype: str = "",
    grid: SpacetimeGrid | None = None,
    metric: MetricBase | None = None,
    coefficients: TransportCoefficients | None = None,
    fields: ISFieldConfiguration | None = None,
    **kwargs: Any,
) -> (
    FiniteDifferenceScheme
    | ImplicitSolverBase
    | OperatorSplittingBase
    | SpectralISolver
    | SpectralISHydrodynamics
):
    """
    Master factory function for creating any type of Israel-Stewart solver.

    This is the main entry point for solver creation, providing a unified interface
    to all available numerical methods in the Israel-Stewart framework.

    Args:
        solver_type: Type of solver to create:
            - "finite_difference": Spatial discretization schemes
            - "implicit": Time integration methods for stiff equations
            - "splitting": Operator splitting methods
            - "spectral": FFT-based high-accuracy solvers
        solver_subtype: Specific solver variant:
            - For finite_difference: "conservative", "upwind", "weno"
            - For implicit: "backward_euler", "imex_rk", "exponential"
            - For splitting: "strang", "lietrotter", "adaptive", "physics"
            - For spectral: "solver" (SpectralISolver) or "hydro" (SpectralISHydrodynamics)
        grid: SpacetimeGrid for discretization (required for all)
        metric: MetricBase for curved spacetime (required for most)
        coefficients: TransportCoefficients for physics (required for implicit/splitting)
        fields: ISFieldConfiguration (required for spectral hydro)
        **kwargs: Additional solver-specific parameters

    Returns:
        Configured solver instance of the appropriate type

    Raises:
        ValueError: If solver_type or solver_subtype is not recognized
        ValueError: If required parameters are missing

    Examples:
        >>> # Conservative finite difference solver
        >>> fd_solver = create_solver("finite_difference", "conservative", grid, metric, order=4)

        >>> # Adaptive operator splitting
        >>> split_solver = create_solver("splitting", "adaptive", grid, metric, coeffs, tolerance=1e-6)

        >>> # Spectral hydrodynamics solver
        >>> spectral_hydro = create_solver(
        ...     "spectral", "hydro", grid, fields=fields, coefficients=coeffs
        ... )
    """
    # Validate inputs
    if grid is None:
        raise ValueError("SpacetimeGrid is required for all solver types")

    solver_type = solver_type.lower()
    solver_subtype = solver_subtype.lower()

    # Route to appropriate factory function
    if solver_type == "finite_difference":
        if metric is None:
            raise ValueError("MetricBase is required for finite difference solvers")
        return create_finite_difference_solver(solver_subtype, grid, metric, **kwargs)

    elif solver_type == "implicit":
        if metric is None or coefficients is None:
            raise ValueError(
                "MetricBase and TransportCoefficients are required for implicit solvers"
            )
        return create_implicit_solver(solver_subtype, grid, metric, coefficients, **kwargs)

    elif solver_type == "splitting":
        if metric is None or coefficients is None:
            raise ValueError(
                "MetricBase and TransportCoefficients are required for splitting solvers"
            )
        return create_splitting_solver(solver_subtype, grid, metric, coefficients, **kwargs)

    elif solver_type == "spectral":
        if solver_subtype in ["", "solver"]:
            # Create SpectralISolver
            if fields is None:
                raise ValueError("ISFieldConfiguration is required for SpectralISolver")
            return SpectralISolver(grid, fields, coefficients)
        elif solver_subtype == "hydro":
            # Create SpectralISHydrodynamics
            if fields is None:
                raise ValueError("ISFieldConfiguration is required for SpectralISHydrodynamics")
            return SpectralISHydrodynamics(grid, fields, coefficients)
        else:
            raise ValueError(
                f"Unknown spectral solver subtype: {solver_subtype}. "
                f"Available: 'solver', 'hydro'"
            )
    else:
        raise ValueError(
            f"Unknown solver type: {solver_type}. "
            f"Available types: 'finite_difference', 'implicit', 'splitting', 'spectral'"
        )


# Complete list of all exported symbols
__all__ = [
    # === Master factory function ===
    "create_solver",
    # === Finite difference methods ===
    # Base classes
    "FiniteDifferenceScheme",
    # Concrete implementations
    "ConservativeFiniteDifference",
    "UpwindFiniteDifference",
    "WENOFiniteDifference",
    # Factory function
    "create_finite_difference_solver",
    # === Implicit time integration ===
    # Base classes
    "ImplicitSolverBase",
    # Concrete implementations
    "BackwardEulerSolver",
    "IMEXRungeKuttaSolver",
    "ExponentialIntegrator",
    # Factory function
    "create_implicit_solver",
    # === Operator splitting methods ===
    # Base classes
    "OperatorSplittingBase",
    # Concrete implementations
    "StrangSplitting",
    "LieTrotterSplitting",
    "AdaptiveSplitting",
    "PhysicsBasedSplitting",
    # Factory function
    "create_splitting_solver",
    # Utility functions
    "solve_hyperbolic_conservative",
    "solve_relaxation_exponential",
    # === Spectral methods ===
    "SpectralISolver",
    "SpectralISHydrodynamics",
]
