"""
Tests for solver integration and cross-compatibility in Israel-Stewart hydrodynamics.

This module tests the integration between different solver types, factory functions,
and the overall solver ecosystem.
"""

import numpy as np
import pytest

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.metrics import MilneMetric, MinkowskiMetric
from israel_stewart.core.spacetime_grid import SpacetimeGrid
from israel_stewart.solvers import (
    BackwardEulerSolver,
    ConservativeFiniteDifference,
    IMEXRungeKuttaSolver,
    SpectralISolver,
    StrangSplitting,
    create_solver,
)


class TestSolverFactoryIntegration:
    """Test master factory function and solver creation."""

    @pytest.fixture
    def setup_basic_problem(
        self,
    ) -> tuple[SpacetimeGrid, MinkowskiMetric, TransportCoefficients, ISFieldConfiguration]:
        """Setup basic problem for integration testing."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(5, 16, 16, 16),
        )

        metric = MinkowskiMetric()

        coefficients = TransportCoefficients(
            shear_viscosity=0.1,
            bulk_viscosity=0.05,
            shear_relaxation_time=0.5,
            bulk_relaxation_time=0.3,
        )

        fields = ISFieldConfiguration(grid)
        fields.rho.fill(1.0)
        fields.pressure.fill(0.33)
        fields.Pi.fill(0.01)
        fields.pi_munu.fill(0.005)

        return grid, metric, coefficients, fields

    def test_master_factory_finite_difference(self, setup_basic_problem) -> None:
        """Test master factory for finite difference solvers."""
        grid, metric, coefficients, fields = setup_basic_problem

        # Test conservative finite difference
        solver = create_solver("finite_difference", "conservative", grid, metric, coefficients)
        assert isinstance(solver, ConservativeFiniteDifference)

        # Test with options
        solver = create_solver(
            "finite_difference", "conservative", grid, metric, coefficients, order=4
        )
        assert solver.order == 4

        # Test different subtypes
        upwind_solver = create_solver("finite_difference", "upwind", grid, metric, coefficients)
        weno_solver = create_solver("finite_difference", "weno", grid, metric, coefficients)

        assert upwind_solver.__class__.__name__ == "UpwindFiniteDifference"
        assert weno_solver.__class__.__name__ == "WENOFiniteDifference"

    def test_master_factory_implicit(self, setup_basic_problem) -> None:
        """Test master factory for implicit solvers."""
        grid, metric, coefficients, fields = setup_basic_problem

        # Test backward Euler
        solver = create_solver("implicit", "backward_euler", grid, metric, coefficients)
        assert isinstance(solver, BackwardEulerSolver)

        # Test IMEX Runge-Kutta
        solver = create_solver("implicit", "imex_rk", grid, metric, coefficients, order=2)
        assert isinstance(solver, IMEXRungeKuttaSolver)
        assert solver.order == 2

        # Test exponential integrator
        solver = create_solver("implicit", "exponential", grid, metric, coefficients)
        assert solver.__class__.__name__ == "ExponentialIntegrator"

    def test_master_factory_splitting(self, setup_basic_problem) -> None:
        """Test master factory for splitting methods."""
        grid, metric, coefficients, fields = setup_basic_problem

        # Test Strang splitting
        solver = create_solver("splitting", "strang", grid, metric, coefficients)
        assert isinstance(solver, StrangSplitting)

        # Test Lie-Trotter splitting
        solver = create_solver("splitting", "lietrotter", grid, metric, coefficients, order="HR")
        assert solver.__class__.__name__ == "LieTrotterSplitting"
        assert solver.order == "HR"

        # Test adaptive splitting
        solver = create_solver("splitting", "adaptive", grid, metric, coefficients, tolerance=1e-6)
        assert solver.__class__.__name__ == "AdaptiveSplitting"
        assert solver.tolerance == 1e-6

    def test_master_factory_spectral(self, setup_basic_problem) -> None:
        """Test master factory for spectral methods."""
        grid, metric, coefficients, fields = setup_basic_problem

        # Test spectral solver
        solver = create_solver("spectral", "solver", grid, fields=fields, coefficients=coefficients)
        assert isinstance(solver, SpectralISolver)

        # Test spectral hydrodynamics
        solver = create_solver("spectral", "hydro", grid, fields=fields, coefficients=coefficients)
        assert solver.__class__.__name__ == "SpectralISHydrodynamics"

    def test_factory_error_handling(self, setup_basic_problem) -> None:
        """Test factory error handling."""
        grid, metric, coefficients, fields = setup_basic_problem

        # Invalid solver type
        with pytest.raises(ValueError, match="Unknown solver type"):
            create_solver("invalid_type", "", grid, metric, coefficients)

        # Invalid subtype
        with pytest.raises(ValueError, match="Unknown.*solver subtype"):
            create_solver("implicit", "invalid_subtype", grid, metric, coefficients)

        # Missing required parameters
        with pytest.raises(ValueError, match="required"):
            create_solver("finite_difference", "conservative", grid=None)

        with pytest.raises(ValueError, match="required"):
            create_solver("spectral", "solver", grid, fields=None)

    def test_factory_parameter_validation(self, setup_basic_problem) -> None:
        """Test factory parameter validation."""
        grid, metric, coefficients, fields = setup_basic_problem

        # All solvers require grid
        with pytest.raises(ValueError, match="SpacetimeGrid is required"):
            create_solver("finite_difference", "conservative", grid=None, metric=metric)

        # Finite difference requires metric
        with pytest.raises(ValueError, match="MetricBase.*required"):
            create_solver("finite_difference", "conservative", grid, metric=None)

        # Implicit requires both metric and coefficients
        with pytest.raises(ValueError, match="MetricBase.*required"):
            create_solver("implicit", "backward_euler", grid, metric=None)

        with pytest.raises(ValueError, match="TransportCoefficients.*required"):
            create_solver("implicit", "backward_euler", grid, metric, coefficients=None)


class TestCrossSolverCompatibility:
    """Test compatibility between different solver types."""

    @pytest.fixture
    def setup_compatibility_test(
        self,
    ) -> tuple[SpacetimeGrid, MinkowskiMetric, TransportCoefficients, ISFieldConfiguration]:
        """Setup for cross-solver compatibility testing."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(5, 32, 8, 8),
        )

        metric = MinkowskiMetric()
        coefficients = TransportCoefficients()
        fields = ISFieldConfiguration(grid)

        # Initialize with simple profile
        coords = grid.coordinates()
        x = coords["x"]
        fields.rho = 1.0 + 0.1 * np.sin(np.pi * x)
        fields.pressure = fields.rho / 3.0
        fields.Pi.fill(0.01)

        return grid, metric, coefficients, fields

    def test_field_format_compatibility(self, setup_compatibility_test) -> None:
        """Test that all solvers work with the same field format."""
        grid, metric, coefficients, fields = setup_compatibility_test

        # Create different solver types
        solvers = [
            create_solver("finite_difference", "conservative", grid, metric, coefficients),
            create_solver("implicit", "backward_euler", grid, metric, coefficients),
            create_solver("splitting", "strang", grid, metric, coefficients),
        ]

        # All should accept the same field configuration
        for solver in solvers:
            if hasattr(solver, "compute_spatial_derivatives"):
                # Finite difference solver
                result = solver.compute_spatial_derivatives(fields.rho, axis=1)
                assert result.shape == fields.rho.shape
            elif hasattr(solver, "solve_step"):
                # Implicit solver
                def simple_rhs(f):
                    return {"Pi": -f.Pi / 0.3}

                result = solver.solve_step(fields, 0.01, simple_rhs)
                assert isinstance(result, ISFieldConfiguration)
            elif hasattr(solver, "advance_timestep"):
                # Splitting solver
                result = solver.advance_timestep(fields, 0.01)
                assert isinstance(result, ISFieldConfiguration)

    def test_solver_chaining(self, setup_compatibility_test) -> None:
        """Test chaining different solver types."""
        grid, metric, coefficients, fields = setup_compatibility_test

        # Create solvers
        fd_solver = create_solver("finite_difference", "conservative", grid, metric, coefficients)
        implicit_solver = create_solver("implicit", "backward_euler", grid, metric, coefficients)

        # Use finite difference result as input to implicit solver
        def rhs_with_derivatives(f):
            drho_dx = fd_solver.compute_spatial_derivatives(f.rho, axis=1)
            # Simple RHS that uses spatial derivatives
            return {"Pi": -f.Pi / 0.3 - 0.1 * drho_dx}

        result = implicit_solver.solve_step(fields, 0.01, rhs_with_derivatives)
        assert isinstance(result, ISFieldConfiguration)
        assert np.all(np.isfinite(result.Pi))

    def test_metric_compatibility(self) -> None:
        """Test solver compatibility with different metrics."""
        # Cartesian grid with Minkowski metric
        grid_cartesian = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(5, 16, 8, 8),
        )

        # Milne grid
        grid_milne = SpacetimeGrid(
            coordinate_system="milne",
            time_range=(0.5, 2.0),
            spatial_ranges=[(-2.0, 2.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(5, 8, 8, 8),
        )

        minkowski_metric = MinkowskiMetric()
        milne_metric = MilneMetric()
        coefficients = TransportCoefficients()

        # Test finite difference with different metrics
        fd_minkowski = create_solver(
            "finite_difference", "conservative", grid_cartesian, minkowski_metric, coefficients
        )
        fd_milne = create_solver(
            "finite_difference", "conservative", grid_milne, milne_metric, coefficients
        )

        assert fd_minkowski.metric == minkowski_metric
        assert fd_milne.metric == milne_metric

    def test_coefficient_compatibility(self, setup_compatibility_test) -> None:
        """Test solver compatibility with different transport coefficients."""
        grid, metric, _, fields = setup_compatibility_test

        # Different coefficient sets
        coeffs_minimal = TransportCoefficients()
        coeffs_full = TransportCoefficients(
            shear_viscosity=0.2,
            bulk_viscosity=0.1,
            shear_relaxation_time=0.8,
            bulk_relaxation_time=0.4,
        )

        # Solvers should work with different coefficient sets
        solver_minimal = create_solver("implicit", "backward_euler", grid, metric, coeffs_minimal)
        solver_full = create_solver("implicit", "backward_euler", grid, metric, coeffs_full)

        def simple_rhs(f):
            return {"Pi": -f.Pi / 0.3}

        result_minimal = solver_minimal.solve_step(fields, 0.01, simple_rhs)
        result_full = solver_full.solve_step(fields, 0.01, simple_rhs)

        assert isinstance(result_minimal, ISFieldConfiguration)
        assert isinstance(result_full, ISFieldConfiguration)


class TestSolverPerformanceIntegration:
    """Test integrated performance of solver combinations."""

    @pytest.fixture
    def setup_performance_test(
        self,
    ) -> tuple[SpacetimeGrid, MinkowskiMetric, TransportCoefficients, ISFieldConfiguration]:
        """Setup for performance integration testing."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(5, 32, 32, 8),
        )

        metric = MinkowskiMetric()
        coefficients = TransportCoefficients()
        fields = ISFieldConfiguration(grid)

        return grid, metric, coefficients, fields

    def test_solver_performance_comparison(self, setup_performance_test) -> None:
        """Compare performance of different solver types."""
        import time

        grid, metric, coefficients, fields = setup_performance_test

        # Create different solvers
        solvers = {
            "finite_difference": create_solver(
                "finite_difference", "conservative", grid, metric, coefficients
            ),
            "implicit": create_solver("implicit", "backward_euler", grid, metric, coefficients),
            "splitting": create_solver("splitting", "strang", grid, metric, coefficients),
        }

        times = {}

        # Test finite difference performance
        start_time = time.time()
        for _ in range(10):
            result = solvers["finite_difference"].compute_spatial_derivatives(fields.rho, axis=1)
        times["finite_difference"] = time.time() - start_time

        # Test implicit solver performance
        def simple_rhs(f):
            return {"Pi": -f.Pi / 0.3}

        start_time = time.time()
        for _ in range(10):
            result = solvers["implicit"].solve_step(fields, 0.01, simple_rhs)
        times["implicit"] = time.time() - start_time

        # Test splitting performance
        start_time = time.time()
        for _ in range(10):
            result = solvers["splitting"].advance_timestep(fields, 0.01)
        times["splitting"] = time.time() - start_time

        # All should complete in reasonable time
        assert all(t < 10.0 for t in times.values()), "All solvers should complete quickly"

    def test_memory_usage_integration(self, setup_performance_test) -> None:
        """Test memory usage in integrated solver scenarios."""
        grid, metric, coefficients, fields = setup_performance_test

        # Create multiple solvers simultaneously
        solvers = [
            create_solver("finite_difference", "conservative", grid, metric, coefficients),
            create_solver("implicit", "backward_euler", grid, metric, coefficients),
            create_solver("splitting", "strang", grid, metric, coefficients),
        ]

        # All solvers should coexist without memory issues
        assert len(solvers) == 3

        # Test that they can all operate on the same fields
        for solver in solvers:
            if hasattr(solver, "advance_timestep"):
                result = solver.advance_timestep(fields, 0.01)
                assert isinstance(result, ISFieldConfiguration)


class TestSolverRobustnessIntegration:
    """Test robustness of integrated solver systems."""

    def test_error_propagation(self) -> None:
        """Test error handling in solver chains."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(5, 16, 8, 8),
        )

        metric = MinkowskiMetric()
        coefficients = TransportCoefficients()
        fields = ISFieldConfiguration(grid)

        # Create solver chain that might fail
        implicit_solver = create_solver("implicit", "backward_euler", grid, metric, coefficients)

        def problematic_rhs(f):
            # RHS that might cause numerical issues
            return {"Pi": np.full_like(f.Pi, np.inf)}

        # Should handle error gracefully
        with pytest.warns(UserWarning):
            result = implicit_solver.solve_step(fields, 0.01, problematic_rhs)
            assert np.all(np.isfinite(result.rho))

    def test_boundary_case_integration(self) -> None:
        """Test solver integration with boundary cases."""
        # Very small grid
        grid_small = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(3, 4, 4, 4),  # Minimal grid
        )

        metric = MinkowskiMetric()
        coefficients = TransportCoefficients()
        fields = ISFieldConfiguration(grid_small)

        # Solvers should handle minimal grids
        solver = create_solver(
            "finite_difference", "conservative", grid_small, metric, coefficients
        )
        result = solver.compute_spatial_derivatives(fields.rho, axis=1)
        assert result.shape == fields.rho.shape

    def test_extreme_parameter_integration(self) -> None:
        """Test solver integration with extreme parameters."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(5, 16, 8, 8),
        )

        metric = MinkowskiMetric()

        # Extreme transport coefficients
        coeffs_extreme = TransportCoefficients(
            shear_viscosity=1e-10,  # Very small
            bulk_viscosity=1e10,  # Very large
            shear_relaxation_time=1e-6,  # Very fast
            bulk_relaxation_time=1e6,  # Very slow
        )

        fields = ISFieldConfiguration(grid)

        # Solver should handle extreme parameters gracefully
        solver = create_solver("implicit", "backward_euler", grid, metric, coeffs_extreme)

        def simple_rhs(f):
            return {"Pi": -f.Pi / coeffs_extreme.bulk_relaxation_time}

        result = solver.solve_step(fields, 0.01, simple_rhs)
        assert isinstance(result, ISFieldConfiguration)
        assert np.all(np.isfinite(result.Pi))


class TestSolverAPIConsistency:
    """Test API consistency across solver types."""

    @pytest.fixture
    def setup_api_test(
        self,
    ) -> tuple[SpacetimeGrid, MinkowskiMetric, TransportCoefficients, ISFieldConfiguration]:
        """Setup for API consistency testing."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(5, 16, 8, 8),
        )

        metric = MinkowskiMetric()
        coefficients = TransportCoefficients()
        fields = ISFieldConfiguration(grid)

        return grid, metric, coefficients, fields

    def test_common_attributes(self, setup_api_test) -> None:
        """Test that all solvers have consistent common attributes."""
        grid, metric, coefficients, fields = setup_api_test

        solvers = [
            create_solver("finite_difference", "conservative", grid, metric, coefficients),
            create_solver("implicit", "backward_euler", grid, metric, coefficients),
            create_solver("splitting", "strang", grid, metric, coefficients),
        ]

        for solver in solvers:
            # All should have grid
            assert hasattr(solver, "grid")
            assert solver.grid == grid

            # All should have metric (where applicable)
            if hasattr(solver, "metric"):
                assert solver.metric == metric

            # All should have coefficients (where applicable)
            if hasattr(solver, "coefficients"):
                assert solver.coefficients == coefficients

    def test_factory_consistency(self, setup_api_test) -> None:
        """Test consistency of factory function interfaces."""
        grid, metric, coefficients, fields = setup_api_test

        # All factory functions should accept same basic parameters
        factory_calls = [
            (
                "finite_difference",
                "conservative",
                {"grid": grid, "metric": metric, "coefficients": coefficients},
            ),
            (
                "implicit",
                "backward_euler",
                {"grid": grid, "metric": metric, "coefficients": coefficients},
            ),
            ("splitting", "strang", {"grid": grid, "metric": metric, "coefficients": coefficients}),
        ]

        for solver_type, subtype, kwargs in factory_calls:
            solver = create_solver(solver_type, subtype, **kwargs)
            assert solver is not None

    def test_error_message_consistency(self, setup_api_test) -> None:
        """Test that error messages are consistent across solver types."""
        grid, metric, coefficients, fields = setup_api_test

        # Test consistent error messages for missing parameters
        error_cases = [
            ("finite_difference", "conservative", {"grid": None}),
            ("implicit", "backward_euler", {"grid": None}),
            ("splitting", "strang", {"grid": None}),
        ]

        for solver_type, subtype, kwargs in error_cases:
            kwargs.setdefault("metric", metric)
            kwargs.setdefault("coefficients", coefficients)

            with pytest.raises(ValueError, match="SpacetimeGrid is required"):
                create_solver(solver_type, subtype, **kwargs)


class TestSolverDocumentationIntegration:
    """Test that solver documentation and examples work correctly."""

    def test_documentation_examples(self) -> None:
        """Test examples from solver module documentation."""
        # Example from __init__.py documentation
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(8, 8, 8, 8),
        )

        coeffs = TransportCoefficients(
            shear_viscosity=0.1,
            bulk_viscosity=0.05,
            shear_relaxation_time=0.5,
            bulk_relaxation_time=0.3,
        )

        fields = ISFieldConfiguration(grid)
        metric = MinkowskiMetric()

        # Test documented examples
        fd_solver = create_solver("finite_difference", "conservative", grid, metric, coeffs)
        assert isinstance(fd_solver, ConservativeFiniteDifference)

        implicit_solver = create_solver("implicit", "backward_euler", grid, metric, coeffs)
        assert isinstance(implicit_solver, BackwardEulerSolver)

        splitting_solver = create_solver("splitting", "strang", grid, metric, coeffs)
        assert isinstance(splitting_solver, StrangSplitting)

    def test_solver_help_strings(self) -> None:
        """Test that solvers have proper documentation strings."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(5, 8, 8, 8),
        )

        metric = MinkowskiMetric()
        coefficients = TransportCoefficients()

        solver = create_solver("finite_difference", "conservative", grid, metric, coefficients)

        # Should have docstrings
        assert solver.__class__.__doc__ is not None
        assert len(solver.__class__.__doc__.strip()) > 50  # Substantial documentation

    def test_solver_string_representations(self) -> None:
        """Test string representations of solvers."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(5, 8, 8, 8),
        )

        metric = MinkowskiMetric()
        coefficients = TransportCoefficients()

        solvers = [
            create_solver("finite_difference", "conservative", grid, metric, coefficients),
            create_solver("implicit", "backward_euler", grid, metric, coefficients),
            create_solver("splitting", "strang", grid, metric, coefficients),
        ]

        for solver in solvers:
            # Should have meaningful string representation
            str_repr = str(solver)
            assert len(str_repr) > 10
            assert solver.__class__.__name__ in str_repr or "Solver" in str_repr
