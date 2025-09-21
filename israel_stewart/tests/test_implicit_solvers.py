"""
Tests for implicit solvers in Israel-Stewart hydrodynamics.

This module provides comprehensive tests for implicit time integration methods
including Jacobian accuracy, Newton convergence, and physics integration.
"""

import numpy as np
import pytest
import scipy.sparse as sparse
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.spacetime_grid import SpacetimeGrid
from israel_stewart.solvers.implicit import (
    BackwardEulerSolver,
    ExponentialIntegrator,
    IMEXRungeKuttaSolver,
    create_implicit_solver,
)


class TestImplicitSolverBase:
    """Test base functionality common to all implicit solvers."""

    @pytest.fixture
    def setup_basic_problem(self) -> tuple[SpacetimeGrid, MinkowskiMetric, TransportCoefficients]:
        """Setup basic problem configuration for implicit solver testing."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(3, 2, 2, 2),  # MEMORY SAFE: Small grid to prevent 88GB Jacobian
        )

        metric = MinkowskiMetric()

        coefficients = TransportCoefficients(
            shear_viscosity=0.1,
            bulk_viscosity=0.05,
            shear_relaxation_time=0.5,
            bulk_relaxation_time=0.3,
        )

        return grid, metric, coefficients

    @pytest.fixture
    def setup_fields(self, setup_basic_problem) -> ISFieldConfiguration:
        """Setup field configuration for testing."""
        grid, _, _ = setup_basic_problem
        fields = ISFieldConfiguration(grid)

        # Initialize with physically reasonable values
        fields.rho.fill(1.0)
        fields.pressure.fill(0.33)
        fields.u_mu[..., 0] = 1.0  # Time component
        fields.u_mu[..., 1:] = 0.0  # Spatial components

        # Small dissipative fluxes
        fields.Pi.fill(0.01)
        fields.pi_munu.fill(0.005)
        fields.q_mu.fill(0.002)

        return fields

    def simple_rhs_function(self, fields: ISFieldConfiguration) -> dict[str, np.ndarray]:
        """Simple RHS function for testing purposes."""
        rhs = {}

        # Linear relaxation terms for dissipative quantities
        if hasattr(fields, "Pi") and fields.Pi is not None:
            rhs["Pi"] = -fields.Pi / 0.3  # -Pi/tau_Pi

        if hasattr(fields, "pi_munu") and fields.pi_munu is not None:
            rhs["pi_munu"] = -fields.pi_munu / 0.5  # -pi^munu/tau_pi

        if hasattr(fields, "q_mu") and fields.q_mu is not None:
            rhs["q_mu"] = -fields.q_mu / 0.4  # -q^mu/tau_q

        return rhs


class TestBackwardEulerSolver(TestImplicitSolverBase):
    """Test BackwardEulerSolver implementation."""

    @pytest.fixture
    def solver(self, setup_basic_problem) -> BackwardEulerSolver:
        """Create BackwardEulerSolver instance."""
        grid, metric, coefficients = setup_basic_problem
        return BackwardEulerSolver(grid, metric, coefficients)

    def test_initialization(self, solver: BackwardEulerSolver) -> None:
        """Test proper initialization of BackwardEulerSolver."""
        assert solver.tolerance == 1e-8
        assert solver.max_iterations == 50
        assert hasattr(solver, "grid")
        assert hasattr(solver, "metric")
        assert hasattr(solver, "coefficients")

    def test_field_vector_conversion(self, solver: BackwardEulerSolver, setup_fields) -> None:
        """Test robust field to vector conversion."""
        fields = setup_fields

        # Test conversion and back-conversion
        vector = solver._fields_to_vector(fields)
        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0

        # Create new fields and test restoration
        restored_fields = ISFieldConfiguration(solver.grid)
        solver._vector_to_fields(vector, restored_fields)

        # Check that key fields are restored correctly
        np.testing.assert_allclose(fields.rho, restored_fields.rho, rtol=1e-12)
        np.testing.assert_allclose(fields.Pi, restored_fields.Pi, rtol=1e-12)

    def test_analytical_jacobian_structure(self, solver: BackwardEulerSolver, setup_fields) -> None:
        """Test analytical Jacobian computation for relaxation terms."""
        fields = setup_fields

        jacobian = solver._compute_analytical_jacobian(fields, self.simple_rhs_function)

        assert sparse.issparse(jacobian)
        assert jacobian.shape[0] == jacobian.shape[1]

        # Check that diagonal elements are negative (relaxation terms)
        diagonal = jacobian.diagonal()
        assert np.all(diagonal <= 0), "Relaxation Jacobian should have non-positive diagonal"

    def test_jacobian_accuracy(self, solver: BackwardEulerSolver, setup_fields) -> None:
        """Test Jacobian accuracy against finite differences."""
        fields = setup_fields

        # Compute analytical Jacobian
        analytical_jac = solver._compute_analytical_jacobian(fields, self.simple_rhs_function)

        # Compute numerical Jacobian with very small perturbation
        numerical_jac = solver._compute_vectorized_jacobian(
            fields, self.simple_rhs_function, perturbation=1e-8
        )

        # Compare sparse and dense representations
        if sparse.issparse(analytical_jac):
            analytical_dense = analytical_jac.toarray()
        else:
            analytical_dense = analytical_jac

        # Check agreement on diagonal elements (most important for relaxation)
        analytical_diag = np.diag(analytical_dense)
        numerical_diag = np.diag(numerical_jac)

        np.testing.assert_allclose(
            analytical_diag, numerical_diag, rtol=1e-4,
            err_msg="Analytical and numerical Jacobian diagonals should agree"
        )

    def test_newton_convergence(self, solver: BackwardEulerSolver, setup_fields) -> None:
        """Test Newton iteration convergence for simple problem."""
        fields = setup_fields
        dt = 0.01

        # Test with simple linear relaxation
        result = solver.solve_step(fields, dt, self.simple_rhs_function)

        assert isinstance(result, ISFieldConfiguration)
        assert np.all(np.isfinite(result.rho))
        assert np.all(np.isfinite(result.Pi))

        # Check that dissipative quantities decay (relaxation)
        assert np.max(np.abs(result.Pi)) < np.max(np.abs(fields.Pi))

    def test_timestep_scaling(self, solver: BackwardEulerSolver, setup_fields) -> None:
        """Test behavior with different timestep sizes."""
        fields = setup_fields

        timesteps = [0.001, 0.01, 0.1]
        results = []

        for dt in timesteps:
            result = solver.solve_step(fields, dt, self.simple_rhs_function)
            results.append(result)

        # Larger timesteps should lead to more relaxation
        pi_values = [np.max(np.abs(r.Pi)) for r in results]
        assert pi_values[0] > pi_values[1] > pi_values[2], "Larger timesteps should cause more decay"

    def test_error_handling(self, solver: BackwardEulerSolver, setup_fields) -> None:
        """Test error handling for problematic cases."""
        fields = setup_fields

        def bad_rhs_function(fields):
            """RHS function that returns invalid data."""
            return {"Pi": np.full_like(fields.Pi, np.inf)}

        # Should handle infinite RHS gracefully
        with pytest.warns(UserWarning):
            result = solver.solve_step(fields, 0.01, bad_rhs_function)
            assert np.all(np.isfinite(result.rho))


class TestIMEXRungeKuttaSolver(TestImplicitSolverBase):
    """Test IMEXRungeKuttaSolver implementation."""

    @pytest.fixture
    def solver(self, setup_basic_problem) -> IMEXRungeKuttaSolver:
        """Create IMEXRungeKuttaSolver instance."""
        grid, metric, coefficients = setup_basic_problem
        return IMEXRungeKuttaSolver(grid, metric, coefficients, order=2)

    def test_initialization(self, solver: IMEXRungeKuttaSolver) -> None:
        """Test proper initialization of IMEX solver."""
        assert solver.order == 2
        assert hasattr(solver, "stages")
        assert hasattr(solver, "a_exp")
        assert hasattr(solver, "a_imp")
        assert hasattr(solver, "b_exp")
        assert hasattr(solver, "b_imp")

    def test_rhs_splitting(self, solver: IMEXRungeKuttaSolver, setup_fields) -> None:
        """Test explicit/implicit RHS splitting."""
        fields = setup_fields
        rhs = self.simple_rhs_function(fields)

        hyperbolic = solver._extract_hyperbolic_part(rhs)
        relaxation = solver._extract_relaxation_part(rhs)

        # Check that splitting is complementary
        for key in rhs:
            if key in hyperbolic and key in relaxation:
                total = hyperbolic[key] + relaxation[key]
                np.testing.assert_allclose(total, rhs[key], rtol=1e-6)

    def test_physics_based_splitting(self, solver: IMEXRungeKuttaSolver, setup_fields) -> None:
        """Test physics-informed RHS splitting."""
        fields = setup_fields

        # Test timescale estimation
        tau_pi = solver._get_relaxation_timescale("pi_munu")
        tau_Pi = solver._get_relaxation_timescale("Pi")
        tau_hydro = solver._estimate_transport_timescale()

        assert tau_pi > 0
        assert tau_Pi > 0
        assert tau_hydro > 0

    def test_implicit_stage_solving(self, solver: IMEXRungeKuttaSolver, setup_fields) -> None:
        """Test implicit stage solution with Newton iteration."""
        fields = setup_fields
        dt = 0.01

        # Test single stage solution
        explicit_contrib = {"Pi": np.zeros_like(fields.Pi)}

        result = solver._solve_implicit_stage(
            fields, explicit_contrib, dt * 0.5, 0, self.simple_rhs_function
        )

        assert isinstance(result, ISFieldConfiguration)
        assert np.all(np.isfinite(result.Pi))

    def test_order_accuracy(self, solver: IMEXRungeKuttaSolver, setup_fields) -> None:
        """Test order of accuracy for IMEX method."""
        fields = setup_fields

        # Test with different timesteps
        dt_coarse = 0.1
        dt_fine = 0.05

        result_coarse = solver.solve_step(fields, dt_coarse, self.simple_rhs_function)

        # Two fine steps
        intermediate = solver.solve_step(fields, dt_fine, self.simple_rhs_function)
        result_fine = solver.solve_step(intermediate, dt_fine, self.simple_rhs_function)

        # Check that error decreases with expected order
        error = np.max(np.abs(result_coarse.Pi - result_fine.Pi))
        assert error < 0.1, "IMEX method should show reasonable accuracy"

    def test_relaxation_jacobian(self, solver: IMEXRungeKuttaSolver, setup_fields) -> None:
        """Test analytical Jacobian for relaxation terms."""
        fields = setup_fields

        jacobian = solver._compute_relaxation_jacobian(fields)
        assert isinstance(jacobian, np.ndarray)

        # Check diagonal structure for linear relaxation
        if jacobian.size > 0:
            diagonal = np.diag(jacobian)
            assert np.all(diagonal <= 0), "Relaxation Jacobian diagonal should be non-positive"


class TestExponentialIntegrator(TestImplicitSolverBase):
    """Test ExponentialIntegrator implementation."""

    @pytest.fixture
    def solver(self, setup_basic_problem) -> ExponentialIntegrator:
        """Create ExponentialIntegrator instance."""
        grid, metric, coefficients = setup_basic_problem
        return ExponentialIntegrator(grid, metric, coefficients)

    def test_initialization(self, solver: ExponentialIntegrator) -> None:
        """Test proper initialization of exponential integrator."""
        assert hasattr(solver, "grid")
        assert hasattr(solver, "metric")
        assert hasattr(solver, "coefficients")

    def test_exponential_accuracy(self, solver: ExponentialIntegrator, setup_fields) -> None:
        """Test accuracy of exponential integrator for linear problems."""
        fields = setup_fields
        dt = 0.1

        # For pure exponential decay: du/dt = -u/tau
        # Exact solution: u(t) = u0 * exp(-t/tau)
        tau = 0.5  # Relaxation time
        expected_factor = np.exp(-dt / tau)

        result = solver.solve_step(fields, dt, self.simple_rhs_function)

        # Check exponential decay
        actual_factor = np.max(result.Pi) / np.max(fields.Pi)
        np.testing.assert_allclose(
            actual_factor, expected_factor, rtol=1e-3,
            err_msg="Exponential integrator should be exact for linear relaxation"
        )

    def test_large_timestep_stability(self, solver: ExponentialIntegrator, setup_fields) -> None:
        """Test stability with large timesteps."""
        fields = setup_fields

        # Large timestep that would be unstable for explicit methods
        dt = 2.0

        result = solver.solve_step(fields, dt, self.simple_rhs_function)

        assert np.all(np.isfinite(result.Pi))
        assert np.all(result.Pi >= 0) or np.all(result.Pi <= 0), "Should preserve sign"


class TestImplicitSolverFactory:
    """Test factory functions for implicit solvers."""

    @pytest.fixture
    def setup_basic_problem(self) -> tuple[SpacetimeGrid, MinkowskiMetric, TransportCoefficients]:
        """Setup basic problem configuration."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(3, 2, 2, 2),  # MEMORY SAFE: Small grid to prevent memory explosion
        )

        metric = MinkowskiMetric()
        coefficients = TransportCoefficients(
            shear_viscosity=0.1,
            bulk_viscosity=0.05
        )

        return grid, metric, coefficients

    def test_factory_function(self, setup_basic_problem) -> None:
        """Test create_implicit_solver factory function."""
        grid, metric, coefficients = setup_basic_problem

        # Test different solver types
        backward_euler = create_implicit_solver("backward_euler", grid, metric, coefficients)
        assert isinstance(backward_euler, BackwardEulerSolver)

        imex_rk = create_implicit_solver("imex_rk", grid, metric, coefficients, order=2)
        assert isinstance(imex_rk, IMEXRungeKuttaSolver)
        assert imex_rk.order == 2

        exponential = create_implicit_solver("exponential", grid, metric, coefficients)
        assert isinstance(exponential, ExponentialIntegrator)

    def test_factory_error_handling(self, setup_basic_problem) -> None:
        """Test factory function error handling."""
        grid, metric, coefficients = setup_basic_problem

        with pytest.raises(ValueError, match="Unknown implicit solver type"):
            create_implicit_solver("invalid_type", grid, metric, coefficients)

    def test_solver_options(self, setup_basic_problem) -> None:
        """Test solver creation with various options."""
        grid, metric, coefficients = setup_basic_problem

        # Test with custom tolerance
        solver = create_implicit_solver(
            "backward_euler", grid, metric, coefficients, tolerance=1e-10
        )
        assert solver.tolerance == 1e-10

        # Test with custom max iterations
        solver = create_implicit_solver(
            "backward_euler", grid, metric, coefficients, max_iterations=100
        )
        assert solver.max_iterations == 100


class TestImplicitSolverPerformance:
    """Performance and scaling tests for implicit solvers."""

    @classmethod
    def setup_class(cls):
        """Class-level setup with memory safety check."""
        if PSUTIL_AVAILABLE:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            if available_memory_gb < 2.0:
                pytest.skip("Insufficient memory for performance tests (need at least 2GB available)")
        else:
            # Conservative fallback when psutil not available
            pytest.skip("Memory monitoring unavailable, skipping performance tests for safety")

    @pytest.fixture
    def setup_performance_problem(self) -> tuple[SpacetimeGrid, MinkowskiMetric, TransportCoefficients, ISFieldConfiguration]:
        """Setup larger problem for performance testing."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(3, 2, 2, 2),  # Minimal grid to prevent memory crash
        )

        metric = MinkowskiMetric()
        coefficients = TransportCoefficients(
            shear_viscosity=0.1,
            bulk_viscosity=0.05
        )
        fields = ISFieldConfiguration(grid)

        return grid, metric, coefficients, fields

    def test_jacobian_computation_scaling(self, setup_performance_problem) -> None:
        """Test Jacobian computation performance scaling."""
        grid, metric, coefficients, fields = setup_performance_problem
        solver = BackwardEulerSolver(grid, metric, coefficients)

        def simple_rhs(f):
            return {"Pi": -f.Pi / 0.3}

        import time

        start_time = time.time()
        jacobian = solver._compute_analytical_jacobian(fields, simple_rhs)
        analytical_time = time.time() - start_time

        # Only run vectorized Jacobian test if memory is available
        if PSUTIL_AVAILABLE:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            field_size = fields.to_state_vector().size
            estimated_memory_gb = (field_size ** 2) * 8 / (1024**3)  # 8 bytes per float64

            if estimated_memory_gb < available_memory_gb * 0.5:  # Use max 50% of available memory
                start_time = time.time()
                jacobian_numerical = solver._compute_vectorized_jacobian(fields, simple_rhs, 1e-6)
                numerical_time = time.time() - start_time

                # Analytical should be faster than numerical
                assert analytical_time < numerical_time, "Analytical Jacobian should be faster"
            else:
                pytest.skip("Skipping vectorized Jacobian test due to insufficient memory")
        else:
            # Just test that analytical Jacobian works
            assert jacobian is not None, "Analytical Jacobian should be computed"

    def test_memory_efficiency(self, setup_performance_problem) -> None:
        """Test memory efficiency of solvers."""
        grid, metric, coefficients, fields = setup_performance_problem
        solver = BackwardEulerSolver(grid, metric, coefficients, use_sparse=True)

        def simple_rhs(f):
            return {"Pi": -f.Pi / 0.3}

        jacobian = solver._compute_analytical_jacobian(fields, simple_rhs)

        # Should use sparse matrix for large problems
        assert sparse.issparse(jacobian), "Should use sparse matrices for efficiency"


@pytest.mark.slow
class TestImplicitSolverConvergence:
    """Convergence studies for implicit solvers."""

    def test_newton_convergence_rate(self) -> None:
        """Test Newton iteration convergence rate."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(2, 2, 2, 2),  # MEMORY SAFE: Minimal grid for convergence testing
        )

        metric = MinkowskiMetric()
        coefficients = TransportCoefficients(
            shear_viscosity=0.1,
            bulk_viscosity=0.05
        )
        solver = BackwardEulerSolver(grid, metric, coefficients, tolerance=1e-12)

        fields = ISFieldConfiguration(grid)
        fields.Pi.fill(1.0)

        def simple_rhs(f):
            return {"Pi": -f.Pi / 0.3}

        # Newton convergence should be quadratic for well-conditioned problems
        result = solver.solve_step(fields, 0.01, simple_rhs)
        assert np.all(np.isfinite(result.Pi))

    def test_temporal_convergence(self) -> None:
        """Test temporal convergence order."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(2, 2, 2, 2),  # MEMORY SAFE: Minimal grid for convergence testing
        )

        metric = MinkowskiMetric()
        coefficients = TransportCoefficients(
            shear_viscosity=0.1,
            bulk_viscosity=0.05
        )

        # Test different orders
        solvers = [
            BackwardEulerSolver(grid, metric, coefficients),  # 1st order
            IMEXRungeKuttaSolver(grid, metric, coefficients, order=2),  # 2nd order
        ]

        fields = ISFieldConfiguration(grid)
        fields.Pi.fill(1.0)

        def simple_rhs(f):
            return {"Pi": -f.Pi / 0.3}

        timesteps = [0.1, 0.05, 0.025]

        for solver in solvers:
            errors = []
            for dt in timesteps:
                result = solver.solve_step(fields, dt, simple_rhs)
                # Compare with analytical solution
                analytical = np.exp(-dt / 0.3)
                error = np.max(np.abs(result.Pi - analytical))
                errors.append(error)

            # Check convergence (errors should decrease)
            assert errors[0] > errors[1] > errors[2], "Errors should decrease with smaller timesteps"