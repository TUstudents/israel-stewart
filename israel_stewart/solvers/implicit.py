"""
Implicit time integration methods for Israel-Stewart hydrodynamics.

This module implements robust implicit solvers specifically designed for the stiff
relaxation equations in Israel-Stewart second-order viscous hydrodynamics.
Key features include Newton-Krylov methods, IMEX schemes, and adaptive timestep control.
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

from ..core.fields import ISFieldConfiguration, TransportCoefficients
from ..core.metrics import MetricBase
from ..core.performance import monitor_performance
from ..core.spacetime_grid import SpacetimeGrid


class ImplicitSolverBase(ABC):
    """
    Abstract base class for implicit time integration methods.

    Defines the interface for solving stiff systems of the form:
    du/dt = f(u, t) where f contains both hyperbolic and relaxation terms.
    """

    def __init__(
        self,
        grid: SpacetimeGrid,
        metric: MetricBase,
        coefficients: TransportCoefficients,
        tolerance: float = 1e-8,
        max_iterations: int = 50,
    ):
        """
        Initialize implicit solver.

        Args:
            grid: Spacetime grid for discretization
            metric: Spacetime metric for covariant operations
            coefficients: Transport coefficients
            tolerance: Convergence tolerance for nonlinear iterations
            max_iterations: Maximum number of nonlinear iterations
        """
        self.grid = grid
        self.metric = metric
        self.coefficients = coefficients
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        # Performance monitoring
        self.iteration_counts: list[int] = []
        self.convergence_history: list[float] = []

    @abstractmethod
    def solve_step(
        self,
        fields: ISFieldConfiguration,
        dt: float,
        rhs_func: Callable[[ISFieldConfiguration], dict[str, np.ndarray]],
    ) -> ISFieldConfiguration:
        """
        Solve one implicit time step.

        Args:
            fields: Current field configuration
            dt: Time step size
            rhs_func: Function computing right-hand side

        Returns:
            Updated field configuration
        """
        pass

    @abstractmethod
    def compute_jacobian(
        self,
        fields: ISFieldConfiguration,
        rhs_func: Callable[[ISFieldConfiguration], dict[str, np.ndarray]],
        perturbation: float = 1e-8,
    ) -> np.ndarray | sparse.spmatrix:
        """
        Compute Jacobian matrix for Newton iteration.

        Args:
            fields: Current field configuration
            rhs_func: Right-hand side function
            perturbation: Finite difference step size

        Returns:
            Jacobian matrix (dense or sparse)
        """
        pass

    def estimate_stiffness_ratio(
        self,
        fields: ISFieldConfiguration,
        rhs_func: Callable[[ISFieldConfiguration], dict[str, np.ndarray]],
    ) -> float:
        """
        Estimate stiffness ratio for adaptive timestep control.

        The stiffness ratio is defined as the ratio of the largest to smallest
        characteristic timescales in the system.
        """
        # Compute Jacobian eigenvalues to estimate stiffness
        jacobian = self.compute_jacobian(fields, rhs_func)

        if sparse.issparse(jacobian):
            # For sparse matrices, compute a few eigenvalues
            try:
                eigenvals = spla.eigs(
                    jacobian, k=min(10, jacobian.shape[0] - 1), return_eigenvectors=False
                )
            except (spla.ArpackNoConvergence, spla.ArpackError):
                # Fallback to condition number estimate
                return 1e6
        else:
            eigenvals = la.eigvals(jacobian)

        # Filter out near-zero eigenvalues and compute stiffness ratio
        real_parts = np.real(eigenvals)
        positive_eigs = real_parts[real_parts > 1e-12]

        if len(positive_eigs) < 2:
            return 1.0

        stiffness_ratio = np.max(positive_eigs) / np.min(positive_eigs)
        return float(stiffness_ratio)

    def recommend_timestep(
        self,
        fields: ISFieldConfiguration,
        rhs_func: Callable[[ISFieldConfiguration], dict[str, np.ndarray]],
        target_ratio: float = 10.0,
    ) -> float:
        """
        Recommend optimal timestep based on stiffness analysis.

        Args:
            fields: Current field configuration
            rhs_func: Right-hand side function
            target_ratio: Target stiffness ratio for timestep selection

        Returns:
            Recommended timestep
        """
        stiffness = self.estimate_stiffness_ratio(fields, rhs_func)

        # Conservative timestep for stiff systems
        if stiffness > target_ratio:
            # For very stiff systems, use relaxation timescales
            tau_pi = self.coefficients.shear_relaxation_time or 0.1
            tau_Pi = self.coefficients.bulk_relaxation_time or 0.1
            min_relaxation_time = min(tau_pi, tau_Pi)
            recommended_dt = 0.1 * min_relaxation_time
        else:
            # For mildly stiff systems, use CFL-like condition
            recommended_dt = 2.0 / np.sqrt(stiffness) if stiffness > 0 else 0.01

        return float(recommended_dt)

    def get_solver_statistics(self) -> dict[str, Any]:
        """Get performance statistics for the solver."""
        if not self.iteration_counts:
            return {"status": "no_steps_taken"}

        return {
            "total_steps": len(self.iteration_counts),
            "avg_iterations": np.mean(self.iteration_counts),
            "max_iterations": np.max(self.iteration_counts),
            "final_residual": self.convergence_history[-1] if self.convergence_history else None,
            "convergence_rate": float(
                np.mean(np.diff(np.log10(np.array(self.convergence_history) + 1e-16)))
            ),
        }


class BackwardEulerSolver(ImplicitSolverBase):
    """
    First-order backward Euler implicit solver.

    Solves: u^{n+1} = u^n + dt * f(u^{n+1}, t^{n+1})
    using Newton-Krylov iteration for the nonlinear system.
    """

    def __init__(
        self,
        grid: SpacetimeGrid,
        metric: MetricBase,
        coefficients: TransportCoefficients,
        tolerance: float = 1e-8,
        max_iterations: int = 50,
        use_sparse: bool = True,
        preconditioner: str | None = "ilu",
    ):
        """
        Initialize backward Euler solver.

        Args:
            grid: Spacetime grid
            metric: Spacetime metric
            coefficients: Transport coefficients
            tolerance: Newton iteration tolerance
            max_iterations: Maximum Newton iterations
            use_sparse: Whether to use sparse linear algebra
            preconditioner: Preconditioning method ('ilu', 'jacobi', None)
        """
        super().__init__(grid, metric, coefficients, tolerance, max_iterations)
        self.use_sparse = use_sparse
        self.preconditioner = preconditioner

    @monitor_performance("backward_euler_step")
    def solve_step(
        self,
        fields: ISFieldConfiguration,
        dt: float,
        rhs_func: Callable[[ISFieldConfiguration], dict[str, np.ndarray]],
    ) -> ISFieldConfiguration:
        """
        Solve one backward Euler step using Newton iteration.

        The nonlinear system is: G(u^{n+1}) = u^{n+1} - u^n - dt * f(u^{n+1}) = 0
        """
        # Create working copy of fields
        new_fields = fields.copy()

        # Store initial state
        initial_state = self._fields_to_vector(fields)

        # Newton iteration
        for iteration in range(self.max_iterations):
            # Compute residual: G(u) = u - u^n - dt * f(u)
            current_rhs = rhs_func(new_fields)
            current_vector = self._fields_to_vector(new_fields)

            residual = current_vector - initial_state - dt * self._rhs_to_vector(current_rhs)
            residual_norm = np.linalg.norm(residual)

            self.convergence_history.append(float(residual_norm))

            # Check convergence
            if residual_norm < self.tolerance:
                self.iteration_counts.append(iteration + 1)
                return new_fields

            # Compute Jacobian: J = I - dt * df/du
            jacobian = self._compute_newton_jacobian(new_fields, rhs_func, dt)

            # Solve linear system: J * delta = -residual
            try:
                if self.use_sparse and sparse.issparse(jacobian):
                    delta = self._solve_sparse_linear_system(jacobian, -residual)
                else:
                    delta = la.solve(jacobian, -residual)
            except (la.LinAlgError, spla.LinAlgError) as e:
                warnings.warn(f"Linear solver failed at iteration {iteration}: {e}", UserWarning)
                break

            # Update solution
            new_vector = current_vector + delta
            self._vector_to_fields(new_vector, new_fields)

        # If we reach here, Newton iteration failed to converge
        warnings.warn(
            f"Newton iteration failed to converge in {self.max_iterations} iterations. "
            f"Final residual: {residual_norm:.2e}",
            UserWarning,
        )
        self.iteration_counts.append(self.max_iterations)
        return new_fields

    def _compute_newton_jacobian(
        self,
        fields: ISFieldConfiguration,
        rhs_func: Callable[[ISFieldConfiguration], dict[str, np.ndarray]],
        dt: float,
    ) -> np.ndarray | sparse.spmatrix:
        """Compute Newton Jacobian: J = I - dt * df/du."""
        # Get system size
        system_vector = self._fields_to_vector(fields)
        n = len(system_vector)

        # Compute RHS Jacobian using finite differences
        rhs_jacobian = self.compute_jacobian(fields, rhs_func)

        # Newton Jacobian: J = I - dt * df/du
        if sparse.issparse(rhs_jacobian):
            format_str = getattr(rhs_jacobian, "format", "csr")
            identity = sparse.identity(n, format=format_str)
            jacobian = identity - dt * rhs_jacobian
        else:
            jacobian = np.eye(n) - dt * rhs_jacobian

        return jacobian

    def _solve_sparse_linear_system(self, matrix: sparse.spmatrix, rhs: np.ndarray) -> np.ndarray:
        """Solve sparse linear system with preconditioning."""
        if self.preconditioner == "ilu":
            try:
                # ILU preconditioning
                ilu = spla.spilu(matrix.tocsc())
                preconditioner = spla.LinearOperator(matrix.shape, ilu.solve)
                solution, info = spla.gmres(matrix, rhs, M=preconditioner, tol=self.tolerance)
            except Exception:
                # Fallback to direct solve
                solution = spla.spsolve(matrix, rhs)
                info = 0
        elif self.preconditioner == "jacobi":
            # Jacobi preconditioning
            diag = matrix.diagonal()
            diag[diag == 0] = 1.0  # Avoid division by zero
            preconditioner = spla.LinearOperator(matrix.shape, lambda x: x / diag)
            solution, info = spla.gmres(matrix, rhs, M=preconditioner, tol=self.tolerance)
        else:
            # No preconditioning
            solution, info = spla.gmres(matrix, rhs, tol=self.tolerance)

        if info != 0:
            warnings.warn(f"GMRES failed with info={info}, using direct solve", UserWarning)
            solution = spla.spsolve(matrix, rhs)

        result: np.ndarray = solution
        return result

    def compute_jacobian(
        self,
        fields: ISFieldConfiguration,
        rhs_func: Callable[[ISFieldConfiguration], dict[str, np.ndarray]],
        perturbation: float = 1e-8,
    ) -> np.ndarray | sparse.spmatrix:
        """
        Compute RHS Jacobian matrix using finite differences.

        df/du ~ (f(u + eps*e_i) - f(u)) / eps
        """
        # Get baseline RHS
        baseline_rhs = rhs_func(fields)
        baseline_vector = self._rhs_to_vector(baseline_rhs)
        n = len(baseline_vector)

        # Initialize Jacobian
        if self.use_sparse:
            jacobian = sparse.lil_matrix((n, n))
        else:
            jacobian = np.zeros((n, n))

        # Compute columns using finite differences
        current_vector = self._fields_to_vector(fields)

        for i in range(n):
            # Perturb i-th component
            perturbed_vector = current_vector.copy()
            perturbed_vector[i] += perturbation

            # Create perturbed fields
            perturbed_fields = ISFieldConfiguration(self.grid)
            self._vector_to_fields(perturbed_vector, perturbed_fields)

            # Compute perturbed RHS
            try:
                perturbed_rhs = rhs_func(perturbed_fields)
                perturbed_vector_rhs = self._rhs_to_vector(perturbed_rhs)

                # Finite difference approximation
                column = (perturbed_vector_rhs - baseline_vector) / perturbation

                if self.use_sparse:
                    jacobian[:, i] = column
                else:
                    jacobian[:, i] = column

            except Exception as e:
                warnings.warn(f"Failed to compute Jacobian column {i}: {e}", UserWarning)
                # Leave column as zeros

        if self.use_sparse:
            return jacobian.tocsr()
        else:
            return jacobian

    def _fields_to_vector(self, fields: ISFieldConfiguration) -> np.ndarray:
        """Convert field configuration to flat vector for linear algebra."""
        vectors = []

        # Add scalar fields
        if hasattr(fields, "Pi") and fields.Pi is not None:
            vectors.append(fields.Pi.flatten())

        # Add tensor fields
        if hasattr(fields, "pi_munu") and fields.pi_munu is not None:
            vectors.append(fields.pi_munu.flatten())

        if hasattr(fields, "q_mu") and fields.q_mu is not None:
            vectors.append(fields.q_mu.flatten())

        return np.concatenate(vectors) if vectors else np.array([])

    def _rhs_to_vector(self, rhs: dict[str, np.ndarray]) -> np.ndarray:
        """Convert RHS dictionary to flat vector."""
        vectors = []

        # Maintain same order as _fields_to_vector
        if "Pi" in rhs:
            vectors.append(rhs["Pi"].flatten())
        if "pi_munu" in rhs:
            vectors.append(rhs["pi_munu"].flatten())
        if "q_mu" in rhs:
            vectors.append(rhs["q_mu"].flatten())

        return np.concatenate(vectors) if vectors else np.array([])

    def _vector_to_fields(self, vector: np.ndarray, fields: ISFieldConfiguration) -> None:
        """Convert flat vector back to field configuration."""
        idx = 0

        # Restore scalar fields
        if hasattr(fields, "Pi") and fields.Pi is not None:
            size = fields.Pi.size
            fields.Pi = vector[idx : idx + size].reshape(fields.Pi.shape)
            idx += size

        # Restore tensor fields
        if hasattr(fields, "pi_munu") and fields.pi_munu is not None:
            size = fields.pi_munu.size
            fields.pi_munu = vector[idx : idx + size].reshape(fields.pi_munu.shape)
            idx += size

        if hasattr(fields, "q_mu") and fields.q_mu is not None:
            size = fields.q_mu.size
            fields.q_mu = vector[idx : idx + size].reshape(fields.q_mu.shape)
            idx += size


class IMEXRungeKuttaSolver(ImplicitSolverBase):
    """
    Implicit-Explicit Runge-Kutta solver for Israel-Stewart equations.

    Treats hyperbolic terms explicitly and relaxation terms implicitly.
    This allows larger timesteps while maintaining stability.
    """

    def __init__(
        self,
        grid: SpacetimeGrid,
        metric: MetricBase,
        coefficients: TransportCoefficients,
        order: int = 2,
        tolerance: float = 1e-8,
        max_iterations: int = 20,
    ):
        """
        Initialize IMEX-RK solver.

        Args:
            grid: Spacetime grid
            metric: Spacetime metric
            coefficients: Transport coefficients
            order: Order of accuracy (1, 2, or 3)
            tolerance: Implicit solve tolerance
            max_iterations: Maximum implicit iterations per stage
        """
        super().__init__(grid, metric, coefficients, tolerance, max_iterations)
        self.order = order

        # Set IMEX-RK coefficients based on order
        if order == 1:
            self._setup_imex_euler()
        elif order == 2:
            self._setup_imex_rk2()
        elif order == 3:
            self._setup_imex_rk3()
        else:
            raise ValueError(f"Unsupported IMEX-RK order: {order}")

    def _setup_imex_euler(self) -> None:
        """Setup coefficients for first-order IMEX Euler scheme."""
        self.stages = 1
        self.a_exp = np.array([[0]])  # Explicit coefficients
        self.a_imp = np.array([[1]])  # Implicit coefficients
        self.b_exp = np.array([1])  # Explicit weights
        self.b_imp = np.array([1])  # Implicit weights
        self.c = np.array([1])  # Stage times

    def _setup_imex_rk2(self) -> None:
        """Setup coefficients for second-order IMEX-RK scheme (L-stable)."""
        gamma = (2 - np.sqrt(2)) / 2

        self.stages = 2
        self.a_exp = np.array([[0, 0], [1, 0]])
        self.a_imp = np.array([[gamma, 0], [1 - gamma, gamma]])
        self.b_exp = np.array([1 - gamma, gamma])
        self.b_imp = np.array([1 - gamma, gamma])
        self.c = np.array([gamma, 1])

    def _setup_imex_rk3(self) -> None:
        """Setup coefficients for third-order IMEX-RK scheme."""
        self.stages = 3
        self.a_exp = np.array([[0, 0, 0], [1 / 2, 0, 0], [11 / 18, 1 / 18, 0]])
        self.a_imp = np.array([[1 / 3, 0, 0], [0, 1 / 3, 0], [1 / 4, 0, 1 / 3]])
        self.b_exp = np.array([1 / 4, 0, 3 / 4])
        self.b_imp = np.array([1 / 4, 0, 3 / 4])
        self.c = np.array([1 / 3, 1 / 3, 2 / 3])

    @monitor_performance("imex_rk_step")
    def solve_step(
        self,
        fields: ISFieldConfiguration,
        dt: float,
        rhs_func: Callable[[ISFieldConfiguration], dict[str, np.ndarray]],
    ) -> ISFieldConfiguration:
        """
        Solve one IMEX-RK step.

        The method splits the RHS into explicit (hyperbolic) and implicit (relaxation) parts:
        du/dt = f_E(u) + f_I(u)
        """
        # Initialize stage values
        stage_fields = [ISFieldConfiguration(self.grid) for _ in range(self.stages)]
        stage_fields[0] = fields.copy()

        # Perform each stage
        for i in range(self.stages):
            # Compute explicit contribution
            explicit_sum = self._compute_explicit_sum(stage_fields, i, dt, rhs_func)

            # Set up implicit solve for this stage
            stage_fields[i] = self._solve_implicit_stage(
                fields, explicit_sum, dt * self.a_imp[i, i], i, rhs_func
            )

        # Final update using stage weights
        return self._compute_final_update(fields, stage_fields, dt, rhs_func)

    def _compute_explicit_sum(
        self,
        stage_fields: list[ISFieldConfiguration],
        stage: int,
        dt: float,
        rhs_func: Callable[[ISFieldConfiguration], dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Compute explicit sum for current stage."""
        explicit_sum = {}

        for j in range(stage):
            if self.a_exp[stage, j] != 0:
                stage_rhs = rhs_func(stage_fields[j])
                # Extract explicit (hyperbolic) part - this is simplified
                # In practice, you'd separate hyperbolic and relaxation terms
                explicit_part = self._extract_hyperbolic_part(stage_rhs)

                weight = dt * self.a_exp[stage, j]
                for key, value in explicit_part.items():
                    if key not in explicit_sum:
                        explicit_sum[key] = np.zeros_like(value)
                    explicit_sum[key] += weight * value

        return explicit_sum

    def _solve_implicit_stage(
        self,
        initial_fields: ISFieldConfiguration,
        explicit_contribution: dict[str, np.ndarray],
        implicit_weight: float,
        stage: int,
        rhs_func: Callable[[ISFieldConfiguration], dict[str, np.ndarray]],
    ) -> ISFieldConfiguration:
        """Solve implicit part of stage using simplified Newton iteration."""
        # For now, use a simplified approach - in practice you'd implement
        # a proper Newton solver for the implicit-explicit system
        result = initial_fields.copy()

        # Add explicit contribution
        for key, value in explicit_contribution.items():
            if hasattr(result, key):
                current_value = getattr(result, key)
                if current_value is not None:
                    setattr(result, key, current_value + value)

        # Simple implicit update (placeholder)
        if implicit_weight > 0:
            stage_rhs = rhs_func(result)
            relaxation_part = self._extract_relaxation_part(stage_rhs)

            for key, value in relaxation_part.items():
                if hasattr(result, key):
                    current_value = getattr(result, key)
                    if current_value is not None:
                        setattr(result, key, current_value + implicit_weight * value)

        return result

    def _compute_final_update(
        self,
        initial_fields: ISFieldConfiguration,
        stage_fields: list[ISFieldConfiguration],
        dt: float,
        rhs_func: Callable[[ISFieldConfiguration], dict[str, np.ndarray]],
    ) -> ISFieldConfiguration:
        """Compute final update using stage weights."""
        result = initial_fields.copy()

        # Weighted sum of stage contributions
        for i, stage_field in enumerate(stage_fields):
            if self.b_exp[i] != 0 or self.b_imp[i] != 0:
                stage_rhs = rhs_func(stage_field)

                # Explicit contribution
                if self.b_exp[i] != 0:
                    hyperbolic_part = self._extract_hyperbolic_part(stage_rhs)
                    weight = dt * self.b_exp[i]
                    for key, value in hyperbolic_part.items():
                        if hasattr(result, key):
                            current_value = getattr(result, key)
                            if current_value is not None:
                                setattr(result, key, current_value + weight * value)

                # Implicit contribution
                if self.b_imp[i] != 0:
                    relaxation_part = self._extract_relaxation_part(stage_rhs)
                    weight = dt * self.b_imp[i]
                    for key, value in relaxation_part.items():
                        if hasattr(result, key):
                            current_value = getattr(result, key)
                            if current_value is not None:
                                setattr(result, key, current_value + weight * value)

        return result

    def _extract_hyperbolic_part(self, rhs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Extract hyperbolic (advection) terms from RHS."""
        # Simplified: return spatial derivative terms
        # In practice, this would be more sophisticated
        hyperbolic = {}
        for key, value in rhs.items():
            # Placeholder: assume 50% of RHS is hyperbolic
            hyperbolic[key] = 0.5 * value
        return hyperbolic

    def _extract_relaxation_part(self, rhs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Extract relaxation terms from RHS."""
        # Simplified: return remaining terms
        # In practice, this would extract the relaxation source terms
        relaxation = {}
        for key, value in rhs.items():
            # Placeholder: assume 50% of RHS is relaxation
            relaxation[key] = 0.5 * value
        return relaxation

    def compute_jacobian(
        self,
        fields: ISFieldConfiguration,
        rhs_func: Callable[[ISFieldConfiguration], dict[str, np.ndarray]],
        perturbation: float = 1e-8,
    ) -> np.ndarray | sparse.spmatrix:
        """Compute Jacobian for IMEX method (focusing on implicit part)."""
        # For IMEX, we primarily need the Jacobian of the relaxation terms
        # This is a simplified implementation
        n = 100  # Placeholder size
        return sparse.identity(n, format="csr")


class ExponentialIntegrator(ImplicitSolverBase):
    """
    Exponential time differencing (ETD) methods for relaxation equations.

    Specially designed for equations with linear relaxation operators,
    providing excellent stability properties for stiff systems.
    """

    def __init__(
        self,
        grid: SpacetimeGrid,
        metric: MetricBase,
        coefficients: TransportCoefficients,
        method: str = "etd2",
        tolerance: float = 1e-8,
    ):
        """
        Initialize exponential integrator.

        Args:
            grid: Spacetime grid
            metric: Spacetime metric
            coefficients: Transport coefficients
            method: ETD method ('etd1', 'etd2', 'etd3')
            tolerance: Tolerance for matrix function computations
        """
        super().__init__(grid, metric, coefficients, tolerance, 1)  # Single iteration
        self.method = method

        # Precompute relaxation timescales
        self.tau_pi = coefficients.shear_relaxation_time or 0.1
        self.tau_Pi = coefficients.bulk_relaxation_time or 0.1
        self.tau_q = getattr(coefficients, "heat_relaxation_time", None) or 0.1

    @monitor_performance("etd_step")
    def solve_step(
        self,
        fields: ISFieldConfiguration,
        dt: float,
        rhs_func: Callable[[ISFieldConfiguration], dict[str, np.ndarray]],
    ) -> ISFieldConfiguration:
        """
        Solve one exponential integrator step.

        For linear relaxation: du/dt = -u/tau + N(u)
        The solution is: u(t+dt) = exp(-dt/tau) * u(t) + phi(dt/tau) * N(u)
        """
        result = fields.copy()

        # Compute current RHS
        current_rhs = rhs_func(fields)

        # Apply exponential integration to each relaxation variable
        if hasattr(result, "Pi") and result.Pi is not None and "Pi" in current_rhs:
            result.Pi = self._etd_update(result.Pi, current_rhs["Pi"], dt, self.tau_Pi)

        if hasattr(result, "pi_munu") and result.pi_munu is not None and "pi_munu" in current_rhs:
            result.pi_munu = self._etd_update(
                result.pi_munu, current_rhs["pi_munu"], dt, self.tau_pi
            )

        if hasattr(result, "q_mu") and result.q_mu is not None and "q_mu" in current_rhs:
            result.q_mu = self._etd_update(result.q_mu, current_rhs["q_mu"], dt, self.tau_q)

        return result

    def _etd_update(
        self, current_value: np.ndarray, source_term: np.ndarray, dt: float, tau: float
    ) -> np.ndarray:
        """
        Apply exponential time differencing update.

        For du/dt = -u/tau + S(t), the solution is:
        u(t+dt) = exp(-dt/tau) * u(t) + tau * (1 - exp(-dt/tau)) * S(t)
        """
        # Avoid division by zero
        if tau <= 0:
            tau = 1e-12

        # Compute exponential factors
        alpha = dt / tau
        exp_factor = np.exp(-alpha)

        # phi function for exponential integrator
        if abs(alpha) < 1e-6:
            # Use Taylor expansion for small alpha to avoid numerical issues
            phi = 1.0 - 0.5 * alpha + alpha**2 / 6.0
        else:
            phi = (1.0 - exp_factor) / alpha

        # ETD update formula
        updated_value = exp_factor * current_value + tau * phi * source_term

        result: np.ndarray = updated_value
        return result

    def compute_jacobian(
        self,
        fields: ISFieldConfiguration,
        rhs_func: Callable[[ISFieldConfiguration], dict[str, np.ndarray]],
        perturbation: float = 1e-8,
    ) -> np.ndarray | sparse.spmatrix:
        """
        Compute Jacobian for exponential integrator.

        For linear relaxation terms, the Jacobian is simply -1/tau on the diagonal.
        """
        # Get system size
        total_size = 0
        if hasattr(fields, "Pi") and fields.Pi is not None:
            total_size += fields.Pi.size
        if hasattr(fields, "pi_munu") and fields.pi_munu is not None:
            total_size += fields.pi_munu.size
        if hasattr(fields, "q_mu") and fields.q_mu is not None:
            total_size += fields.q_mu.size

        if total_size == 0:
            return sparse.csr_matrix((0, 0))

        # Build diagonal Jacobian for relaxation terms
        diag_values = []

        if hasattr(fields, "Pi") and fields.Pi is not None:
            diag_values.extend([-1.0 / self.tau_Pi] * fields.Pi.size)
        if hasattr(fields, "pi_munu") and fields.pi_munu is not None:
            diag_values.extend([-1.0 / self.tau_pi] * fields.pi_munu.size)
        if hasattr(fields, "q_mu") and fields.q_mu is not None:
            diag_values.extend([-1.0 / self.tau_q] * fields.q_mu.size)

        return sparse.diags(diag_values, format="csr")


# Factory function for creating solvers
def create_implicit_solver(
    solver_type: str,
    grid: SpacetimeGrid,
    metric: MetricBase,
    coefficients: TransportCoefficients,
    **kwargs: Any,
) -> ImplicitSolverBase:
    """
    Factory function to create implicit solvers.

    Args:
        solver_type: Type of solver ('backward_euler', 'imex_rk', 'exponential')
        grid: Spacetime grid
        metric: Spacetime metric
        coefficients: Transport coefficients
        **kwargs: Additional solver-specific parameters

    Returns:
        Configured implicit solver

    Raises:
        ValueError: If solver_type is not recognized
    """
    if solver_type.lower() == "backward_euler":
        return BackwardEulerSolver(grid, metric, coefficients, **kwargs)
    elif solver_type.lower() == "imex_rk":
        return IMEXRungeKuttaSolver(grid, metric, coefficients, **kwargs)
    elif solver_type.lower() == "exponential":
        return ExponentialIntegrator(grid, metric, coefficients, **kwargs)
    else:
        raise ValueError(
            f"Unknown solver type: {solver_type}. "
            f"Available types: 'backward_euler', 'imex_rk', 'exponential'"
        )
