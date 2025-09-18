"""
Bjorken flow benchmark for Israel-Stewart hydrodynamics validation.

This module implements the canonical 1+1D boost-invariant expansion benchmark
with exact analytical solutions for both ideal and viscous Israel-Stewart cases.
It provides rigorous validation of numerical implementations against known solutions.
"""

import warnings
from typing import Any, Optional, Union

import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize

from ..core.constants import HBAR, KBOLTZ
from ..core.fields import ISFieldConfiguration, TransportCoefficients
from ..core.metrics import MilneMetric
from ..core.performance import monitor_performance
from ..core.spacetime_grid import SpacetimeGrid
from ..equations.relaxation import ISRelaxationEquations


class BjorkenFlowSolution:
    """
    Analytical solutions for Bjorken flow in Israel-Stewart hydrodynamics.

    Implements exact solutions for 1+1D boost-invariant expansion with
    viscous corrections and relaxation dynamics.
    """

    def __init__(
        self,
        initial_temperature: float = 0.3,  # GeV
        initial_time: float = 0.6,  # fm/c
        equation_of_state: str = "ideal",
        degrees_of_freedom: float = 37.5,  # QGP DOF
    ):
        """
        Initialize Bjorken flow analytical solution.

        Args:
            initial_temperature: Initial temperature T_0 in GeV
            initial_time: Initial time tau_0 in fm/c
            equation_of_state: EOS type ('ideal', 'realistic')
            degrees_of_freedom: Effective degrees of freedom g*
        """
        self.T0 = initial_temperature
        self.tau0 = initial_time
        self.eos = equation_of_state
        self.g_eff = degrees_of_freedom

        # Thermodynamic constants for ideal gas
        self.a = (np.pi**2 / 90) * self.g_eff  # Energy density coefficient
        self.b = (2 * np.pi**2 / 90) * self.g_eff  # Entropy density coefficient

        # Initial conditions
        self.epsilon0 = self.a * self.T0**4  # Initial energy density
        self.s0 = self.b * self.T0**3  # Initial entropy density

    @monitor_performance("bjorken_ideal_solution")
    def ideal_solution(self, tau: float | np.ndarray) -> dict[str, np.ndarray]:
        """
        Compute ideal Bjorken flow solution.

        For ideal hydrodynamics with adiabatic expansion:
        T(τ) = T₀ * (τ₀/τ)^(1/3)
        ρ(τ) = ρ₀ * (τ₀/τ)^(4/3)
        """
        tau_array = np.atleast_1d(tau)

        # Temperature evolution
        temperature = self.T0 * (self.tau0 / tau_array) ** (1 / 3)

        # Energy density evolution
        energy_density = self.epsilon0 * (self.tau0 / tau_array) ** (4 / 3)

        # Pressure from ideal gas EOS
        pressure = energy_density / 3.0

        # Entropy density
        entropy_density = self.s0 * (self.tau0 / tau_array)

        # Four-velocity (purely temporal in Milne coordinates)
        u_tau = np.ones_like(tau_array)
        u_eta = np.zeros_like(tau_array)

        return {
            "time": tau_array,
            "temperature": temperature,
            "energy_density": energy_density,
            "pressure": pressure,
            "entropy_density": entropy_density,
            "u_tau": u_tau,
            "u_eta": u_eta,
            "expansion_rate": -1.0 / tau_array,  # θ = -1/τ for Bjorken flow
        }

    @monitor_performance("bjorken_first_order_solution")
    def first_order_viscous_solution(
        self,
        tau: float | np.ndarray,
        shear_viscosity: float,
        bulk_viscosity: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """
        Compute first-order viscous Bjorken flow solution.

        Includes Navier-Stokes corrections:
        Π = -ζ * θ = ζ/τ
        π^μν = (4η/3τ) * (dT/dτ) * (1/T)
        """
        # Get ideal solution as baseline
        ideal = self.ideal_solution(tau)
        tau_array = ideal["time"]

        # First-order viscous corrections
        expansion_rate = ideal["expansion_rate"]

        # Bulk pressure: Π = -ζ * θ
        bulk_pressure = -bulk_viscosity * expansion_rate

        # Shear stress: π^μν ∇⟨μ u^ν⟩
        # For Bjorken flow: π^ητ = (4η/3τ) * (1/T) * (dT/dτ)
        temperature = ideal["temperature"]
        dT_dtau = -(1 / 3) * self.T0 * (self.tau0) ** (1 / 3) * tau_array ** (-4 / 3)

        shear_stress = (4 * shear_viscosity / (3 * tau_array)) * (dT_dtau / temperature)

        # Energy density with viscous corrections
        energy_density_corrected = ideal["energy_density"] - bulk_pressure

        return {
            **ideal,
            "bulk_pressure": bulk_pressure,
            "shear_stress": shear_stress,
            "energy_density_corrected": energy_density_corrected,
            "viscous_contribution": -bulk_pressure,
        }

    @monitor_performance("bjorken_second_order_solution")
    def israel_stewart_solution(
        self,
        tau: float | np.ndarray,
        coefficients: TransportCoefficients,
        numerical_method: str = "odeint",
    ) -> dict[str, np.ndarray]:
        """
        Compute second-order Israel-Stewart Bjorken flow solution.

        Solves the coupled ODEs for temperature and dissipative fluxes:
        dT/dτ + T/(3τ) = -Π/(3ρτ)
        dΠ/dτ + Π/τ_Π = -θ/τ - λ_ΠΠ Π²/(ρτ)
        """
        tau_array = np.atleast_1d(tau)
        tau_min, tau_max = float(np.min(tau_array)), float(np.max(tau_array))

        # Extract coefficients
        eta = coefficients.shear_viscosity or 0.0
        zeta = coefficients.bulk_viscosity or 0.0
        tau_pi = coefficients.shear_relaxation_time or 0.1
        tau_Pi = coefficients.bulk_relaxation_time or 0.1

        # Second-order coefficients
        lambda_PiPi = getattr(coefficients, "lambda_Pi_Pi", 0.0)
        xi_1 = getattr(coefficients, "xi_1", 0.0)

        def ode_system(y: np.ndarray, tau_val: float) -> np.ndarray:
            """ODE system for Israel-Stewart Bjorken flow. Variables: y = [T, Pi, pi^eta_tau]"""
            T, Pi, pi_tau_eta = y

            # Thermodynamic quantities
            epsilon = self.a * T**4
            pressure = epsilon / 3.0

            # Temperature evolution with viscous corrections
            dT_dtau = -(T / (3 * tau_val)) - Pi / (3 * epsilon * tau_val)

            # Bulk pressure evolution
            theta = -1.0 / tau_val  # Expansion rate
            source_Pi = -zeta * theta
            nonlinear_Pi = -lambda_PiPi * Pi**2 / (epsilon * tau_val) if lambda_PiPi > 0 else 0.0
            dPi_dtau = -Pi / tau_Pi + source_Pi + nonlinear_Pi

            # Shear stress evolution (simplified for 1+1D)
            # In full 3+1D, this would be more complex
            shear_rate = (4.0 / (3 * tau_val)) * (dT_dtau / T)
            source_pi = eta * shear_rate
            dpi_dtau = -pi_tau_eta / tau_pi + source_pi

            return np.array([dT_dtau, dPi_dtau, dpi_dtau])

        # Initial conditions
        T_init = self.T0
        Pi_init = 0.0  # Start with no bulk pressure
        pi_init = 0.0  # Start with no shear stress

        y0 = np.array([T_init, Pi_init, pi_init])

        # Solve ODE system
        if numerical_method == "odeint":
            tau_dense = np.linspace(self.tau0, tau_max, 1000)
            solution = integrate.odeint(ode_system, y0, tau_dense, rtol=1e-8, atol=1e-10)

            # Interpolate to requested time points
            T_interp = np.interp(tau_array, tau_dense, solution[:, 0])
            Pi_interp = np.interp(tau_array, tau_dense, solution[:, 1])
            pi_interp = np.interp(tau_array, tau_dense, solution[:, 2])

        elif numerical_method == "solve_ivp":
            sol = integrate.solve_ivp(
                lambda t, y: ode_system(y, t),
                [self.tau0, tau_max],
                y0,
                t_eval=tau_array,
                method="DOP853",
                rtol=1e-8,
                atol=1e-10,
            )

            if not sol.success:
                warnings.warn("ODE integration failed", UserWarning, stacklevel=2)

            T_interp = sol.y[0]
            Pi_interp = sol.y[1]
            pi_interp = sol.y[2]

        else:
            raise ValueError(f"Unknown numerical method: {numerical_method}")

        # Compute derived quantities
        energy_density = self.a * T_interp**4
        pressure = energy_density / 3.0
        entropy_density = self.b * T_interp**3

        # Four-velocity (Milne coordinates)
        u_tau = np.ones_like(tau_array)
        u_eta = np.zeros_like(tau_array)

        return {
            "time": tau_array,
            "temperature": T_interp,
            "energy_density": energy_density,
            "pressure": pressure,
            "entropy_density": entropy_density,
            "bulk_pressure": Pi_interp,
            "shear_stress": pi_interp,
            "u_tau": u_tau,
            "u_eta": u_eta,
            "expansion_rate": -1.0 / tau_array,
        }

    def effective_temperature(
        self,
        tau: float | np.ndarray,
        total_energy_density: np.ndarray,
    ) -> np.ndarray:
        """
        Compute effective temperature from total energy density.

        T_eff = (rho_total / a)^(1/4)
        """
        tau_array = np.atleast_1d(tau)
        return (total_energy_density / self.a) ** (1 / 4)

    def compute_observables(
        self,
        solution: dict[str, np.ndarray],
        coefficients: TransportCoefficients | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Compute physical observables from Bjorken flow solution.

        Includes temperature, particle multiplicity, elliptic flow, etc.
        """
        tau = solution["time"]
        T = solution["temperature"]
        s = solution["entropy_density"]

        # Particle multiplicity (assuming pions)
        # dN/dy ∝ s at freeze-out
        tau_fo = 5.0  # fm/c (typical freeze-out time)
        T_fo = 0.14  # GeV (freeze-out temperature)

        # Interpolate to freeze-out
        if np.min(T) <= T_fo <= np.max(T):
            tau_fo_actual = np.interp(T_fo, T[::-1], tau[::-1])
            s_fo = np.interp(T_fo, T[::-1], s[::-1])
            multiplicity = tau_fo_actual * s_fo
        else:
            multiplicity = np.nan

        # Momentum anisotropy (for viscous case)
        if "shear_stress" in solution:
            pi_tau_eta = solution["shear_stress"]
            epsilon = solution["energy_density"]
            # Simplified anisotropy measure
            momentum_anisotropy = pi_tau_eta / (epsilon + 1e-12)
        else:
            momentum_anisotropy = np.zeros_like(tau)

        # Energy loss rate
        energy_loss_rate = np.gradient(solution["energy_density"], tau)

        return {
            "multiplicity": multiplicity,
            "momentum_anisotropy": momentum_anisotropy,
            "energy_loss_rate": energy_loss_rate,
            "freeze_out_time": tau_fo_actual if not np.isnan(multiplicity) else np.nan,
        }


class BjorkenBenchmark:
    """
    Comprehensive benchmark suite for Bjorken flow validation.

    Compares numerical implementations against analytical solutions
    and provides convergence analysis and performance metrics.
    """

    def __init__(
        self,
        grid: SpacetimeGrid,
        coefficients: TransportCoefficients,
        analytical_solution: BjorkenFlowSolution,
    ):
        """
        Initialize Bjorken flow benchmark.

        Args:
            grid: Spacetime grid for numerical simulation
            coefficients: Transport coefficients
            analytical_solution: Analytical solution for comparison
        """
        self.grid = grid
        self.coefficients = coefficients
        self.analytical = analytical_solution

        # Numerical simulation setup
        self.metric = MilneMetric()
        self.relaxation_eq = ISRelaxationEquations(grid, self.metric, coefficients)

        # Results storage
        self.results: dict[str, Any] = {}

    @monitor_performance("bjorken_numerical_simulation")
    def run_numerical_simulation(
        self,
        final_time: float = 10.0,
        timestep: float = 0.01,
        solver_method: str = "explicit",
    ) -> dict[str, np.ndarray]:
        """
        Run numerical Bjorken flow simulation.

        Args:
            final_time: Final simulation time in fm/c
            timestep: Integration timestep
            solver_method: Numerical method ('explicit', 'implicit')

        Returns:
            Dictionary with numerical solution
        """
        # Initialize fields with Bjorken flow conditions
        fields = ISFieldConfiguration(self.grid)
        self._setup_bjorken_initial_conditions(fields)

        # Time evolution
        time_points: list[float] = []
        solutions: dict[str, list[float]] = {
            "temperature": [],
            "energy_density": [],
            "pressure": [],
            "bulk_pressure": [],
            "shear_stress": [],
        }

        current_time = self.analytical.tau0
        time_points.append(current_time)

        # Record initial conditions
        self._record_solution_state(fields, solutions)

        # Time integration loop
        while current_time < final_time:
            dt = min(timestep, final_time - current_time)

            # Evolve fields
            if solver_method == "explicit":
                self.relaxation_eq.evolve_relaxation(fields, dt, method="explicit")
            elif solver_method == "implicit":
                self.relaxation_eq.evolve_relaxation(fields, dt, method="implicit")
            else:
                raise ValueError(f"Unknown solver method: {solver_method}")

            current_time += dt
            time_points.append(current_time)
            self._record_solution_state(fields, solutions)

        # Convert lists to arrays
        result = {
            "time": np.array(time_points),
        }
        for key, values in solutions.items():
            result[key] = np.array(values)

        return result

    def _setup_bjorken_initial_conditions(self, fields: ISFieldConfiguration) -> None:
        """Setup initial conditions for Bjorken flow."""
        # Get ideal solution at initial time
        ideal_ic = self.analytical.ideal_solution(self.analytical.tau0)

        # Set thermodynamic quantities
        fields.temperature.fill(ideal_ic["temperature"][0])
        fields.rho.fill(ideal_ic["energy_density"][0])
        fields.pressure.fill(ideal_ic["pressure"][0])

        # Set four-velocity (boost-invariant)
        fields.u_mu[..., 0] = 1.0  # u^τ = 1
        fields.u_mu[..., 1] = 0.0  # u^η = 0
        fields.u_mu[..., 2] = 0.0  # u^x = 0
        fields.u_mu[..., 3] = 0.0  # u^y = 0

        # Initialize dissipative fluxes to zero
        fields.Pi.fill(0.0)
        fields.pi_munu.fill(0.0)
        if hasattr(fields, "q_mu"):
            fields.q_mu.fill(0.0)

    def _record_solution_state(
        self, fields: ISFieldConfiguration, solutions: dict[str, list[float]]
    ) -> None:
        """Record current solution state."""
        # Take spatial average (for Bjorken flow, fields should be uniform)
        solutions["temperature"].append(float(np.mean(fields.temperature)))
        solutions["energy_density"].append(float(np.mean(fields.rho)))
        solutions["pressure"].append(float(np.mean(fields.pressure)))
        solutions["bulk_pressure"].append(float(np.mean(fields.Pi)))

        # Shear stress (take trace norm for scalar measure)
        pi_norm = np.sqrt(np.mean(fields.pi_munu**2))
        solutions["shear_stress"].append(float(pi_norm))

    @monitor_performance("bjorken_comparison")
    def compare_solutions(
        self,
        numerical_result: dict[str, np.ndarray],
        analytical_type: str = "israel_stewart",
        relative_tolerance: float = 0.05,
    ) -> dict[str, Any]:
        """
        Compare numerical and analytical solutions.

        Args:
            numerical_result: Numerical simulation results
            analytical_type: Type of analytical solution to compare against
            relative_tolerance: Tolerance for "good" agreement

        Returns:
            Comparison metrics and analysis
        """
        # Get analytical solution at numerical time points
        if analytical_type == "ideal":
            analytical_result = self.analytical.ideal_solution(numerical_result["time"])
        elif analytical_type == "first_order":
            eta = self.coefficients.shear_viscosity or 0.0
            zeta = self.coefficients.bulk_viscosity or 0.0
            analytical_result = self.analytical.first_order_viscous_solution(
                numerical_result["time"], eta, zeta
            )
        elif analytical_type == "israel_stewart":
            analytical_result = self.analytical.israel_stewart_solution(
                numerical_result["time"], self.coefficients
            )
        else:
            raise ValueError(f"Unknown analytical type: {analytical_type}")

        # Compute errors for each quantity
        errors = {}
        for key in ["temperature", "energy_density", "pressure"]:
            if key in numerical_result and key in analytical_result:
                numerical = numerical_result[key]
                analytical = analytical_result[key]

                # Relative error
                rel_error = np.abs(numerical - analytical) / (np.abs(analytical) + 1e-12)
                errors[key] = {
                    "max_relative_error": np.max(rel_error),
                    "mean_relative_error": np.mean(rel_error),
                    "rms_relative_error": np.sqrt(np.mean(rel_error**2)),
                    "passes_tolerance": np.max(rel_error) < relative_tolerance,
                }

        # Special handling for dissipative quantities
        if "bulk_pressure" in numerical_result and "bulk_pressure" in analytical_result:
            num_Pi = numerical_result["bulk_pressure"]
            ana_Pi = analytical_result["bulk_pressure"]

            # Use absolute error for small quantities
            abs_error = np.abs(num_Pi - ana_Pi)
            characteristic_scale = np.max(np.abs(ana_Pi)) + 1e-6

            errors["bulk_pressure"] = {
                "max_absolute_error": np.max(abs_error),
                "scaled_error": np.max(abs_error) / characteristic_scale,
                "passes_tolerance": np.max(abs_error) / characteristic_scale < relative_tolerance,
            }

        # Overall assessment
        all_pass = all(err_data.get("passes_tolerance", True) for err_data in errors.values())

        comparison_result = {
            "errors": errors,
            "overall_agreement": "good" if all_pass else "poor",
            "analytical_type": analytical_type,
            "relative_tolerance": relative_tolerance,
            "comparison_time_points": len(numerical_result["time"]),
        }

        return comparison_result

    @monitor_performance("bjorken_convergence_study")
    def convergence_study(
        self,
        timesteps: list[float],
        final_time: float = 5.0,
        reference_solution: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """
        Perform convergence study with different timesteps.

        Args:
            timesteps: List of timesteps to test
            final_time: Final simulation time
            reference_solution: High-accuracy reference solution

        Returns:
            Convergence analysis results
        """
        if reference_solution is None:
            # Use analytical solution as reference
            time_ref = np.linspace(self.analytical.tau0, final_time, 1000)
            reference_solution = self.analytical.israel_stewart_solution(
                time_ref, self.coefficients
            )

        convergence_data = []

        for dt in timesteps:
            # Run simulation with this timestep
            numerical_result = self.run_numerical_simulation(
                final_time=final_time, timestep=dt, solver_method="explicit"
            )

            # Interpolate to reference time points for comparison
            ref_time = reference_solution["time"]
            interpolated_result = {}

            for key in ["temperature", "energy_density", "pressure"]:
                if key in numerical_result:
                    interpolated_result[key] = np.interp(
                        ref_time, numerical_result["time"], numerical_result[key]
                    )

            # Compute errors
            errors = {}
            for key, values in interpolated_result.items():
                if key in reference_solution:
                    ref_values = reference_solution[key]
                    rel_error = np.abs(values - ref_values) / (np.abs(ref_values) + 1e-12)
                    errors[key] = np.max(rel_error)

            convergence_data.append(
                {
                    "timestep": dt,
                    "errors": errors,
                    "max_error": max(errors.values()) if errors else np.inf,
                }
            )

        # Estimate convergence order
        convergence_order = self._estimate_convergence_order(convergence_data)

        return {
            "convergence_data": convergence_data,
            "convergence_order": convergence_order,
            "is_converging": convergence_order > 0.5,  # At least first-order
        }

    def _estimate_convergence_order(self, convergence_data: list[dict[str, Any]]) -> float:
        """Estimate convergence order from error data."""
        if len(convergence_data) < 2:
            return 0.0

        # Use temperature error for convergence order estimation
        timesteps = [data["timestep"] for data in convergence_data]
        errors = [data["errors"].get("temperature", np.inf) for data in convergence_data]

        # Filter out infinite errors
        valid_indices = [i for i, err in enumerate(errors) if np.isfinite(err) and err > 0]

        if len(valid_indices) < 2:
            return 0.0

        # Fit log(error) vs log(timestep)
        log_dt = np.log([timesteps[i] for i in valid_indices])
        log_err = np.log([errors[i] for i in valid_indices])

        # Linear regression
        if len(log_dt) >= 2:
            convergence_order = np.polyfit(log_dt, log_err, 1)[0]
            return float(abs(convergence_order))
        else:
            return 0.0

    def run_full_benchmark(
        self,
        timestep: float = 0.01,
        final_time: float = 10.0,
        convergence_timesteps: list[float] | None = None,
    ) -> dict[str, Any]:
        """
        Run complete Bjorken flow benchmark suite.

        Args:
            timestep: Default timestep for main simulation
            final_time: Final simulation time
            convergence_timesteps: Timesteps for convergence study

        Returns:
            Complete benchmark results
        """
        if convergence_timesteps is None:
            convergence_timesteps = [0.02, 0.01, 0.005, 0.002]

        # Main numerical simulation
        numerical_result = self.run_numerical_simulation(
            final_time=final_time, timestep=timestep, solver_method="explicit"
        )

        # Compare against different analytical solutions
        ideal_comparison = self.compare_solutions(numerical_result, "ideal")
        is_comparison = self.compare_solutions(numerical_result, "israel_stewart")

        # Convergence study
        convergence_results = self.convergence_study(convergence_timesteps, final_time=final_time)

        # Performance metrics
        performance_stats = {
            "simulation_timestep": timestep,
            "total_time_points": len(numerical_result["time"]),
            "final_time": final_time,
        }

        # Overall benchmark assessment
        benchmark_result = {
            "numerical_solution": numerical_result,
            "ideal_comparison": ideal_comparison,
            "israel_stewart_comparison": is_comparison,
            "convergence_study": convergence_results,
            "performance": performance_stats,
            "benchmark_passed": (
                is_comparison["overall_agreement"] == "good"
                and convergence_results["is_converging"]
            ),
        }

        # Store results
        self.results = benchmark_result

        return benchmark_result

    def generate_report(self) -> str:
        """Generate human-readable benchmark report."""
        if not self.results:
            return "No benchmark results available. Run benchmark first."

        report = []
        report.append("=" * 60)
        report.append("BJORKEN FLOW BENCHMARK REPORT")
        report.append("=" * 60)

        # Overall status
        passed = self.results.get("benchmark_passed", False)
        status = "PASSED" if passed else "FAILED"
        report.append(f"Overall Status: {status}")
        report.append("")

        # Israel-Stewart comparison
        is_comp = self.results.get("israel_stewart_comparison", {})
        report.append("Israel-Stewart Comparison:")
        for quantity, error_data in is_comp.get("errors", {}).items():
            max_err = error_data.get("max_relative_error", 0.0)
            passes = error_data.get("passes_tolerance", False)
            report.append(f"  {quantity}: {max_err:.3e} ({'PASS' if passes else 'FAIL'})")
        report.append("")

        # Convergence study
        conv_study = self.results.get("convergence_study", {})
        conv_order = conv_study.get("convergence_order", 0.0)
        is_conv = conv_study.get("is_converging", False)
        report.append(f"Convergence Order: {conv_order:.2f} ({'PASS' if is_conv else 'FAIL'})")
        report.append("")

        # Performance
        perf = self.results.get("performance", {})
        report.append(f"Simulation Points: {perf.get('total_time_points', 0)}")
        report.append(f"Final Time: {perf.get('final_time', 0.0):.2f} fm/c")

        return "\n".join(report)


# Utility functions for setting up benchmarks
def create_standard_bjorken_benchmark(
    tau0: float = 0.6,
    T0: float = 0.3,
    eta_over_s: float = 0.08,
    grid_points: tuple[int, int, int, int] = (8, 4, 4, 4),
) -> BjorkenBenchmark:
    """
    Create a standard Bjorken flow benchmark setup.

    Args:
        tau0: Initial time in fm/c
        T0: Initial temperature in GeV
        eta_over_s: Shear viscosity to entropy ratio
        grid_points: Grid resolution

    Returns:
        Configured Bjorken benchmark
    """
    # Create grid (only tau direction matters for Bjorken flow)
    grid = SpacetimeGrid(
        coordinate_system="milne",
        time_range=(tau0, 10.0),
        spatial_ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        grid_points=grid_points,
    )

    # Transport coefficients
    s0 = (2 * np.pi**2 / 90) * 37.5 * T0**3  # Initial entropy density
    eta = eta_over_s * s0  # Shear viscosity
    tau_pi = 5 * eta / (s0 * T0)  # Relaxation time

    coefficients = TransportCoefficients(
        shear_viscosity=eta,
        bulk_viscosity=0.0,  # Start with zero bulk viscosity
        shear_relaxation_time=tau_pi,
        bulk_relaxation_time=0.1,
    )

    # Analytical solution
    analytical = BjorkenFlowSolution(
        initial_temperature=T0,
        initial_time=tau0,
        equation_of_state="ideal",
    )

    return BjorkenBenchmark(grid, coefficients, analytical)
