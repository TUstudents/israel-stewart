"""
Diffusion Flow Benchmark for Landau Frame Validation.

This benchmark tests particle diffusion driven by chemical potential gradients,
validating the Landau frame formulation where V^μ (particle diffusion current)
replaces q^μ (heat flux).

Physical Setup:
    - Background: Uniform radiation fluid at rest
    - Perturbation: Sinusoidal chemical potential δμ(x,t)
    - Diffusion: V^i = -D Δ^ij ∂_j(μ/T) (Fick's law)
    - Evolution: ∂_t n + ∇·V = 0 (particle conservation)

Key Tests:
    1. Landau frame constraint: V^μ u_μ = 0 maintained throughout
    2. Fick's law: V^i matches -D ∇^i(μ/T)
    3. Diffusion equation: n(x,t) = n₀ + δn₀ exp(-Dk²t) sin(kx)
    4. Particle conservation: ∫n d³x constant

Reference:
    - Landau-Lifshitz Vol. 6 (Fluid Mechanics)
    - IReD paper: Wagner, Palermo, Ambrus (2022)

Usage:
    >>> benchmark = create_standard_diffusion_benchmark()
    >>> result = benchmark.run_numerical_simulation(final_time=1.0)
    >>> benchmark.validate_landau_frame_constraint(result)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core import ISFieldConfiguration, TransportCoefficients
from ..core.spacegrid import SpaceGrid
from ..equations.ired_simple import HardSphereIReD
from ..solvers.spectral import SpectralISHydrodynamics
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class AnalyticalDiffusionSolution:
    """
    Analytical solution for diffusion flow.

    For small perturbations around equilibrium:
        n(x,t) = n₀ + δn₀ exp(-Dk²t) sin(kx)
        V^x(x,t) = -D ∂_x(μ/T) ≈ -D k δμ₀ exp(-Dk²t) cos(kx)

    where:
        - n: particle number density
        - μ: chemical potential
        - D: diffusion coefficient
        - k: wave number

    Attributes:
        temperature: Background temperature T₀ (GeV)
        particle_density_0: Background particle density n₀ (GeV³)
        diffusion_coefficient: Diffusion coefficient D (GeV²)
        perturbation_amplitude: δn₀/n₀ (dimensionless)
        wave_number: Wave vector magnitude k (GeV)
    """

    temperature: float
    particle_density_0: float
    diffusion_coefficient: float
    perturbation_amplitude: float
    wave_number: float

    def particle_density(self, x: NDArray[np.floating], t: float) -> NDArray[np.floating]:
        """
        Particle number density n(x,t).

        Args:
            x: Spatial coordinate (GeV⁻¹)
            t: Time (GeV⁻¹)

        Returns:
            n(x,t) in GeV³
        """
        k = self.wave_number
        D = self.diffusion_coefficient
        delta_n0 = self.perturbation_amplitude * self.particle_density_0

        damping = np.exp(-D * k**2 * t)
        return self.particle_density_0 + delta_n0 * damping * np.sin(k * x)

    def diffusion_current(self, x: NDArray[np.floating], t: float) -> NDArray[np.floating]:
        """
        Diffusion current V^x(x,t) from Fick's law.

        V^x = -D ∂_x(μ/T)

        For radiation fluid with small perturbations:
            μ/T ∝ n/T³ → ∂_x(μ/T) ∝ ∂_x n

        Args:
            x: Spatial coordinate (GeV⁻¹)
            t: Time (GeV⁻¹)

        Returns:
            V^x(x,t) in GeV²
        """
        k = self.wave_number
        D = self.diffusion_coefficient
        delta_n0 = self.perturbation_amplitude * self.particle_density_0

        damping = np.exp(-D * k**2 * t)
        # V^x = -D ∂_x n (for small perturbations)
        return -D * delta_n0 * damping * k * np.cos(k * x)

    def chemical_potential(self, x: NDArray[np.floating], t: float) -> NDArray[np.floating]:
        """
        Chemical potential μ(x,t).

        For massless particles (radiation): μ/T ∝ n/T³

        Args:
            x: Spatial coordinate (GeV⁻¹)
            t: Time (GeV⁻¹)

        Returns:
            μ(x,t) in GeV
        """
        # For radiation fluid: μ = T * (n/n_eq)^(1/3) in leading order
        # Here we use linearized: μ ≈ (T/3) * (δn/n₀)
        n = self.particle_density(x, t)
        delta_n = n - self.particle_density_0
        return (self.temperature / 3.0) * (delta_n / self.particle_density_0)

    def energy_density(self, x: NDArray[np.floating], t: float) -> NDArray[np.floating]:
        """
        Energy density ε(x,t).

        For isentropic diffusion: ε stays constant to leading order.

        Args:
            x: Spatial coordinate (GeV⁻¹)
            t: Time (GeV⁻¹)

        Returns:
            ε(x,t) in GeV⁴
        """
        # For radiation: ε = (π²/30) T⁴ (constant for isentropic flow)
        return (np.pi**2 / 30.0) * self.temperature**4

    def pressure(self, x: NDArray[np.floating], t: float) -> NDArray[np.floating]:
        """
        Pressure P(x,t).

        For radiation: P = ε/3

        Args:
            x: Spatial coordinate (GeV⁻¹)
            t: Time (GeV⁻¹)

        Returns:
            P(x,t) in GeV⁴
        """
        return self.energy_density(x, t) / 3.0

    def damping_rate(self) -> float:
        """
        Diffusion damping rate Γ = D k².

        Returns:
            Γ in GeV
        """
        return self.diffusion_coefficient * self.wave_number**2


class DiffusionBenchmark:
    """
    Benchmark for testing particle diffusion in Landau frame.

    This validates:
        1. V^μ evolution from relaxation equation
        2. Landau frame constraint V^μ u_μ = 0
        3. Fick's law V^i = -D ∇^i(μ/T)
        4. Particle conservation ∂_t n + ∇·V = 0

    Attributes:
        grid: Spatial grid
        analytical: Analytical diffusion solution
        coefficients: Transport coefficients
        initial_fields: Initial field configuration
    """

    def __init__(
        self,
        grid: SpaceGrid,
        analytical: AnalyticalDiffusionSolution,
        coefficients: TransportCoefficients,
    ):
        """
        Initialize diffusion benchmark.

        Args:
            grid: Spatial grid (must be periodic)
            analytical: Analytical solution
            coefficients: Transport coefficients
        """
        self.grid = grid
        self.analytical = analytical
        self.coefficients = coefficients

        # Validate grid
        if grid.boundary_conditions != "periodic":
            logger.warning(
                f"Diffusion benchmark expects periodic boundaries, got {grid.boundary_conditions}"
            )

        # Initialize fields
        self.initial_fields = self._setup_initial_fields()

        # Create solver (compatible with NumericalSoundWaveBenchmark API)
        self.fields = self.initial_fields  # Alias for consistency with other benchmarks
        self.solver = SpectralISHydrodynamics(self.grid, self.fields, self.coefficients)

    def _setup_initial_fields(self) -> ISFieldConfiguration:
        """
        Set up initial conditions at t=0.

        Returns:
            Initial field configuration
        """
        fields = ISFieldConfiguration(self.grid)
        X, Y, Z = self.grid.meshgrid()

        # Background state (uniform radiation fluid at rest)
        rho_0 = self.analytical.energy_density(X, 0.0)
        p_0 = self.analytical.pressure(X, 0.0)
        n_0 = self.analytical.particle_density(X, 0.0)

        fields.rho[:] = rho_0
        fields.pressure[:] = p_0
        fields.n[:] = n_0  # Initialize particle density!

        # Four-velocity: Rest frame u^μ = (1, 0, 0, 0)
        fields.u_mu[..., 0] = 1.0
        fields.u_mu[..., 1:] = 0.0

        # Diffusion current V^x from initial chemical potential gradient
        # V^x(x,0) = -D ∂_x(μ/T)|_{t=0}
        V_x_initial = self.analytical.diffusion_current(X, 0.0)
        fields.V_mu[..., 1] = V_x_initial  # V^x component
        fields.V_mu[..., 0] = 0.0  # V^0 = 0 (Landau frame constraint)
        fields.V_mu[..., 2:] = 0.0  # V^y = V^z = 0

        # Bulk and shear start at zero (equilibrium)
        fields.Pi[:] = 0.0
        fields.pi_munu[:] = 0.0

        logger.info(
            f"Initialized diffusion benchmark:\n"
            f"  T = {self.analytical.temperature:.4f} GeV\n"
            f"  n₀ = {self.analytical.particle_density_0:.4e} GeV³\n"
            f"  k = {self.analytical.wave_number:.4f} GeV\n"
            f"  D = {self.analytical.diffusion_coefficient:.4e} GeV²\n"
            f"  Damping rate Γ = {self.analytical.damping_rate():.4e} GeV\n"
            f"  Perturbation amplitude = {self.analytical.perturbation_amplitude:.2%}"
        )

        return fields

    def run_numerical_simulation(
        self,
        final_time: float,
        timestep: float | None = None,
        snapshot_interval: float | None = None,
    ) -> dict[str, Any]:
        """
        Run numerical simulation of diffusion flow.

        Args:
            final_time: Final time (GeV⁻¹)
            timestep: Time step (if None, auto-determined)
            snapshot_interval: Time between snapshots (if None, uses 10 snapshots)

        Returns:
            Dictionary with simulation results
        """
        # Use existing solver
        solver = self.solver

        # Determine timestep if not provided
        if timestep is None:
            # CFL condition: dt < dx / c_s
            dx_min = self.grid.dx  # Uniform grid spacing
            c_s = 1.0 / np.sqrt(3.0)  # Sound speed for radiation
            timestep = 0.1 * dx_min / c_s
            logger.info(f"Auto-determined timestep: dt = {timestep:.4e} GeV⁻¹")

        # Determine snapshot interval
        if snapshot_interval is None:
            snapshot_interval = final_time / 10

        # Storage for snapshots
        times = []
        particle_densities = []
        diffusion_currents_x = []
        energy_densities = []
        constraint_violations = []

        # Record initial state
        times.append(0.0)
        particle_densities.append(self._compute_particle_density(solver.fields))
        diffusion_currents_x.append(solver.fields.V_mu[..., 1].copy())
        energy_densities.append(solver.fields.rho.copy())
        constraint_violations.append(self._check_landau_frame_constraint(solver.fields))

        # Track next snapshot time
        next_snapshot = snapshot_interval

        # Callback function to record snapshots
        def record_snapshot(t: float, fields: ISFieldConfiguration) -> None:
            nonlocal next_snapshot
            if t >= next_snapshot - timestep / 2:
                times.append(t)
                particle_densities.append(self._compute_particle_density(fields))
                diffusion_currents_x.append(fields.V_mu[..., 1].copy())
                energy_densities.append(fields.rho.copy())
                constraint_violations.append(self._check_landau_frame_constraint(fields))
                next_snapshot += snapshot_interval

        logger.info(f"Running diffusion simulation: t ∈ [0, {final_time}], dt = {timestep}")

        # Run evolution with callback
        solver.evolve(t_final=final_time, dt=timestep, callback=record_snapshot)

        logger.info(f"Simulation complete: {len(times)} snapshots")

        return {
            "time": np.array(times),
            "particle_density": np.array(particle_densities),
            "diffusion_current_x": np.array(diffusion_currents_x),
            "energy_density": np.array(energy_densities),
            "constraint_violation": np.array(constraint_violations),
        }

    def _compute_particle_density(self, fields: ISFieldConfiguration) -> NDArray[np.floating]:
        """
        Compute particle density n from thermodynamic state.

        For radiation fluid: n = (ζ(3)/π²) T³ where ζ(3) ≈ 1.202

        Args:
            fields: Field configuration

        Returns:
            Particle density (GeV³)
        """
        # Extract temperature from energy density: ε = (π²/30) T⁴
        T = (30.0 * fields.rho / np.pi**2) ** 0.25
        return (1.202 / np.pi**2) * T**3

    def _check_landau_frame_constraint(self, fields: ISFieldConfiguration) -> NDArray[np.floating]:
        """
        Check Landau frame constraint V^μ u_μ = 0.

        Args:
            fields: Field configuration

        Returns:
            Constraint violation |V^μ u_μ|
        """
        # V^μ u_μ with Minkowski metric (-,+,+,+)
        constraint = (
            -fields.V_mu[..., 0] * fields.u_mu[..., 0]
            + fields.V_mu[..., 1] * fields.u_mu[..., 1]
            + fields.V_mu[..., 2] * fields.u_mu[..., 2]
            + fields.V_mu[..., 3] * fields.u_mu[..., 3]
        )
        return np.abs(constraint)

    def validate_fick_law(
        self, result: dict[str, Any], time_index: int = -1, tolerance: float = 0.1
    ) -> bool:
        """
        Validate that diffusion current follows Fick's law.

        V^x = -D ∂_x(μ/T)

        Args:
            result: Simulation result dictionary
            time_index: Time index to check (-1 for final time)
            tolerance: Relative error tolerance

        Returns:
            True if Fick's law is satisfied
        """
        t = result["time"][time_index]
        X, _, _ = self.grid.meshgrid()

        # Numerical diffusion current
        V_x_numerical = result["diffusion_current_x"][time_index]

        # Analytical from Fick's law
        V_x_analytical = self.analytical.diffusion_current(X, t)

        # Compute relative error
        relative_error = np.abs(V_x_numerical - V_x_analytical) / np.max(np.abs(V_x_analytical))
        max_error = np.max(relative_error)

        logger.info(
            f"Fick's law validation at t={t:.4f}:\n"
            f"  Max relative error: {max_error:.2%}\n"
            f"  Tolerance: {tolerance:.2%}\n"
            f"  Status: {'PASS' if max_error < tolerance else 'FAIL'}"
        )

        return max_error < tolerance

    def validate_landau_frame_constraint(
        self, result: dict[str, Any], tolerance: float = 1e-10
    ) -> bool:
        """
        Validate that Landau frame constraint V^μ u_μ = 0 is maintained.

        Args:
            result: Simulation result dictionary
            tolerance: Absolute error tolerance

        Returns:
            True if constraint is satisfied at all times
        """
        violations = result["constraint_violation"]
        max_violation = np.max(violations)

        logger.info(
            f"Landau frame constraint validation:\n"
            f"  Max |V^μ u_μ|: {max_violation:.2e}\n"
            f"  Tolerance: {tolerance:.2e}\n"
            f"  Status: {'PASS' if max_violation < tolerance else 'FAIL'}"
        )

        return max_violation < tolerance

    def validate_particle_conservation(
        self, result: dict[str, Any], tolerance: float = 1e-6
    ) -> bool:
        """
        Validate global particle conservation ∫n d³x = const.

        Args:
            result: Simulation result dictionary
            tolerance: Relative change tolerance

        Returns:
            True if particle number is conserved
        """
        particle_densities = result["particle_density"]
        times = result["time"]

        # Integrate over volume (cell_volume = dx³ for uniform 3D grid)
        cell_volume = self.grid.dx**3
        total_particles = np.array([np.sum(n) * cell_volume for n in particle_densities])

        # Check relative change
        initial_particles = total_particles[0]
        relative_change = np.abs(total_particles - initial_particles) / initial_particles
        max_change = np.max(relative_change)

        logger.info(
            f"Particle conservation validation:\n"
            f"  Initial: {initial_particles:.4e}\n"
            f"  Final: {total_particles[-1]:.4e}\n"
            f"  Max relative change: {max_change:.2%}\n"
            f"  Tolerance: {tolerance:.2%}\n"
            f"  Status: {'PASS' if max_change < tolerance else 'FAIL'}"
        )

        return max_change < tolerance


def create_standard_diffusion_benchmark(
    temperature: float = 0.4,
    perturbation_amplitude: float = 0.01,
    wave_number: float = 1.0,
    diffusion_coefficient: float = 0.1,
    diffusion_relaxation_time: float = 0.5,
    domain_size: float = 2 * np.pi,
    grid_points: tuple[int, int, int] = (64, 64, 16),
) -> DiffusionBenchmark:
    """
    Create standard diffusion benchmark with phenomenological coefficients.

    Args:
        temperature: Background temperature T₀ (GeV)
        perturbation_amplitude: δn₀/n₀ (dimensionless)
        wave_number: Wave vector magnitude k (GeV)
        diffusion_coefficient: Diffusion coefficient D (GeV²)
        diffusion_relaxation_time: Relaxation time τ_V (GeV⁻¹)
        domain_size: Domain size L (GeV⁻¹)
        grid_points: Grid resolution (nx, ny, nz)

    Returns:
        DiffusionBenchmark instance
    """
    # Create spatial grid (periodic for spectral methods)
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, domain_size)] * 3,
        grid_points=grid_points,
        boundary_conditions="periodic",
    )

    # Compute background particle density for radiation fluid
    # n = (ζ(3)/π²) T³ where ζ(3) ≈ 1.202
    particle_density_0 = (1.202 / np.pi**2) * temperature**3

    # Create analytical solution
    analytical = AnalyticalDiffusionSolution(
        temperature=temperature,
        particle_density_0=particle_density_0,
        diffusion_coefficient=diffusion_coefficient,
        perturbation_amplitude=perturbation_amplitude,
        wave_number=wave_number,
    )

    # Transport coefficients (minimal for diffusion test)
    coefficients = TransportCoefficients(
        shear_viscosity=0.05,  # Small but non-zero
        bulk_viscosity=0.0,  # Zero for conformal
        diffusion_coefficient=diffusion_coefficient,
        shear_relaxation_time=0.5,
        bulk_relaxation_time=0.3,
        diffusion_relaxation_time=diffusion_relaxation_time,
    )

    return DiffusionBenchmark(grid, analytical, coefficients)


def create_diffusion_benchmark_with_ired(
    temperature: float = 0.4,
    cross_section: float = 1.0,
    truncation: str = "41",
    perturbation_amplitude: float = 0.01,
    wave_number: float = 1.0,
    domain_size: float = 2 * np.pi,
    grid_points: tuple[int, int, int] = (64, 64, 16),
) -> tuple[DiffusionBenchmark, HardSphereIReD]:
    """
    Create diffusion benchmark with IReD transport coefficients.

    This uses quantitatively accurate coefficients from kinetic theory
    (Wagner et al. 2022) instead of phenomenological values.

    Args:
        temperature: Background temperature T₀ (GeV)
        cross_section: Hard sphere cross-section σ (fm²)
        truncation: IReD moment truncation ('14', '23', '32', '41')
        perturbation_amplitude: δn₀/n₀ (dimensionless)
        wave_number: Wave vector magnitude k (GeV)
        domain_size: Domain size L (GeV⁻¹)
        grid_points: Grid resolution (nx, ny, nz)

    Returns:
        Tuple of (DiffusionBenchmark, HardSphereIReD)
    """
    # Create IReD transport coefficient model
    ired_model = HardSphereIReD(
        temperature=temperature, cross_section=cross_section, truncation=truncation
    )

    # Create spatial grid
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, domain_size)] * 3,
        grid_points=grid_points,
        boundary_conditions="periodic",
    )

    # Particle density for radiation fluid
    particle_density_0 = (1.202 / np.pi**2) * temperature**3

    # Create analytical solution with IReD diffusion coefficient
    analytical = AnalyticalDiffusionSolution(
        temperature=temperature,
        particle_density_0=particle_density_0,
        diffusion_coefficient=ired_model.diffusion_coefficient(),
        perturbation_amplitude=perturbation_amplitude,
        wave_number=wave_number,
    )

    # Extract IReD transport coefficients
    coefficients = TransportCoefficients(
        shear_viscosity=ired_model.shear_viscosity(),
        bulk_viscosity=ired_model.bulk_viscosity(),  # Zero for conformal
        diffusion_coefficient=ired_model.diffusion_coefficient(),
        shear_relaxation_time=ired_model.shear_relaxation_time(),
        bulk_relaxation_time=ired_model.bulk_relaxation_time(),
        diffusion_relaxation_time=ired_model.diffusion_relaxation_time(),
        # Second-order IReD coefficients
        tau_pi_pi=ired_model.tau_pi_pi(),
        lambda_pi_V=ired_model.lambda_pi_V(),
        lambda_V_pi=ired_model.lambda_V_pi(),
    )

    logger.info(
        f"Created diffusion benchmark with IReD coefficients:\n"
        f"  D = {ired_model.diffusion_coefficient():.4e} GeV²\n"
        f"  τ_V = {ired_model.diffusion_relaxation_time():.4f} fm/c\n"
        f"  Damping rate Γ = Dk² = {analytical.damping_rate():.4e} GeV"
    )

    return DiffusionBenchmark(grid, analytical, coefficients), ired_model


if __name__ == "__main__":
    # Example: Run standard diffusion benchmark
    print("Diffusion Flow Benchmark\n" + "=" * 60)

    benchmark = create_standard_diffusion_benchmark(
        temperature=0.4,
        perturbation_amplitude=0.05,
        wave_number=1.0,
        grid_points=(32, 32, 16),
    )

    result = benchmark.run_numerical_simulation(final_time=1.0, timestep=0.01)

    # Validate
    print("\n" + "=" * 60)
    print("Validation Results:")
    print("=" * 60)
    fick_ok = benchmark.validate_fick_law(result, tolerance=0.2)
    constraint_ok = benchmark.validate_landau_frame_constraint(result, tolerance=1e-8)
    conservation_ok = benchmark.validate_particle_conservation(result, tolerance=0.01)

    print(f"\nFick's Law: {'✓ PASS' if fick_ok else '✗ FAIL'}")
    print(f"Landau Frame Constraint: {'✓ PASS' if constraint_ok else '✗ FAIL'}")
    print(f"Particle Conservation: {'✓ PASS' if conservation_ok else '✗ FAIL'}")
