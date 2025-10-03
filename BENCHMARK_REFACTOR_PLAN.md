# Benchmark Refactor Plan: SpaceGrid + Spectral Solver Integration

## Executive Summary

Update all three benchmark modules (`bjorken_flow.py`, `sound_waves.py`, `equilibration.py`) to use the new pure 3D SpaceGrid architecture with SpectralISHydrodynamics solver, replacing the legacy 4D SpacetimeGrid approach. Create executable benchmark scripts that validate the spectral solver against exact analytical solutions.

**Status:** Phase 1 in progress
**Target Completion:** All phases
**Expected Outcome:** Quantitative validation of spectral solver accuracy against analytical solutions

---

## Current State Analysis

### Existing Benchmarks (Legacy Architecture)

**1. `bjorken_flow.py` (761 lines)**
- **Status:** Uses SpacetimeGrid
- **Analytical Solutions:** Complete (ideal, first-order, Israel-Stewart)
- **Numerical Solver:** BjorkenBenchmark class with manual time evolution
- **Issues:**
  - Uses SpacetimeGrid with 4D indexing
  - Manual time integration loop
  - Not using SpectralISHydrodynamics

**2. `sound_waves.py` (1719 lines)**
- **Status:** Uses SpacetimeGrid
- **Analytical Solutions:** Dispersion relation solver (robust)
- **Numerical Solver:** NumericalSoundWaveBenchmark class
- **Issues:**
  - Uses SpacetimeGrid: `grid_points=(64, 64, 16, 16)` (nt, nx, ny, nz)
  - 4D field initialization: `self.fields.rho[t_idx, ...] = ...`
  - Manual time stepping loop
  - Not leveraging spectral methods properly

**3. `equilibration.py` (962 lines)**
- **Status:** Uses SpacetimeGrid
- **Analytical Solutions:** None (empirical validation)
- **Numerical Solver:** EquilibrationAnalysis class
- **Issues:**
  - Uses SpacetimeGrid for evolution
  - Manual relaxation evolution
  - Not using spectral solver capabilities

### Existing Test Files (Also Legacy)

- `test_bjorken_flow_validation.py` - Uses SpacetimeGrid
- `test_sound_wave_validation.py` - Uses SpacetimeGrid
- `test_equilibration_validation.py` - Uses SpacetimeGrid

### Memory and Performance Impact

**Before (4D with nt=20, nx=ny=nz=32):**
- Grid storage: ~1500 MB
- Field evolution: Growing memory with time steps

**After (Pure 3D with nx=ny=nz=32):**
- Grid storage: ~75 MB (95% reduction)
- Field evolution: Constant memory with streaming snapshots

---

## Architecture Changes

### Core Transformation

**Old Pattern (4D Spacetime):**
```python
# Create 4D grid
grid = SpacetimeGrid(
    coordinate_system="cartesian",
    time_range=(0.0, 10.0),
    spatial_ranges=[(0.0, 2*np.pi)] * 3,
    grid_points=(100, 32, 32, 32),  # (nt, nx, ny, nz)
)

# Initialize 4D fields
fields = ISFieldConfiguration(grid)
for t_idx in range(grid.nt):
    fields.rho[t_idx, :, :, :] = initial_value

# Manual time stepping
t = 0.0
while t < t_final:
    # Custom integration logic
    fields = update_fields(fields, dt)
    t += dt
```

**New Pattern (3D Spatial):**
```python
# Create pure 3D spatial grid
grid = SpaceGrid(
    coordinate_system="cartesian",
    spatial_ranges=[(0.0, 2*np.pi)] * 3,
    grid_points=(32, 32, 32),  # (nx, ny, nz) only
    boundary_conditions="periodic"  # Required for spectral!
)

# Initialize 3D fields (direct indexing)
fields = ISFieldConfiguration(grid)
fields.rho[:] = initial_value  # Pure 3D array

# Use spectral solver with built-in time integration
solver = SpectralISHydrodynamics(grid, fields, coeffs)
solver.evolve(
    t_final=10.0,
    dt=0.01,  # Optional (adaptive if None)
    snapshot_config={
        "filename": "output.h5",
        "interval": 0.1,
        "buffer_size": 20
    }
)
```

### Key Benefits

1. **Memory Efficiency:** 95% reduction in field storage
2. **Code Clarity:** No dimension checking, pure 3D indexing
3. **Spectral Methods:** FFT-based derivatives with periodic boundaries
4. **Streaming Architecture:** Constant memory for long simulations
5. **Built-in Time Integration:** RK4, IMEX methods in solver

---

## Detailed Phase Breakdown

## Phase 1: Update `bjorken_flow.py`

**File:** `israel_stewart/benchmarks/bjorken_flow.py`

### Changes Required

#### 1.1 Update `BjorkenBenchmark.__init__`

**Current (lines 321-344):**
```python
class BjorkenBenchmark:
    def __init__(
        self,
        grid: SpacetimeGrid,  # 4D grid
        coefficients: TransportCoefficients,
        analytical_solution: BjorkenFlowSolution,
    ):
        self.grid = grid
        self.coefficients = coefficients
        self.analytical = analytical_solution

        # Numerical simulation setup
        self.metric = MilneMetric()
        self.relaxation_eq = ISRelaxationEquations(grid, self.metric, coefficients)

        # Results storage
        self.results: dict[str, Any] = {}
```

**Updated:**
```python
class BjorkenBenchmark:
    def __init__(
        self,
        grid: SpaceGrid,  # Pure 3D spatial grid
        coefficients: TransportCoefficients,
        analytical_solution: BjorkenFlowSolution,
    ):
        self.grid = grid
        self.coefficients = coefficients
        self.analytical = analytical_solution

        # Initialize fields
        self.fields = ISFieldConfiguration(grid)

        # Create spectral solver
        self.metric = MilneMetric()
        self.solver = SpectralISHydrodynamics(grid, self.fields, coefficients)

        # Results storage
        self.results: dict[str, Any] = {}
```

**Lines affected:** 321-344 (24 lines modified)

---

#### 1.2 Update `run_numerical_simulation`

**Current (lines 346-407):**
```python
def run_numerical_simulation(
    self,
    final_time: float = 10.0,
    timestep: float = 0.01,
    solver_method: str = "explicit",
) -> dict[str, np.ndarray]:
    # Initialize fields
    fields = ISFieldConfiguration(self.grid)
    self._setup_bjorken_initial_conditions(fields)

    # Time evolution
    time_points: list[float] = []
    solutions: dict[str, list[float]] = {...}

    current_time = self.analytical.tau0

    # Time integration loop
    while current_time < final_time:
        dt = min(timestep, final_time - current_time)

        # Evolve fields
        if solver_method == "explicit":
            self.relaxation_eq.evolve_relaxation(fields, dt, method="explicit")
        # ...

        current_time += dt
        self._record_solution_state(fields, solutions)

    return result
```

**Updated:**
```python
def run_numerical_simulation(
    self,
    final_time: float = 10.0,
    timestep: float | None = None,
    method: str = "spectral_imex",
) -> dict[str, np.ndarray]:
    """
    Run numerical Bjorken flow simulation using spectral solver.

    Args:
        final_time: Final simulation time in fm/c
        timestep: Integration timestep (adaptive if None)
        method: Solver method ('spectral_imex', 'rk4', etc.)

    Returns:
        Dictionary with numerical solution
    """
    # Setup initial conditions
    self._setup_bjorken_initial_conditions(self.fields)

    # Storage for monitoring during evolution
    time_points: list[float] = []
    solutions: dict[str, list[float]] = {
        "temperature": [],
        "energy_density": [],
        "pressure": [],
        "bulk_pressure": [],
        "shear_stress": [],
    }

    # Callback to record solution state during evolution
    def record_state(t: float, fields: ISFieldConfiguration) -> None:
        time_points.append(t)
        self._record_solution_state(fields, solutions)

    # Record initial state
    record_state(self.analytical.tau0, self.fields)

    # Evolve using spectral solver
    self.solver.evolve(
        t_final=final_time,
        dt=timestep,
        method=method,
        callback=record_state,  # Called at each timestep
    )

    # Convert lists to arrays
    result = {
        "time": np.array(time_points),
    }
    for key, values in solutions.items():
        result[key] = np.array(values)

    return result
```

**Lines affected:** 346-407 (62 lines → ~50 lines, simplified)

**New callback feature needed in SpectralISHydrodynamics.evolve():**
```python
# In spectral.py evolve() method (will add in Phase 1)
def evolve(
    self,
    t_final: float,
    dt: float | None = None,
    method: str = "spectral_imex",
    snapshot_config: dict | None = None,
    callback: Callable[[float, ISFieldConfiguration], None] | None = None,
) -> None:
    """
    Time evolution with optional callback for monitoring.

    Args:
        callback: Optional function called as callback(t, fields) at each timestep
    """
    t = 0.0
    while t < t_final:
        # ... time stepping ...

        if callback is not None:
            callback(t, self.fields)
```

---

#### 1.3 Update `_setup_bjorken_initial_conditions`

**Current (lines 409-430):**
```python
def _setup_bjorken_initial_conditions(self, fields: ISFieldConfiguration) -> None:
    """Setup initial conditions for Bjorken flow."""
    ideal_ic = self.analytical.ideal_solution(self.analytical.tau0)

    # Set thermodynamic quantities
    fields.temperature.fill(ideal_ic["temperature"][0])
    fields.rho.fill(ideal_ic["energy_density"][0])
    fields.pressure.fill(ideal_ic["pressure"][0])

    # Set four-velocity (boost-invariant)
    fields.u_mu[..., 0] = 1.0  # u^τ = 1
    fields.u_mu[..., 1] = 0.0  # u^η = 0
    # ...
```

**Updated:**
```python
def _setup_bjorken_initial_conditions(self, fields: ISFieldConfiguration) -> None:
    """Setup initial conditions for Bjorken flow (pure 3D)."""
    # Get ideal solution at initial time
    ideal_ic = self.analytical.ideal_solution(self.analytical.tau0)

    # Set thermodynamic quantities (pure 3D arrays)
    fields.rho[:] = ideal_ic["energy_density"][0]
    fields.pressure[:] = ideal_ic["pressure"][0]

    if hasattr(fields, "temperature"):
        fields.temperature[:] = ideal_ic["temperature"][0]

    # Set four-velocity (boost-invariant, rest frame)
    fields.u_mu[:] = 0.0
    fields.u_mu[..., 0] = 1.0  # u^t = 1 in rest frame

    # Initialize dissipative fluxes to zero
    fields.Pi[:] = 0.0
    fields.pi_munu[:] = 0.0
    if hasattr(fields, "q_mu"):
        fields.q_mu[:] = 0.0
```

**Lines affected:** 409-430 (21 lines, simplified indexing)

**Note:** Fields are now pure 3D, so `fields.rho` has shape `(nx, ny, nz)` not `(nt, nx, ny, nz)`.

---

#### 1.4 Update `_record_solution_state`

**Current (lines 431-444):**
```python
def _record_solution_state(
    self, fields: ISFieldConfiguration, solutions: dict[str, list[float]]
) -> None:
    """Record current solution state."""
    # Take spatial average (Bjorken flow should be uniform)
    solutions["temperature"].append(float(np.mean(fields.temperature)))
    solutions["energy_density"].append(float(np.mean(fields.rho)))
    solutions["pressure"].append(float(np.mean(fields.pressure)))
    solutions["bulk_pressure"].append(float(np.mean(fields.Pi)))

    # Shear stress (take trace norm for scalar measure)
    pi_norm = np.sqrt(np.mean(fields.pi_munu**2))
    solutions["shear_stress"].append(float(pi_norm))
```

**Updated:**
```python
def _record_solution_state(
    self, fields: ISFieldConfiguration, solutions: dict[str, list[float]]
) -> None:
    """Record current solution state (pure 3D fields)."""
    # Take spatial average (Bjorken flow should be uniform)
    if hasattr(fields, "temperature"):
        solutions["temperature"].append(float(np.mean(fields.temperature)))
    else:
        # Compute temperature from energy density
        T = self._compute_temperature_from_rho(float(np.mean(fields.rho)))
        solutions["temperature"].append(T)

    solutions["energy_density"].append(float(np.mean(fields.rho)))
    solutions["pressure"].append(float(np.mean(fields.pressure)))
    solutions["bulk_pressure"].append(float(np.mean(fields.Pi)))

    # Shear stress magnitude (Frobenius norm)
    pi_norm = np.sqrt(np.mean(fields.pi_munu**2))
    solutions["shear_stress"].append(float(pi_norm))

def _compute_temperature_from_rho(self, rho: float) -> float:
    """Compute temperature from energy density (ideal gas EOS)."""
    g_eff = 37.5  # QGP degrees of freedom
    a = (np.pi**2 / 90.0) * g_eff
    return (rho / a) ** (1.0 / 4.0)
```

**Lines affected:** 431-444 (14 lines + helper function)

---

#### 1.5 Update `create_standard_bjorken_benchmark`

**Current (lines 715-761):**
```python
def create_standard_bjorken_benchmark(
    tau0: float = 0.6,
    T0: float = 0.3,
    eta_over_s: float = 0.08,
    grid_points: tuple[int, int, int, int] = (8, 4, 4, 4),
) -> BjorkenBenchmark:
    """Create standard Bjorken benchmark setup."""

    # Create grid (only tau direction matters for Bjorken flow)
    grid = SpacetimeGrid(
        coordinate_system="milne",
        time_range=(tau0, 10.0),
        spatial_ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        grid_points=grid_points,  # (nt, nx, ny, nz)
    )

    # Transport coefficients
    # ...

    return BjorkenBenchmark(grid, coefficients, analytical)
```

**Updated:**
```python
def create_standard_bjorken_benchmark(
    tau0: float = 0.6,
    T0: float = 0.3,
    eta_over_s: float = 0.08,
    grid_points: tuple[int, int, int] = (32, 32, 32),  # Pure 3D
    domain_size: float = 2.0 * np.pi,
) -> BjorkenBenchmark:
    """
    Create standard Bjorken benchmark setup with pure 3D spatial grid.

    Args:
        tau0: Initial proper time in fm/c
        T0: Initial temperature in GeV
        eta_over_s: Shear viscosity to entropy ratio
        grid_points: Spatial grid resolution (nx, ny, nz)
        domain_size: Spatial domain size (periodic)

    Returns:
        Configured Bjorken benchmark with SpaceGrid
    """
    # Create pure 3D spatial grid
    grid = SpaceGrid(
        coordinate_system="cartesian",  # Use Cartesian for simplicity
        spatial_ranges=[(0.0, domain_size)] * 3,
        grid_points=grid_points,  # (nx, ny, nz) - pure 3D
        boundary_conditions="periodic",  # Required for spectral methods
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
```

**Lines affected:** 715-761 (47 lines, updated signature and implementation)

---

#### 1.6 Add Callback Support to `SpectralISHydrodynamics.evolve()`

**File:** `israel_stewart/solvers/spectral.py`

**Current evolve() signature (approximate line 1850):**
```python
def evolve(
    self,
    t_final: float,
    dt: float | None = None,
    snapshot_config: dict | None = None,
) -> None:
    """Time evolution with streaming snapshots."""
```

**Updated:**
```python
def evolve(
    self,
    t_final: float,
    dt: float | None = None,
    method: str = "spectral_imex",
    snapshot_config: dict | None = None,
    callback: Callable[[float, ISFieldConfiguration], None] | None = None,
) -> None:
    """
    Time evolution with optional monitoring callback.

    Args:
        t_final: Final simulation time
        dt: Timestep (adaptive if None)
        method: Integration method ('spectral_imex', 'rk4', etc.)
        snapshot_config: Configuration for streaming snapshots
        callback: Optional function called as callback(t, fields) at each timestep
                 for monitoring observables during evolution

    Example with callback:
        >>> def monitor(t, fields):
        ...     print(f"t={t:.2f}, rho={np.mean(fields.rho):.3f}")
        >>> solver.evolve(t_final=10.0, callback=monitor)
    """
    # Setup streaming if requested
    stream = None
    if snapshot_config:
        from ..utils.streaming import SnapshotStream
        stream = SnapshotStream(
            filename=snapshot_config['filename'],
            grid=self.grid,
            coeffs=self.coeffs,
            interval=snapshot_config.get('interval', 0.1),
            buffer_size=snapshot_config.get('buffer_size', 10)
        )
        if snapshot_config.get('save_initial', True):
            stream.save(0.0, self.fields)

    # Time evolution loop
    t = 0.0
    try:
        while t < t_final:
            # Determine timestep
            if dt is None:
                dt_step = self.adaptive_time_step()
            else:
                dt_step = dt

            dt_step = min(dt_step, t_final - t)

            # Advance fields one timestep
            self.time_step(dt_step, method=method)
            t += dt_step

            # Call monitoring callback
            if callback is not None:
                callback(t, self.fields)

            # Save snapshot if needed
            if stream and stream.should_save(t):
                stream.save(t, self.fields)

    finally:
        # Ensure final snapshot is saved
        if stream:
            stream.flush()
            stream.close()
```

**Lines affected:** ~1850-1920 (add callback parameter and invocation)

---

### Summary of Phase 1 Changes

**Files Modified:**
1. `israel_stewart/benchmarks/bjorken_flow.py` (~150 lines changed)
   - Update imports: `SpaceGrid` instead of `SpacetimeGrid`
   - Update `BjorkenBenchmark.__init__` (24 lines)
   - Update `run_numerical_simulation` (62 lines)
   - Update `_setup_bjorken_initial_conditions` (21 lines)
   - Update `_record_solution_state` (14 lines)
   - Update `create_standard_bjorken_benchmark` (47 lines)

2. `israel_stewart/solvers/spectral.py` (~30 lines changed)
   - Add `callback` parameter to `evolve()` method
   - Add callback invocation in time loop
   - Update docstring with callback example

**Key Changes:**
- ✅ SpacetimeGrid → SpaceGrid
- ✅ grid_points: `(nt, nx, ny, nz)` → `(nx, ny, nz)`
- ✅ Remove time dimension from field initialization
- ✅ Use `SpectralISHydrodynamics.evolve()` instead of manual loop
- ✅ Add callback support for monitoring during evolution
- ✅ Simplify code by removing 4D indexing

**Validation:**
- Analytical solutions remain unchanged
- Numerical simulations use spectral solver with pure 3D fields
- Direct comparison: analytical vs spectral results

---

## Phase 2: Update `sound_waves.py`

**File:** `israel_stewart/benchmarks/sound_waves.py`

### Changes Required

#### 2.1 Update `SoundWaveAnalysis.__init__`

**Current (lines 106-133):**
```python
class SoundWaveAnalysis:
    def __init__(
        self,
        grid: SpacetimeGrid,  # 4D grid
        metric: GeneralMetric,
        transport_coeffs: TransportCoefficients,
        background_fields: ISFieldConfiguration | None = None,
    ):
        self.grid = grid
        # ...
```

**Updated:**
```python
class SoundWaveAnalysis:
    def __init__(
        self,
        grid: SpaceGrid,  # Pure 3D spatial grid
        metric: GeneralMetric,
        transport_coeffs: TransportCoefficients,
        background_fields: ISFieldConfiguration | None = None,
    ):
        self.grid = grid
        # ... (rest unchanged - analysis is grid-agnostic)
```

**Lines affected:** 106-133 (just type annotation)

---

#### 2.2 Update `NumericalSoundWaveBenchmark.__init__`

**Current (lines 960-993):**
```python
class NumericalSoundWaveBenchmark:
    def __init__(
        self,
        domain_size: float = 2 * np.pi,
        grid_points: tuple[int, int, int, int] = (64, 64, 16, 16),  # (Nt, Nx, Ny, Nz)
        transport_coeffs: TransportCoefficients | None = None,
        metric: GeneralMetric | None = None,
    ):
        self.domain_size = domain_size
        self.grid_points = grid_points

        # Create periodic grid for spectral simulation
        time_range = (0.0, 10.0)
        spatial_ranges = [(0.0, domain_size)] * 3
        self.grid = create_periodic_grid("cartesian", time_range, spatial_ranges, grid_points)

        # Initialize spectral solver
        self.fields = ISFieldConfiguration(self.grid)
        self.solver = SpectralISHydrodynamics(self.grid, self.fields, self.transport_coeffs)
```

**Updated:**
```python
class NumericalSoundWaveBenchmark:
    def __init__(
        self,
        domain_size: float = 2 * np.pi,
        grid_points: tuple[int, int, int] = (64, 64, 16),  # (Nx, Ny, Nz) - pure 3D
        transport_coeffs: TransportCoefficients | None = None,
        metric: GeneralMetric | None = None,
    ):
        """
        Initialize numerical sound wave benchmark with pure 3D spatial grid.

        Args:
            domain_size: Spatial domain size (periodic)
            grid_points: Spatial grid resolution (nx, ny, nz)
            transport_coeffs: Transport coefficients for viscosity
            metric: Spacetime metric (defaults to Minkowski)
        """
        self.domain_size = domain_size
        self.grid_points = grid_points

        # Create pure 3D spatial grid for spectral simulation
        spatial_ranges = [(0.0, domain_size)] * 3
        self.grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=spatial_ranges,
            grid_points=grid_points,  # (nx, ny, nz)
            boundary_conditions="periodic",  # Required for FFT
        )

        # Physics setup
        self.metric = metric or MinkowskiMetric()
        self.transport_coeffs = transport_coeffs or self._default_transport_coeffs()

        # Initialize analytical analysis for comparison
        self.analytical = SoundWaveAnalysis(self.grid, self.metric, self.transport_coeffs)

        # Initialize fields and spectral solver
        self.fields = ISFieldConfiguration(self.grid)
        self.solver = SpectralISHydrodynamics(self.grid, self.fields, self.transport_coeffs)
```

**Lines affected:** 960-993 (33 lines updated)

---

#### 2.3 Update `setup_initial_conditions`

**Current (lines 1006-1057):**
```python
def setup_initial_conditions(
    self,
    wave_number: float,
    amplitude: float = 0.01,
    background_density: float = 1.0,
    background_pressure: float = 1.0 / 3.0,
) -> None:
    # Get spatial coordinates
    x = self.grid.coordinates["x"]
    y = self.grid.coordinates["y"]
    z = self.grid.coordinates["z"]

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Sound wave perturbation
    delta_rho = amplitude * np.sin(wave_number * X)
    delta_ux = amplitude * 0.5 * np.sin(wave_number * X)

    # Apply perturbations to 4D fields
    for t_idx in range(self.grid_points[0]):  # Loop over time
        self.fields.rho[t_idx, ...] = background_density + delta_rho
        self.fields.u_mu[t_idx, ..., 1] = delta_ux

    # Update pressure
    self.fields.pressure[:] = self.fields.rho / 3.0

    # Zero dissipative fluxes
    self.fields.Pi.fill(0.0)
    self.fields.pi_munu.fill(0.0)
    self.fields.q_mu.fill(0.0)
```

**Updated:**
```python
def setup_initial_conditions(
    self,
    wave_number: float,
    amplitude: float = 0.01,
    background_density: float = 1.0,
    background_pressure: float = 1.0 / 3.0,
) -> None:
    """
    Setup sinusoidal perturbation initial conditions (pure 3D).

    Args:
        wave_number: Wave number k for the perturbation
        amplitude: Perturbation amplitude (should be small for linear regime)
        background_density: Background energy density ρ₀
        background_pressure: Background pressure P₀
    """
    # Get spatial coordinates and create meshgrid
    X, Y, Z = self.grid.meshgrid()

    # Sound wave perturbation along x-direction
    # δρ = A * sin(k*x), δuₓ = A' * sin(k*x)
    delta_rho = amplitude * np.sin(wave_number * X)
    delta_ux = amplitude * 0.5 * np.sin(wave_number * X)

    # Initialize pure 3D fields (no time loop!)
    self.fields.rho[:] = background_density + delta_rho
    self.fields.pressure[:] = (background_density + delta_rho) / 3.0  # P = ρ/3

    # Velocity: u^μ = (γ, γvˣ, 0, 0) where γ ≈ 1 for small velocities
    self.fields.u_mu[:] = 0.0
    self.fields.u_mu[..., 0] = 1.0  # u^t = 1 in rest frame
    self.fields.u_mu[..., 1] = delta_ux  # u^x = δuₓ

    # Zero dissipative fluxes initially
    self.fields.Pi[:] = 0.0
    self.fields.pi_munu[:] = 0.0
    if hasattr(self.fields, "q_mu"):
        self.fields.q_mu[:] = 0.0
```

**Lines affected:** 1006-1057 (52 lines → ~35 lines, much simpler!)

**Key change:** No more `for t_idx in range(...)` loop! Fields are pure 3D.

---

#### 2.4 Update `run_simulation`

**Current (lines 1058-1168):**
```python
def run_simulation(
    self,
    wave_number: float,
    simulation_time: float = 10.0,
    n_periods: int = 5,
    dt_factor: float = 0.1,
) -> NumericalWaveResults:
    # Setup initial conditions
    self.setup_initial_conditions(wave_number)

    # ... determine timestep ...

    # Time evolution loop
    n_steps = int(simulation_time / dt_cfl)
    time_points = np.linspace(0, simulation_time, n_steps)

    rho_time_series = []
    ux_time_series = []

    monitor_idx = (
        self.grid_points[1] // 2,  # Note: using grid_points[1] for nx
        self.grid_points[2] // 2,
        self.grid_points[3] // 2,
    )

    current_time = 0.0
    for _step in range(n_steps):
        # Record at monitor point
        rho_monitor = self.fields.rho[0, monitor_idx[0], monitor_idx[1], monitor_idx[2]]
        ux_monitor = self.fields.u_mu[0, monitor_idx[0], monitor_idx[1], monitor_idx[2], 1]

        rho_time_series.append(rho_monitor)
        ux_time_series.append(ux_monitor)

        # Manual time step
        self.solver.time_step(dt_cfl, method="spectral_imex")
        current_time += dt_cfl

    # Analyze time series
    measured_freq, measured_damping = self._extract_frequency_damping(...)

    return NumericalWaveResults(...)
```

**Updated:**
```python
def run_simulation(
    self,
    wave_number: float,
    simulation_time: float = 10.0,
    n_periods: int = 5,
    dt_factor: float = 0.1,
) -> NumericalWaveResults:
    """
    Run numerical simulation of sound wave evolution.

    Args:
        wave_number: Wave number to simulate
        simulation_time: Total simulation time
        n_periods: Number of wave periods to evolve
        dt_factor: Timestep factor (fraction of CFL limit)

    Returns:
        Numerical wave simulation results with frequency and damping
    """
    # Setup initial conditions
    self.setup_initial_conditions(wave_number)

    # Get analytical prediction for comparison
    wave_vector = np.array([wave_number, 0.0, 0.0])
    analytical_modes = self.analytical.analyze_dispersion_relation(wave_vector)

    if not analytical_modes:
        raise ValueError(f"Could not find analytical mode for k={wave_number}")

    analytical_mode = analytical_modes[0]
    analytical_freq = analytical_mode.frequency
    analytical_damping = analytical_mode.attenuation

    # Adjust simulation time based on wave properties
    if analytical_freq > 0:
        period = 2 * np.pi / analytical_freq
        simulation_time = max(simulation_time, n_periods * period)

    # Determine timestep (CFL condition)
    dx = self.grid.spatial_spacing[0]
    sound_speed = analytical_mode.sound_speed
    dt_cfl = dt_factor * dx / max(sound_speed, 0.1)

    # Monitor point for time series (center of domain)
    monitor_idx = tuple(n // 2 for n in self.grid_points)

    # Storage for time series
    time_points = []
    rho_time_series = []
    ux_time_series = []

    # Callback to record time series during evolution
    def record_time_series(t: float, fields: ISFieldConfiguration) -> None:
        time_points.append(t)
        # Extract at monitor point (pure 3D indexing)
        rho_monitor = fields.rho[monitor_idx]
        ux_monitor = fields.u_mu[monitor_idx + (1,)]  # u^x component
        rho_time_series.append(rho_monitor)
        ux_time_series.append(ux_monitor)

    # Evolve using spectral solver with callback
    try:
        self.solver.evolve(
            t_final=simulation_time,
            dt=dt_cfl,
            method="spectral_imex",
            callback=record_time_series,
        )
    except Exception as e:
        warnings.warn(f"Simulation failed: {e}", stacklevel=2)
        # Return empty result
        return self._create_failed_result(wave_number, analytical_freq, analytical_damping)

    # Analyze time series for frequency and damping
    time_array = np.array(time_points)
    rho_array = np.array(rho_time_series)
    ux_array = np.array(ux_time_series)

    measured_freq, measured_damping = self._extract_frequency_damping(time_array, rho_array)

    # Calculate errors
    freq_error = abs(measured_freq - analytical_freq) / max(analytical_freq, 1e-10)
    damping_error = abs(measured_damping - analytical_damping) / max(analytical_damping, 1e-10)

    # Check convergence
    convergence_achieved = (
        freq_error < 0.1  # 10% frequency error
        and damping_error < 0.2  # 20% damping error
    )

    return NumericalWaveResults(
        wave_number=wave_number,
        measured_frequency=measured_freq,
        measured_damping_rate=measured_damping,
        analytical_frequency=analytical_freq,
        analytical_damping_rate=analytical_damping,
        frequency_error=freq_error,
        damping_error=damping_error,
        simulation_time=simulation_time,
        grid_resolution=self.grid_points[0],
        convergence_achieved=convergence_achieved,
        time_series_data={
            "time": time_array,
            "density": rho_array,
            "velocity": ux_array,
        },
    )

def _create_failed_result(
    self, wave_number: float, analytical_freq: float, analytical_damping: float
) -> NumericalWaveResults:
    """Create result object for failed simulation."""
    return NumericalWaveResults(
        wave_number=wave_number,
        measured_frequency=0.0,
        measured_damping_rate=0.0,
        analytical_frequency=analytical_freq,
        analytical_damping_rate=analytical_damping,
        frequency_error=np.inf,
        damping_error=np.inf,
        simulation_time=0.0,
        grid_resolution=self.grid_points[0],
        convergence_achieved=False,
        time_series_data={},
    )
```

**Lines affected:** 1058-1168 (110 lines → ~120 lines, cleaner implementation)

**Key changes:**
- No more manual time loop
- Pure 3D indexing: `fields.rho[monitor_idx]` not `fields.rho[0, monitor_idx[0], ...]`
- Use callback for time series recording
- Cleaner separation of simulation and analysis

---

#### 2.5 Update `create_numerical_benchmark`

**Current (lines 1680-1719):**
```python
def create_numerical_benchmark(
    domain_size: float = 2 * np.pi,
    grid_points: tuple[int, int, int, int] = (64, 64, 16, 16),  # (Nt, Nx, Ny, Nz)
    transport_coeffs: TransportCoefficients | None = None,
    metric: GeneralMetric | None = None,
    **kwargs,
) -> NumericalSoundWaveBenchmark:
```

**Updated:**
```python
def create_numerical_benchmark(
    domain_size: float = 2 * np.pi,
    grid_points: tuple[int, int, int] = (64, 64, 16),  # (Nx, Ny, Nz) - pure 3D
    transport_coeffs: TransportCoefficients | None = None,
    metric: GeneralMetric | None = None,
    **kwargs,
) -> NumericalSoundWaveBenchmark:
    """
    Factory function for creating numerical sound wave benchmark instances.

    Args:
        domain_size: Spatial domain size (periodic)
        grid_points: Spatial grid resolution (nx, ny, nz)
        transport_coeffs: Transport coefficients for viscosity
        metric: Spacetime metric (defaults to Minkowski)
        **kwargs: Additional arguments passed to NumericalSoundWaveBenchmark

    Returns:
        Configured numerical benchmark instance with SpaceGrid
    """
    return NumericalSoundWaveBenchmark(
        domain_size=domain_size,
        grid_points=grid_points,
        transport_coeffs=transport_coeffs,
        metric=metric,
        **kwargs,
    )
```

**Lines affected:** 1680-1719 (update signature and docstring)

---

### Summary of Phase 2 Changes

**Files Modified:**
1. `israel_stewart/benchmarks/sound_waves.py` (~200 lines changed)
   - Update `SoundWaveAnalysis.__init__` type annotations
   - Update `NumericalSoundWaveBenchmark.__init__` (33 lines)
   - Update `setup_initial_conditions` (52 → 35 lines, remove time loop)
   - Update `run_simulation` (110 → 120 lines, use callback)
   - Update `create_numerical_benchmark` (signature change)
   - Add `_create_failed_result` helper

**Key Changes:**
- ✅ Remove 4D time indexing from field initialization
- ✅ Use callback for time series monitoring
- ✅ Simplify code by using `solver.evolve()` instead of manual loop
- ✅ Pure 3D indexing throughout

---

## Phase 3: Update `equilibration.py`

**File:** `israel_stewart/benchmarks/equilibration.py`

### Changes Required

#### 3.1 Update `EquilibrationAnalysis.__init__`

**Current (lines 89-115):**
```python
class EquilibrationAnalysis:
    def __init__(
        self,
        grid: SpacetimeGrid,  # 4D grid
        metric: GeneralMetric,
        transport_coeffs: TransportCoefficients,
        equation_of_state: str = "ideal",
    ):
        self.grid = grid
        self.metric = metric
        self.transport_coeffs = transport_coeffs
        self.eos = equation_of_state

        # Initialize physics modules
        self.conservation = ConservationLaws(grid, metric)
        self.relaxation = ISRelaxationEquations(grid, metric, transport_coeffs)
```

**Updated:**
```python
class EquilibrationAnalysis:
    def __init__(
        self,
        grid: SpaceGrid,  # Pure 3D spatial grid
        metric: GeneralMetric,
        transport_coeffs: TransportCoefficients,
        equation_of_state: str = "ideal",
    ):
        """
        Initialize equilibration analysis with pure 3D spatial grid.

        Args:
            grid: SpaceGrid defining spatial computational domain
            metric: Spacetime metric
            transport_coeffs: Transport coefficients
            equation_of_state: Equation of state type ("ideal", etc.)
        """
        self.grid = grid
        self.metric = metric
        self.transport_coeffs = transport_coeffs
        self.eos = equation_of_state

        # Analysis results cache
        self._equilibration_cache: dict[str, EquilibrationProperties] = {}
```

**Lines affected:** 89-115 (remove ConservationLaws, ISRelaxationEquations - will use solver)

---

#### 3.2 Update `analyze_relaxation_to_equilibrium`

**Current (lines 117-216):**
```python
def analyze_relaxation_to_equilibrium(
    self,
    initial_fields: ISFieldConfiguration,
    final_time: float = 10.0,
    timestep: float = 0.01,
    method: str = "implicit",
) -> EquilibrationProperties:
    # Validate initial state
    self._validate_initial_state(initial_fields)

    # ... setup ...

    fields = self._copy_fields(initial_fields)
    current_time = 0.0

    # Integration loop
    while current_time < final_time:
        dt = min(timestep, final_time - current_time)

        # Evolve system
        self.relaxation.evolve_relaxation(fields, dt, method=method)

        current_time += dt
        time_points.append(current_time)

        # Record thermodynamic quantities
        temp = self._compute_temperature(fields)
        entropy = self._compute_entropy_density(fields)
        temperature_data.append(temp)
        # ...

    # ... analysis ...
```

**Updated:**
```python
def analyze_relaxation_to_equilibrium(
    self,
    initial_fields: ISFieldConfiguration,
    final_time: float = 10.0,
    timestep: float | None = None,
    method: str = "spectral_imex",
) -> EquilibrationProperties:
    """
    Analyze complete relaxation process to thermal equilibrium.

    Args:
        initial_fields: Initial non-equilibrium state (pure 3D)
        final_time: Final simulation time
        timestep: Integration timestep (adaptive if None)
        method: Integration method ("spectral_imex", "rk4", etc.)

    Returns:
        Equilibration properties and evolution data
    """
    # Validate initial state
    self._validate_initial_state(initial_fields)

    # Store initial thermodynamic quantities
    initial_state = self._extract_thermodynamic_state(initial_fields)

    # Create solver with copy of initial fields
    fields = self._copy_fields(initial_fields)
    solver = SpectralISHydrodynamics(self.grid, fields, self.transport_coeffs)

    # Storage for time evolution
    time_points = []
    temperature_data = []
    entropy_data = []
    bulk_pressure_data = []
    shear_stress_data = []

    # Callback to record thermodynamic quantities during evolution
    def record_thermodynamics(t: float, fields: ISFieldConfiguration) -> None:
        time_points.append(t)
        temperature_data.append(self._compute_temperature(fields))
        entropy_data.append(self._compute_entropy_density(fields))
        bulk_pressure_data.append(float(np.mean(np.abs(fields.Pi))))
        shear_stress_data.append(self._compute_shear_stress_magnitude(fields))

    # Record initial state
    record_thermodynamics(0.0, fields)

    # Evolve to equilibrium using spectral solver
    solver.evolve(
        t_final=final_time,
        dt=timestep,
        method=method,
        callback=record_thermodynamics,
    )

    # Convert to arrays
    time_array = np.array(time_points)
    temperature_evolution = np.array(temperature_data)
    entropy_evolution = np.array(entropy_data)
    bulk_pressure_evolution = np.array(bulk_pressure_data)
    shear_stress_evolution = np.array(shear_stress_data)

    # Analyze relaxation timescales
    relaxation_times = self._extract_relaxation_times(
        time_array, bulk_pressure_evolution, shear_stress_evolution
    )

    # Compute decay rates
    decay_rates = {key: 1.0 / tau for key, tau in relaxation_times.items()}

    # Analyze entropy production
    entropy_production_rate = self._compute_entropy_production_rate(
        time_array, entropy_evolution
    )

    # Fit approach to equilibrium
    approach_exponent = self._fit_approach_exponent(time_array, bulk_pressure_evolution)

    # Final equilibrium state
    final_state = self._extract_thermodynamic_state(fields)

    return EquilibrationProperties(
        initial_state=initial_state,
        final_state=final_state,
        relaxation_times=relaxation_times,
        decay_rates=decay_rates,
        entropy_production_rate=entropy_production_rate,
        approach_exponent=approach_exponent,
        temperature_evolution=temperature_evolution,
        entropy_evolution=entropy_evolution,
        bulk_pressure_evolution=bulk_pressure_evolution,
        shear_stress_evolution=shear_stress_evolution,
    )
```

**Lines affected:** 117-216 (100 lines, restructured to use solver)

---

#### 3.3 Update field helper methods

**Update `_validate_initial_state` (lines 218-232):**
```python
def _validate_initial_state(self, fields: ISFieldConfiguration) -> None:
    """Validate that initial state is physically reasonable (pure 3D)."""
    # Check energy density positivity
    if np.any(fields.rho <= 0):
        raise ValueError("Energy density must be positive")

    # Check pressure positivity
    if np.any(fields.pressure <= 0):
        raise ValueError("Pressure must be positive")

    # Check four-velocity normalization (u^μ u_μ = 1)
    # For Minkowski: u^0 u^0 - u^i u^i = 1
    u0_sq = fields.u_mu[..., 0]**2
    ui_sq = np.sum(fields.u_mu[..., 1:]**2, axis=-1)
    if not np.allclose(u0_sq - ui_sq, 1.0, rtol=1e-3):
        warnings.warn("Four-velocity not properly normalized", stacklevel=2)
```

**Update `_copy_fields` (lines 234-246):**
```python
def _copy_fields(self, fields: ISFieldConfiguration) -> ISFieldConfiguration:
    """Create a deep copy of field configuration (pure 3D)."""
    new_fields = ISFieldConfiguration(self.grid)

    # Copy all field data (pure 3D arrays)
    new_fields.rho[:] = fields.rho[:]
    new_fields.pressure[:] = fields.pressure[:]
    new_fields.u_mu[:] = fields.u_mu[:]
    new_fields.Pi[:] = fields.Pi[:]
    new_fields.pi_munu[:] = fields.pi_munu[:]
    if hasattr(fields, "q_mu"):
        new_fields.q_mu[:] = fields.q_mu[:]

    return new_fields
```

**Update `_compute_shear_stress_magnitude` (lines 290-296):**
```python
def _compute_shear_stress_magnitude(self, fields: ISFieldConfiguration) -> float:
    """Compute magnitude of shear stress tensor (pure 3D)."""
    # Compute Frobenius norm: sqrt(π^{μν} π_{μν})
    magnitude_squared = np.sum(fields.pi_munu**2, axis=(-2, -1))
    return float(np.sqrt(np.mean(magnitude_squared)))
```

**Lines affected:** ~50 lines of helper methods (simplified)

---

### Summary of Phase 3 Changes

**Files Modified:**
1. `israel_stewart/benchmarks/equilibration.py` (~100 lines changed)
   - Update `EquilibrationAnalysis.__init__` (remove old physics modules)
   - Update `analyze_relaxation_to_equilibrium` (use SpectralISHydrodynamics)
   - Update `_validate_initial_state` (pure 3D validation)
   - Update `_copy_fields` (pure 3D copying)
   - Update helper methods for pure 3D fields

**Key Changes:**
- ✅ Use `SpectralISHydrodynamics` instead of `ISRelaxationEquations`
- ✅ Callback-based thermodynamic monitoring
- ✅ Pure 3D field operations throughout
- ✅ Simpler code without manual time evolution

---

## Phase 4: Update Test Files

### 4.1 Update `test_bjorken_flow_validation.py`

**Changes Required:**
- Replace `SpacetimeGrid` with `SpaceGrid` in fixtures
- Update grid_points: `(nt, nx, ny, nz)` → `(nx, ny, nz)`
- Add `boundary_conditions="periodic"`
- Update field shape assertions

**Example changes:**
```python
# OLD:
grid = SpacetimeGrid(
    coordinate_system="cartesian",
    time_range=(0.0, 2.0),
    spatial_ranges=[(0.0, 2*np.pi)] * 3,
    grid_points=(20, 16, 16, 16),
    boundary_conditions="periodic",
)

# NEW:
grid = SpaceGrid(
    coordinate_system="cartesian",
    spatial_ranges=[(0.0, 2*np.pi)] * 3,
    grid_points=(16, 16, 16),
    boundary_conditions="periodic",
)
```

**Lines affected:** ~50 lines across test file

---

### 4.2 Update `test_sound_wave_validation.py`

Similar changes as above for all test fixtures and grid creation.

**Lines affected:** ~50 lines

---

### 4.3 Update `test_equilibration_validation.py`

Similar changes as above for all test fixtures and grid creation.

**Lines affected:** ~50 lines

---

## Phase 5: Create Executable Benchmark Scripts

### 5.1 Create `examples/run_bjorken_benchmark.py` (NEW)

**Purpose:** Standalone script to run and validate Bjorken flow benchmark

**Structure:** ~300 lines
- Main function with argument parsing
- Benchmark execution
- Results analysis and plotting
- Report generation
- HDF5 output

**Key features:**
```python
#!/usr/bin/env python3
"""Bjorken Flow Benchmark Runner"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--timestep", type=float, default=0.01)
    parser.add_argument("--output", default="bjorken_results.h5")
    args = parser.parse_args()

    # Create benchmark
    benchmark = create_standard_bjorken_benchmark(
        grid_points=(args.grid_size,) * 3
    )

    # Run full benchmark
    results = benchmark.run_full_benchmark(timestep=args.timestep)

    # Generate plots
    create_bjorken_plots(results)

    # Print report
    print(benchmark.generate_report())

    # Save results
    save_results_hdf5(args.output, results)

    # Exit with status
    sys.exit(0 if results["benchmark_passed"] else 1)
```

---

### 5.2 Create `examples/run_sound_wave_benchmark.py` (NEW)

**Purpose:** Standalone script to validate sound wave propagation

**Structure:** ~400 lines
- Dispersion relation validation
- Frequency/damping measurements
- Multiple wave numbers
- Causality checks
- Comprehensive plotting

---

### 5.3 Create `examples/run_equilibration_benchmark.py` (NEW)

**Purpose:** Standalone script to validate equilibration and entropy production

**Structure:** ~350 lines
- Relaxation timescale validation
- Entropy production analysis
- Second law verification
- Temperature evolution plots

---

## Phase 6: Validation and Testing

### 6.1 Run All Benchmarks

```bash
# Run each benchmark
python examples/run_bjorken_benchmark.py
python examples/run_sound_wave_benchmark.py
python examples/run_equilibration_benchmark.py

# Run tests
pytest test_bjorken_flow_validation.py -v
pytest test_sound_wave_validation.py -v
pytest test_equilibration_validation.py -v
```

### 6.2 Expected Results

**Bjorken Flow:**
- Temperature evolution error < 1%
- Bulk pressure error < 5%
- Convergence order ≥ 2

**Sound Waves:**
- Sound speed: |c_s - 1/√3| < 1%
- Frequency error < 10%
- Damping error < 20%
- k² scaling validated

**Equilibration:**
- Exponential relaxation confirmed
- Timescale error < 20%
- Second law: dS/dt ≥ 0
- Fluxes decay to < 1%

---

## Phase 7: Documentation

### 7.1 Update CLAUDE.md

Add benchmark examples section:

```markdown
## Benchmarks

### Running Benchmarks

The codebase includes three comprehensive benchmarks for validation:

**Bjorken Flow:**
```bash
python examples/run_bjorken_benchmark.py --grid-size 64 --timestep 0.005
```

Validates against exact boost-invariant expansion solution.

**Sound Waves:**
```bash
python examples/run_sound_wave_benchmark.py --grid-size 128
```

Validates dispersion relations and wave propagation.

**Equilibration:**
```bash
python examples/run_equilibration_benchmark.py
```

Validates approach to equilibrium and entropy production.
```

---

## Implementation Summary

### Total Changes

**Files Modified:** 7
- `israel_stewart/benchmarks/bjorken_flow.py` (~150 lines)
- `israel_stewart/benchmarks/sound_waves.py` (~200 lines)
- `israel_stewart/benchmarks/equilibration.py` (~100 lines)
- `israel_stewart/solvers/spectral.py` (~30 lines - add callback)
- `test_bjorken_flow_validation.py` (~50 lines)
- `test_sound_wave_validation.py` (~50 lines)
- `test_equilibration_validation.py` (~50 lines)

**Files Created:** 4
- `examples/run_bjorken_benchmark.py` (~300 lines)
- `examples/run_sound_wave_benchmark.py` (~400 lines)
- `examples/run_equilibration_benchmark.py` (~350 lines)
- `BENCHMARK_REFACTOR_PLAN.md` (this document)

**Total:** ~1680 lines of changes/additions

---

## Breaking Changes

### For Benchmark Users

❌ **Old:**
```python
from israel_stewart.core.spacetime_grid import SpacetimeGrid

grid = SpacetimeGrid(..., grid_points=(20, 32, 32, 32))
benchmark = BjorkenBenchmark(grid, coeffs, analytical)
```

✅ **New:**
```python
from israel_stewart.core.spacegrid import SpaceGrid

grid = SpaceGrid(..., grid_points=(32, 32, 32), boundary_conditions="periodic")
benchmark = BjorkenBenchmark(grid, coeffs, analytical)
```

### Migration Path

1. Use factory functions (automatically create SpaceGrid)
2. Update custom benchmarks to use SpaceGrid
3. Add `boundary_conditions="periodic"` for spectral methods
4. Remove time dimension from field initialization

---

## Success Criteria

✅ All benchmarks use SpaceGrid
✅ All tests pass with pure 3D architecture
✅ Executable scripts run successfully
✅ Numerical accuracy maintained vs analytical solutions
✅ Documentation complete
✅ No performance degradation

**Target:** Quantitative validation of SpectralISHydrodynamics against exact solutions with pure 3D architecture.

---

## Timeline

- **Phase 1:** 1-2 hours (bjorken_flow.py + callback in spectral.py)
- **Phase 2:** 1-2 hours (sound_waves.py)
- **Phase 3:** 1 hour (equilibration.py)
- **Phase 4:** 1 hour (test files)
- **Phase 5:** 2-3 hours (executable scripts)
- **Phase 6:** 1 hour (validation)
- **Phase 7:** 30 minutes (documentation)

**Total estimated time:** 8-10 hours

---

*End of Benchmark Refactor Plan*
