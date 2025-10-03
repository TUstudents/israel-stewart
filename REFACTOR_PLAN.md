# Pure 3D Refactor Plan: SpaceGrid Architecture

## Core Philosophy
**Separation of Concerns**: Space (grid) vs Time (evolution parameter) vs History (trajectory files)

---

## Phase 1: New SpaceGrid Module

### 1.1 Create `core/spacegrid.py`
New pure 3D spatial grid class to replace SpacetimeGrid for evolution:

```python
class SpaceGrid:
    """Pure 3D spatial grid for hydrodynamic evolution."""

    def __init__(
        self,
        coordinate_system: str,          # 'cartesian', 'spherical', 'cylindrical'
        spatial_ranges: list[tuple[float, float]],  # [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        grid_points: tuple[int, int, int],          # (nx, ny, nz) - pure 3D
        metric: Optional[MetricBase] = None,
        boundary_conditions: Literal["periodic", "dirichlet", "neumann"] = "periodic"
    ):
        self.coordinate_system = coordinate_system
        self.spatial_ranges = spatial_ranges
        self.grid_points = grid_points  # (nx, ny, nz)
        self.metric = metric
        self.boundary_conditions = boundary_conditions

        # Grid properties
        self.shape = grid_points  # (nx, ny, nz)
        self.ndim = 3
        self.nx, self.ny, self.nz = grid_points

        # Spatial spacing
        self.dx, self.dy, self.dz = self._compute_spacing()
        self.spatial_spacing = (self.dx, self.dy, self.dz)

        # Create coordinate arrays (3D only)
        self.coordinates = self._create_coordinate_arrays()
```

**Key features:**
- No time dimension in grid
- Pure 3D spatial operations
- Coordinate arrays: `{'x': ndarray(nx), 'y': ndarray(ny), 'z': ndarray(nz)}`
- Methods: `meshgrid()`, `gradient(field, axis)`, `divergence(vector_field)`, `laplacian(field)`
- ~400 lines (simpler than SpacetimeGrid)

### 1.2 Keep SpacetimeGrid for Trajectory Metadata
- SpacetimeGrid remains for HDF5 metadata storage only
- Used by TrajectoryWriter to record simulation domain info
- NOT used for field evolution

---

## Phase 2: Field Storage Refactor

### 2.1 Update `ISFieldConfiguration`
Complete rewrite to pure 3D storage:

```python
class ISFieldConfiguration:
    def __init__(self, grid: SpaceGrid):
        self.grid = grid
        nx, ny, nz = grid.shape

        # All fields are pure 3D spatial
        self.rho = np.zeros((nx, ny, nz))           # Energy density
        self.n = np.zeros((nx, ny, nz))             # Particle density
        self.u_mu = np.zeros((nx, ny, nz, 4))       # Four-velocity
        self.Pi = np.zeros((nx, ny, nz))            # Bulk pressure
        self.pi_munu = np.zeros((nx, ny, nz, 4, 4)) # Shear tensor
        self.q_mu = np.zeros((nx, ny, nz, 4))       # Heat flux

        # Thermodynamic fields
        self.pressure = np.zeros((nx, ny, nz))
        self.temperature = np.zeros((nx, ny, nz))

        # Initialize to rest frame
        self.u_mu[..., 0] = 1.0
```

**Changes:**
- Remove all `grid.shape` → use `(nx, ny, nz)` directly
- Remove MSRJD noise fields (not needed yet)
- Remove transport coefficient fields (belong in TransportCoefficients)
- Pure 3D state vector packing/unpacking

**Impact:** ~300 lines modified in fields.py

---

## Phase 3: Spectral Solver Refactor

### 3.1 Update `SpectralISolver.__init__`
```python
class SpectralISolver:
    def __init__(self, grid: SpaceGrid, fields: ISFieldConfiguration, coeffs):
        self.grid = grid
        self.fields = fields
        self.coeffs = coeffs

        # Store ONLY spatial dimensions
        self.nx, self.ny, self.nz = grid.shape
        self.dx, self.dy, self.dz = grid.spatial_spacing

        # NO nt, NO dt attributes

        # Precompute wave vectors for FFT
        self.k_vectors = self._compute_wave_vectors()
        self.k_squared = self._compute_k_squared()
```

### 3.2 Simplify All Methods
Remove every instance of:
- `field[-1, :, :, :]` → just `field`
- 4D shape checking
- Time slice extraction logic
- `if field.ndim == 4:` branches

**Key methods to update:**
- `spatial_derivative()`: Pure 3D input/output
- `_compute_laplacian()`: Remove 4D branch entirely
- `spatial_divergence()`: Pure 3D vector field input
- `time_step()`: Direct 3D field updates

**Impact:** ~500 lines modified/deleted in spectral.py

---

## Phase 4: Physics Equations Update

### 4.1 `conservation.py`
- `stress_energy_tensor()`: Returns `(nx, ny, nz, 4, 4)` tensor
- `evolution_equations()`: Returns dict with `(nx, ny, nz)` spatial fields
- Already correct: uses spatial divergence only
- Just remove 4D field handling

**Impact:** ~100 lines

### 4.2 `relaxation.py`
- All methods operate on pure 3D fields
- Update field access: no more `[-1,:,:,:]`
- Tensor contractions remain same (just 3D base)

**Impact:** ~150 lines

---

## Phase 5: Streaming Architecture

### 5.1 New `utils/streaming.py`
```python
class SnapshotStream:
    """Buffered snapshot streaming with automatic flushing."""

    def __init__(self, filename: str, grid: SpaceGrid, coeffs,
                 interval: float = 0.1, buffer_size: int = 10):
        # Create SpacetimeGrid for trajectory metadata
        spacetime_grid = self._spacegrid_to_spacetimegrid(grid)
        self.writer = TrajectoryWriter(filename, spacetime_grid, coeffs)

        self.interval = interval
        self.buffer_size = buffer_size
        self.buffer = []
        self.last_snapshot_time = -np.inf

    def should_save(self, t: float) -> bool:
        return (t - self.last_snapshot_time) >= self.interval

    def save(self, t: float, fields: ISFieldConfiguration):
        snapshot = self._copy_fields(fields)
        self.buffer.append((t, snapshot))

        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        for t, fields_copy in self.buffer:
            self.writer.write_snapshot(t, fields_copy)
        self.buffer.clear()
        self.last_snapshot_time = self.buffer[-1][0] if self.buffer else self.last_snapshot_time

    def _spacegrid_to_spacetimegrid(self, grid: SpaceGrid) -> SpacetimeGrid:
        """Convert SpaceGrid to SpacetimeGrid for metadata."""
        return SpacetimeGrid(
            coordinate_system=grid.coordinate_system,
            time_range=(0.0, 1.0),  # Dummy range
            spatial_ranges=grid.spatial_ranges,
            grid_points=(1, *grid.grid_points),  # nt=1 for metadata
            metric=grid.metric,
            boundary_conditions=grid.boundary_conditions
        )
```

### 5.2 Update `evolve()` in spectral.py
```python
def evolve(self, t_final: float, dt: float | None = None,
           snapshot_config: dict | None = None):
    """Pure 3D time evolution with streaming snapshots."""

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

    t = 0.0
    try:
        while t < t_final:
            if dt is None:
                dt_step = self.adaptive_time_step()
            else:
                dt_step = dt

            dt_step = min(dt_step, t_final - t)
            self.time_step(dt_step)
            t += dt_step

            if stream and stream.should_save(t):
                stream.save(t, self.fields)
    finally:
        if stream:
            stream.flush()
            stream.close()
```

**Impact:** ~250 lines new file, ~100 lines modified in spectral.py

---

## Phase 6: Test Suite Complete Rewrite

### 6.1 New Fixtures
```python
@pytest.fixture
def test_grid():
    """Pure 3D spatial grid."""
    return SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(8, 8, 8),
        boundary_conditions="periodic"
    )

@pytest.fixture
def test_fields(test_grid):
    """3D field configuration."""
    fields = ISFieldConfiguration(test_grid)
    fields.rho[:] = 1.0  # Direct 3D indexing
    fields.u_mu[..., 0] = 1.0
    return fields
```

### 6.2 Update All Tests
- `test_fields.py`: All shape assertions → 3D
- `test_conservation.py`: Update tensor shapes
- `test_relaxation.py`: Update field operations
- `test_spectral.py`: Rewrite for pure 3D
- `test_trajectory_io.py`: Update field extraction (now writes 3D, reads 3D)

**Impact:** ~800 lines across test suite

---

## Phase 7: Examples Rewrite

### 7.1 `save_wave_evolution.py`
```python
def main():
    # Pure 3D spatial grid
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2*np.pi)] * 3,
        grid_points=(32, 32, 32),
        boundary_conditions="periodic"
    )

    # Initialize 3D fields
    fields = ISFieldConfiguration(grid)
    x, y, z = grid.coordinates['x'], grid.coordinates['y'], grid.coordinates['z']
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Direct 3D initialization
    fields.rho[:] = rho_0 + amplitude * np.sin(k * X)
    fields.u_mu[..., 0] = np.sqrt(1.0 + u_x**2)
    fields.u_mu[..., 1] = u_x

    # Evolve with streaming
    hydro = SpectralISHydrodynamics(grid, fields, coeffs)
    hydro.evolve(
        t_final=5.0,
        dt=0.01,
        snapshot_config={
            'filename': 'wave_evolution.h5',
            'interval': 0.1,
            'buffer_size': 20,
            'save_initial': True
        }
    )
```

### 7.2 New `streaming_long_run.py`
Demonstrate 10,000 timestep evolution with constant memory.

**Impact:** ~400 lines across examples/

---

## Phase 8: Documentation

### 8.1 Update `CLAUDE.md`
- Document SpaceGrid vs SpacetimeGrid distinction
- Update all example code to 3D
- Clarify architecture: "3D spatial grid + time evolution parameter"

### 8.2 Update Docstrings
- `spacegrid.py`: Complete module docstring
- `spectral.py`: Remove "4D" references, clarify "3+1D"
- `fields.py`: Document pure 3D storage model
- `streaming.py`: Comprehensive streaming docs

**Impact:** ~200 lines

---

## Implementation Order

1. **Create `core/spacegrid.py`** (~400 lines new file)
2. **Update `fields.py`** (~300 lines modified)
3. **Update `spectral.py`** (~500 lines modified/deleted)
4. **Update `conservation.py` + `relaxation.py`** (~250 lines)
5. **Create `utils/streaming.py`** (~250 lines new file)
6. **Update tests/** (~800 lines)
7. **Update examples/** (~400 lines)
8. **Update documentation** (~200 lines)

**Total:** ~3100 lines, 2 new files

---

## Breaking Changes (Complete)

❌ **SpacetimeGrid for evolution** - Use SpaceGrid instead
❌ **grid_points=(nt,nx,ny,nz)** → **grid_points=(nx,ny,nz)**
❌ **field shapes (nt,nx,ny,nz)** → **(nx,ny,nz)**
❌ **field[-1,:,:,:]** → **field**
❌ **grid.nt, solver.nt** - Removed entirely
❌ **grid.dt** - Removed (dt is evolution parameter, not grid property)
❌ **evolve(save_trajectory={...})** → **evolve(snapshot_config={...})**

---

## Memory Savings

**Before:** `fields.rho` = (20, 32, 32, 32) × 8 bytes = **5.24 MB**
**After:** `fields.rho` = (32, 32, 32) × 8 bytes = **0.26 MB**

**Full ISFieldConfiguration:** ~100 MB → ~5 MB = **95% reduction**

---

## File Structure After Refactor

```
israel_stewart/
├── core/
│   ├── spacegrid.py          # NEW: Pure 3D spatial grid
│   ├── spacetime_grid.py     # KEPT: For trajectory metadata only
│   ├── fields.py             # MODIFIED: Pure 3D storage
│   └── ...
├── solvers/
│   └── spectral.py           # MODIFIED: Pure 3D operations
├── equations/
│   ├── conservation.py       # MODIFIED: Remove 4D branches
│   └── relaxation.py         # MODIFIED: Pure 3D operations
├── utils/
│   ├── streaming.py          # NEW: Snapshot streaming system
│   └── io.py                 # KEPT: Trajectory I/O (unchanged)
└── ...
```
