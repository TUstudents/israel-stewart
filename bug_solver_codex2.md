# Solver Module Defects (Codex Review)

## `israel_stewart/solvers/finite_difference.py`

### Conservative flux slicing drops interior cells
- **Location**: `israel_stewart/solvers/finite_difference.py:299-333`
- **Severity**: High (breaks discrete conservation and corrupts array shapes)
- **Impact**: The left/right interface slices each produce `N-1` entries, and the subsequent `slice(self.ghost_points, -self.ghost_points)` trims two additional cells. The returned derivative is shorter than the underlying field, so conservation updates either broadcast into the wrong locations or raise shape mismatches once ghost padding is removed. This is visible immediately for small grids (e.g. three spatial points) where the derivative becomes empty.
- **Recommendation**: Build face flux arrays that already match the interior extent, then trim exactly once. A simple fix is to slice the interfaces with `slice(self.ghost_points-1, -self.ghost_points-1)` / `slice(self.ghost_points, -self.ghost_points)` (or equivalent explicit indices) so the final difference retains all interior cells.

### Fourth-order conservative scheme never activates
- **Location**: `israel_stewart/solvers/finite_difference.py:284-291`, `:335-338`
- **Severity**: High (silent order reduction to second order)
- **Impact**: The `order==4` branch requests numerical fluxes at offsets ±1.5, but `_compute_numerical_flux` only understands ±0.5. The helper emits a warning and returns zero arrays, so the four-point stencil collapses to the second-order derivative. Users who select fourth order therefore run a materially different scheme without notification.
- **Recommendation**: Either implement the required higher-order interface reconstructions (e.g. biased four-point averages or a proper flux-splitting routine) or fail fast with a `NotImplementedError` until wider offsets are supported.

### Second-derivative and WENO stencils overshoot the interior
- **Location**: `israel_stewart/solvers/finite_difference.py:353-358`, `:676-682`
- **Severity**: Medium/High (shape corruption and ghost contamination)
- **Impact**: For non-negative offsets the code leaves the stop index as `None`, so the slice includes the padded ghost cells. The resulting arrays are longer than the physical field and either misalign when assembled or silently reintroduce ghosts into the interior. This is especially problematic in the WENO reconstruction where weight normalisation assumes matching shapes.
- **Recommendation**: Compute explicit `start` and `stop` indices for every offset (e.g. `start = self.ghost_points + offset`, `stop = -self.ghost_points + offset` for all sign combinations) or use `np.take` with validated index ranges.

### Dimension bookkeeping references a missing attribute
- **Location**: `israel_stewart/solvers/finite_difference.py:447-488`, `:636-641`
- **Severity**: High (crashes on first call in default configuration)
- **Impact**: Both the upwind and WENO paths call `self.grid.spatial_dimensions`, but `SpacetimeGrid` does not define that attribute. Any attempt to build derivatives triggers an `AttributeError`, so the solvers fail before doing useful work.
- **Recommendation**: Derive the spatial dimension count from existing data (`len(self.dx)` or `len(self.grid.spatial_ranges)`) and cache it on the scheme instance.

### Characteristic selection ignores left-going families
- **Location**: `israel_stewart/solvers/finite_difference.py:479-488`
- **Severity**: Medium (loss of upwind stability for one characteristic family)
- **Impact**: The decision logic compares `lambda_plus` with zero, but negative characteristic speeds (from `lambda_minus`) still use the “upwind” coefficients. That selects the wrong stencil for left-moving waves, introducing numerical diffusion and, in the worst case, instability in shocks propagating against the flow.
- **Recommendation**: Evaluate both characteristic families and choose the stencil based on the sign of the relevant eigenvalue for the evolved quantity, or restructure the flux split so that `lambda_plus` and `lambda_minus` weight different biased derivatives.

### Curved-space divergence ignores Christoffel terms
- **Location**: `israel_stewart/solvers/finite_difference.py:399-407`
- **Severity**: High for non-Cartesian grids (breaks energy-momentum conservation)
- **Impact**: `_compute_christoffel_contributions` always returns zero. In Milne, spherical, or any curved metric, the true covariant divergence requires the connection term `Γ^μ_{αβ} T^{αβ}`; omitting it introduces large systematic errors and violates discrete conservation laws.
- **Recommendation**: Request the connection coefficients from the metric object (the metric API already provides them) and contract with `tensor_field`. Only skip the term when the metric is explicitly flat.

## `israel_stewart/solvers/implicit.py`

### Stiffness estimator ignores stabilising eigenvalues
- **Location**: `israel_stewart/solvers/implicit.py:137-145`
- **Severity**: Medium (misleading timestep recommendations)
- **Impact**: The estimator drops eigenvalues with negative real part, yet those modes define the fastest decay scales. For purely dissipative systems the ratio collapses to unity, causing `recommend_timestep` to select time steps orders of magnitude too large. This destabilises implicit-explicit integrations and defeats adaptive control.
- **Recommendation**: Base the stiffness ratio on the absolute real part (or magnitude) of all eigenvalues whose modulus exceeds a small tolerance. That preserves information about strongly damped modes.

### Relaxation Jacobian populates wrong rows
- **Location**: `israel_stewart/solvers/implicit.py:965-988`
- **Severity**: High (singular Newton systems, stalled convergence)
- **Impact**: The cursor `idx` is not advanced past the conserved variables before the Π block is written. As a result the Π relaxation rates overwrite the `[rho, n, u_mu]` rows, while the actual Π rows remain zero. During Newton iteration the Jacobian loses the diagonal relaxation entries, making the linear solves ill-conditioned or singular.
- **Recommendation**: Advance `idx` by each preceding field’s size (`rho`, `n`, `u_mu`) before inserting the dissipative blocks, or compute offsets from the field sizes directly to avoid manual bookkeeping errors.

### Vectorised Jacobian path materialises a dense matrix
- **Location**: `israel_stewart/solvers/implicit.py:597-604`
- **Severity**: Medium/High for large grids (defeats sparse memory safeguards)
- **Impact**: Even when `use_sparse` is `True`, the code assembles columns via `np.column_stack`, producing a fully dense `n×n` array before wrapping it in CSR. This negates the earlier 1 GB guard and can exhaust memory for moderate lattice sizes.
- **Recommendation**: Accumulate `(data, row, col)` triplets per batch and build the CSR matrix directly (`sparse.csr_matrix((data, (rows, cols)), shape=(n, n))`). Alternatively, append CSR blocks with `sparse.hstack` without ever creating a dense intermediate.

## `israel_stewart/solvers/spectral.py`

### Periodic spacing inconsistent with stored data
- **Location**: `israel_stewart/solvers/spectral.py:55-63`
- **Severity**: Medium (spectral derivatives lose accuracy, Gibbs artefacts)
- **Impact**: `SpacetimeGrid` populates coordinates with `np.linspace(..., N)` (spacing `L/(N-1)`), but `SpectralISolver` overrides `dx` to `L/N` without relocating the field values. The Fourier wave numbers no longer match the sample positions, so derivatives and dealiasing masks drift, especially near boundaries.
- **Recommendation**: Either rebuild the grid using spectral spacing (`grid._create_spectral_coordinates()` and update the field arrays accordingly) or require the caller to supply fields already sampled on the periodic mesh before overriding `dx`.

## `israel_stewart/solvers/splitting.py`

### Spatial gradients accidentally differentiate in time
- **Location**: `israel_stewart/solvers/splitting.py:427-434`, `:471-473`
- **Severity**: High (expansion/shear sources computed from wrong axis)
- **Impact**: The helper uses `np.roll(..., axis=i)` inside a loop over spatial directions. When `i == 0` the roll operates along axis 0 of `u_mu`, which is the time dimension. Consequently the expansion scalar and shear magnitude mingle time and space derivatives, producing large, unphysical source terms.
- **Recommendation**: Restrict the roll to spatial axes (e.g. `axis=i+1` after slicing off the time dimension) or operate on `u_mu[..., 1:4]` with explicit axis names. This keeps the diagnostic consistent with the intended spatial gradients.
