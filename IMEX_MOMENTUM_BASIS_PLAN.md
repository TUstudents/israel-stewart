# Fully Coupled IMEX for Israel-Stewart Hydrodynamics

## Problem Analysis

The Israel-Stewart system has **bidirectional coupling**:

### Equations:
```
∂_t ρ = -∂_i T^{i0}(ρ, u, Π, π, q)           [Conservation, explicit]
∂_t(ρu^j) = -∂_i T^{ij}(ρ, u, Π, π, q)       [Conservation, explicit]
∂_t Π = -Π/τ_Π - ζ∇·u + nonlinear(Π, π)     [Relaxation: stiff + explicit]
∂_t π = -π/τ_π + 2η σ(u) + nonlinear(π,q,u)  [Relaxation: stiff + explicit]
∂_t q = -q/τ_q + κ∇T + nonlinear(q, π, u)    [Relaxation: stiff + explicit]
```

### Coupling:
- **Conservation → Relaxation**: ∂_t ρ and ∂_t u appear in stress tensor T^{μν}
- **Relaxation → Conservation**: Π, π, q appear in stress tensor T^{μν}
- **Velocity coupling**: u appears in ∇·u and σ for relaxation sources

## Proposed Solution: Extended IMEX with Momentum Density Variables

### Variable Reformulation

Instead of trying to convert ∂_t(ρu) → ∂_t(u), **keep momentum density as primary variable**:

**State vector**: `y = [ρ, mom_x, mom_y, mom_z, Π, π_00, π_01, ..., q_0, q_1, q_2, q_3]`

Where:
- `mom_i = ρ u^i` (momentum density, NOT velocity!)
- Compute `u^i = mom_i / ρ` when needed (derived quantity)

**Advantages**:
1. No product rule conversion needed
2. Conservation equations are natural: ∂_t(ρ) and ∂_t(mom_i)
3. Coupling through u = mom/ρ is explicit algebraic relation
4. Numerically stable: no division in time derivatives

### IMEX Splitting

**Explicit terms F(y)**:
```python
F(y) = [
    F_ρ = -∂_i T^{i0}(ρ, mom/ρ, Π, π, q),      # Energy conservation
    F_mom_x = -∂_i T^{ix}(ρ, mom/ρ, Π, π, q),  # x-momentum conservation
    F_mom_y = -∂_i T^{iy}(ρ, mom/ρ, Π, π, q),  # y-momentum conservation
    F_mom_z = -∂_i T^{iz}(ρ, mom/ρ, Π, π, q),  # z-momentum conservation
    F_Π = -ζ∇·(mom/ρ) + nonlinear_Π(Π, π),    # Bulk source
    F_π = 2η σ(mom/ρ) + nonlinear_π(π, Π, q), # Shear source
    F_q = κ∇T + nonlinear_q(q, π),            # Heat source
]
```

**Implicit (stiff) terms G(y)**:
```python
G(y) = [
    0,         # No stiff term for ρ
    0,         # No stiff term for mom_x
    0,         # No stiff term for mom_y
    0,         # No stiff term for mom_z
    -Π/τ_Π,    # Bulk relaxation (stiff)
    -π/τ_π,    # Shear relaxation (stiff)
    -q/τ_q,    # Heat relaxation (stiff)
]
```

## Implementation Plan

### Step 1: Field Conversion Functions

Create functions to convert between velocity basis and momentum basis:

```python
def _fields_to_momentum_basis(self) -> dict[str, np.ndarray]:
    """Convert ISFieldConfiguration to momentum-density basis."""
    rho = self.fields.rho
    u_spatial = self.fields.u_mu[..., 1:4]

    return {
        "rho": rho.copy(),
        "mom_x": (rho * u_spatial[..., 0]).copy(),
        "mom_y": (rho * u_spatial[..., 1]).copy(),
        "mom_z": (rho * u_spatial[..., 2]).copy(),
        "Pi": self.fields.Pi.copy(),
        "pi_munu": self.fields.pi_munu.copy(),
        "q_mu": self.fields.q_mu.copy(),
    }

def _momentum_basis_to_fields(self, mom_dict: dict[str, np.ndarray]) -> None:
    """Update ISFieldConfiguration from momentum-density basis."""
    rho = mom_dict["rho"]

    # Avoid division by zero
    rho_safe = np.where(np.abs(rho) > 1e-14, rho, 1e-14)

    # Update density
    self.fields.rho[:] = rho

    # Update four-velocity
    self.fields.u_mu[..., 0] = 1.0  # Time component (rest frame approximation)
    self.fields.u_mu[..., 1] = mom_dict["mom_x"] / rho_safe
    self.fields.u_mu[..., 2] = mom_dict["mom_y"] / rho_safe
    self.fields.u_mu[..., 3] = mom_dict["mom_z"] / rho_safe

    # Update dissipative fluxes
    self.fields.Pi[:] = mom_dict["Pi"]
    self.fields.pi_munu[:] = mom_dict["pi_munu"]
    self.fields.q_mu[:] = mom_dict["q_mu"]
```

### Step 2: Compute Explicit RHS in Momentum Basis

```python
def _compute_explicit_rhs_momentum(self) -> dict[str, np.ndarray]:
    """Compute F(y) with y in momentum-density basis."""
    explicit_rhs = {}

    # Conservation laws (already return ∂_t ρ and ∂_t(ρu^i))
    if self.conservation is not None:
        conservation_rhs = self.conservation.evolution_equations()
        explicit_rhs["rho"] = conservation_rhs["drho_dt"]
        # dmom_dt is ALREADY ∂_t(ρu^i) - perfect for momentum basis!
        explicit_rhs["mom_x"] = conservation_rhs["dmom_dt"][..., 0]
        explicit_rhs["mom_y"] = conservation_rhs["dmom_dt"][..., 1]
        explicit_rhs["mom_z"] = conservation_rhs["dmom_dt"][..., 2]

    # Relaxation sources (self.fields already has u = mom/ρ)
    if self.relaxation is not None:
        relaxation_rhs = self._compute_relaxation_sources()
        explicit_rhs["Pi"] = relaxation_rhs.get("Pi", np.zeros_like(self.fields.Pi))
        explicit_rhs["pi_munu"] = relaxation_rhs.get("pi_munu", np.zeros_like(self.fields.pi_munu))
        explicit_rhs["q_mu"] = relaxation_rhs.get("q_mu", np.zeros_like(self.fields.q_mu))

    return explicit_rhs
```

### Step 3: Compute Stiff Terms in Momentum Basis

```python
def _compute_stiff_terms_momentum(self, fields: ISFieldConfiguration) -> dict[str, np.ndarray]:
    """Compute G(y) - only relaxation terms are stiff."""
    stiff_terms = {}

    # Hydrodynamic fields have NO stiff terms
    stiff_terms["rho"] = np.zeros_like(fields.rho)
    stiff_terms["mom_x"] = np.zeros_like(fields.rho)
    stiff_terms["mom_y"] = np.zeros_like(fields.rho)
    stiff_terms["mom_z"] = np.zeros_like(fields.rho)

    # Relaxation terms (stiff linear parts only - sources are in explicit)
    if self.coeffs is not None:
        if hasattr(self.coeffs, "bulk_relaxation_time") and self.coeffs.bulk_relaxation_time:
            stiff_terms["Pi"] = -fields.Pi / self.coeffs.bulk_relaxation_time
        else:
            stiff_terms["Pi"] = np.zeros_like(fields.Pi)

        if hasattr(self.coeffs, "shear_relaxation_time") and self.coeffs.shear_relaxation_time:
            stiff_terms["pi_munu"] = -fields.pi_munu / self.coeffs.shear_relaxation_time
        else:
            stiff_terms["pi_munu"] = np.zeros_like(fields.pi_munu)

        # Heat flux relaxation (if implemented)
        stiff_terms["q_mu"] = np.zeros_like(fields.q_mu)

    return stiff_terms
```

### Step 4: Update IMEX RK2 Step

Modify `_imex_rk2_step()` to work in momentum basis:

```python
def _imex_rk2_step_momentum(self, dt: float) -> None:
    """ARS(2,2,2) IMEX-RK in momentum-density basis."""
    h = dt
    gamma = 1.0 - 1.0 / np.sqrt(2.0)

    # Store initial state in momentum basis
    y_n_dict = self._fields_to_momentum_basis()

    # === Stage 1: Y₁ = y^n + h·γ·G(Y₁) ===
    Y1_dict = self._solve_implicit_stage_momentum(y_n_dict, gamma * h)
    # Convert back to update self.fields for RHS computation
    self._momentum_basis_to_fields(Y1_dict)

    # Compute explicit RHS F(Y₁)
    F_Y1_dict = self._compute_explicit_rhs_momentum()

    # Compute implicit terms G(Y₁) from stage equation
    G_Y1_scaled_dict = self._add_fields_momentum(Y1_dict, y_n_dict, scale=-1.0)
    G_Y1_dict = self._scale_fields_momentum(G_Y1_scaled_dict, scale=1.0 / (gamma * h))

    # === Stage 2 ===
    rhs2_dict = self._add_fields_momentum(y_n_dict, F_Y1_dict, scale=h)
    rhs2_dict = self._add_fields_momentum(rhs2_dict, G_Y1_dict, scale=h * (1.0 - gamma))

    Y2_dict = self._solve_implicit_stage_momentum(rhs2_dict, gamma * h)
    self._momentum_basis_to_fields(Y2_dict)

    # Compute explicit RHS F(Y₂)
    F_Y2_dict = self._compute_explicit_rhs_momentum()

    # Compute implicit terms G(Y₂)
    G_Y2_scaled_dict = self._add_fields_momentum(Y2_dict, rhs2_dict, scale=-1.0)
    G_Y2_dict = self._scale_fields_momentum(G_Y2_scaled_dict, scale=1.0 / (gamma * h))

    # === Final Update ===
    final_dict = y_n_dict.copy()
    final_dict = self._add_fields_momentum(final_dict, F_Y1_dict, scale=h / 2.0)
    final_dict = self._add_fields_momentum(final_dict, F_Y2_dict, scale=h / 2.0)
    final_dict = self._add_fields_momentum(final_dict, G_Y1_dict, scale=h * (1.0 - gamma))
    final_dict = self._add_fields_momentum(final_dict, G_Y2_dict, scale=h * gamma)

    # Convert final result back to velocity basis
    self._momentum_basis_to_fields(final_dict)
```

### Step 5: Helper Functions for Momentum Basis

```python
def _add_fields_momentum(self, base_dict, add_dict, scale=1.0):
    """Add fields in momentum basis."""
    result = {}
    for key in base_dict:
        if key in add_dict:
            result[key] = base_dict[key] + scale * add_dict[key]
        else:
            result[key] = base_dict[key].copy()
    return result

def _scale_fields_momentum(self, field_dict, scale):
    """Scale fields in momentum basis."""
    return {key: scale * field_array for key, field_array in field_dict.items()}

def _solve_implicit_stage_momentum(self, rhs_dict, gamma_dt):
    """Solve implicit stage in momentum basis using Newton-Krylov."""
    # Similar to existing _solve_implicit_stage but works with momentum dict
    # Uses _compute_stiff_terms_momentum instead of _compute_stiff_terms
    ...
```

## Files to Modify

1. **spectral.py** (~line 1280-1400):
   - Add momentum-basis conversion functions
   - Add `_imex_rk2_step_momentum()`
   - Update `_spectral_imex_advance()` to use momentum version

2. **spectral.py** (~line 1360-1400):
   - Re-enable conservation laws in explicit RHS
   - Use momentum-basis formulation (no conversion needed!)

3. **spectral.py** (~line 1955-2030):
   - Add `_compute_stiff_terms_momentum()`
   - Add momentum-basis implicit solver

## Expected Benefits

1. **Correctness**: Conservation laws properly coupled to relaxation
2. **Stability**: No product rule conversion, no field configuration mismatch
3. **Performance**: Sound wave benchmark should complete in ~30s instead of timeout
4. **Accuracy**: Frequency error < 10%, damping error < 20%

## Verification Tests

After implementation:
```bash
# Should complete quickly and show good agreement
uv run python run_sound_wave_benchmark.py --method spectral_imex --wave-number 8.0 --simulation-time 1.0 --no-plot

# All IMEX tests should still pass
uv run pytest israel_stewart/tests/test_spectral_solver.py::TestARS22IMEXRK -v
```
