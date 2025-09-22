# Fix Fundamental Physics Issues in Analytical Dispersion Solver

## Problem Analysis

After detailed code review, the analytical solver in `sound_waves.py` has fundamental physics issues:

### 1. **Incorrect Variable Choice**
Current matrix uses `[drho, dp, dPi, dpi]` where:
- `drho` (energy density) and `dp` (pressure) are NOT independent
- They're linked by equation of state: `δp = c_s² δε`
- This makes the system ill-defined

### 2. **Incorrect Matrix Elements**
- Row 0: `iω·δρ + ik·(ρ/(ρ+p))·δp = 0` - not a valid physical equation
- Missing proper conservation law structure
- Incorrect coupling between variables

### 3. **Wrong Physical Foundation**
- Current approach mixes thermodynamic and hydrodynamic variables incorrectly
- Doesn't properly represent linearized Israel-Stewart equations

## Solution Plan

### Phase 1: Implement Correct Physics Foundation

**1.1 Replace Variable Set**
- **OLD**: `[drho, dp, dPi, dpi]`
- **NEW**: `[δε, δv_x, δΠ, δπ_xx]`
  - `δε`: Energy density perturbation
  - `δv_x`: Velocity perturbation (longitudinal)
  - `δΠ`: Bulk pressure perturbation
  - `δπ_xx`: Shear stress perturbation (longitudinal component)

**1.2 Build Correct 4×4 Dispersion Matrix**
For plane wave `exp(-iωt + ikx)`, implement correct linearized equations:

1. **Energy Conservation** (`∂_μ T^μ0 = 0`):
   ```
   (-iω)·δε + ik·(ε₀+p₀)·δv_x = 0
   ```

2. **Momentum Conservation** (`∂_μ T^μx = 0`):
   ```
   ik·c_s²·δε - iω·(ε₀+p₀)·δv_x + ik·δΠ + ik·δπ_xx = 0
   ```

3. **Bulk Pressure Relaxation**:
   ```
   (1 - iωτ_Π)·δΠ + iζk·δv_x = 0
   ```

4. **Shear Stress Relaxation**:
   ```
   (1 - iωτ_π)·δπ_xx + i·(4/3)ηk·δv_x = 0
   ```

### Phase 2: Implement Robust Numerical Solution

**2.1 Replace Determinant Scanning with Root Finding**
- **Current**: Scan frequencies, check `|det| < tolerance`
- **New**: Use `scipy.optimize.root_scalar` to find `det(ω) = 0`

**2.2 Create Determinant Function**
```python
def _determinant_function(self, omega: complex, k: float) -> complex:
    matrix = self._build_dispersion_matrix(omega, k)
    return np.linalg.det(matrix)
```

**2.3 Robust Root Finding**
- Find complex roots ω(k) for given real k
- Handle both propagating and evanescent modes
- Return physical solutions with proper error handling

### Phase 3: Validation and Testing

**3.1 Ideal Limit Validation**
- Test η=ζ=0 case should give ω = c_s·k
- Verify c_s = 1/√3 for radiation equation of state

**3.2 Viscous Corrections**
- Check first-order damping: γ ∝ k²
- Validate relaxation time effects
- Test causality constraints

**3.3 Second-Order Effects**
- Verify coupling coefficients λ_ππ, ξ₁ contributions
- Check dispersion relation accuracy vs literature

### Phase 4: Integration with Numerical Benchmark

**4.1 Update Method Signatures**
- Ensure compatibility with existing `analyze_dispersion_relation` interface
- Maintain `WaveProperties` dataclass structure

**4.2 Enhanced Error Handling**
- Graceful degradation when roots not found
- Clear physics-based error messages
- Robust handling of edge cases

## Implementation Strategy

### Priority Order:
1. **`_build_dispersion_matrix`** - Core physics fix
2. **`_solve_single_mode`** - Root-finding implementation
3. **`analyze_dispersion_relation`** - Integration layer
4. **Validation tests** - Physics verification
5. **Numerical benchmark integration** - End-to-end testing

### Files Modified:
- `israel_stewart/benchmarks/sound_waves.py`
  - `_build_dispersion_matrix()` method (complete rewrite)
  - `_solve_single_mode()` method (root-finding approach)
  - New helper method `_determinant_function()`

### Dependencies:
- `scipy.optimize` for root finding
- Verify transport coefficient handling

## Expected Outcome

✅ **Physically correct analytical dispersion solver**
✅ **Accurate sound speed and damping predictions**
✅ **Reliable benchmark for numerical validation**
✅ **Foundation for Israel-Stewart physics verification**

This addresses the core physics issues while maintaining the existing interface for the numerical benchmark infrastructure.

## Matrix Elements Detail

The correct 4×4 matrix M for variables [δε, δv_x, δΠ, δπ_xx] should be:

```
M = [
    [-iω,           ik(ε₀+p₀),    0,    0     ],  # Energy conservation
    [ikc_s²,        -iω(ε₀+p₀),  ik,   ik    ],  # Momentum conservation
    [0,             iζk,         1-iωτ_Π, 0  ],  # Bulk relaxation
    [0,             i(4/3)ηk,    0,    1-iωτ_π]   # Shear relaxation
]
```

Where:
- ε₀, p₀: Background energy density and pressure
- c_s²: Sound speed squared (typically 1/3 for radiation)
- ζ, η: Bulk and shear viscosity
- τ_Π, τ_π: Bulk and shear relaxation times
- ω, k: Frequency and wave number

The dispersion relation is: det(M) = 0