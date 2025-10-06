# Israel-Stewart Damping Debug Guide

Quick reference for debugging the γ_analytical vs γ_measured discrepancy.

---

## Quick Diagnosis

### 1. Print Dispersion Matrix (10 seconds)

```python
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients
import numpy as np

coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=0.5,
    bulk_relaxation_time=0.3
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2*np.pi,
    grid_points=(32, 32, 16),
    transport_coeffs=coeffs
)

# Test matrix at trial frequency
omega = complex(6.0, -0.5)  # ω - iγ
k_vec = np.array([8.0, 0.0, 0.0])

matrix = benchmark.analytical._build_dispersion_matrix(omega, k_vec)
det = np.linalg.det(matrix)

print("Dispersion Matrix M(ω=6-0.5i, k=8):")
print(matrix)
print(f"\nDeterminant: {det}")
print(f"|det|: {abs(det)}")

# Expected elements:
# matrix[0,0] = -iω = -6i
# matrix[0,1] = ikh = 10.67i  (h=4/3, k=8)
# matrix[2,1] = iζk = 0.32i   (ζ=0.04, k=8)
# matrix[2,2] = 1-iωτ_Π = 1-1.8i  (τ_Π=0.3)
```

**Check**:
- Is `matrix[2,2]` = `1 - iωτ_Π` = `1 - 6i×0.3` = `1 - 1.8i`? ✓
- Is `matrix[2,1]` = `iζk` = `i×0.04×8` = `0.32i`? ✓

### 2. Test Expansion Scalar Sign (30 seconds)

```python
from israel_stewart.solvers.spectral import SpectralISHydrodynamics
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.core.fields import ISFieldConfiguration
import numpy as np

# Create grid
grid = SpaceGrid(
    coordinate_system="cartesian",
    spatial_ranges=[(0, 2*np.pi)]*3,
    grid_points=(32, 32, 16),
    boundary_conditions="periodic"
)

# Initialize fields
fields = ISFieldConfiguration(grid)
fields.rho[:] = 1.0
fields.pressure[:] = 1/3
fields.u_mu[..., 0] = 1.0

# Create spectral solver (no coeffs needed for this test)
from israel_stewart.solvers.spectral_ops import SpectralOperators
spectral = SpectralOperators(grid)

# Test velocity: u^x = A sin(kx)
A = 0.01
k = 8.0
X, Y, Z = grid.meshgrid()
fields.u_mu[..., 1] = A * np.sin(k * X)

# Compute expansion scalar
velocity_spatial = fields.u_mu[..., 1:4]
theta = spectral.spatial_divergence(velocity_spatial)

# Expected: θ = ∂_x u^x = Ak cos(kx)
theta_expected = A * k * np.cos(k * X)

print("Expansion Scalar Test:")
print(f"At x=0 (max): θ_computed = {theta[0, 0, 0]:.6f}")
print(f"              θ_expected = {theta_expected[0, 0, 0]:.6f}")
print(f"At x=π/k (zero): θ_computed = {theta[16, 0, 0]:.6f}")
print(f"                 θ_expected = {theta_expected[16, 0, 0]:.6f}")

# CRITICAL: Both should match!
# If signs differ → FFT derivative has wrong sign
```

**Expected**:
- At x=0: θ = +0.08 (for A=0.01, k=8)
- At x=π/k: θ ≈ 0

**If wrong**:
- θ = -0.08 at x=0 → FFT derivative has wrong sign!

### 3. Test Bulk RHS (1 minute)

```python
from israel_stewart.equations.relaxation import ISRelaxationEquations
from israel_stewart.core.metrics import MinkowskiMetric

# Create relaxation module
metric = MinkowskiMetric()
relaxation = ISRelaxationEquations(grid, metric, coeffs)

# Set test state
fields.Pi[:] = 0.1
fields.pi_munu[:] = 0.0
fields.u_mu[..., 1] = 0.01 * np.sin(8.0 * X)

# Compute expansion
theta = spectral.spatial_divergence(fields.u_mu[..., 1:4])

# Compute bulk RHS
dPi_dt = relaxation._bulk_rhs(fields.Pi, fields.pi_munu, theta)

print("\nBulk RHS Test:")
print(f"Π = {fields.Pi[0, 0, 0]:.3f}")
print(f"θ(x=0) = {theta[0, 0, 0]:.6f}")
print(f"dΠ/dt(x=0) = {dPi_dt[0, 0, 0]:.6f}")

# Expected at x=0:
# dΠ/dt = -Π/τ_Π - ζθ
#       = -0.1/0.3 - 0.04×0.08
#       = -0.333 - 0.0032
#       = -0.336

print(f"Expected: -0.336")
print(f"Error: {abs(dPi_dt[0, 0, 0] + 0.336):.6f}")
```

### 4. Monitor Fourier Mode Directly (during simulation)

```python
# In run_simulation, add diagnostic:
time_points = []
rho_k_amplitudes = []

k_index = 8  # For k=8 with domain 2π
rho_fft = np.fft.fftn(fields.rho - 1.0)
initial_amplitude = abs(rho_fft[k_index, 0, 0])

def callback(t, fields):
    rho_fft = np.fft.fftn(fields.rho - 1.0)
    amplitude = abs(rho_fft[k_index, 0, 0])

    time_points.append(t)
    rho_k_amplitudes.append(amplitude)

    if len(time_points) % 20 == 0:
        gamma_instant = -np.log(amplitude/initial_amplitude) / t if t > 0 else 0
        print(f"t={t:.3f}: |ρ_k|={amplitude:.6e}, γ_inst={gamma_instant:.6f}")

# After simulation:
import matplotlib.pyplot as plt
plt.semilogy(time_points, rho_k_amplitudes, 'o-')
plt.xlabel('Time')
plt.ylabel('|ρ_k|')
plt.title('Fourier Mode Amplitude')
plt.grid(True)
plt.savefig('/tmp/mode_amplitude.png')
```

---

## Critical Checks

### Sign Convention Audit

| Location | Expected | Check Command |
|----------|----------|---------------|
| Energy row [0,0] | `-iω` | `matrix[0,0]` |
| Energy row [0,1] | `+ikh` | `matrix[0,1]` |
| Bulk row [2,1] | `+iζk` | `matrix[2,1]` (NOT `-iζk`) |
| Bulk row [2,2] | `1-iωτ_Π` | `matrix[2,2]` |
| Expansion θ at x=0 | `> 0` for `u^x=sin(kx)` | `theta[0,0,0] > 0` |
| Bulk RHS at equilibrium | `< 0` for θ>0 | `dPi_dt < 0` when Π=0, θ>0 |

### File & Line Numbers

| Component | File | Function | Line |
|-----------|------|----------|------|
| **Matrix [2,1]** | sound_waves.py | `_build_dispersion_matrix` | 425 |
| **Matrix [2,2]** | sound_waves.py | `_build_dispersion_matrix` | 426 |
| **Expansion** | spectral.py | `_compute_expansion_scalar` | 1887 |
| **Bulk RHS linear** | relaxation.py | `_bulk_rhs` | 203-207 |
| **Bulk RHS source** | relaxation.py | `_bulk_rhs` | 209-210 |

---

## Common Bugs & Fixes

### Bug 1: Wrong FFT Derivative Sign

**Symptom**: θ has opposite sign from expected

**Cause**: FFT using `exp(-ikx)` instead of `exp(+ikx)`

**Fix**: In `spatial_divergence()`:
```python
# WRONG:
div_k = -1j * (kx*ux_k + ky*uy_k + kz*uz_k)

# CORRECT:
div_k = 1j * (kx*ux_k + ky*uy_k + kz*uz_k)
```

**Test**:
```python
u^x = sin(kx) → θ = k cos(kx)
At x=0: θ should be +k (positive!)
```

### Bug 2: Missing Minus Sign in Bulk Source

**Symptom**: Bulk pressure grows instead of relaxing

**Cause**: Source term has wrong sign

**Current** (`relaxation.py:210`):
```python
first_order = -self.coeffs.bulk_viscosity * theta  # CORRECT
```

**Check**: For compression (θ>0), source should reduce Π (negative)

### Bug 3: IMEX Double-Counting

**Symptom**: Relaxation too fast or too slow in IMEX mode

**Cause**: Linear terms included in both F(y) and G(y)

**Fix** (`spectral.py:1311-1315`):
```python
if self._integration_mode == "spectral_imex":
    # Remove stiff term from F (already in G)
    dPi_dt += self.fields.Pi / self.coeffs.bulk_relaxation_time
```

**Verify**: Full RHS = F + G should equal `compute_relaxation_rhs()` output

### Bug 4: Damping Extraction from Wrong Signal

**Symptom**: Measured γ ≈ 0 or negative

**Cause**: Using point measurement instead of Fourier mode

**Fix**: Track `|ρ_k(t)|` directly, not `ρ(x_monitor, t)`

**Reason**: Point measurement affected by phase evolution; Fourier mode shows pure exponential decay

---

## Expected Values (k=8.0 Test Case)

### Analytical

```
ω_analytical ≈ 6.0 /time
γ_analytical ≈ 0.51 /time
ωτ_Π ≈ 1.8 (moderate relaxation)
ωτ_π ≈ 3.0 (strong relaxation)
```

### Numerical (if correct)

```
ω_measured ≈ 5.9-6.1 /time (1-2% error acceptable)
γ_measured ≈ 0.4-0.6 /time (10-20% error acceptable)
|ρ_k(t)| should decay exponentially
```

### Current Bug

```
ω_measured ≈ 5.9 /time ✓ (frequency is fine)
γ_measured ≈ -0.025 /time ✗ (WRONG SIGN!)
|ρ_k(t)| is GROWING → instability
```

---

## Investigation Priority

### High Priority (Most Likely Bugs)

1. **Expansion scalar sign** (`spectral.py:1887`)
   - Test with `u^x = sin(kx)` → expect `θ = k cos(kx) > 0` at x=0
   - If negative → FFT derivative wrong

2. **Bulk source term sign** (`relaxation.py:210`)
   - Should be `-ζθ` (negative for compression)
   - Test: Π=0, θ>0 → dΠ/dt < 0

3. **Dispersion matrix bulk row** (`sound_waves.py:425-426`)
   - Should be `+iζk` for source (NOT `-iζk`)
   - Should be `1-iωτ_Π` for relaxation

### Medium Priority

4. **IMEX source term handling** (`spectral.py:1311-1315`)
   - Verify linear terms removed from F in IMEX mode
   - Compare split_step vs spectral_imex results

5. **Initial conditions** (`sound_waves.py:1239-1259`)
   - Verify eigenmode ratios match analytical solution
   - Check Π, π initialized correctly

### Low Priority

6. **Root finding damping cap** (`sound_waves.py:328-329`)
   - Capping at 30% might exclude correct solution
   - Try without cap for k=8

7. **Transport coefficients** (test different values)
   - Try τ_Π = 0.1, τ_π = 0.1 (weaker relaxation)
   - Try η = 0.2, ζ = 0.1 (stronger viscosity)

---

## Quick Fix Template

```python
# 1. Test expansion scalar
theta_test = """
fields.u_mu[..., 1] = 0.01 * np.sin(8*X)
theta = spectral.spatial_divergence(fields.u_mu[..., 1:4])
print(f"θ(x=0) = {theta[0,0,0]:.6f} (expect +0.08)")
"""

# 2. Test bulk RHS
bulk_test = """
fields.Pi[:] = 0.1
theta[:] = 0.08  # Compression
dPi = relaxation._bulk_rhs(fields.Pi, fields.pi_munu, theta)
print(f"dΠ/dt(x=0) = {dPi[0,0,0]:.6f} (expect -0.336)")
"""

# 3. Monitor mode amplitude
mode_monitor = """
k_idx = 8
rho_k = np.fft.fftn(fields.rho - 1.0)[k_idx, 0, 0]
print(f"t={t:.3f}, |ρ_k|={abs(rho_k):.6e}")
# Should decay: |ρ_k| = A₀ exp(-γt)
"""
```

---

## Success Criteria

After fixes:
- [ ] Expansion scalar: θ > 0 at x=0 for `u^x = sin(kx)`
- [ ] Bulk RHS: dΠ/dt < 0 when Π=0, θ>0
- [ ] Dispersion matrix: `matrix[2,1] = +iζk` (positive imaginary)
- [ ] Fourier mode: |ρ_k(t)| decays exponentially
- [ ] Measured damping: 0.4 < γ < 0.6 /time
- [ ] Damping error: < 20%

**Current Status**: ❌ All checks failing
**Root Cause**: Likely expansion scalar sign or bulk source term sign
**Next Step**: Run tests 1 & 2 above to isolate bug
