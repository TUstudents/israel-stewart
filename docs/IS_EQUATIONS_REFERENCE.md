# Israel-Stewart Equations Reference

Complete mathematical reference for all IS equations and their implementation.

---

## 1. Background Thermodynamics

### Equation of State (Conformal Radiation Fluid)

**Pressure-energy relation**:
```
p = ε/3
```

**Implementation**: `israel_stewart/core/fields.py`
```python
def update_pressure_from_eos(self, eos_type="radiation"):
    if eos_type == "radiation":
        self.pressure = self.rho / 3.0  # p = ε/3
```

**Sound speed**:
```
c_s² = ∂p/∂ε = 1/3
c_s = 1/√3 ≈ 0.5773
```

**Enthalpy density**:
```
h = ε + p = ε + ε/3 = (4/3)ε
```

For background ε₀ = 1, p₀ = 1/3:
- h₀ = 4/3
- c_s = 1/√3

---

## 2. Conservation Laws

### Energy-Momentum Tensor

**Perfect fluid contribution**:
```
T^μν_perfect = (ε + p)u^μ u^ν + p g^μν
```

**Viscous corrections**:
```
T^μν = T^μν_perfect + Π Δ^μν + π^μν
```

where:
- Π = bulk viscous pressure (scalar)
- π^μν = shear stress tensor (traceless, symmetric)
- Δ^μν = g^μν + u^μ u^ν (spatial projection)

**Implementation**: `israel_stewart/equations/conservation.py:136-168`

### Energy Conservation

**Covariant form**:
```
∇_μ T^μ0 = 0
```

**Component form** (Minkowski, ∂_t = -∂_τ):
```
∂_t ε + ∂_i[(ε+p)u^i] + Π ∂_i u^i + π^ij ∂_i u_j = 0
```

**Linearized** (around equilibrium u^i=0):
```
∂_t δε + (ε₀+p₀)∂_i δu^i + ∂_i δΠ = 0
```

**For plane wave** `exp(-iωt + ikx)`:
```
-iω δε + (ε₀+p₀)(ik)δv_x = 0
```

**Implementation**: Dispersion matrix row 0
```python
matrix[0,0] = -1j*omega       # δε coefficient
matrix[0,1] = 1j*k*enthalpy   # δv_x coefficient
```

### Momentum Conservation

**Covariant form**:
```
∇_μ T^μi = 0
```

**Component form** (x-direction):
```
(ε+p)∂_t u^x + ∂_x p + ∂_x Π + ∂_x π^xx = 0
```

**Linearized**:
```
(ε₀+p₀)∂_t δu^x + c_s² ∂_x δε + ∂_x δΠ + ∂_x δπ^xx = 0
```

**For plane wave**:
```
-iω(ε₀+p₀)δv_x + (ik)c_s² δε + (ik)δΠ + (ik)δπ_xx = 0
```

**Implementation**: Dispersion matrix row 1
```python
matrix[1,0] = 1j*k*cs_squared    # δε coefficient (pressure gradient)
matrix[1,1] = -1j*omega*enthalpy # δv_x coefficient (inertia)
matrix[1,2] = 1j*k               # δΠ coefficient (bulk force)
matrix[1,3] = 1j*k               # δπ_xx coefficient (shear force)
```

---

## 3. Israel-Stewart Relaxation Equations

### Bulk Viscous Pressure

**Covariant IS equation**:
```
τ_Π D_u Π + Π = -ζθ + ξ₁Πθ + ξ₂Π²/(ζτ_Π) + λ_Ππ π^μν σ_μν
```

where:
- `D_u = u^μ ∇_μ` = comoving derivative
- `θ = ∇_μ u^μ` = expansion scalar
- `ξ₁, ξ₂` = second-order coupling coefficients
- `λ_Ππ` = shear-bulk coupling

**In rest frame** (u^μ = (1,0,0,0)):
```
D_u → ∂_t
θ → ∂_i u^i (spatial divergence)
```

**Linearized** (drop quadratic terms):
```
τ_Π ∂_t δΠ + δΠ = -ζ θ
```

**For plane wave**:
```
-iωτ_Π δΠ + δΠ = -ζ(ik)δv_x
δΠ(1 - iωτ_Π) = -iζk δv_x
```

**Implementation**:

*Dispersion matrix row 2*:
```python
matrix[2,1] = 1j*zeta*k           # Source: -ζθ where θ=ikv_x
matrix[2,2] = 1 - 1j*omega*tau_Pi # Relaxation: (1-iωτ_Π)
```

*Numerical RHS* (`relaxation.py:200-234`):
```python
def _bulk_rhs(Pi, pi_munu, theta):
    linear = -Pi / tau_Pi                      # -Π/τ_Π
    first_order = -bulk_viscosity * theta      # -ζθ
    nonlinear = xi_1*Pi*theta + xi_2*Pi**2/... # Second-order
    return linear + first_order + nonlinear
```

### Shear Stress Tensor

**Covariant IS equation** (simplified, longitudinal mode):
```
τ_π D_u π^μν + π^μν = 2η σ^μν + λ_ππ π^μν θ + λ_πΠ Π σ^μν + ...
```

where:
- `σ^μν` = shear tensor (traceless, symmetric)
- `λ_ππ, λ_πΠ` = second-order couplings

**Shear tensor** (velocity gradients):
```
σ^μν = Δ^μα Δ^νβ (∇_α u_β + ∇_β u_α)/2 - (1/3)θ Δ^μν
```

**For longitudinal wave** (x-direction only):
```
σ_xx = (2/3)∂_x u^x
σ_yy = σ_zz = -(1/3)∂_x u^x
π_xx component only (π_yy = π_zz = -π_xx/2)
```

**Linearized**:
```
τ_π ∂_t δπ_xx + δπ_xx = 2η σ_xx = 2η(2/3)∂_x δu^x
```

**For plane wave**:
```
-iωτ_π δπ_xx + δπ_xx = 2η(2/3)(ik)δv_x
δπ_xx(1 - iωτ_π) = i(4/3)ηk δv_x
```

**Implementation**:

*Dispersion matrix row 3*:
```python
matrix[3,1] = 1j*(4/3)*eta*k        # Source: 2η(2/3)ikv_x = (4/3)ηikv_x
matrix[3,3] = 1 - 1j*omega*tau_pi  # Relaxation: (1-iωτ_π)
```

*Numerical RHS* (`relaxation.py:236-308`):
```python
def _shear_rhs(pi_munu, Pi, q_mu, theta, sigma_munu, omega_munu, nabla_T):
    linear = -pi_munu / tau_pi                  # -π/τ_π
    first_order = 2*shear_viscosity * sigma_munu # 2ησ
    nonlinear = (lambda_pi_pi*pi_munu*theta +    # Second-order
                 lambda_pi_Pi*Pi*sigma_munu + ...)
    return linear + first_order + nonlinear
```

---

## 4. Kinematic Quantities

### Expansion Scalar

**Definition**:
```
θ = ∇_μ u^μ
```

**In flat spacetime** (Minkowski):
```
θ = ∂_μ u^μ = ∂_t u^0 + ∂_i u^i
```

**Rest frame** (u^0 = 1 constant):
```
θ = ∂_i u^i  (spatial divergence)
```

**For plane wave** u^i = δu^i exp(-iωt+ikx):
```
θ = ∂_x δu^x = (ik)δv_x
```

**Implementation** (`spectral.py:1872-1895`):
```python
def _compute_expansion_scalar():
    velocity_spatial = u_mu[..., 1:4]  # u^1, u^2, u^3
    theta = spectral.spatial_divergence(velocity_spatial)
    return theta
```

**Spectral divergence** (Fourier space):
```python
# FFT convention: f(x) = ∫ f̃(k) exp(ikx) dk
# → ∂_i f(x) = (ik_i) f̃(k)
theta_k = 1j*kx*ux_k + 1j*ky*uy_k + 1j*kz*uz_k
```

**CRITICAL**: Check FFT convention!
- If using `exp(-ikx)`: `∂_x → -ik` (WRONG for our equations)
- If using `exp(+ikx)`: `∂_x → +ik` (CORRECT)

### Shear Tensor

**Definition**:
```
σ^μν = (1/2)Δ^μα Δ^νβ (∇_α u_β + ∇_β u_α) - (1/3)θ Δ^μν
```

where `Δ^μν = g^μν + u^μ u^ν` is spatial projector.

**Rest frame, longitudinal wave** (∂_y = ∂_z = 0):
```
σ_xx = ∂_x u^x - (1/3)θ = ∂_x u^x - (1/3)∂_x u^x = (2/3)∂_x u^x
σ_yy = -(1/3)θ = -(1/3)∂_x u^x
σ_zz = -(1/3)θ = -(1/3)∂_x u^x
σ_ij = 0 for i≠j
```

**For plane wave**:
```
σ_xx = (2/3)(ik)δv_x
```

**Implementation** (`relaxation.py:491-561`):
```python
def _compute_shear_tensor(u_mu):
    # Compute velocity gradients
    nabla_u = compute_covariant_derivative(u_mu)

    # Symmetrize: σ = (1/2)(∇u + ∇u^T)
    sigma = 0.5*(nabla_u + np.swapaxes(nabla_u, -1, -2))

    # Subtract trace: σ -= (1/3)θ Δ
    theta = expansion_scalar(u_mu)
    for i in range(3):
        sigma[..., i, i] -= theta/3

    return sigma
```

---

## 5. Dispersion Relation

### Linearized System

Variables: **[δε, δv_x, δΠ, δπ_xx]**

Equations (plane wave `exp(-iωt+ikx)`):
1. Energy: `-iω δε + ik h δv_x = 0`
2. Momentum: `ik c_s² δε - iω h δv_x + ik δΠ + ik δπ_xx = 0`
3. Bulk: `iζk δv_x + (1-iωτ_Π) δΠ = 0`
4. Shear: `i(4/3)ηk δv_x + (1-iωτ_π) δπ_xx = 0`

**Matrix form**:
```
┌                                              ┐ ┌ δε     ┐   ┌ 0 ┐
│  -iω       ikh         0          0          │ │ δv_x   │   │ 0 │
│  ikc_s²    -iωh        ik         ik         │ │ δΠ     │ = │ 0 │
│  0         iζk         (1-iωτ_Π)  0          │ │ δπ_xx  │   │ 0 │
│  0         i(4/3)ηk    0          (1-iωτ_π)  │ └        ┘   └ 0 ┘
└                                              ┘
```

**Dispersion relation**: `det(M) = 0`

### Analytical Solution (Approximate)

**From equations 3 & 4**:
```
δΠ = -iζk δv_x / (1-iωτ_Π)
δπ_xx = -i(4/3)ηk δv_x / (1-iωτ_π)
```

**Substitute into equation 2**:
```
ikc_s² δε - iωh δv_x + ik[-iζk/(1-iωτ_Π) - i(4/3)ηk/(1-iωτ_π)]δv_x = 0
```

**From equation 1**: `δε = (kh/ω) δv_x`

**Substitute**:
```
ik c_s²(kh/ω) - iωh + ik²[ζ/(1-iωτ_Π) + (4/3)η/(1-iωτ_π)] = 0
```

**Simplify** (divide by `ikh`):
```
(c_s²k/ω) - ω + (k²/h)[ζ/(1-iωτ_Π) + (4/3)η/(1-iωτ_π)] = 0
```

**Multiply by ω**:
```
c_s²k - ω² + (ωk²/h)[ζ/(1-iωτ_Π) + (4/3)η/(1-iωτ_π)] = 0
```

**Solve for ω²**:
```
ω² = c_s²k + (ωk²/h)[ζ/(1-iωτ_Π) + (4/3)η/(1-iωτ_π)]
```

**Perturbative solution** (ω = ω₀ + δω, ω₀ = c_s k):
```
ω ≈ c_s k - i(ζ + 4η/3)k²/(2hc_s) × [1/(1+(c_s k τ)²)]
```

**Real/imaginary parts**:
```
Re(ω) = c_s k  (frequency)
Im(ω) = -(ζ + 4η/3)k²/(2hc_s) × [1/(1+(c_s k τ)²)]  (damping)
```

**Damping rate**:
```
γ = -Im(ω) = (ζ + 4η/3)k²/(2hc_s) × [1/(1+(c_s k τ)²)]
```

### Test Case (k=8.0)

**Parameters**:
- ε₀ = 1, p₀ = 1/3, h = 4/3
- c_s = 1/√3 ≈ 0.577
- η = 0.08, ζ = 0.04
- τ_π = 0.5, τ_Π = 0.3

**Calculations**:
```
k = 8.0
ω₀ = c_s k = 0.577 × 8 = 4.62

Navier-Stokes damping:
γ_NS = (ζ + 4η/3)k² / h
     = (0.04 + 0.107)×64 / 1.33
     = 7.05 /time

Relaxation suppression:
ωτ_Π = 4.62 × 0.3 = 1.39 → 1/(1+1.39²) = 0.34
ωτ_π = 4.62 × 0.5 = 2.31 → 1/(1+2.31²) = 0.16

Effective damping:
γ_eff ≈ 7.05 × 0.25 ≈ 1.76 /time  (order of magnitude)
```

**Numerical root finding gives**: γ = 0.51 /time

**Simulation measures**: γ ≈ -0.025 /time (WRONG SIGN!)

---

## 6. Sign Convention Summary

### Time Derivatives

**Convention**: `-iω` for `exp(-iωt)`

```
∂_t f = ∂_t[A exp(-iωt)] = -iω A exp(-iωt) = -iω f
```

**In matrix**: Use `-iω` for `∂_t` terms

### Spatial Derivatives

**Convention**: `+ik` for `exp(+ikx)`

```
∂_x f = ∂_x[A exp(ikx)] = ik A exp(ikx) = ik f
```

**In matrix**: Use `+ik` for `∂_x` terms

### Damping Sign

**Physical damping** (decay):
```
f(t) = A exp(-γt) exp(-iωt) = A exp(-iωt - γt)
→ ω_complex = ω - iγ  (Im < 0 for damping)
```

**Attenuation**:
```
γ = -Im(ω)  (positive for decay)
```

**Growth** (instability):
```
Im(ω) > 0 → γ < 0  (mode grows)
```

---

## 7. Implementation Checklist

### Dispersion Matrix (`_build_dispersion_matrix`)
- [ ] Energy row: `-iω, +ikh, 0, 0`
- [ ] Momentum row: `+ikc_s², -iωh, +ik, +ik`
- [ ] Bulk row: `0, +iζk, (1-iωτ_Π), 0`
- [ ] Shear row: `0, +i(4/3)ηk, 0, (1-iωτ_π)`

### Expansion Scalar (`_compute_expansion_scalar`)
- [ ] FFT uses `exp(+ikx)` convention
- [ ] Derivative: `∂_x → +ik` in Fourier space
- [ ] Test: `u^x=sin(kx)` → `θ=k cos(kx)` (positive at x=0)

### Bulk RHS (`_bulk_rhs`)
- [ ] Linear: `-Π/τ_Π` (negative for damping)
- [ ] Source: `-ζθ` (negative for compression θ>0)
- [ ] Test: Π→0 at equilibrium with θ≠0

### Shear RHS (`_shear_rhs`)
- [ ] Linear: `-π/τ_π` (negative for damping)
- [ ] Source: `+2ησ` (drives toward Navier-Stokes)
- [ ] Test: π→2ησ at equilibrium

### IMEX Splitting
- [ ] G(y) = `-Π/τ, -π/τ` (stiff, implicit)
- [ ] F(y) = `-ζθ + nonlinear` (non-stiff, explicit)
- [ ] Verify: F + G = full RHS

---

## Quick Reference

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| ε | Energy density | 1.0 |
| p | Pressure | 0.333 (=ε/3) |
| h | Enthalpy | 1.333 (=ε+p) |
| c_s | Sound speed | 0.577 (=1/√3) |
| θ | Expansion | `∇·u` |
| σ^μν | Shear tensor | `∇u - (1/3)θ` |
| Π | Bulk pressure | `-ζθ/τ_Π` (equilibrium) |
| π^μν | Shear stress | `2ησ/τ_π` (equilibrium) |
| η | Shear viscosity | 0.08 |
| ζ | Bulk viscosity | 0.04 |
| τ_π | Shear relaxation | 0.5 |
| τ_Π | Bulk relaxation | 0.3 |
| ω | Frequency | `c_s k` |
| γ | Damping | `-Im(ω)` |
