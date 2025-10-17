# Landau Frame Formulation for Israel-Stewart Hydrodynamics

**Author**: Physics design document for Landau frame refactoring
**Date**: 2025-10-11
**Status**: Reference document for implementation
**Based on**: Wagner & Gavassino (2024), arXiv:2203.12608v2 (IReD paper)

## 1. Frame Choice in Relativistic Hydrodynamics

### 1.1 The Frame Problem

In relativistic hydrodynamics, the decomposition of the stress-energy tensor T^μν into "ideal" and "dissipative" parts is **frame-dependent**. Different frames define the fluid four-velocity u^μ differently, leading to different sets of independent dissipative fluxes.

The two most common frames are:

1. **Eckart Frame**: Zero particle flux in fluid rest frame
   - Condition: J^μ = n u^μ (no particle diffusion)
   - Consequence: V^μ = 0 (particle diffusion current is zero)
   - Dissipative flux: Heat flux q^μ (energy diffusion)

2. **Landau Frame**: Zero energy flux in fluid rest frame
   - Condition: T^μν u_ν = ε u^μ (no energy flux)
   - Consequence: q^μ = 0 (heat flux is zero)
   - Dissipative flux: Particle diffusion current V^μ

### 1.2 Why Landau Frame?

**This project uses the Landau frame** for the following reasons:

1. **Energy-momentum conservation**: More natural formulation (energy flux condition built into frame definition)
2. **Thermodynamic consistency**: Energy density ε is a well-defined thermodynamic variable
3. **Numerical stability**: Energy conservation is better controlled
4. **IReD theory**: Wagner & Gavassino (2024) use Landau frame throughout
5. **Relativistic heavy-ion physics**: Standard choice in QGP studies

**Critical**: The current implementation incorrectly uses Eckart frame variables (q^μ) while assuming Landau frame elsewhere. This refactoring fixes this inconsistency.

---

## 2. Landau Frame: Fundamental Equations

### 2.1 Frame Definition

The Landau frame four-velocity u^μ is defined by the **energy flux condition**:

```
T^μν u_ν = ε u^μ
```

where:
- T^μν is the total stress-energy tensor
- ε is the energy density in the fluid rest frame
- u^μ is the fluid four-velocity (u^μ u_μ = -1 in signature (-,+,+,+))

**Physical interpretation**: In the local rest frame (u^μ = (1,0,0,0)), the energy flux T^0i = 0. All energy is carried by the fluid motion, not by heat conduction.

### 2.2 Stress-Energy Tensor Decomposition

In Landau frame, the stress-energy tensor has the form:

```
T^μν = (ε + p) u^μ u^ν + p g^μν + Π Δ^μν + π^μν
```

where:
- **ε**: Energy density
- **p**: Thermodynamic pressure
- **Π**: Bulk viscous pressure (scalar dissipative correction)
- **π^μν**: Shear stress tensor (traceless, spatial, symmetric)
- **Δ^μν = g^μν + u^μ u^ν**: Spatial projector

**Key observation**: NO heat flux term q^μ u^ν + q^ν u^μ. Heat flux is identically zero in Landau frame.

### 2.3 Particle Number Current

The particle number current J^μ is decomposed as:

```
J^μ = n u^μ + V^μ
```

where:
- **n**: Particle number density in fluid rest frame
- **V^μ**: Particle diffusion current (orthogonal to u^μ)

**Orthogonality condition**: V^μ u_μ = 0

This means V^μ is a purely spatial vector in the local rest frame.

### 2.4 Conservation Laws

**Energy-momentum conservation**:
```
∇_μ T^μν = 0
```

Expanding in Landau frame:
```
∇_μ[(ε+p)u^μ u^ν + p g^μν + Π Δ^μν + π^μν] = 0
```

**Particle number conservation**:
```
∇_μ J^μ = ∇_μ(n u^μ + V^μ) = 0
```

In Landau frame, both conservation laws are independent (unlike Eckart frame where particle conservation defines the frame).

---

## 3. Thermodynamic Quantities

### 3.1 Equation of State (Radiation Fluid)

For a relativistic radiation-dominated fluid (conformal symmetry):

```
p = ε/3                    (Equation of state)
c_s² = ∂p/∂ε = 1/3        (Sound speed squared)
h = (ε + p)/n = 4ε/(3n)    (Enthalpy per particle)
```

### 3.2 Chemical Potential

The baryon chemical potential μ_B is defined thermodynamically:

```
dε = T ds + μ_B dn
```

For a radiation fluid in equilibrium:
```
μ_B/T = ln(n/n_eq)
```

where n_eq is the equilibrium particle density at temperature T.

**For ideal gas** (alternative formulation):
```
μ_B = k_B T ln(n/n_Q)
```

where n_Q = (m k_B T/(2π ℏ²))^(3/2) is the quantum concentration.

### 3.3 Thermodynamic Driving Forces

In Landau frame, the **thermodynamic forces** driving dissipation are:

1. **Bulk viscosity**: Expansion scalar θ = ∇_μ u^μ
2. **Shear viscosity**: Shear tensor σ^μν (traceless, symmetric part of velocity gradient)
3. **Particle diffusion**: Chemical potential gradient ∇^μ(μ_B/T)

The last point is crucial: **particle diffusion is driven by ∇^μ(μ_B/T), not by ∇^μ T**.

---

## 4. Israel-Stewart Relaxation Equations (Landau Frame)

### 4.1 Bulk Viscous Pressure

```
τ_Π (u^μ ∇_μ Π + Π) = -ζ θ + J_Π + R_Π
```

Simplified (first-order):
```
dΠ/dτ + Π/τ_Π = -ζ θ
```

where:
- τ_Π: Bulk relaxation time
- ζ: Bulk viscosity coefficient
- θ = ∇_μ u^μ: Expansion scalar
- J_Π, R_Π: Second-order coupling terms (expansion in gradients and fluxes)

**Second-order terms** (from IReD/DNMR formulation):
```
J_Π = ξ_1 Π θ                           (Π-θ coupling)
R_Π = ξ_2 Π²/(ζ τ_Π)                   (Π nonlinearity)
```

Note: No shear-bulk coupling in standard formulation (shear is traceless).

### 4.2 Shear Stress Tensor

```
τ_π (Δ^μα Δ^νβ u^λ ∇_λ π_αβ + π^μν) = 2η σ^μν + J_π^μν + R_π^μν
```

Simplified (first-order):
```
dπ^μν/dτ + π^μν/τ_π = 2η σ^μν
```

where:
- τ_π: Shear relaxation time
- η: Shear viscosity coefficient
- σ^μν: Shear tensor (traceless, symmetric, spatial velocity gradient)

**Second-order terms** (Landau frame):
```
J_π^μν = λ_ππ π^μν θ                    (Shear-expansion coupling)
       + λ_πΠ Π σ^μν                     (Shear-bulk coupling)
       + λ_πV (V^μ ∇^ν(μ_B/T) + V^ν ∇^μ(μ_B/T))/2  (Shear-diffusion coupling)

R_π^μν = -τ_ππ π^μ_α π^α_ν/(η τ_π)    (Shear nonlinearity, O(Re^-2))
       + τ_πω (π^μ_α ω^α_ν - ω^μ_α π^α_ν)  (Vorticity coupling)
```

where ω^μν is the vorticity tensor (antisymmetric velocity gradient).

**Key change**: Shear-diffusion coupling uses **V^μ ∇^μ(μ_B/T)**, not q^μ ∇^μ T.

### 4.3 Particle Diffusion Current (NEW)

This is the equation that replaces heat flux evolution in Eckart frame:

```
τ_V (Δ^μν u^λ ∇_λ V_ν + V^μ) = D ∇^μ(μ_B/T) + J_V^μ + R_V^μ
```

Simplified (first-order):
```
dV^μ/dτ + V^μ/τ_V = -D ∇^μ(μ_B/T)
```

**Fick's law**: V^μ = -D ∇^μ(μ_B/T) (particles flow down chemical potential gradient)

where:
- τ_V: Diffusion relaxation time
- D: Diffusion coefficient (replaces thermal conductivity κ)
- ∇^μ(μ_B/T): Chemical potential gradient (projected to spatial hypersurface)
- Negative sign: ensures particles flow from high μ to low μ

**Second-order terms** (Landau frame):
```
J_V^μ = -τ_Vπ V^μ θ                     (Diffusion-expansion coupling)
      + λ_Vπ π^μν ∇_ν(μ_B/T)           (Diffusion-shear coupling)

R_V^μ = [Higher-order terms, typically neglected]
```

**Physical interpretation**:
- First-order term: Fick's law (diffusion driven DOWN chemical potential gradient)
- Second-order terms: Couplings to expansion and shear (small corrections)

**Orthogonality**: After evolution, V^μ must be projected to ensure V^μ u_μ = 0.

---

## 5. Coupling Coefficients (Landau Frame)

### 5.1 Transport Coefficients (First-Order)

| Coefficient | Symbol | Physical Meaning | Units |
|-------------|--------|------------------|-------|
| Shear viscosity | η | Resistance to shear flow | Energy·Time/Volume |
| Bulk viscosity | ζ | Resistance to compression | Energy·Time/Volume |
| Diffusion coefficient | D | Particle diffusion rate | Length²/Time |

### 5.2 Relaxation Times (First-Order)

| Time | Symbol | Typical Estimate | Physical Meaning |
|------|--------|------------------|------------------|
| Shear relaxation | τ_π | η/(ε+p) | Time for shear stress to equilibrate |
| Bulk relaxation | τ_Π | ζ/(ε+p) | Time for bulk pressure to equilibrate |
| Diffusion relaxation | τ_V | D·(ε+p)/T² | Time for particle flux to equilibrate |

### 5.3 Second-Order Coupling Coefficients

**J terms (O(Re^-1 Kn))**: First-order gradients times dissipative fluxes

| Coefficient | Symbol | Couples | Form |
|-------------|--------|---------|------|
| Shear-expansion | λ_ππ | π^μν, θ | λ_ππ π^μν θ |
| Shear-bulk | λ_πΠ | π^μν, Π | λ_πΠ Π σ^μν |
| **Shear-diffusion** | **λ_πV** | **π^μν, V^μ** | **λ_πV (V^μ ∇^ν(μ/T) + V^ν ∇^μ(μ/T))/2** |
| Bulk-expansion | ξ_1 | Π, θ | ξ_1 Π θ |
| **Diffusion-shear** | **λ_Vπ** | **V^μ, π^μν** | **λ_Vπ π^μν ∇_ν(μ/T)** |
| Diffusion-expansion | τ_Vπ | V^μ, θ | -τ_Vπ V^μ θ |

**R terms (O(Re^-2))**: Quadratic in dissipative fluxes

| Coefficient | Symbol | Couples | Form |
|-------------|--------|---------|------|
| Bulk nonlinearity | ξ_2 | Π, Π | ξ_2 Π²/(ζ τ_Π) |
| Shear nonlinearity | τ_ππ | π^μν, π^αβ | -τ_ππ π^μ_α π^α_ν/(η τ_π) |
| Shear-vorticity | τ_πω | π^μν, ω^αβ | τ_πω (π^μ_α ω^α_ν - ω^μ_α π^α_ν) |

**Note**: Bold entries are **new in Landau frame**, replacing Eckart frame heat flux couplings.

### 5.4 Coefficient Estimates (Kinetic Theory)

For a relativistic ideal gas near equilibrium:

```
η ∼ 0.1 (ε + p) τ_micro
ζ ∼ 0 (vanishes for conformal fluids)
D ∼ 0.1 T/n τ_micro

τ_π ∼ τ_micro ∼ λ_mfp/c
τ_Π ∼ τ_micro (if ζ ≠ 0)
τ_V ∼ τ_micro

λ_ππ ∼ 1
λ_πΠ ∼ 1
λ_πV ∼ 1
```

where τ_micro is the microscopic collision time and λ_mfp is the mean free path.

---

## 6. Kinematic Decomposition

### 6.1 Velocity Gradient Decomposition

The velocity gradient ∇_μ u_ν is decomposed into:

```
∇_μ u_ν = σ_μν + ω_μν + (1/3) θ Δ_μν + u_μ a_ν
```

where:
- **θ = ∇_μ u^μ**: Expansion scalar (trace)
- **σ_μν**: Shear tensor (symmetric, traceless, spatial)
- **ω_μν**: Vorticity tensor (antisymmetric, spatial)
- **a^μ = u^ν ∇_ν u^μ**: Four-acceleration

### 6.2 Shear Tensor

The shear tensor is defined as:

```
σ^μν = Δ^μα Δ^νβ ∇^(α u^β) - (1/3) Δ^μν θ
```

where ∇^(α u^β) = (∇^α u^β + ∇^β u^α)/2 is the symmetrized covariant derivative.

**Properties**:
- Symmetric: σ^μν = σ^νμ
- Traceless: g_μν σ^μν = 0
- Spatial: σ^μν u_μ = 0

### 6.3 Vorticity Tensor

The vorticity tensor is:

```
ω^μν = Δ^μα Δ^νβ ∇^[α u^β]
```

where ∇^[α u^β] = (∇^α u^β - ∇^β u^α)/2 is the antisymmetrized covariant derivative.

**Properties**:
- Antisymmetric: ω^μν = -ω^νμ
- Spatial: ω^μν u_μ = 0

### 6.4 Expansion Scalar

The expansion scalar is:

```
θ = ∇_μ u^μ = Δ^μν ∇_μ u_ν
```

**Physical interpretation**:
- θ > 0: Fluid expands (decreasing density)
- θ < 0: Fluid contracts (increasing density)
- θ = 0: Incompressible flow

---

## 7. Projected Gradients

### 7.1 Spatial Projection Operator

The spatial projector Δ^μν projects tensors onto the spatial hypersurface orthogonal to u^μ:

```
Δ^μν = g^μν + u^μ u^ν
```

**Properties**:
- Δ^μν u_ν = 0 (projects out time direction)
- Δ^μ_μ = 3 (three spatial dimensions)
- Δ^μα Δ^α_ν = Δ^μ_ν (projector identity)

### 7.2 Projected Temperature Gradient

The spatial temperature gradient is:

```
∇^μ T = Δ^μν ∇_ν T
```

This ensures ∇^μ T is orthogonal to u^μ: (∇^μ T) u_μ = 0.

### 7.3 Projected Chemical Potential Gradient (Landau Frame)

The crucial thermodynamic force for particle diffusion:

```
∇^μ(μ_B/T) = Δ^μν ∇_ν(μ_B/T)
```

**Computation**:
1. Compute μ_B from particle density: μ_B = μ_B(n, T)
2. Compute μ_B/T
3. Take covariant derivative: ∇_ν(μ_B/T)
4. Project to spatial hypersurface: Δ^μν ∇_ν(μ_B/T)

**For radiation fluid**:
```
μ_B/T = ln(n/n_eq)
∇^μ(μ_B/T) = Δ^μν (∇_ν n)/n  (if n_eq is constant)
```

---

## 8. Implementation Formulas

### 8.1 Chemical Potential (Radiation Fluid)

For a conformal radiation fluid:

```python
def compute_chemical_potential(n, T, eos_type="radiation"):
    """
    Compute μ_B/T for given particle density and temperature.

    For radiation fluid: μ_B/T = ln(n/n_eq)
    where n_eq is equilibrium density.
    """
    if eos_type == "radiation":
        # Assume equilibrium density n_eq = 1 (dimensionless)
        n_eq = 1.0
        mu_over_T = np.log(n / n_eq)
    elif eos_type == "ideal_gas":
        # Quantum concentration: n_Q = (m k_B T / (2π ℏ²))^(3/2)
        # For simplicity, use n_Q = T^(3/2) in natural units
        n_Q = T**(3/2)
        mu_over_T = np.log(n / n_Q)
    else:
        raise ValueError(f"Unknown EOS type: {eos_type}")

    return mu_over_T
```

### 8.2 Chemical Potential Gradient

```python
def compute_chemical_potential_gradient(mu_over_T, u_mu, grid, spectral_solver=None):
    """
    Compute projected gradient ∇^μ(μ_B/T) = Δ^μν ∇_ν(μ_B/T).

    Returns:
        nabla_mu_over_T: Shape (nx, ny, nz, 4)
    """
    # Compute ∇_ν(μ_B/T) using spectral derivatives
    grad_mu_over_T_lower = np.zeros(mu_over_T.shape + (4,))

    for mu in range(1, 4):  # Only spatial derivatives for 3D grid
        spatial_axis = mu - 1
        if spectral_solver is not None:
            grad_mu_over_T_lower[..., mu] = spectral_solver.spatial_derivative(
                mu_over_T, direction=spatial_axis
            )
        else:
            grad_mu_over_T_lower[..., mu] = np.gradient(
                mu_over_T, axis=spatial_axis
            )

    # Get metric inverse for raising indices
    g_inv = np.diag([-1, 1, 1, 1])  # Minkowski for simplicity

    # Raise indices: ∇^μ(μ/T) = g^μν ∇_ν(μ/T)
    grad_mu_over_T_up = np.einsum('ab,...b->...a', g_inv, grad_mu_over_T_lower)

    # Compute spatial projector: Δ^μν = g^μν + u^μ u^ν
    u_outer = np.einsum('...i,...j->...ij', u_mu, u_mu)
    Delta = g_inv + u_outer

    # Project: ∇^μ(μ/T) = Δ^μν ∇_ν(μ/T)
    nabla_mu_over_T = np.einsum('...ab,...b->...a', Delta, grad_mu_over_T_lower)

    return nabla_mu_over_T
```

### 8.3 Diffusion Current Evolution

```python
def diffusion_rhs(V_mu, pi_munu, theta, nabla_mu_over_T, coeffs):
    """
    Compute RHS of particle diffusion evolution equation.

    dV^μ/dτ = -V^μ/τ_V - D ∇^μ(μ_B/T) + coupling terms

    Fick's law: V^μ = -D ∇^μ(μ_B/T) (particles flow down gradient)
    """
    # Linear relaxation
    linear = -V_mu / coeffs.diffusion_relaxation_time

    # First-order source: -D ∇^μ(μ_B/T) (Fick's law)
    # Negative sign: particles flow DOWN chemical potential gradient
    first_order = -coeffs.diffusion_coefficient * nabla_mu_over_T

    # Second-order coupling terms
    nonlinear = np.zeros_like(V_mu)

    # Expansion coupling: -τ_Vπ V^μ θ
    if coeffs.tau_V_pi != 0:
        nonlinear += -coeffs.tau_V_pi * V_mu * theta[..., np.newaxis]

    # Shear coupling: λ_Vπ π^μν ∇_ν(μ/T)
    if coeffs.lambda_V_pi != 0:
        shear_coupling = coeffs.lambda_V_pi * np.einsum(
            '...ij,...j->...i', pi_munu, nabla_mu_over_T
        )
        nonlinear += shear_coupling

    return linear + first_order + nonlinear
```

---

## 9. Landau Frame Validation Criteria

### 9.1 Energy Flux Condition

The defining condition of Landau frame:

```
T^μν u_ν = ε u^μ
```

**Numerical test**:
```python
def validate_landau_frame_energy_flux(fields):
    """Check that T^μν u_ν = ε u^μ to machine precision."""
    T_munu = fields.compute_stress_energy_tensor()
    energy_flux = np.einsum('...ij,...j->...i', T_munu, fields.u_mu)
    expected = fields.rho * fields.u_mu

    error = np.max(np.abs(energy_flux - expected))
    assert error < 1e-10, f"Energy flux condition violated: error = {error}"
```

### 9.2 Orthogonality Conditions

All dissipative fluxes must be orthogonal to u^μ:

```
π^μν u_ν = 0
V^μ u_μ = 0
```

**Numerical test**:
```python
def validate_orthogonality(fields):
    """Check orthogonality of dissipative fluxes."""
    # Shear tensor orthogonality
    pi_u = np.einsum('...ij,...j->...i', fields.pi_munu, fields.u_mu)
    assert np.max(np.abs(pi_u)) < 1e-12

    # Diffusion current orthogonality
    V_u = np.einsum('...i,...i->...', fields.V_mu, fields.u_mu)
    assert np.max(np.abs(V_u)) < 1e-12
```

### 9.3 Particle Number Conservation

```
∂_μ J^μ = ∂_μ (n u^μ + V^μ) = 0
```

**Numerical test**:
```python
def validate_particle_conservation(fields, dt):
    """Check particle number conservation during evolution."""
    # Compute ∇_μ J^μ
    J_mu = fields.n * fields.u_mu + fields.V_mu
    div_J = compute_divergence(J_mu)

    # Should be zero (up to numerical truncation)
    assert np.max(np.abs(div_J)) < 1e-8 * fields.n
```

### 9.4 Chemical Equilibrium Approach

At equilibrium, diffusion current should vanish:

```
V^μ → 0  as  ∇_μ(μ_B/T) → 0
```

**Numerical test**:
```python
def validate_equilibrium_approach(fields, t_final=10.0):
    """Check that V^μ → 0 when ∇(μ/T) → 0."""
    # Start with non-zero V_mu, zero gradient
    fields.V_mu[:] = 0.01  # Small initial diffusion
    # Let system evolve with no driving force

    # After relaxation, V should decay exponentially
    V_final = np.max(np.abs(fields.V_mu))
    V_expected = 0.01 * np.exp(-t_final / tau_V)

    assert np.abs(V_final - V_expected) / V_expected < 0.1
```

---

## 10. Comparison: Eckart vs Landau Frame

### 10.1 Variable Correspondence

| Quantity | Eckart Frame | Landau Frame |
|----------|--------------|--------------|
| Frame definition | J^μ = n u^μ | T^μν u_ν = ε u^μ |
| Energy flux | q^μ ≠ 0 | q^μ = 0 |
| Particle flux | V^μ = 0 | V^μ ≠ 0 |
| Stress tensor | T^μν = ... + q^μu^ν + q^νu^μ | T^μν = ... (no q^μ term) |
| Particle current | J^μ = n u^μ | J^μ = n u^μ + V^μ |
| Thermodynamic force | ∇^μ T | ∇^μ(μ_B/T) |
| Transport coefficient | κ (thermal conductivity) | D (diffusion coefficient) |

### 10.2 Relaxation Equations

**Eckart Frame (OLD - INCORRECT in this project)**:
```
dq^μ/dτ + q^μ/τ_q = κ ∇^μ T + coupling terms
```

**Landau Frame (NEW - CORRECT)**:
```
dV^μ/dτ + V^μ/τ_V = D ∇^μ(μ_B/T) + coupling terms
```

### 10.3 When They Coincide

For a **neutral fluid** (no net particle number), or when particle diffusion is negligible:
- μ_B = 0 or n = constant
- V^μ ≈ 0
- Landau and Eckart frames approximately coincide

For **heavy-ion collisions** and **QGP physics**:
- Strong baryon number density gradients
- Significant difference between frames
- Landau frame is standard choice

---

## 11. References

1. **Wagner & Gavassino (2024)**
   "Inverse-Reynolds-Dominance approach to transient fluid dynamics"
   arXiv:2203.12608v2
   *Primary reference for IReD formulation in Landau frame*

2. **Denicol, Niemi, Molnar, Rischke (2012)**
   "Derivation of transient relativistic fluid dynamics from the Boltzmann equation"
   Phys. Rev. D 85:114047
   *Complete derivation of second-order coefficients*

3. **Landau & Lifshitz**
   "Fluid Mechanics" (2nd edition), Section 127
   *Original definition of Landau frame*

4. **Romatschke & Romatschke (2019)**
   "Relativistic Fluid Dynamics In and Out of Equilibrium"
   Cambridge University Press
   *Modern textbook covering both frames*

5. **Bemfica, Disconzi, Noronha (2020)**
   "Causality and existence of solutions of relativistic viscous fluid dynamics"
   Phys. Rev. D 98:104064
   *Causality and well-posedness in Landau frame*

---

## Appendix A: Notation Conventions

### A.1 Metric Signature
- **(-,+,+,+)**: This project uses "mostly plus" signature
- g^μν = diag(-1, +1, +1, +1) for Minkowski spacetime

### A.2 Index Conventions
- **Greek indices** (μ, ν, α, β): Spacetime indices (0,1,2,3)
- **Latin indices** (i, j, k): Spatial indices (1,2,3)
- **Parentheses**: Symmetrization T^(μν) = (T^μν + T^νμ)/2
- **Brackets**: Antisymmetrization T^[μν] = (T^μν - T^νμ)/2

### A.3 Covariant Derivative
- ∇_μ: Covariant derivative compatible with metric g_μν
- For Minkowski: ∇_μ = ∂_μ (ordinary partial derivative)
- For curved spacetime: ∇_μ V^ν = ∂_μ V^ν + Γ^ν_μλ V^λ

### A.4 Four-Velocity Normalization
- u^μ u_μ = -1 (timelike normalization in signature (-,+,+,+))
- u^0 = γ (Lorentz factor) in local rest frame
- u^i = γ v^i where v^i is spatial three-velocity

---

## Appendix B: Numerical Implementation Notes

### B.1 Spectral Derivatives

For periodic boundary conditions, use FFT-based spectral derivatives:

```python
def spectral_derivative(field, direction, wave_numbers):
    """
    Compute derivative using spectral method (machine precision).

    ∂f/∂x = FFT^{-1}[ik_x * FFT[f]]
    """
    field_k = np.fft.fftn(field)
    k = wave_numbers[direction]
    deriv_k = 1j * k * field_k
    deriv = np.fft.ifftn(deriv_k).real
    return deriv
```

### B.2 Chemical Potential Edge Cases

Handle numerical edge cases:

```python
# Avoid log(0) when n → 0
n_safe = np.maximum(n, 1e-15)
mu_over_T = np.log(n_safe / n_eq)

# For very small n, use asymptotic: μ/T → -∞
# In practice, cap the gradient to avoid numerical overflow
nabla_mu_over_T = np.clip(nabla_mu_over_T, -1e3, 1e3)
```

### B.3 Projection Operations

Ensure exact orthogonality after projection:

```python
# Project V_mu to be orthogonal to u_mu
V_u_contraction = np.einsum('...i,...i->...', V_mu, u_mu)
V_mu_projected = V_mu - V_u_contraction[..., np.newaxis] * u_mu

# Verify orthogonality
assert np.max(np.abs(np.einsum('...i,...i', V_mu_projected, u_mu))) < 1e-14
```

---

**End of Landau Frame Formulation Document**

This document serves as the physics reference for the Landau frame refactoring. All implementation decisions should be traceable back to equations and principles documented here.
