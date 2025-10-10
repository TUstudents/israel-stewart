# IReD Quick Reference

**Purpose**: Concise reference for Israel-Stewart hydrodynamics using the IReD (Inverse-Reynolds-Dominance) formulation.

**Key Reference**: Wagner, Palermo, Ambrus (2022) - "Inverse-Reynolds-Dominance approach to transient fluid dynamics" ([arXiv:2208.02506](https://arxiv.org/abs/2208.02506))

---

## Core IReD Relaxation Equations (Form B)

**Bulk viscous pressure Π**:
```
dΠ/dt = -Π/τ_Π - ζθ + J^Π_terms
```

**Shear stress tensor π^μν**:
```
Dπ^μν/Dt = -π^μν/τ_π - 2ησ^μν + J^π_terms
```

**Heat flux q^μ**:
```
Dq^μ/Dt = -q^μ/τ_q - κΔ^μν∂_νT + J^q_terms
```

**Key property**: K^{μ₁...μℓ} = 0 by construction (no O(Kn²) parabolic terms).

---

## IReD Relaxation Times (Weighted Averages)

**NOT inverse eigenvalues!** IReD relaxation times are weighted averages:

```
τ_Π = Σ_r τ^(0)_{0r} C^(0)_r

τ_π = Σ_r τ^(1)_{1r} C^(1)_r

τ_q = Σ_r τ^(1)_{1r} C^(1)_r   (same basis as shear)
```

where:
- τ^(ℓ)_{ℓr} are DNMR relaxation times for mode r at rank ℓ
- C^(ℓ)_r = (transport_coeff_r)/(transport_coeff_0) are dimensionless ratios

**Ultrarelativistic hard sphere gas** (m → 0):
```
τ_Π = 5τ₀/3,   τ_π = 5τ₀/3,   τ_q = 5τ₀/3
```
where τ₀ is the fundamental microscopic relaxation time.

---

## First-Order Transport Coefficients

**Bulk viscosity ζ**:
```
ζ = Σ_r β^(0)_{0r}   (sum over all scalar modes)
```

**Shear viscosity η**:
```
η = Σ_r β^(1)_{1r}   (sum over all traceless symmetric tensor modes)
```

**Thermal conductivity κ** (Landau frame):
```
κ = 0   (by Landau frame definition: u_μq^μ = 0)
```

**For hard sphere gas**:
```
ζ/s = 0   (conformal, no bulk viscosity)
η/s = 1/(4π)   (famous result)
```

---

## Second-Order Transport Coefficients (J terms)

**Bulk sector** (5 coefficients):
```
J^Π_terms = δ_ΠΠ Π²/β̄ + τ_ΠΠ ΠΘ + λ_Πq l_Π^μq_μ + λ_ΠπJ_Π^{μν}π_{μν} + ξ_Π Πθ
```

**Shear sector** (10 coefficients):
```
J^π_terms = δ_ππ π^{μν}π_{μν}/β̄ + τ_ππ π^{μν}Θ_{μν}
           + λ_πΠ Πσ^{μν} + λ_πq I_π^{μν}
           + φ₁ π^{μν}θ + φ₂ π^μ_α σ^{αν} + φ₃ π^μ_α Ω^{αν}
           + φ₄ ∇^{μ}ln(T) ∇^{ν}ln(T) + φ₅ ∇^{μ}ln(T) ∇^{ν}ln(n)
```

**Heat flux sector** (similar structure with coupling to Π, π^μν).

**See IRED_THEORY.md Appendix C or IReD.pdf Appendix B for explicit formulas**.

---

## Regime of Applicability

**Fundamental criterion** (Wagner & Gavassino 2024):
```
|τω| ≲ 1
```
where τ = max(τ_Π, τ_π, τ_q) and ω is characteristic frequency.

**For plane wave modes**: ω ≈ k·c_s, so:
```
k ≲ 1/(τ·c_s)
```

**Radiation fluid** (c_s = 1/√3 ≈ 0.577) with τ ~ 0.5:
```
k_max ≈ 3.5   →   Recommended: k_max ≤ 4
```

**Outside regime** (|τω| > 1): Expect instabilities, unphysical results. This is fundamental physics, not a numerical bug!

---

## IReD ↔ DNMR Equivalence

**Formal equivalence** up to second order in gradients:
```
IReD ≡ DNMR   (same predictions for observables)
```

**How they differ**:
- **DNMR**: Eigenmode decomposition, K^{μ₁...μℓ} ≠ 0
- **IReD**: Direct asymptotic matching, K^{μ₁...μℓ} = 0

**Transport coefficient mapping** (Table II from IReD.pdf):

| IReD | DNMR Expansion |
|------|----------------|
| τ_Π  | Σ_r τ^(0)_{0r} C^(0)_r |
| ζ    | Σ_r β^(0)_{0r} |
| δ_ΠΠ | Σ_r (β^(0)_{0r})²/β^(0)_{00} τ^(0)_{0r} |
| τ_ΠΠ | Σ_r β^(0)_{0r} |
| λ_Πq | Σ_r (2/3) β^(0)_{0r} β^(1)_{1r}/β^(1)_{11} |

*(See IRED_THEORY.md Table 6 for complete mapping of all 31 coefficients)*

---

## Stress-Energy Tensor Decomposition

**Landau frame** (+,-,-,-) signature:
```
T^μν = ε u^μ u^ν − (P + Π) Δ^μν + π^{μν}
```

**Landau frame** (-,+,+,+) signature:
```
T^μν = ε u^μ u^ν + (P + Π) Δ^μν + π^{μν}
```

where:
- ε = energy density
- P = equilibrium pressure
- Π = bulk viscous pressure (Π < 0 for expansion)
- π^μν = shear stress tensor (traceless, space-like, symmetric)
- Δ^μν = g^μν + u^μ u^ν = spatial projection tensor

**Conservation laws**:
```
∇_μ T^μν = 0
∇_μ (n u^μ) = 0   (optional, for conserved particle number)
```

---

## Code Variable Cross-Reference

| Theory | Code (`israel_stewart/`) | Location |
|--------|--------------------------|----------|
| τ_Π | `coeffs.bulk_relaxation_time` | `core/constants.py:275` |
| τ_π | `coeffs.shear_relaxation_time` | `core/constants.py:283` |
| ζ | `coeffs.bulk_viscosity` | `core/constants.py:251` |
| η | `coeffs.shear_viscosity` | `core/constants.py:259` |
| λ_ππ | `coeffs.lambda_pipi` | `core/constants.py:299` |
| λ_πΠ | `coeffs.lambda_piPi` | `core/constants.py:307` |
| ξ₁ | `coeffs.xi_1` | `core/constants.py:331` |
| ξ₂ | `coeffs.xi_2` | `core/constants.py:339` |
| Π | `fields.Pi` | `core/fields.py:228` |
| π^μν | `fields.pi_munu` | `core/fields.py:243` |
| θ = ∇_μ u^μ | `theta` | `equations/conservation.py:316` |
| σ^μν | `sigma_munu` | `equations/conservation.py:348` |

**Relaxation equation implementation**:
- `israel_stewart/equations/relaxation.py:200-348` implements Form B (correct IReD)

**Regime checking**:
- `israel_stewart/solvers/spectral.py:119-185` implements |τω| criterion

---

## Verification Checklist

| Item | Status | Evidence |
|------|--------|----------|
| ✅ Form B relaxation equations | CORRECT | `relaxation.py:227,291` |
| ✅ No `/τ` in source terms | CORRECT | `-ζθ` not `-ζθ/τ_Π` |
| ✅ Regime checking | IMPLEMENTED | `spectral.py:119-185` |
| ⚠️ Full IReD coefficients | PHENOMENOLOGICAL | `coefficients.py` uses simple ratios |

**Current status**: Code structure is correct (Form B). Transport coefficients are phenomenological but acceptable for exploratory work. For quantitative comparison with kinetic theory, implement full IReD coefficients from Appendix B.

---

## Key Equations for Implementation

**Expansion scalar**:
```
θ = ∇_μ u^μ = ∂_μ u^μ + Γ^μ_{μλ} u^λ
```

**Shear tensor** (symmetric, traceless, spatial):
```
σ^μν = Δ^μ_α Δ^ν_β ∇^(α u^β) - (1/3) Δ^μν θ
```

**Vorticity tensor** (antisymmetric, spatial):
```
Ω^μν = Δ^μ_α Δ^ν_β ∇^[α u^β]
```

**Comoving derivative**:
```
D/Dt = u^μ ∇_μ
```

**Spatial projection of derivative**:
```
∇^μ A = Δ^μν ∂_ν A
```

---

## Common Pitfalls

1. **Relaxation times ≠ inverse eigenvalues**: In IReD, they're weighted averages (Eq. 38 in IReD.pdf)
2. **Form A vs Form B**: Only Form B (`-ζθ` without `/τ`) is correct for IReD
3. **Regime boundary**: High-k instabilities at |τω| > 1 are EXPECTED physics, not bugs
4. **Metric signatures**: Signs flip between (+,-,-,-) and (-,+,+,+) conventions
5. **K terms**: Should be ZERO in IReD (that's the whole point!)
6. **DNMR equivalence**: Only up to second order; higher orders differ
7. **Landau frame**: q^μ u_μ = 0 and π^μν u_ν = 0 by construction

---

## Further Reading

- **Primary reference**: Wagner, Palermo, Ambrus (2022) [arXiv:2208.02506](https://arxiv.org/abs/2208.02506) - IReD formulation
- **Regime applicability**: Wagner, Gavassino (2024) [arXiv:2309.14828v2](https://arxiv.org/abs/2309.14828) - |τω| ≲ 1 criterion
- **DNMR formulation**: Denicol et al. (2012) [arXiv:1202.4551](https://arxiv.org/abs/1202.4551) - Original DNMR approach
- **Conformal case**: Baier et al. (2008) [arXiv:0712.2451](https://arxiv.org/abs/0712.2451) - Second-order hydrodynamics for QGP
- **Detailed derivations**: See `docs/IRED_THEORY.md` in this repository

---

## Quick Start: Checking Code

```python
# Verify Form B structure
from israel_stewart.equations import ISRelaxationEquations

# Check source terms have Form B structure:
# - Bulk: -Π/τ_Π - ζθ (NOT -ζθ/τ_Π)
# - Shear: -π^μν/τ_π - 2ησ^μν (NOT -2ησ^μν/τ_π)

# Check regime for your simulation
from israel_stewart.solvers.spectral import SpectralISHydrodynamics

hydro = SpectralISHydrodynamics(grid, fields, coeffs)
regime_status = hydro.check_ired_regime()  # Should show k_max vs regime limit
```

---

**Last updated**: 2025-10-10
**Codebase version**: israel-stewart v0.1.0
**For full theory**: See `docs/IRED_THEORY.md`
