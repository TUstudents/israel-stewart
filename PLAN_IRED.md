# Plan: Comprehensive IReD Theory Report

## Document Structure

### 1. **Executive Summary** (`IRED_THEORY.md`)
- Brief overview of IReD vs DNMR approaches
- Key result: K^{μ₁...μₗ} = 0 by construction
- Practical implications for our spectral solver
- Summary of verification checklist for codebase

### 2. **Part I: Theoretical Foundation**

#### Section 2.1: Starting Point - Boltzmann Equation
- Document equation (2) from IReD.pdf: k^μ∂_μf = C[f]
- Irreducible moments ρ^{μ₁...μₗ}_r definition (equation 3)
- Connection to dissipative quantities (equation 4)

#### Section 2.2: Landau Frame Decomposition
- Stress-energy tensor in BOTH metric signatures:
  - (+,−,−,−): T^μν = εu^μu^ν − (P+Π)Δ^μν + π^μν
  - (−,+,+,+): T^μν = εu^μu^ν + (P+Π)Δ^μν + π^μν
- Particle current: N^μ = nu^μ + n^μ
- Landau matching conditions (equation 6)

#### Section 2.3: IReD vs DNMR Matching
- **DNMR approach**: eigenmode decomposition with Ω^(ℓ)_{rn} (equation 13)
- **IReD approach**: direct asymptotic matching (equation 35):
  ```
  ρ_n ≃ −(3/m²)C^(0)_n Π
  ρ^μ_n ≃ C^(1)_n n^μ
  ρ^{μν}_n ≃ C^(2)_n π^{μν}
  ```
- Why this eliminates K^{μ₁...μₗ} terms (Section III explanation)

### 3. **Part II: Relaxation Equations**

#### Section 3.1: General Form
- Equation (1a,b,c) structure:
  ```
  τ_Π Π̇ + Π = −ζθ + J
  τ_n ṅ^⟨μ⟩ + n^μ = κI^μ + J^μ
  τ_π π̇^⟨μν⟩ + π^{μν} = 2ησ^{μν} + J^{μν}
  ```
- Note: K terms absent by construction

#### Section 3.2: J^{μ₁...μₗ} Terms
- Full expressions from equations (29a,b,c)
- Physical interpretation of each term
- Vorticity, expansion, gradients

### 4. **Part III: Transport Coefficients**

#### Section 4.1: First-Order Coefficients
- Bulk viscosity: ζ_n (equation 19a)
- Diffusivity: κ_n (equation 19b)
- Shear viscosity: η_n (equation 19c)
- Note: Same in IReD and DNMR

#### Section 4.2: Relaxation Times (CRITICAL!)
- **IReD formulation** (equation 38):
  ```
  τ_Π = Σ_{r≠1,2} τ^(0)_{0r} C^(0)_r
  τ_n = Σ_{r≠1} τ^(1)_{0r} C^(1)_r
  τ_π = Σ_r τ^(2)_{0r} C^(2)_r
  ```
- **Weighted average interpretation**: τ_Π = (Σ_n ζ_n τ_n)/(Σ_m ζ_m)
- **Relationship to DNMR** (equation 51):
  ```
  τ_Π = τ̃_Π + ζ̃₁/ζ
  τ_n = τ̃_n + κ̃₅/(2κ)
  τ_π = τ̃_π + η̃₁/(2η)
  ```

#### Section 4.3: Second-Order Coefficients
- Complete listing from Appendix B (equations B1-B20)
- Substitution rule: Ω^(ℓ)_{r0} → C^(ℓ)_r
- Table II mapping from paper

#### Section 4.4: C^(ℓ)_r Coefficients
- Definition (equation 26): C^(ℓ)_n = (transport_coeff_n)/(transport_coeff_0)
- For negative indices (equation 36): C^(ℓ)_{−r} = Σ_n F^(ℓ)_{rn} C^(ℓ)_n
- Connection to microscopic collision matrix

### 5. **Part IV: Regime of Applicability**

#### Section 5.1: Physical Boundary
- Wagner & Gavassino (2024) criterion: **|τω| ≲ 1**
- For plane waves: ω ≈ k·c_s
- Maximum wavenumber: k_max ≲ 1/(τ·c_s)

#### Section 5.2: Practical Limits
- For radiation fluid (c_s = 1/√3 ≈ 0.577):
  ```
  k_max ≈ 1/(0.5 × 0.577) ≈ 3.5
  Recommended: k_max ≤ 4
  ```
- Explains why k=8 tests failed in our codebase
- Not a bug, but fundamental physics limit!

#### Section 5.3: Separation of Scales Discussion
- DNMR: τ_π,0 ≤ τ_π,1 ≤ ... (based on eigenvalues)
- IReD: τ_π,0 ≤ τ_π,1 ≤ ... (but physically different meaning)
- Table I comparison from paper

### 6. **Part V: Formal Equivalence with DNMR**

#### Section 6.1: Proof of Equivalence
- Equation (44): τ_Π Π̇ − J = τ̃_Π Π̇ − J̃ − K̃
- How K̃ terms get absorbed into modified τ and J coefficients
- Appendix A walkthrough (key steps only)

#### Section 6.2: Complete Coefficient Mapping
- Reproduce Table II with explanations
- Partial derivatives notation: ∂/∂α, ∂/∂β
- H and H̄ definitions (equation A2)

### 7. **Part VI: Implementation Guide**

#### Section 7.1: Current Codebase Status
- Review `israel_stewart/equations/relaxation.py`
- Review `israel_stewart/solvers/spectral.py`
- Current transport coefficient implementation in `coefficients.py`

#### Section 7.2: Verification Checklist
- [ ] Relaxation equation structure matches IReD (no K terms)
- [ ] Sign conventions for T^μν (check metric signature)
- [ ] Transport coefficient formulation (phenomenological vs IReD)
- [ ] Regime checking implemented (|τω| < 1)
- [ ] Boundary conditions (periodic for spectral)

#### Section 7.3: What Needs Implementation
**Option A: Keep phenomenological coefficients (current)**
- ✓ Relaxation equation Form B is correct
- ✓ Regime checking is implemented
- → No changes needed, just document equivalence

**Option B: Full IReD coefficients (future enhancement)**
- Implement C^(ℓ)_r computation (Appendix B formulas)
- Implement inverse collision matrix τ^(ℓ)_{rs}
- Add IReD coefficient model class
- This is for quantitative accuracy vs experiments

### 8. **Part VII: Numerical Examples**

#### Section 8.1: Ultrarelativistic Hard Sphere Gas
- Tables III and IV from paper (diffusion, shear)
- Convergence behavior with N_ℓ
- Compare 14, 23, 32, 41 moment truncations

#### Section 8.2: Test Cases for Our Code
- Bjorken flow with IReD coefficients
- Sound wave propagation (k=1 vs k=4 vs k=8)
- Verify regime violations trigger warnings

### 9. **Appendices**

#### Appendix A: Notation Conventions
- Metric signatures and conversions
- Greek indices (μ, ν, ...) vs Latin indices (i, j, ...)
- Projection operators Δ^{μν}
- Angle brackets ⟨μν⟩ notation

#### Appendix B: Derivation Details
- Comoving derivatives (equations A10)
- Matching condition algebra
- Second-order coefficient derivations

#### Appendix C: Code Cross-Reference
- Map paper equations to code locations
- Variable name dictionary
- Function call graph for transport coefficients

## Deliverables

1. **Main document**: `docs/IRED_THEORY.md` (8000-10000 words)
2. **Quick reference**: `docs/IRED_QUICK_REFERENCE.md` (key equations only)
3. **Verification script**: `verify_ired_implementation.py` (checks codebase)
4. **Update**: `CLAUDE.md` (add IReD-specific guidance)

## Expected Outcomes

- Complete theoretical reference for IReD approach
- Clear understanding of current code status
- Decision point: keep phenomenological or implement full IReD
- Verification that our Form B implementation is correct
- Documentation of regime limits for future users
