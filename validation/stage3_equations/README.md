# Stage 3: Equation Components

**Status**: 🔴 30% Complete (need isolated unit tests)

**Priority**: HIGH (blocks Stage 6 benchmark validation)

## Goal

Validate conservation laws and Israel-Stewart relaxation equations in isolation, before testing them in full simulation benchmarks.

## Why This Stage Matters

**"If individual equations are wrong, the full system will be wrong."**

Current problem: When benchmarks fail, we can't tell if the issue is:
- Conservation law implementation ❌
- Relaxation equation implementation ❌
- Numerical solver ❌
- Boundary conditions ❌

**Solution**: Test each equation component independently with known inputs/outputs.

## Acceptance Criteria

- ❌ Conservation laws pass in isolation
- ❌ Relaxation equations pass unit tests
- ✅ Form B structure verified (no /τ in sources)
- ❌ Equilibrium RHS = 0

## Current Status

### ✅ Completed

1. **Form B Structure Verified**
   - Relaxation equations use correct Form B: `dΠ/dt = -Π/τ_Π - ζθ + J_terms`
   - NOT Form A (causes instability): `dΠ/dt = -Π/τ_Π - ζθ/τ_Π + J_terms`
   - Implementation: `israel_stewart/equations/relaxation.py:200-348`
   - Verified in `verify_ired_implementation.py` (12/16 checks passing)

2. **Implementation Exists**
   - Conservation laws: `israel_stewart/equations/conservation.py`
   - Relaxation equations: `israel_stewart/equations/relaxation.py`
   - Both used successfully in benchmarks (sound waves, Bjorken flow)

### ⚠️ In Progress

**Need isolated unit tests** that check:

3a. **Conservation Laws** (no tests exist)
   - Stress-energy tensor construction: T^μν = (ε+p)u^μu^ν + p·g^μν + π^μν + ...
   - Sign conventions: ALL dissipative terms have PLUS signs (see `CLAUDE.md`)
   - Expansion scalar: θ = ∇_μ u^μ
   - Spatial divergence: ∇_i T^{μi}
   - Metric compatibility

3b. **Relaxation Equations** (no tests exist)
   - Each coupling term in isolation
   - Equilibrium condition: θ=0, σ=0 → RHS=0
   - Regime warning triggers (|τω| > 1)
   - All second-order terms present
   - Normalization factors correct

### ❌ TODO

**Create systematic component-level validation**:

## Test Scripts

### To Be Created (Conservation Laws)

**Priority 1: Stress-Energy Tensor**
- `conservation/test_stress_tensor_components.py`
  - Test ideal part: (ε+p)u^μu^ν + p·g^μν
  - Test viscous part: π^μν (shear stress)
  - Test dissipative additions: Π·Δ^μν (bulk), q^μu^ν (heat flux), V^μu^ν (diffusion)
  - Verify sign conventions (CRITICAL: all dissipative terms positive)

**Priority 2: Geometric Quantities**
- `conservation/test_expansion_scalar.py`
  - Test θ = ∇_μ u^μ on known flows
  - Bjorken: θ = 1/τ (boost-invariant expansion)
  - Minkowski rest frame: θ = ∇·v
  - FLRW: θ = 3H (Hubble expansion)

**Priority 3: Divergence Operators**
- `conservation/verify_divergence_operators.py`
  - Test spatial divergence: ∇_i T^{μi}
  - Test covariant derivative with Christoffel symbols
  - Verify metric compatibility: ∇_μ g^αβ = 0

**Priority 4: Sign Convention**
- `conservation/verify_sign_conventions.py`
  - CRITICAL: Check (-,+,+,+) signature throughout
  - Verify T^μν matches IReD paper eq. (5) after metric conversion
  - See `docs/IRED_THEORY.md` Section 1.3 for derivation

### To Be Created (Relaxation Equations)

**Priority 1: Equilibrium Test**
- `relaxation/verify_equilibrium_rhs.py`
  - Set θ = 0 (no expansion)
  - Set σ^μν = 0 (no shear)
  - Set ∇^μ(μ/T) = 0 (no gradients)
  - **Expected**: All RHS = 0 (equilibrium preserved)
  - **Implementation**: Test `_bulk_rhs()`, `_shear_rhs()`, `_diffusion_rhs()` return zero

**Priority 2: Individual Coupling Terms**
- `relaxation/test_coupling_terms.py`
  - Test each J-term in isolation
  - Bulk: ξ₁·Π·θ, ξ₂·Π²/τ_Π, λ_ΠΠ·π^μν·σ_μν
  - Shear: τ_ππ·π^μλ·π_λ^ν/τ_π, λ_πΠ·Π·σ^μν, λ_πV·(V^μ∇^ν(μ/T) + ...)
  - **Diffusion (Landau frame)**: All 6 coupling terms now validated in Stage 2 ✅
    - δ_VV·V^μ·θ (expansion coupling) - **CRITICAL**: Fixed in Stage 1 (was using wrong coefficient)
    - λ_Vπ·T²·π^μν·∇_ν(μ/T) (diffusion-shear) - **CRITICAL**: Fixed T² scaling in Stage 1
    - λ_πV·(V^μ∇^ν(μ/T) + V^ν∇^μ(μ/T))/2 (shear-diffusion) - **CRITICAL**: Fixed T scaling in Stage 1
    - λ_VV·V^μ·V_μ/τ_V (diffusion-diffusion)
    - τ_Vπ·π^μν·F_ν (diffusion-shear force, NOT YET IMPLEMENTED)
    - ℓ_Vπ·∇^μ∇^ν(μ/T) (diffusion-shear gradient, NOT YET IMPLEMENTED)

**Priority 3: Regime Warning**
- `relaxation/test_regime_warnings.py`
  - Set k = 10 GeV (high wavenumber)
  - Set τ = 0.5 GeV⁻¹ (typical relaxation time)
  - Compute |τω| with ω ≈ k·c_s
  - **Expected**: Warning logged when |τω| > 1

**Priority 4: Form B Structure**
- `relaxation/verify_form_b_structure.py`
  - Parse source code for relaxation equations
  - **Check**: No `/τ` factors multiply first-order terms (ζθ, ησ^μν, κ∇T)
  - **Correct**: `dΠ/dt = -Π/τ_Π - ζθ + ...`
  - **Wrong**: `dΠ/dt = -Π/τ_Π - ζθ/τ_Π + ...`

### Existing Diagnostic Scripts (to be moved from root)

- `audit_coupling.py` - Check coupling term structure
- `verify_ired_implementation.py` - Overall IReD validation (includes Form B check)
- `check_equilibrium_*.py` - Various equilibrium verification scripts

## Detailed Test Plan

### 3a. Conservation Laws (3 days)

**Test 1: Stress-Energy Tensor Construction**
```python
def test_stress_tensor_ideal():
    """Test ideal part: (ε+p)u^μu^ν + p·g^μν"""
    # Known values
    epsilon = 1.0  # Energy density
    pressure = epsilon / 3.0  # Radiation EOS
    u_mu = np.array([1, 0, 0, 0])  # Rest frame

    # Compute T^μν (ideal part only)
    T_ideal = stress_energy_ideal(epsilon, pressure, u_mu)

    # Expected (rest frame, Minkowski)
    expected = np.diag([epsilon, pressure, pressure, pressure])

    np.testing.assert_allclose(T_ideal, expected, rtol=1e-14)

def test_stress_tensor_viscous():
    """Test viscous corrections: π^μν, Π·Δ^μν, q^μu^ν"""
    # Set viscous fields
    pi_munu = known_shear_tensor()
    Pi = known_bulk_pressure()
    q_mu = known_heat_flux()

    # Compute full T^μν
    T_full = stress_energy_full(epsilon, pressure, u_mu, pi_munu, Pi, q_mu)

    # Check: Dissipative terms have CORRECT SIGN
    # From IReD paper eq. (5) with (-,+,+,+) signature:
    # T^μν = (ε+p)u^μu^ν + p·g^μν + Π·Δ^μν + π^μν + q^μu^ν + q^νu^μ
    # All dissipative terms: PLUS signs
```

**Test 2: Expansion Scalar**
```python
def test_expansion_bjorken():
    """Test θ = ∇_μ u^μ for Bjorken flow"""
    tau = 2.0  # Proper time
    u_mu = bjorken_four_velocity(tau)

    # Compute expansion
    theta = compute_expansion_scalar(u_mu, metric="Milne")

    # Analytical: θ = 1/τ
    expected = 1.0 / tau

    assert abs(theta - expected) / expected < 1e-12
```

**Test 3: Divergence Operators**
```python
def test_divergence_flat_space():
    """Test ∇_i T^{μi} in Minkowski"""
    # Uniform field (no gradients)
    T_munu = uniform_stress_tensor()

    # Compute divergence
    div_T = spatial_divergence(T_munu, metric="Minkowski")

    # Expected: zero (no spatial variation)
    np.testing.assert_allclose(div_T, 0.0, atol=1e-15)
```

### 3b. Relaxation Equations (3 days)

**Test 1: Equilibrium RHS = 0**
```python
def test_equilibrium_bulk():
    """Equilibrium: θ=0 → dΠ/dt=0"""
    # Setup equilibrium state
    theta = 0.0  # No expansion
    Pi = 0.0     # No bulk pressure

    # Compute RHS
    relaxation = ISRelaxation(coeffs)
    dPi_dt = relaxation._bulk_rhs(Pi, theta, pi_munu=zeros, V_mu=zeros)

    # Expected: RHS = 0
    assert abs(dPi_dt) < 1e-15

def test_equilibrium_shear():
    """Equilibrium: σ^μν=0 → dπ^μν/dt=0"""
    sigma_munu = np.zeros((4, 4))  # No shear
    pi_munu = np.zeros((4, 4))     # No shear stress

    dpi_dt = relaxation._shear_rhs(pi_munu, sigma_munu, Pi=0, V_mu=zeros)

    np.testing.assert_allclose(dpi_dt, 0.0, atol=1e-15)
```

**Test 2: Individual Coupling Terms**
```python
def test_lambda_pi_V_coupling():
    """Test λ_πV · (V^μ∇^ν(μ/T) + ...) term"""
    # Known inputs
    V_mu = np.array([0, 0.1, 0, 0])  # Small diffusion current
    nabla_mu_over_T = np.array([0, 0.01, 0, 0])  # Gradient
    lambda_pi_V = 0.5

    # Compute coupling term
    coupling = lambda_pi_V_term(V_mu, nabla_mu_over_T, lambda_pi_V)

    # Verify structure and sign
    assert coupling.shape == (4, 4)  # Tensor
    # Check specific components...
```

**Test 3: Regime Warning**
```python
def test_regime_warning_trigger():
    """Warning when |τω| > 1"""
    with pytest.warns(UserWarning, match="Outside Israel-Stewart regime"):
        k = 10.0  # High wavenumber
        coeffs = TransportCoefficients(
            shear_relaxation_time=0.5,  # τ = 0.5 GeV⁻¹
            # ...
        )
        # Create solver (should trigger warning during initialization)
        hydro = SpectralISHydrodynamics(grid_high_k, fields, coeffs)
```

## Landau Frame Diffusion Coefficient Status

**Updated**: 2025-10-19

### Stage 2 Completion ✅

All 6 Landau frame diffusion coupling coefficients now have **value validation** against IReD Table III:

| Coefficient | IReD Formula | Stage 2 Test | Status |
|-------------|--------------|--------------|--------|
| `lambda_pi_V` | 0.20890 × τ_π / β | ✅ test_lambda_pi_V_value | **PASS** |
| `lambda_V_V` | 0.89501 × τ_V | ✅ test_lambda_V_V_value | **PASS** |
| `delta_V_V` | 1.0 (exact) | ✅ test_delta_V_V_value | **PASS** |
| `lambda_V_pi` | 0.069240 × β × τ_V | ✅ test_lambda_V_pi_value | **PASS** (added 2025-10-19) |
| `tau_V_pi` | 0.0071692 × β × τ_V / P | ✅ test_tau_V_pi_value | **PASS** (added 2025-10-19) |
| `ell_V_pi` | 0.028677 × β × τ_V | ✅ test_ell_V_pi_value | **PASS** (added 2025-10-19) |

### Stage 3 Requirements ❌ (This Stage)

Need **equation usage** validation for diffusion relaxation RHS terms:

| Term | Implementation | Stage 3 Test | Status |
|------|----------------|--------------|--------|
| `-δ_VV × V^μ × θ` | ✅ relaxation.py:459-465 | ❌ **MISSING** | Fixed coeff in Stage 1, not tested |
| `-λ_Vπ × T² × π^μν × ∇_ν(μ/T)` | ✅ relaxation.py:471-481 | ❌ **MISSING** | Fixed T² scaling in Stage 1, not tested |
| `-λ_πV × (V^μ∇^ν + V^ν∇^μ)/2` | ✅ relaxation.py:355-371 | ❌ **MISSING** | Fixed T scaling in Stage 1, not tested |
| `-λ_VV/(D·τ_V) × (V·V) × V^μ` | ✅ relaxation.py:483-506 | ❌ **MISSING** | **Implemented 2025-10-19**, not tested |
| `-τ_Vπ × π^μν × F_ν` | ❌ **NOT IMPLEMENTED** | ❌ N/A | Blocked: needs pressure gradient computation |
| `-ℓ_Vπ × ∇^μ∇^ν(μ/T)` | ❌ **NOT IMPLEMENTED** | ❌ N/A | Blocked: needs second derivative infrastructure |

**Critical Gap**: Stage 1 fixed 3 dimensional errors in Landau diffusion terms, but Stage 3 has **no tests** to verify the equation usage is correct!

**Priority**: Add relaxation RHS component tests for all implemented Landau diffusion terms.

### Recent Implementation (2025-10-19)

**Added λ_VV term**: Implemented nonlinear diffusion self-coupling in `relaxation.py:483-506`
- **Formula**: `-λ_VV/(D·τ_V) × (V·V) × V^μ`
- **Pattern**: Analogous to shear self-coupling τ_ππ
- **Units**: λ_VV = 0.89501 × τ_V (GeV⁻¹) from IReD Table III
- **Physics**: Higher-order correction (O(Re⁻²) R term), suppresses diffusion at large current magnitudes
- **Status**: Implemented and passes all existing tests ✅ (needs dedicated component test)

**Blocked terms** (require additional infrastructure):
- τ_Vπ: Needs pressure gradient ∇_ν P computation
- ℓ_Vπ: Needs second derivative ∇^μ∇^ν infrastructure

**Next steps**:
1. Create component-level tests for 4 implemented Landau diffusion terms
2. Implement pressure gradient computation for τ_Vπ
3. Implement second derivative infrastructure for ℓ_Vπ

## Remaining Work

**Time estimate**: 6-7 days total
- 3 days: Conservation law tests (Tests 1-3)
- 3-4 days: Relaxation equation tests (Tests 1-3) + **Landau diffusion term tests**

**Breakdown**:
1. Create 8 test scripts (1 day)
2. Implement test cases (3 days)
3. Debug failures and fix implementation (2 days)

## References

- **Conservation**: `israel_stewart/equations/conservation.py`
- **Relaxation**: `israel_stewart/equations/relaxation.py:200-348`
- **Form B structure**: `DISPERSION_MATRIX_PARADOX.md`, `HIGH_K_INSTABILITY_RESOLUTION.md`
- **Sign conventions**: `CLAUDE.md`, `docs/IRED_THEORY.md` Section 1.3
- **IReD equations**: `docs/IReD.pdf` Appendix B

## Success Metrics

**Before** (2025-10-18):
- Form B structure verified ✓
- Implementation exists and used in benchmarks ✓
- **But**: No isolated unit tests ❌
- **Impact**: Can't debug failures systematically

**Target**:
- 10+ tests for conservation laws ✓
- 10+ tests for relaxation equations ✓
- All equilibrium tests passing ✓
- Regime warnings working ✓
- Can isolate equation bugs from solver bugs ✓

## Next Steps

1. **Create conservation test suite** (3 days)
   - Stress-energy tensor
   - Expansion scalar
   - Divergence operators
   - Sign conventions

2. **Create relaxation test suite** (3 days)
   - Equilibrium RHS = 0
   - Individual coupling terms
   - Regime warnings
   - Form B structure

3. **Document findings** (0.5 days)
   - Write `results/equation_validation.md`
   - Update VALIDATION_ROADMAP.md

**After Stage 3 complete**: Can proceed to Stage 6 (Benchmarks) with confidence that equations are correct.
