# Landau Frame Refactoring Plan

**Status**: Planning Phase
**Priority**: Critical - Core physics implementation error
**Estimated Effort**: 2-3 weeks
**Risk Level**: High (touches core modules, requires careful validation)

## Executive Summary

The current implementation uses **Eckart frame variables** (heat flux q^μ) with **Landau frame assumptions** elsewhere in the code. This is fundamentally inconsistent. The Landau frame is defined by **zero energy flux** (T^μν u_ν = ε u^μ), which means q^μ = 0. The actual dissipative flux in Landau frame is the **particle diffusion current** V^μ (or n^μ in some notations).

This refactoring will convert the entire codebase to use proper Landau frame variables.

## Physics Background

### Landau Frame Definition
- **Energy flux condition**: T^μν u_ν = ε u^μ (zero energy flux in fluid rest frame)
- **Consequence**: Heat flux q^μ = 0 (identically)
- **Dissipative flux**: Particle diffusion current V^μ, orthogonal to u^μ (V^μ u_μ = 0)

### Eckart Frame (Current Implementation - WRONG)
- **Particle flux condition**: J^μ = n u^μ (zero particle flux in fluid rest frame)
- **Consequence**: Diffusion current V^μ = 0
- **Dissipative flux**: Heat flux q^μ, orthogonal to u^μ (q^μ u_μ = 0)

### Why This Matters
The choice of frame affects:
1. Which dissipative fluxes are independent variables
2. The coupling structure in relaxation equations
3. The thermodynamic driving forces (∇T vs ∇(μ/T))
4. Energy-momentum conservation structure

**Current code mismatch**: Uses Eckart variables (q^μ) but assumes Landau frame in stress-energy tensor construction.

## Refactoring Phases

### Phase 1: Design and Documentation (1-2 days)

**Goal**: Establish clear specifications before touching code.

**Tasks**:
1. Document Landau frame formulation from IReD paper (Wagner & Gavassino 2024)
2. Derive particle diffusion evolution equation:
   ```
   dV^μ/dτ + V^μ/τ_V = D ∇^μ(μ_B/T) + coupling terms
   ```
3. Identify all coupling terms for Landau frame (λ_πV, λ_ΠV, etc.)
4. Define chemical potential calculation for radiation fluid
5. Write validation criteria for Landau frame conditions

**Output**:
- `docs/LANDAU_FRAME_FORMULATION.md` with complete equations
- Physics validation checklist

**Files to create/modify**:
- Create: `docs/LANDAU_FRAME_FORMULATION.md`
- Update: `docs/IRED_THEORY.md` (Section 3: Landau Frame)

---

### Phase 2: Field Variables (2-3 days)

**Goal**: Rename q_mu → V_mu throughout ISFieldConfiguration.

**Tasks**:
1. Rename `q_mu` → `V_mu` in `ISFieldConfiguration` (fields.py:583, 739)
2. Update docstrings to refer to "particle diffusion current"
3. Update `to_state_vector()` and `from_state_vector()` methods
4. Update `to_dissipative_vector()` and `from_dissipative_vector()` methods
5. Update constraint projection: `_project_heat_flux()` → `_project_diffusion_current()`
6. Update validation methods

**Files to modify**:
- `israel_stewart/core/fields.py` (lines 583, 739, 849-850, 994, 997-999)

**Breaking changes**: YES - All code using `fields.q_mu` must be updated

**Migration strategy**:
```python
# Add deprecation warning in transition period
@property
def q_mu(self):
    warnings.warn("q_mu is deprecated. Use V_mu (particle diffusion current) instead.",
                  DeprecationWarning, stacklevel=2)
    return self.V_mu
```

---

### Phase 3: Chemical Potential (1-2 days)

**Goal**: Add chemical potential calculation and gradient computation.

**Tasks**:
1. Add method `ISFieldConfiguration.compute_chemical_potential()`:
   ```python
   def compute_chemical_potential(self, eos_type="radiation") -> np.ndarray:
       """
       Compute baryon chemical potential μ_B.

       For radiation fluid: μ_B/T = ln(n/n_eq) where n_eq is equilibrium density
       For ideal gas: μ_B = k_B T ln(n/n_Q) where n_Q is quantum concentration

       Returns:
           Chemical potential μ_B with shape (nx, ny, nz)
       """
   ```

2. Add method `ISRelaxationEquations._compute_chemical_potential_gradient()`:
   ```python
   def _compute_chemical_potential_gradient(self, mu: np.ndarray, T: np.ndarray, u_mu: np.ndarray) -> np.ndarray:
       """
       Compute projected gradient ∇^μ(μ_B/T) = Δ^μν ∇_ν(μ_B/T).

       Returns:
           Gradient of μ_B/T with shape (nx, ny, nz, 4)
       """
   ```

3. Handle edge cases: zero particle density, equilibrium states

**Files to modify**:
- `israel_stewart/core/fields.py` (add new methods)
- `israel_stewart/equations/relaxation.py` (add gradient computation)

**Physics validation**:
- Check μ_B/T → -∞ as n → 0 (correct thermodynamic limit)
- Verify gradient calculation uses spectral derivatives when available

---

### Phase 4: Transport Coefficients (1 day)

**Goal**: Rename thermal_conductivity → diffusion_coefficient.

**Tasks**:
1. Rename in `TransportCoefficients.__init__()`:
   - `thermal_conductivity` → `diffusion_coefficient`
   - `heat_relaxation_time` → `diffusion_relaxation_time`

2. Rename coupling coefficients:
   - `lambda_pi_q` → `lambda_pi_V` (shear-diffusion coupling)
   - `lambda_q_pi` → `lambda_V_pi` (diffusion-shear coupling)
   - `tau_q_pi` → `tau_V_pi` (diffusion relaxation coupling)

3. Update coefficient validation and docstrings

4. Add deprecation warnings for old names

**Files to modify**:
- `israel_stewart/core/fields.py` (TransportCoefficients class, lines 242-418)

**Breaking changes**: YES - All code using coefficient names must be updated

---

### Phase 5: Diffusion Evolution Equation (2-3 days)

**Goal**: Replace heat flux evolution with particle diffusion evolution.

**Tasks**:
1. Rename `_heat_rhs()` → `_diffusion_rhs()` in `ISRelaxationEquations`

2. Implement proper diffusion equation:
   ```python
   def _diffusion_rhs(
       self,
       V_mu: np.ndarray,
       pi_munu: np.ndarray,
       theta: np.ndarray,
       nabla_mu_over_T: np.ndarray,  # ∇^μ(μ_B/T)
   ) -> np.ndarray:
       """
       Compute particle diffusion current evolution RHS.

       Landau frame equation:
       dV^μ/dτ = -V^μ/τ_V + D ∇^μ(μ_B/T) + coupling terms

       First-order: D ∇^μ(μ_B/T) (Fick's law generalization)
       Second-order: -τ_Vπ V^μ θ, λ_Vπ π^μν ∇_ν(μ_B/T)
       """
   ```

3. Update `compute_relaxation_rhs()` to call new method

4. Remove all references to "heat flux" in docstrings

**Files to modify**:
- `israel_stewart/equations/relaxation.py` (lines 372-430, module docstring)

**Physics validation**:
- Verify V^μ u_μ = 0 after projection
- Check diffusion evolves toward chemical equilibrium (∇μ = 0)

---

### Phase 6: Coupling Terms (1-2 days)

**Goal**: Update all coupling terms to use ∇^μ(μ_B/T) instead of ∇^μ T.

**Tasks**:
1. Update shear-diffusion coupling in `_shear_rhs()`:
   ```python
   # OLD: λ_πq * (q^μ ∇^ν T + q^ν ∇^μ T) / 2
   # NEW: λ_πV * (V^μ ∇^ν(μ/T) + V^ν ∇^μ(μ/T)) / 2
   ```

2. Update bulk-diffusion coupling (if exists)

3. Update vorticity-diffusion coupling (if exists)

4. Remove temperature gradient computation where no longer needed

5. Update all coefficient names in coupling terms

**Files to modify**:
- `israel_stewart/equations/relaxation.py` (lines 318-345, 416-427)

**Physics validation**:
- All coupling terms should be symmetric in μ↔ν indices where required
- Check sign conventions match IReD paper

---

### Phase 7: Spectral Solver (3-4 days)

**Goal**: Update IMEX spectral solver to evolve V^μ instead of q^μ.

**Tasks**:
1. Rename all `q_mu` references → `V_mu` in spectral solver

2. Update implicit/explicit splitting for diffusion relaxation:
   ```python
   # Implicit: -V^μ/τ_V
   # Explicit: D ∇^μ(μ_B/T) + coupling terms
   ```

3. Update field packing/unpacking in state vectors

4. Modify `_apply_dissipative_rhs()` to compute chemical potential gradient

5. Update `_convert_momentum_to_velocity_derivative_with_fields()` (if needed)

6. Ensure spectral derivatives used for ∇^μ(μ_B/T)

**Files to modify**:
- `israel_stewart/solvers/spectral.py` (scattered throughout, especially lines 1200-1400)
- `israel_stewart/solvers/implicit.py` (if used)
- `israel_stewart/solvers/splitting.py` (if used)

**Critical**: This is the most complex phase - requires careful testing at each step

**Testing strategy**:
- Unit test: Evolve pure diffusion mode without coupling
- Integration test: Verify spectral accuracy of chemical potential gradients
- Regression test: Compare total energy conservation before/after

---

### Phase 8: Analytical Dispersion Relations (2-3 days)

**Goal**: Update analytical sound wave solutions for Landau frame.

**Tasks**:
1. Rebuild dispersion matrix in Landau frame variables:
   ```
   Variables: [δε, δv_x, δΠ, δπ_xx, δV_x]  (not δq_x)
   ```

2. Update momentum conservation equation row

3. Update energy conservation equation row (Landau frame: no q^μ contribution)

4. Add particle diffusion evolution equation row:
   ```
   (-iω + 1/τ_V) δV_x - iD k δ(μ/T) + coupling terms = 0
   ```

5. Derive new analytical eigenmode structure

6. Update eigenmode initialization in `NumericalSoundWaveBenchmark`

**Files to modify**:
- `israel_stewart/benchmarks/sound_waves.py` (lines 350-450, eigenmode initialization)

**Physics validation**:
- Sound mode should remain stable in Landau frame
- Diffusion mode should appear as additional slow mode
- Check c_s^2 = 1/3 for radiation fluid

---

### Phase 9: Test Updates (2-3 days)

**Goal**: Update all tests to use V_mu and verify Landau frame physics.

**Tasks**:
1. **Field tests**:
   - Update `test_field_configuration.py`: q_mu → V_mu
   - Add test for chemical potential calculation
   - Add test for ∇^μ(μ_B/T) gradient computation

2. **Relaxation tests**:
   - Update `test_relaxation_equations.py`: coupling terms with V_mu
   - Add test for Landau frame orthogonality (V^μ u_μ = 0)
   - Test diffusion evolution toward equilibrium

3. **Solver tests**:
   - Update `test_spectral_solver.py`: V_mu evolution
   - Add test for chemical potential gradient accuracy
   - Verify IMEX splitting with diffusion relaxation

4. **Eigenmode tests**:
   - Update `test_eigenmode_preservation.py`: 5-variable system [δε, δv, δΠ, δπ, δV]
   - Test analytical eigenmode ratios include V_x/ρ
   - Verify eigenmode stability at k=1

5. **Benchmark tests**:
   - Update `test_sound_waves.py`: Landau frame dispersion
   - Add test for particle diffusion mode
   - Check energy flux T^μν u_ν = ε u^μ (Landau condition)

**Files to modify**:
- `tests/test_field_configuration.py`
- `tests/test_relaxation_equations.py`
- `tests/test_spectral_solver.py`
- `tests/test_eigenmode_preservation.py`
- `tests/test_sound_waves.py`
- `tests/test_conservation_laws.py` (verify Landau frame)

**Critical validation**:
- ALL tests must pass before merging
- Coverage should remain >80%

---

### Phase 10: Validation and Documentation (2 days)

**Goal**: Comprehensive validation of Landau frame implementation.

**Tasks**:
1. **Physics validation**:
   - Verify T^μν u_ν = ε u^μ (zero energy flux condition)
   - Check V^μ u_μ = 0 (orthogonality)
   - Confirm particle number conservation: ∂_μ(n u^μ + V^μ) = 0
   - Test approach to equilibrium: V^μ → 0 as ∇μ_B → 0

2. **Benchmark comparisons**:
   - Sound wave propagation: Compare Eckart vs Landau frame
   - Bjorken flow: Should be identical (no gradients)
   - Shock waves: Check dissipative structure

3. **Create validation script**:
   ```python
   # verify_landau_frame_physics/validate_landau_conditions.py
   """
   Comprehensive validation of Landau frame implementation:
   1. Energy flux condition
   2. Orthogonality conditions
   3. Particle conservation
   4. Chemical equilibrium approach
   5. Sound mode dispersion relation
   """
   ```

4. **Update documentation**:
   - Update `CLAUDE.md`: Landau frame convention
   - Update `README.md`: Mention Landau frame
   - Update `docs/IRED_THEORY.md`: Complete Landau frame section
   - Add migration guide for users of old API

**Files to create/modify**:
- Create: `verify_landau_frame_physics/validate_landau_conditions.py`
- Create: `docs/LANDAU_FRAME_MIGRATION_GUIDE.md`
- Update: `CLAUDE.md`, `README.md`, `docs/IRED_THEORY.md`

**Success criteria**:
- ✅ All tests pass
- ✅ T^μν u_ν = ε u^μ to machine precision
- ✅ Sound modes stable and accurate
- ✅ Documentation complete and clear

---

## Risk Assessment and Mitigation

### High Risk Areas

1. **Spectral solver changes** (Phase 7)
   - **Risk**: Breaking time evolution, numerical instabilities
   - **Mitigation**:
     - Test each change incrementally
     - Keep old code commented out until validation complete
     - Use git branches for each phase

2. **Dispersion relation changes** (Phase 8)
   - **Risk**: Incorrect analytical predictions, test failures
   - **Mitigation**:
     - Derive dispersion matrix analytically before coding
     - Cross-check with IReD paper equations
     - Test against known limiting cases (no viscosity → perfect fluid)

3. **API breaking changes** (Phases 2, 4)
   - **Risk**: Downstream code breakage
   - **Mitigation**:
     - Add deprecation warnings for 1-2 releases
     - Provide clear migration guide
     - Search codebase for all uses of renamed variables

### Medium Risk Areas

1. **Chemical potential calculation** (Phase 3)
   - **Risk**: Incorrect thermodynamics, wrong equilibrium states
   - **Mitigation**:
     - Test against known EOS (ideal gas, radiation)
     - Verify thermodynamic identities (e.g., Gibbs-Duhem)
     - Check limiting behaviors (n→0, n→∞)

2. **Coupling term signs** (Phase 6)
   - **Risk**: Wrong signs lead to instabilities
   - **Mitigation**:
     - Carefully compare with IReD paper Table I
     - Test stability with different coefficient combinations
     - Verify energy dissipation is non-negative

### Low Risk Areas

1. **Documentation updates** (Phases 1, 10)
   - **Risk**: Inconsistent or unclear docs
   - **Mitigation**: Multiple review passes

2. **Variable renaming** (Phases 2, 4)
   - **Risk**: Missing some references
   - **Mitigation**: Use find-and-replace with regex, run full test suite

---

## Testing Strategy

### Unit Tests (Per Phase)
- Test each modified method in isolation
- Mock dependencies when possible
- Aim for >90% coverage of new/modified code

### Integration Tests (After Major Phases)
- Phase 3: Chemical potential + gradients
- Phase 5: Diffusion evolution standalone
- Phase 7: Full spectral solver with diffusion
- Phase 8: Analytical + numerical eigenmode comparison

### Regression Tests (After All Phases)
- Bjorken flow benchmark (should be unchanged)
- Sound wave propagation (dispersion relation should shift slightly)
- Energy conservation (should improve - Landau frame has cleaner energy flux)
- Particle conservation (should be exact)

### Performance Benchmarks
- Time evolution with V_mu should have same performance as with q_mu
- Chemical potential gradient computation should use spectral derivatives (O(N log N))
- Memory usage should be identical (same number of fields)

---

## Success Criteria

### Physics Correctness
- ✅ Landau frame condition: T^μν u_ν = ε u^μ (error < 10^-10)
- ✅ Orthogonality: V^μ u_μ = 0 (error < 10^-12)
- ✅ Particle conservation: ∂_μ J^μ = 0 where J^μ = n u^μ + V^μ
- ✅ Sound mode stability: attenuation > 0 for all physical k
- ✅ Eigenmode preservation: ratios stable to <10% over τ=1

### Code Quality
- ✅ All tests pass (>100 tests)
- ✅ Test coverage >80%
- ✅ No deprecation warnings in tests (after migration period)
- ✅ Linting passes (ruff check)
- ✅ Type checking passes (mypy)

### Documentation
- ✅ LANDAU_FRAME_FORMULATION.md complete with equations
- ✅ Migration guide for users
- ✅ CLAUDE.md updated with Landau frame convention
- ✅ Docstrings accurate for all modified methods

---

## Timeline Estimate

| Phase | Duration | Dependencies | Risk |
|-------|----------|--------------|------|
| 1. Design | 1-2 days | None | Low |
| 2. Fields | 2-3 days | Phase 1 | Low |
| 3. Chemical Potential | 1-2 days | Phase 2 | Medium |
| 4. Coefficients | 1 day | Phase 1 | Low |
| 5. Diffusion Evolution | 2-3 days | Phases 2,3,4 | Medium |
| 6. Coupling Terms | 1-2 days | Phase 5 | Medium |
| 7. Spectral Solver | 3-4 days | Phases 2,3,5 | **High** |
| 8. Dispersion Relations | 2-3 days | Phases 2,3,5 | **High** |
| 9. Test Updates | 2-3 days | All previous | Medium |
| 10. Validation | 2 days | All previous | Low |

**Total**: 17-25 days (3.5-5 weeks with buffer)

**Critical path**: Phase 1 → 2 → 3 → 5 → 7 → 8 → 9 → 10

**Parallel opportunities**: Phases 4 and 3 can be done simultaneously

---

## Rollout Strategy

### Development Approach
1. Create feature branch: `feature/landau-frame-refactoring`
2. Complete phases 1-6 (core physics changes)
3. Create checkpoint: `checkpoint/landau-frame-core-physics`
4. Complete phases 7-8 (solver and analytics)
5. Create checkpoint: `checkpoint/landau-frame-numerics`
6. Complete phases 9-10 (tests and validation)
7. Merge to main after all tests pass

### Code Review Strategy
- Review after each phase completion
- Physics review for Phases 1, 3, 5, 8
- Code review for all phases
- Final comprehensive review before merge

### Deployment
1. Merge to `main` after validation
2. Tag release: `v2.0.0-landau-frame`
3. Update documentation site
4. Announce breaking changes in release notes
5. Provide migration support for 2 releases

---

## References

1. Wagner & Gavassino (2024), "Inverse-Reynolds-Dominance approach to transient fluid dynamics", arXiv:2203.12608v2
2. Denicol, Niemi, Molnar, Rischke (2012), "Derivation of transient relativistic fluid dynamics from the Boltzmann equation", PRD 85:114047
3. Landau & Lifshitz, "Fluid Mechanics" (2nd edition), Section 127
4. IReD theory documentation: `docs/IRED_THEORY.md`

---

## Appendix A: Variable Mapping

| Old (Eckart) | New (Landau) | Description |
|--------------|--------------|-------------|
| `q_mu` | `V_mu` | Heat flux → Particle diffusion current |
| `thermal_conductivity` | `diffusion_coefficient` | κ → D |
| `heat_relaxation_time` | `diffusion_relaxation_time` | τ_q → τ_V |
| `lambda_pi_q` | `lambda_pi_V` | Shear-heat → Shear-diffusion coupling |
| `lambda_q_pi` | `lambda_V_pi` | Heat-shear → Diffusion-shear coupling |
| `tau_q_pi` | `tau_V_pi` | Heat relaxation → Diffusion relaxation |
| `∇^μ T` | `∇^μ(μ_B/T)` | Temperature gradient → Chemical potential gradient |

## Appendix B: Equation Summary

### Old (Eckart Frame - INCORRECT)
```
dq^μ/dτ = -q^μ/τ_q + κ ∇^μ T + coupling terms
T^μν = (ε+p)u^μu^ν + p g^μν + Π Δ^μν + π^μν + q^μu^ν + q^νu^μ
```

### New (Landau Frame - CORRECT)
```
dV^μ/dτ = -V^μ/τ_V + D ∇^μ(μ_B/T) + coupling terms
T^μν = (ε+p)u^μu^ν + p g^μν + Π Δ^μν + π^μν
Particle flux: J^μ = n u^μ + V^μ
```

### Landau Frame Constraints
```
Energy flux condition: T^μν u_ν = ε u^μ  (defines frame)
Orthogonality: V^μ u_μ = 0
Particle conservation: ∂_μ J^μ = 0
Chemical equilibrium: V^μ → 0 as ∇_μ(μ_B/T) → 0
```
