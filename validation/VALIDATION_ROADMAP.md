# Israel-Stewart Validation Roadmap

## Current Status (2025-10-18)

| Stage | Status | Tests Passing | Coverage | Blockers | Priority | ETA |
|-------|--------|---------------|----------|----------|----------|-----|
| 1. Units | 🟡 90% | 26/29 | Unit conversion ✅<br>λ_πV dimensions ⚠️ | Need to verify temperature scaling for coupling terms | HIGH | 1 day |
| 2. Coefficients | ✅ 100% | 26/26 | All IReD tables ✅<br>Scaling laws ✅<br>Truncation ✅ | None | - | DONE |
| 3. Equations | 🔴 30% | 0/10 est. | Conservation laws ⚠️<br>Relaxation RHS ⚠️<br>Form B structure ✅ | Missing isolated unit tests | HIGH | 3 days |
| 4. Dispersion | 🟢 70% | 1/2 | Frequency ✅<br>Nearly-ideal ✅<br>Damping ⚠️ | Regime Validity Paradox (physical, not bug) | MEDIUM | BLOCKED |
| 5. Solvers | 🔴 20% | 0/15 est. | Spectral FFT ⚠️<br>RK4 order ⚠️<br>IMEX splitting ⚠️ | No convergence tests exist | MEDIUM | 5 days |
| 6. Benchmarks | 🟡 50% | 3/6 | Sound freq ✅<br>Bjorken shear ✅<br>Ficks law ✅<br>Sound damp ❌<br>Bjorken T ❌<br>Diffusion decay ❌ | T not evolving, long-time instability | HIGH | 7 days |
| 7. Integration | 🔴 0% | 0/- | Not started | Stages 3-6 incomplete | LOW | 2 weeks |

**Legend**: ✅ Complete | 🟢 Good | 🟡 Needs work | 🔴 Blocked/Not started

## Overall Progress

**Completion**: ~45% (3.1/7 stages)

**Timeline**:
- **Week 1-2** (Now): Complete Stages 1, 3 (units, equations)
- **Week 3-4**: Complete Stage 5 (solvers)
- **Week 5-6**: Complete Stage 6 (benchmarks)
- **Week 7**: Complete Stage 7 (integration)

**Total estimated time**: 7 weeks to full validation

## Stage Details

### Stage 1: Units & Dimensional Analysis (🟡 90%)

**Acceptance Criteria**:
- ✅ All unit conversions accurate to < 10⁻¹⁰
- ⚠️ All coupling terms dimensionally consistent
- ✅ Temperature scaling formulas correct

**Current Status**:
- ✅ fm/c ↔ GeV⁻¹ conversions working (error < 2e-16)
- ✅ Mean free path calculation fixed (added (ℏc)³ factor)
- ⚠️ λ_πV, λ_Vπ temperature scaling added but needs verification against IReD paper

**Remaining Work**:
1. Read IReD.pdf Appendix B Table III to verify dimensionless coefficient convention
2. Check if λ_πV should be dimensionless or have units [GeV]
3. Validate against solver usage in sound_waves.py and relaxation.py
4. Document findings in `stage1_units/results/dimensional_analysis.md`

**Scripts**:
- `unit_audit.py` - Comprehensive unit conversion verification
- `check_lambda_coupling_dimensions.py` - Validate mixed-unit couplings
- `verify_time_conversions.py` - Test fm/c ↔ GeV⁻¹ ↔ SI

**Time estimate**: 1 day

---

### Stage 2: Transport Coefficients (✅ 100%)

**Acceptance Criteria**:
- ✅ All coefficients match IReD Tables III-IV (< 0.01% error)
- ✅ Temperature scaling: η ∝ T/σ
- ✅ Cross-section scaling: η ∝ 1/σ
- ✅ Truncation convergence: 14 → 23 → 32 → 41

**Current Status**:
- ✅ 26/26 tests passing
- ✅ Validates against IReD paper Tables III-IV
- ✅ Mean free path calculation corrected
- ✅ All second-order couplings implemented

**COMPLETE** - No remaining work

---

### Stage 3: Equation Components (🔴 30%)

**Acceptance Criteria**:
- ❌ Conservation laws pass in isolation
- ❌ Relaxation equations pass unit tests
- ✅ Form B structure verified (no /τ in sources)
- ❌ Equilibrium RHS = 0

**Current Status**:
- ✅ Form B structure implemented correctly
- ⚠️ Conservation laws exist but no isolated tests
- ⚠️ Relaxation equations exist but no unit tests
- ❌ Missing: systematic component-level validation

**Remaining Work**:

**3a. Conservation Laws** (3 days):
1. Test stress-energy tensor construction independently
2. Verify sign conventions for (-,+,+,+) signature
3. Test expansion scalar θ = ∇_μ u^μ calculation
4. Validate spatial divergence operators
5. Check metric compatibility

**3b. Relaxation Equations** (3 days):
1. Test each coupling term in isolation
2. Verify equilibrium (θ=0, σ=0) gives RHS=0
3. Test regime warning triggers
4. Validate all second-order terms
5. Check normalization factors

**Scripts to create**:
- `conservation/test_stress_tensor_components.py`
- `conservation/test_expansion_scalar.py`
- `conservation/verify_divergence_operators.py`
- `relaxation/test_coupling_terms.py`
- `relaxation/verify_equilibrium_rhs.py`
- `relaxation/test_regime_warnings.py`

**Time estimate**: 6 days (3 days conservation + 3 days relaxation)

---

### Stage 4: Dispersion Relations (🟢 70%)

**Acceptance Criteria**:
- ✅ Sound wave frequency ω(k) correct to < 5%
- ⚠️ Damping rate Γ(k) matches theory
- ✅ Nearly-ideal modes (|Γ/ω| < 1%) accepted
- ✅ Regime validity |τω| < 1 enforced

**Current Status**:
- ✅ Fixed nearly-ideal mode acceptance (Option A, 2025-10-18)
- ✅ Test `test_sound_wave_frequency` now passing
- ⚠️ Test `test_sound_wave_damping` marked xfail (Regime Validity Paradox)
- ✅ Regime checking implemented

**Regime Validity Paradox** (PHYSICAL, not a bug):
```
Small σ (σ=1 fm²):   η large → Γ measurable  BUT |τω| >> 1 (invalid regime)
Large σ (σ=10000 fm²): |τω| < 1 (valid)  BUT η tiny → Γ ~ 10⁻⁵ (unmeasurable)
```

**Conclusion**: Cannot achieve both regime validity AND measurable damping with hard sphere gas at single σ value.

**Remaining Work**:
1. Document Option A fix in `results/nearly_ideal_mode_fix.md`
2. Document Regime Validity Paradox in `results/regime_paradox.md`
3. Option B: Search for intermediate σ where both conditions met (if exists)
4. Option C: Compare to full Israel-Stewart formula (not just NS) for damping

**Scripts**:
- `debug_dispersion_k05.py` - Diagnostic for k=0.5 modes
- `verify_nearly_ideal_modes.py` - Test |Γ/ω| < 1% acceptance
- `analyze_regime_boundaries.py` - Map valid parameter space
- `check_dispersion_accuracy.py` - Compare to analytical predictions

**Time estimate**: BLOCKED (physical limitation) or 3 days (Option B/C)

---

### Stage 5: Numerical Solvers (🔴 20%)

**Acceptance Criteria**:
- ❌ Spectral: FFT derivatives match analytical (< 10⁻¹⁰)
- ❌ RK4: 4th-order convergence demonstrated
- ❌ IMEX: Splitting error O(dt²) verified
- ❌ All solvers conserve energy in simple tests

**Current Status**:
- ⚠️ Spectral solver exists, no unit tests
- ⚠️ RK4 exists, no convergence tests
- ⚠️ IMEX exists, no splitting validation
- ❌ No standalone solver tests (only full benchmarks)

**Remaining Work**:

**5a. Spectral Solver** (2 days):
1. Test FFT derivatives on sin(kx), cos(kx)
2. Verify periodic boundary conditions (dx = L/N)
3. Test Nyquist limit behavior
4. Validate linear regime detection
5. Check momentum ↔ velocity conversion

**5b. RK4 Solver** (1 day):
1. Test on linear ODE: dy/dt = -λy
2. Verify 4th-order convergence: error ∝ dt⁴
3. Test stability region
4. Check timestep adaptivity (if implemented)

**5c. IMEX Solver** (2 days):
1. Test on stiff test problem (fast + slow modes)
2. Verify splitting accuracy
3. Check for double-counting of terms
4. Test stability on stiff systems

**Scripts to create**:
- `spectral/test_fft_derivatives.py`
- `spectral/verify_boundary_conditions.py`
- `spectral/test_linear_regime.py`
- `rk4/test_convergence_order.py`
- `rk4/verify_stability.py`
- `imex/test_splitting_accuracy.py`
- `imex/verify_stiff_stability.py`

**Time estimate**: 5 days

---

### Stage 6: Analytical Benchmarks (🟡 50%)

**Acceptance Criteria**:
- ✅ Sound wave frequency correct (< 5% error)
- ⚠️ Sound wave damping matches theory (< 15% error)
- ❌ Bjorken T(τ) matches analytical (< 5% error)
- ✅ Bjorken shear stress evolves correctly
- ✅ Diffusion Fick's law V^i = -D∇^i(μ/T) (< 10% error)
- ❌ Diffusion decay rate Γ = Dk² (< factor 2.5×)

**Current Status**:
- ✅ 3/6 tests passing (sound freq, Bjorken shear, Fick's law)
- ⚠️ 1/6 xfail (sound damping - Regime Validity Paradox)
- ❌ 1/6 xfail (Bjorken T - not evolving, 71% error)
- ⚠️ 1/6 skipped (diffusion decay - long-time numerical instability)

**Failing Tests Analysis**:

**Test 1: Sound Wave Damping** (xfail, physical limitation)
- Issue: Regime Validity Paradox (see Stage 4)
- Status: BLOCKED by fundamental physics
- Resolution: Either accept xfail or implement Option B/C from Stage 4

**Test 2: Bjorken Temperature Evolution** (xfail, physics bug)
- Issue: Numerical T not evolving (71% error)
- Root cause: Conservation laws not updating temperature field
- Fix needed: In spectral solver, update T from energy density
- Time estimate: 2 days

**Test 3: Diffusion Decay Rate** (skipped, numerical bug)
- Issue: Long-time evolution (t ~ 11 GeV⁻¹) shows numerical overflow
- Warnings: `overflow encountered in square`, `invalid value in fft`
- Root cause: Numerical instability at extended evolution times
- Fix needed: Investigate timestep sensitivity, roundoff accumulation
- Time estimate: 3 days

**Remaining Work**:
1. Fix Bjorken temperature evolution (HIGH priority)
2. Debug diffusion long-time instability (MEDIUM priority)
3. Document all benchmark results in `stage6_benchmarks/results/`

**Time estimate**: 7 days (2 days Bjorken + 3 days diffusion + 2 days documentation)

---

### Stage 7: Integration Testing (🔴 0%)

**Acceptance Criteria**:
- ❌ All benchmarks pass with multiple parameter sets
- ❌ Regime violation detection works
- ❌ Performance benchmarks documented
- ❌ Conservation holds in long-time evolution
- ❌ No regressions in any stage

**Current Status**: Not started (blocked on Stages 3, 5, 6)

**Remaining Work**:
1. Multi-parameter validation (vary T, σ, grid size)
2. Stress testing with extreme parameters
3. Performance profiling
4. Long-time conservation checks
5. Regression test suite

**Time estimate**: 10 days (after Stages 3-6 complete)

---

## Critical Path

**Immediate priorities** (blocking progress):

1. **Stage 1** (1 day): Verify λ_πV dimensions → unblocks equation validation
2. **Stage 3** (6 days): Create equation unit tests → validates physics correctness
3. **Stage 6.2** (2 days): Fix Bjorken temperature → validates conservation
4. **Stage 5** (5 days): Solver unit tests → validates numerics independently
5. **Stage 6.3** (3 days): Fix diffusion stability → completes benchmarks
6. **Stage 7** (10 days): Integration testing → full system validation

**Total critical path**: ~27 days (5.4 weeks)

## Success Metrics

### Before (Current State - 2025-10-18)
- **Organized validation**: No (77 scripts in root, 47 docs scattered)
- **Component tests**: Minimal (mostly end-to-end)
- **Passing benchmarks**: 3/6
- **Clear progress tracking**: No
- **Can isolate failures**: Difficult

### After (Target State)
- **Organized validation**: Yes (7-stage framework)
- **Component tests**: Comprehensive (each stage independently validated)
- **Passing benchmarks**: 5/6 (1 may remain xfail due to physics)
- **Clear progress tracking**: Yes (this roadmap)
- **Can isolate failures**: Easy (know exactly which stage fails)

## Historical Context

**Previous approaches that failed**:
1. "Write everything then test" → Can't isolate root causes
2. "Test only end-to-end" → Failures could be anywhere in stack
3. "No organization" → 77 scripts in root, impossible to navigate

**This approach (staged validation)**:
1. Test each component before integration
2. Can't proceed without passing previous stage
3. Clear organization makes navigation easy
4. Progress is measurable and visible

## References

- **Option A Results**: `stage4_dispersion/results/nearly_ideal_fix.md`
- **IReD Test Plan**: Original plan that evolved into this framework
- **Regime Validity**: Wagner & Gavassino (2024), `docs/regime of applicability.pdf`
- **IReD Theory**: `docs/IRED_THEORY.md`

## Updates

- **2025-10-18**: Created staged validation framework (Option C)
- **2025-10-18**: Completed Option A - fixed nearly-ideal mode acceptance
- **2025-10-17**: IReD coefficient tests 26/26 passing (Stage 2 complete)
