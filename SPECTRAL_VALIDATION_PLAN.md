# Spectral Solver Validation & Integration Plan

## Current Status
- ✅ All 8 spectral solver test failures resolved
- ✅ 71% code coverage in spectral.py with 56 passing tests
- ✅ Performance optimized: 2.04s per timestep, 1.06GB peak memory
- ✅ Physics validation: Spectral accuracy for derivatives, Laplacians, wave propagation

## Phase 1: Physics Benchmarks Integration (HIGH PRIORITY)

### 1.1 Sound Wave Benchmark Test
**Objective**: Validate linear wave propagation in relativistic viscous fluids
**Files**: `israel_stewart/benchmarks/sound_waves.py`
**Implementation**:
- Create comprehensive test using existing SoundWaveAnalysis framework
- Validate dispersion relations: ω(k) for viscous modes
- Test causality constraints and stability properties
- Verify second-order Israel-Stewart corrections
- Compare analytical vs numerical spectral solutions

**Success Criteria**:
- Dispersion relation accurate to <1e-10
- Causality violations detected and prevented
- Viscous damping rates match theoretical predictions
- Second-order corrections properly implemented

### 1.2 Bjorken Flow Validation
**Objective**: Test boost-invariant expansion exact solution
**Files**: `israel_stewart/benchmarks/bjorken_flow.py`
**Implementation**:
- Use existing BjorkenFlow analytical framework
- Test 1D boost-invariant relativistic expansion
- Compare exact analytical solution vs spectral numerical implementation
- Validate proper time evolution and viscous corrections
- Benchmark computational performance vs analytical expectations

**Success Criteria**:
- Numerical solution matches analytical within <1e-12
- Proper time evolution correctly implemented
- Viscous stress evolution follows analytical predictions
- Performance benchmarks established

### 1.3 Equilibration Dynamics Testing
**Objective**: Validate relaxation to equilibrium
**Files**: `israel_stewart/benchmarks/equilibration.py`
**Implementation**:
- Leverage existing EquilibrationAnalysis framework
- Test approach to thermal equilibrium from non-equilibrium initial conditions
- Validate relaxation timescales (τ_π, τ_Π, τ_q)
- Test approach to Navier-Stokes limit (τ → 0)
- Verify thermodynamic consistency throughout evolution

**Success Criteria**:
- Exponential relaxation with correct timescales
- Smooth approach to Navier-Stokes limit
- Energy and momentum conservation maintained
- Thermodynamic consistency preserved

## Phase 2: Integration & Cross-Validation

### 2.1 Cross-Solver Validation
- Compare spectral vs finite difference on identical problems
- Use `/tests/test_solvers/test_finite_difference.py` as reference
- Validate convergence properties and accuracy trade-offs

### 2.2 Full Israel-Stewart System Integration
- Test conservation + relaxation equation coupling
- Validate with ConservationLaws and ISRelaxationEquations
- Test stiff equation handling and numerical stability

## Phase 3: Production Readiness

### 3.1 Test Suite Cleanup
- Fix boundary condition warnings in test setups
- Ensure periodic boundaries by default
- Add proper metric initialization

### 3.2 Extended Performance Testing
- Benchmark larger grids (64³, 128³)
- Memory scaling analysis
- Parallel performance evaluation

### 3.3 Documentation & Error Handling
- Physics validation documentation
- Enhanced error messages
- User-friendly benchmark runners

## Implementation Priority
1. **Sound Wave Benchmark** - Critical physics validation
2. **Bjorken Flow Test** - Exact solution validation
3. **Equilibration Testing** - Thermodynamic validation
4. **Cross-solver comparison** - Implementation verification
5. **Full system integration** - Production readiness

## Success Metrics
- All physics benchmarks pass with spectral accuracy
- Cross-validation between solvers successful
- Performance targets met (sub-second per timestep for 32³)
- Comprehensive documentation and error handling
- Production-ready Israel-Stewart spectral solver