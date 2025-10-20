# Israel-Stewart Validation Framework

## Overview

This directory contains a comprehensive, staged validation framework for the Israel-Stewart hydrodynamics implementation. The approach validates each component systematically before integration, ensuring correctness at every level.

## Philosophy

**"Validate components in isolation before integration"**

Traditional approach (broken):
```
Write everything → Test full system → Debug when it fails → Can't isolate root cause
```

Staged validation approach (working):
```
Stage 1: Units → Stage 2: Coefficients → Stage 3: Equations → ... → Stage 7: Integration
   ✓            ✓                    ✓                              ✓
(Can't proceed to next stage until current stage passes)
```

## Directory Structure

```
validation/
├── README.md (this file)
├── VALIDATION_ROADMAP.md (master progress tracker)
│
├── stage1_units/          # Unit & dimensional analysis
├── stage2_coefficients/   # IReD transport coefficients
├── stage3_equations/      # Conservation & relaxation equations
│   ├── conservation/      # ∇_μ T^μν = 0
│   └── relaxation/        # Israel-Stewart evolution
├── stage4_dispersion/     # Analytical dispersion relations
├── stage5_solvers/        # Numerical integrators
│   ├── spectral/          # FFT-based methods
│   ├── rk4/               # Runge-Kutta 4th order
│   └── imex/              # Implicit-explicit methods
├── stage6_benchmarks/     # Full analytical validation
│   ├── bjorken/           # 1D boost-invariant expansion
│   ├── sound_waves/       # Linear wave propagation
│   └── diffusion/         # Particle diffusion
├── stage7_integration/    # Full system testing
│
├── scripts/               # Utility scripts
└── archive/               # Historical debugging artifacts
```

## Validation Stages

### Stage 1: Units & Dimensional Analysis
**Goal**: Verify all quantities have correct physical dimensions

**Status**: 🟡 90% complete (26/29 tests passing)

**Blockers**: Need to verify λ_πV temperature scaling

See `stage1_units/README.md` for details.

### Stage 2: Transport Coefficients
**Goal**: Validate IReD coefficient implementation against kinetic theory

**Status**: ✅ 100% complete (26/26 tests passing)

See `stage2_coefficients/README.md` for details.

### Stage 3: Equation Components
**Goal**: Test conservation laws and relaxation equations in isolation

**Status**: ✅ 100% complete (58/60 tests passing, 2 skipped)

**Completion Date**: 2025-10-20

See `stage3_equations/README.md` for details.

### Stage 4: Dispersion Relations
**Goal**: Validate analytical eigenmode finding

**Status**: 🟢 70% complete (major fix from Option A)

**Recent**: Fixed nearly-ideal mode acceptance (|Γ/ω| < 1%)

See `stage4_dispersion/README.md` for details.

### Stage 5: Numerical Solvers
**Goal**: Validate integrators on simple test problems

**Status**: 🔴 20% complete (no unit tests yet)

**Blockers**: Need convergence tests for each solver

See `stage5_solvers/README.md` for details.

### Stage 6: Analytical Benchmarks
**Goal**: Full system validation against known solutions

**Status**: 🟡 50% complete (3/6 benchmarks passing)

**Passing**: Sound wave frequency, Bjorken shear, Diffusion Fick's law
**Failing**: Sound damping (regime paradox), Bjorken T (not evolving), Diffusion decay (long-time instability)

See `stage6_benchmarks/README.md` for details.

### Stage 7: Integration Testing
**Goal**: Comprehensive system testing with realistic scenarios

**Status**: 🔴 0% complete (blocked on Stages 3-6)

See `stage7_integration/README.md` for details.

## How to Use This Framework

### For Developers

1. **Adding new physics**: Start at Stage 1 (check dimensions), work through each stage
2. **Debugging failures**: Find the lowest stage that fails, fix there before proceeding
3. **Adding tests**: Place in appropriate stage based on what's being validated
4. **Running validation**: `python validation/scripts/check_all_stages.py`

### For Reviewers

1. **Check VALIDATION_ROADMAP.md** for current status
2. **Each stage has acceptance criteria** - tests must pass before proceeding
3. **Results are documented** in `stage*/results/` directories
4. **Historical context** preserved in `archive/` if needed

## Status Legend

- ✅ Complete: All tests passing, acceptance criteria met
- 🟢 Good: Most tests passing, minor issues
- 🟡 Needs Work: Some tests passing, significant gaps
- 🔴 Blocked: Major issues preventing progress

## Quick Links

- [Master Roadmap](VALIDATION_ROADMAP.md) - Overall progress and timeline
- [Stage 1: Units](stage1_units/README.md)
- [Stage 2: Coefficients](stage2_coefficients/README.md)
- [Stage 3: Equations](stage3_equations/README.md)
- [Stage 4: Dispersion](stage4_dispersion/README.md)
- [Stage 5: Solvers](stage5_solvers/README.md)
- [Stage 6: Benchmarks](stage6_benchmarks/README.md)
- [Stage 7: Integration](stage7_integration/README.md)

## Key Principles

1. **No skipping stages**: Can't proceed to Stage N+1 without passing Stage N
2. **Document everything**: Each stage has README explaining goals and acceptance
3. **Preserve history**: Archive old scripts/notes for context
4. **Clear ownership**: Each validation script has specific purpose
5. **Measure progress**: VALIDATION_ROADMAP.md tracks completion

## Recent Progress

- **2025-10-20**: Stage 3 completed - all equation validation tests passing
- **2025-10-20**: Fixed critical covariant divergence bug in conservation.py
- **2025-10-20**: Added Stage 3A.3 (shear tensor) and 3A.4 (curved spacetime) tests
- **2025-10-18**: Option A completed - fixed nearly-ideal mode acceptance
- **2025-10-18**: Disabled coverage by default (400× faster tests)
- **2025-10-17**: Phase 4 consolidation - removed 8 duplicate tests
- **2025-10-17**: IReD coefficient tests 26/26 passing

## Contributing

When adding validation scripts:
1. Determine which stage it belongs to
2. Add to appropriate subdirectory
3. Update stage README.md
4. Update VALIDATION_ROADMAP.md if completing a task
5. Store results in `results/` subdirectory

## References

- **IReD Theory**: `../docs/IRED_THEORY.md`
- **Test Plan**: Original plan evolved into this staged framework
- **Regime Validity**: Wagner & Gavassino (2024), see `stage4_dispersion/`
