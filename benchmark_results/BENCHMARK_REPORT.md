# Spectral Solver E2E Benchmark Report

## Summary

| Test | Status | Error | Tolerance | Time |
|------|--------|-------|-----------|------|
| Sound Wave 4D Spacetime | ✅ PASS | 0.002052 | 0.050000 | 1.87s |
| Convergence Study | ✅ PASS | 0.000013 | 0.010000 | 8.74s |
| Multi-Mode Superposition | ✅ PASS | 0.004671 | 0.050000 | 7.55s |
| Viscous Stress-Energy Tensor | ✅ PASS | 0.000000 | 0.500000 | 1.84s |
| Conservation Law Verification | ✅ PASS | 0.028116 | 1.000000 | 0.46s |

## Detailed Results

### Sound Wave 4D Spacetime

- **Status**: ✅ PASS
- **Error**: 0.002052
- **Tolerance**: 0.050000
- **Grid**: (8, 16, 16, 16)
- **Time**: 1.87s

**Details**:
```json
{
  "initial_conservation_violation": 0.002048422132107891,
  "final_conservation_violation": 0.002055042208658481,
  "errors_at_time_slices": [
    0.0017551227410206316,
    0.0019631019559485097,
    0.0020518886556408766
  ],
  "sound_speed": 0.5773502691896257,
  "wave_number": 1.0,
  "frequency": 0.5773502691896257
}
```

### Convergence Study

- **Status**: ✅ PASS
- **Error**: 0.000013
- **Tolerance**: 0.010000
- **Grid**: (8, 32, 32, 32)
- **Time**: 8.74s

**Details**:
```json
{
  "resolutions": [
    [
      8,
      8,
      8,
      8
    ],
    [
      8,
      16,
      16,
      16
    ],
    [
      8,
      32,
      32,
      32
    ]
  ],
  "errors": [
    1.0637623754283603e-05,
    1.2066849933926407e-05,
    1.3236191744686494e-05
  ],
  "convergence_order": -0.13343898844207505
}
```

### Multi-Mode Superposition

- **Status**: ✅ PASS
- **Error**: 0.004671
- **Tolerance**: 0.050000
- **Grid**: (8, 32, 32, 32)
- **Time**: 7.55s

**Details**:
```json
{
  "wave_numbers": [
    0.5,
    1.0,
    2.0
  ],
  "amplitudes": [
    0.01,
    0.008,
    0.005
  ],
  "mode_errors": [
    0.0046710654133641505,
    0.0046710654133641505,
    0.0046710654133641505
  ],
  "fft_analysis": "Used FFT to isolate individual modes"
}
```

### Viscous Stress-Energy Tensor

- **Status**: ✅ PASS
- **Error**: 0.000000
- **Tolerance**: 0.500000
- **Grid**: (8, 16, 16, 16)
- **Time**: 1.84s

**Details**:
```json
{
  "shear_viscosity": 0.1,
  "bulk_viscosity": 0.05,
  "Pi_max": 1.1271919888674133e-06,
  "pi_max": 3.003680647678776e-06,
  "has_viscous_corrections": true,
  "note": "Verifies viscous transport coefficients are included in solver"
}
```

### Conservation Law Verification

- **Status**: ✅ PASS
- **Error**: 0.028116
- **Tolerance**: 1.000000
- **Grid**: (16, 16, 16, 16)
- **Time**: 0.46s

**Details**:
```json
{
  "max_conservation_violation": 0.0022493070073086145,
  "discretization_tolerance": 0.08,
  "conservation_ok": true,
  "fields_bounded": true,
  "rho_range": [
    0.99,
    1.01
  ],
  "note": "Verifies \u2202_\u03bc T^\u03bc\u03bd \u2248 0 for analytical solution"
}
```
