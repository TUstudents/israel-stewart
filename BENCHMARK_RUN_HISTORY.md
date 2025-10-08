# Benchmark Run History

This document summarizes the key experimental runs performed during the debugging session for the sound wave benchmark (`k=8.0`).

## Key Parameters

- **Viscosities (Constant):**
  - Shear Viscosity (η): `0.08`
  - Bulk Viscosity (ζ): `0.04`
- **Grid Resolution (Standard):** `(32, 32, 16)`
- **Integrator:** `spectral_imex`

## Run Summary

| Run ID                      | `τ_π` | `τ_Π` | 2nd-Order Coeffs | Analytical `ω` | Measured `ω` | Freq. Error | Damping Error | Observation                                                                                             |
|-----------------------------|-------|-------|------------------|----------------|--------------|-------------|---------------|---------------------------------------------------------------------------------------------------------|
| 1. Initial Failing Run      | 0.5   | 0.3   | Default          | `6.006`        | `3.997`      | `33.4%`     | `73.7%`       | The analytical dispersion matrix had two compensating sign errors, leading to an incorrect prediction.      |
| 2. Stable Parameter Run     | 1.0   | 0.5   | Default          | `5.457`        | `5.448`      | `0.2%`      | `75.7%`       | **Frequency converged.** The remaining large error in damping suggests interference from second-order terms. |
