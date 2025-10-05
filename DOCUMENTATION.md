# Solver Architecture and Flow

This document provides a high-level overview of the spectral solver's architecture and the program flow during a single time step evolution.

## 1. Component Map

The solver is composed of several interacting classes, each with a distinct responsibility.

| Class | File | Role | Inputs | Key Outputs |
| :--- | :--- | :--- | :--- | :--- |
| **`SpectralISHydrodynamics`** | `solvers/spectral.py` | **The Orchestrator.** Manages the overall time-stepping process and owns all other components. | `grid`, `fields`, `coeffs` | Evolves the `fields` object over time. |
| **`SpectralISolver`** | `solvers/spectral.py` | **The Spectral Engine.** Provides functions for performing high-precision spatial derivatives using FFTs. | `grid`, `fields`, `coeffs` | `spatial_derivative()`, `spatial_divergence()` |
| **`ConservationLaws`** | `equations/conservation.py`| **The Ideal Fluid.** Calculates the evolution of the ideal part of the fluid (density, velocity) based on the conservation of energy and momentum. | `fields`, `coeffs`, `spectral_solver` | `d(ρ)/dt`, `d(ρu)/dt` |
| **`ISRelaxationEquations`** | `equations/relaxation.py`| **The Viscous Physics.** Calculates the source terms for the viscous parts of the fluid (shear stress, bulk pressure) based on velocity gradients. | `grid`, `metric`, `coeffs`, `spectral_solver` | `d(π)/dt`, `d(Π)/dt` |
| **`ISFieldConfiguration`** | `core/fields.py` | **The State Vector.** A container that holds all the physical fields (`ρ`, `u`, `π`, `Π`) as NumPy arrays. | `grid` | The current state of the simulation. |

<br>

## 2. Time Step Flowchart (`split_step` method)

This flowchart describes the sequence of operations during a single call to `hydro.time_step(dt)`.

**START: `time_step(dt)`**
> The main `SpectralISHydrodynamics` object begins the time evolution.

1.  **`advance_linear_terms(dt/2)`**
    *   **Who:** `SpectralISolver`
    *   **What:** The linear part of the relaxation equations is solved exactly for a half time step.
    *   **Action:** The viscous fields `π` and `Π` are multiplied by `exp(-dt / 2τ)`. This is the **first half of the damping**.
    *   **Input:** `fields`, `dt/2`
    *   **Output:** `fields` are updated in-place.

2.  **`_advance_conservation_laws(dt)`**
    *   **Who:** `SpectralISHydrodynamics` calls `ConservationLaws`.
    *   **What:** The ideal, non-dissipative part of the fluid evolution is calculated.
    *   **Action:**
        *   `ConservationLaws` calculates the divergence of the stress-energy tensor (`∂ᵢTⁱᶝ`) using the **spectral solver**.
        *   `SpectralISHydrodynamics` takes the result and uses a 2nd-order Runge-Kutta method to update the density `ρ` and velocity `u`.
    *   **Input:** `fields`, `dt`
    *   **Output:** `fields.rho` and `fields.u_mu` are updated in-place.

3.  **`_advance_relaxation_terms(dt)`**
    *   **Who:** `SpectralISHydrodynamics` calls `ISRelaxationEquations`.
    *   **What:** The source terms for the viscous effects are calculated and applied.
    *   **Action:**
        *   `ISRelaxationEquations` calculates velocity gradients (`∇u`) using the **spectral solver** to get the shear tensor `σ` and expansion `θ`.
        *   It then calculates the source terms (e.g., `2ησ`).
        *   `SpectralISHydrodynamics` applies these source terms to the viscous fields `π` and `Π` using a simple Euler step (`π_new = π_old + dt * source`).
    *   **Input:** `fields`, `dt`
    *   **Output:** `fields.pi_munu` and `fields.Pi` are updated in-place.

4.  **`advance_linear_terms(dt/2)`**
    *   **Who:** `SpectralISolver`
    *   **What:** The linear part of the relaxation is applied again for the second half of the time step.
    *   **Action:** The viscous fields `π` and `Π` are multiplied by `exp(-dt / 2τ)`. This is the **second half of the damping**.
    *   **Input:** `fields`, `dt/2`
    *   **Output:** `fields` are updated in-place.

**END OF TIME STEP**
