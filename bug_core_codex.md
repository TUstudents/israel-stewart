# Core Module Code Review

## constants.py
- No blocking issues observed; conversions and constant definitions look consistent with the documented unit system.

## derivatives.py
- **High** – israel_stewart/core/derivatives.py:68 – The symbolic gradient path wraps components in an `sp.Matrix`, yielding a `(4, 1)` shape that `FourVector` rejects; any symbolic use immediately raises `ValueError`.
- **Medium** – israel_stewart/core/derivatives.py:105 – `np.gradient(..., edge_order=2)` fails for axes with only two grid points (which `SpacetimeGrid` allows), so divergence crashes on minimally resolved domains.
- **Medium** – israel_stewart/core/derivatives.py:395 – The parallel projector hardcodes `V_∥^μ = -(u·V)u^μ`; this flips the sign when the metric uses the mostly-minus convention, violating orthogonality.

## fields.py
- **High** – israel_stewart/core/fields.py:842 – `ISFieldConfiguration.apply_constraints()` raises for any grid whose `metric` is `None` (the default Minkowski case), preventing constraint enforcement in the most common setup.
- **High** – israel_stewart/core/fields.py:853 – `_project_shear_tensor` assumes `self.grid.metric.inverse` has `ndim`; with symbolic metrics this is a `sympy.Matrix` and the attribute access raises, breaking Milne/FLRW grids.

## four_vectors.py
- **High** – israel_stewart/core/four_vectors.py:35 – `FourVector` still demands an exact `(4,)` shape; SymPy column vectors `(4,1)` from other APIs cannot be constructed, so symbolic workflows fail outright.
- **Medium** – israel_stewart/core/four_vectors.py:252 – `is_timelike` / `is_spacelike` cast results to `float`, which raises `TypeError` for symbolic magnitudes, disabling signature checks for SymPy vectors.
- **Medium** – israel_stewart/core/four_vectors.py:253 – `boost_to_rest_frame` calls `is_timelike()` even when the vector lacks an attached metric (the default), so Minkowski boosts raise instead of defaulting to flat-space behavior.

## metrics.py
- **High** – israel_stewart/core/metrics.py:963 – `SchwarzschildMetric.components` introduces a fresh symbol `rs` and ignores `self.rs`; callers never see their configured radius reflected in the metric.

## performance.py
- No issues noted.

## spacetime_grid.py
- **High** – israel_stewart/core/spacetime_grid.py:364 – The periodic-boundary axis map sets `'phi'` → 2 even in spherical coordinates, so φ-boundaries actually wrap θ.
- **High** – israel_stewart/core/spacetime_grid.py:482 – For metrics with symbolic determinants (Milne/FLRW), `np.sqrt` on a SymPy expression raises immediately, making `volume_element()` unusable once a metric is attached.

## stress_tensors.py
- **High** – israel_stewart/core/stress_tensors.py:99 – The SymPy branch builds a `(4,1)` momentum matrix that `FourVector` refuses, so symbolic stress tensors can’t extract momentum density.

## tensor_base.py
- **Medium** – israel_stewart/core/tensor_base.py:192 – `TensorField.copy()` always returns a bare `TensorField`, stripping subclass behavior (`FourVector`, `StressEnergyTensor`, etc.) and breaking method chaining.

## tensor_utils.py
- **High** – israel_stewart/core/tensor_utils.py:55 – `convert_to_sympy` blindly calls `sp.Matrix` on higher-rank arrays; SymPy cannot build matrices from >2D data, so conversions for grid-based tensors crash.
- **High** – israel_stewart/core/tensor_utils.py:234 – The strict `expected_shape` check still rejects `(4,1)` column vectors, which is why several SymPy pathways fail despite newer validation helpers.

## tensors.py
- No issues noted (re-export module only).

## transformations.py
- **High** – israel_stewart/core/transformations.py:198 – The SymPy tensor branch applies either Λ or Λ⁻¹ identically to both indices, ignoring their variance; mixed-type tensors transform incorrectly.
- **Medium** – israel_stewart/core/transformations.py:145 – SymPy vector transformations return `(4,1)` matrices, immediately triggering the FourVector shape guard and breaking symbolic Lorentz boosts.
- **Medium** – israel_stewart/core/transformations.py:253 – As with `FourVector`, `boost_to_rest_frame` can’t operate without a metric because it forces a timelike check; default Minkowski use raises instead of falling back to flat space.
