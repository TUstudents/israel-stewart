# Spectral Solver Critical Bug Fix Plan

## Status Assessment

Based on analysis of `israel_stewart/solvers/spectral.py`, the following critical bugs have been identified:

### **Priority 1: Critical Physics Fixes**

#### 1. **CRITICAL: Incorrect Conservation Law in Fallback** ❌ **CONFIRMED BUG**
- **Location**: `_fallback_conservation_advance` method (~line 863)
- **Current Code**: `∂_t ρ = -∂_i T^0i`
- **Should Be**: `∂_t ρ = -∂_i T^i0`
- **Impact**: Fundamentally incorrect energy conservation physics
- **Fix**: Change stress tensor component indexing

#### 2. **MAJOR: Flawed Dealiasing Implementation** ❌ **CONFIRMED BUG**
- **Location**: `_apply_dealiasing` method
- **Current Code**: Zeros entire slices `result[kx_max:, :, :] = 0`
- **Problem**: Doesn't respect `fftfreq` layout - should handle positive/negative frequencies
- **Impact**: Aliasing errors not properly removed
- **Fix**: Implement proper 2/3 rule respecting frequency ordering

### **Priority 2: Numerical Method Improvements**

#### 3. **MAJOR: IMEX-RK2 Scheme Validation** ⚠️ **NEEDS INVESTIGATION**
- **Location**: `_imex_rk2_step` method
- **Current Code**: Custom IMEX scheme implementation
- **Concern**: May not follow established IMEX-RK standards
- **Impact**: Potentially poor accuracy/stability
- **Fix**: Validate against standard Butcher tableau

#### 4. **MEDIUM: Ad-hoc Bulk Viscous Operator** ❌ **CONFIRMED ISSUE**
- **Location**: `apply_bulk_viscous_operator` method
- **Current Code**: `relaxation_factor * damped_field` (multiplicative)
- **Problem**: May not accurately model bulk viscosity physics
- **Impact**: Potentially incorrect bulk viscosity evolution
- **Fix**: Research proper Israel-Stewart bulk viscosity evolution

### **Priority 3: Performance & Robustness**

#### 5. **LOW: Real FFT Usage** ⚠️ **PARTIALLY ADDRESSED**
- **Location**: Throughout solver, FFT function calls
- **Current Code**: Uses complex FFTs even for real fields
- **Problem**: Suboptimal performance for real-valued fields
- **Impact**: 2x memory usage and slower performance
- **Fix**: Switch to `rfftn/irfftn` for real fields

#### 6. **LOW: Grid Spacing Fallback Warning** ⚠️ **SILENT ERROR RISK**
- **Location**: `__init__` method spacing calculation
- **Current Code**: Falls back to `grid.spatial_spacing` without warning
- **Problem**: Risk of silent numerical errors
- **Impact**: Incorrect derivatives if grid misconfigured
- **Fix**: Add warning when using potentially incorrect spacing

### **Priority 4: Architecture Enhancement**

#### 7. **MEDIUM: Flat Spacetime Limitation** ❌ **CONFIRMED LIMITATION**
- **Location**: `_init_physics_modules` method
- **Current Code**: Defaults to `MinkowskiMetric()` if no metric found
- **Problem**: Cannot handle curved spacetime problems
- **Impact**: Limits solver to flat spacetime only
- **Fix**: Add better metric handling and warnings

## Implementation Plan

### **Phase 1: Critical Physics Fixes (Immediate)**

1. **Fix Conservation Law Bug**
   ```python
   # Change from: energy_flux_div += self.spectral.spatial_derivative(T_munu[..., 0, i + 1], i)
   # To:         energy_flux_div += self.spectral.spatial_derivative(T_munu[..., i + 1, 0], i)
   ```

2. **Fix Dealiasing Implementation**
   ```python
   def _apply_dealiasing(self, field_k: np.ndarray) -> np.ndarray:
       # Proper implementation respecting fftfreq layout
       # Handle both positive and negative frequencies correctly
   ```

### **Phase 2: Numerical Method Validation (High Priority)**

3. **Validate IMEX-RK2 Scheme**
   - Review against standard IMEX-RK Butcher tableaux
   - Implement proper second-order IMEX scheme if needed
   - Add accuracy tests

4. **Improve Bulk Viscous Operator**
   - Research proper Israel-Stewart bulk viscosity evolution
   - Replace multiplicative approach with physics-based method

### **Phase 3: Performance Optimizations (Medium Priority)**

5. **Implement Real FFT Usage**
   - Switch to `rfftn/irfftn` for real-valued fields
   - Update all FFT calls consistently
   - Maintain interface compatibility

6. **Add Grid Spacing Warnings**
   - Issue warning when falling back to potentially incorrect spacing
   - Add validation checks for periodic domain setup

### **Phase 4: Architecture Enhancement (Lower Priority)**

7. **Address Flat Spacetime Limitation**
   - Add better metric handling for curved spacetime
   - Issue warning when defaulting to Minkowski metric
   - Document limitations clearly

## Validation Strategy

### **Testing Requirements**
- Add comprehensive test suite for each fix
- Verify physics correctness with analytical solutions
- Performance benchmarking for FFT optimizations
- Cross-validation with other solver methods

### **Physics Validation**
- Energy conservation tests
- Aliasing error measurement
- Accuracy convergence tests
- Stability analysis

### **Performance Validation**
- Memory usage benchmarks
- Computation time comparisons
- Scalability tests

## Expected Outcomes

### **Immediate Benefits**
- ✅ Correct energy conservation physics
- ✅ Proper aliasing error removal
- ✅ Improved numerical accuracy

### **Performance Improvements**
- 🚀 ~50% memory reduction from real FFTs
- 🚀 ~30% speed improvement
- 🚀 Better scalability for large problems

### **Scientific Impact**
- 🔬 Reliable spectral solver for Israel-Stewart equations
- 🔬 Production-ready for relativistic hydrodynamics research
- 🔬 Foundation for curved spacetime extensions

## Risk Mitigation

### **Backward Compatibility**
- Maintain existing API interfaces
- Add feature flags for new functionality
- Comprehensive regression testing

### **Validation Safety**
- Cross-check results with analytical solutions
- Compare with other numerical methods
- Extensive edge case testing

### **Performance Safety**
- Benchmark before/after performance
- Memory usage monitoring
- Fallback to safe methods if needed

This plan systematically addresses all critical bugs while maintaining scientific rigor and code quality.
