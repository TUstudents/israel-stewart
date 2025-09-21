# Memory-Efficient Jacobian Implementation Plan

## Current Problem Analysis

### Memory Crisis Identified
- **Root Cause**: Dense Jacobian matrices scale O(n²) where n = state vector size
- **Critical Example**: Grid (10,8,8,8) → 107,520 state vector → 88GB Jacobian matrix
- **Field Structure**: 27 variables per grid point:
  - ρ (energy density): 1 component
  - n (particle density): 1 component
  - u_μ (four-velocity): 4 components
  - Π (bulk pressure): 1 component
  - π_μν (shear tensor): 16 components
  - q_μ (heat flux): 4 components
- **Current Protection**: Hard 6GB limit prevents crashes but blocks legitimate large computations

### Current Implementation Issues
- `_compute_vectorized_jacobian()` creates full dense n×n matrices
- Memory requirements grow quadratically with grid resolution
- Batch processing still accumulates massive memory usage
- No exploitation of Israel-Stewart equation structure

## Solution Strategy: Jacobian-Free Newton-Krylov (JFNK) Methods

### Overview
Jacobian-Free Newton-Krylov methods combine:
- **Newton's Method**: Quadratic convergence for nonlinear systems
- **Krylov Subspace Methods**: Memory-efficient linear system solving
- **Matrix-Free Operations**: Avoid explicit Jacobian formation

Key insight: Only Jacobian-vector products J·v are needed, not the full matrix J.

## Approach 1: Matrix-Free Jacobian-Vector Products (HIGH PRIORITY)

### Memory Advantage
- **Current**: O(n²) storage for full Jacobian matrix
- **Matrix-Free**: O(n) storage for vectors only
- **Scaling**: Grid (20,16,16,16) becomes feasible within 6GB

### Technical Implementation

#### Core Algorithm
```python
def jacobian_vector_product(x, v, f, epsilon=1e-8):
    """Approximate J·v using finite differences"""
    f_x = f(x)
    f_x_plus_ev = f(x + epsilon * v)
    return (f_x_plus_ev - f_x) / epsilon
```

#### Integration with Existing Code
- Extend `scipy.sparse.linalg.LinearOperator`
- Replace explicit Jacobian in Newton iteration
- Maintain compatibility with GMRES/BiCGSTAB solvers

#### New Classes to Implement
```python
class JacobianFreeLinearOperator(LinearOperator):
    """Matrix-free Jacobian operator for JFNK methods"""

class MatrixFreeNewtonSolver(ImplicitSolverBase):
    """Newton solver using matrix-free Jacobian-vector products"""
```

## Approach 2: Block-Structured Analytical Jacobians (MEDIUM PRIORITY)

### Physics-Based Structure
Israel-Stewart equations have natural block structure:

#### Relaxation Blocks (Diagonal)
- **Bulk pressure**: ∂(dΠ/dt)/∂Π = -1/τ_Π
- **Shear tensor**: ∂(dπ^μν/dt)/∂π^μν = -1/τ_π
- **Heat flux**: ∂(dq^μ/dt)/∂q^μ = -1/τ_q

#### Conservation Blocks (Sparse)
- **Energy-momentum**: Spatial derivatives, local coupling
- **Particle number**: Advection terms, sparse structure

#### Coupling Blocks (Sparse)
- **Cross-field coupling**: Second-order Israel-Stewart terms
- **Spatial gradients**: Finite difference stencils

### Memory Efficiency
- **Block-diagonal**: O(n) storage for relaxation terms
- **Sparse blocks**: O(n) storage for spatial coupling
- **Total**: O(n) instead of O(n²)

### Implementation Strategy
```python
class BlockStructuredJacobian:
    """Exploit Israel-Stewart equation structure"""
    relaxation_blocks: List[sparse.spmatrix]
    conservation_blocks: List[sparse.spmatrix]
    coupling_blocks: List[sparse.spmatrix]
```

## Approach 3: Hierarchical Preconditioning (MEDIUM PRIORITY)

### Preconditioning Challenges
JFNK methods require effective preconditioning for fast convergence:
- **Physics-based**: Exploit Israel-Stewart timescales
- **Multigrid-inspired**: Handle spatial scales efficiently
- **Adaptive**: Adjust to local stiffness

### Israel-Stewart Preconditioners

#### Relaxation Preconditioner
```python
class RelaxationPreconditioner:
    """Block-diagonal preconditioner for relaxation terms"""
    # P^{-1} ≈ diag(τ_Π, τ_π, τ_q)
```

#### Schur Complement Preconditioner
```python
class SchurComplementPreconditioner:
    """Approximate Schur complement for field coupling"""
    # Separate fast (relaxation) and slow (hydrodynamic) modes
```

#### Spatial Multigrid Preconditioner
```python
class SpatialMultigridPreconditioner:
    """Multigrid-inspired spatial preconditioning"""
    # Handle different length scales in spatial gradients
```

## Approach 4: Adaptive Memory Management (LOW PRIORITY)

### Runtime Method Selection
```python
def select_jacobian_method(grid_size, available_memory):
    """Choose optimal method based on resources"""
    if memory_required(analytical) < available_memory:
        return AnalyticalJacobian()
    elif memory_required(block_structured) < available_memory:
        return BlockStructuredJacobian()
    else:
        return JacobianFreeOperator()
```

### Memory-Aware Grid Management
- **Automatic coarsening**: Reduce grid resolution if memory insufficient
- **Adaptive refinement**: Increase resolution where possible
- **Load balancing**: Distribute memory usage across processes

## Implementation Roadmap

### Phase 1: Jacobian-Free Foundation (Weeks 1-2)

#### Week 1: Core Implementation
1. **Create `JacobianFreeLinearOperator`**
   - Implement `_matvec` method using finite differences
   - Handle perturbation parameter selection automatically
   - Add memory monitoring and safety checks

2. **Integrate with Newton Solver**
   - Modify `BackwardEulerSolver` to support matrix-free mode
   - Add command-line/config option to enable JFNK
   - Maintain backward compatibility with existing tests

#### Week 2: Testing and Validation
3. **Comprehensive Testing**
   - Unit tests for Jacobian-vector products
   - Integration tests with Israel-Stewart equations
   - Performance benchmarks against analytical methods

4. **Memory Validation**
   - Verify O(n) memory scaling
   - Test with large grids (16×16×16×16 and larger)
   - Benchmark against 6GB memory limit

### Phase 2: Physics-Informed Optimization (Weeks 3-4)

#### Week 3: Block Structure
5. **Implement Block-Structured Jacobians**
   - Code relaxation blocks analytically
   - Implement sparse conservation law blocks
   - Add coupling terms for second-order IS theory

6. **Memory Layout Optimization**
   - Optimize data structures for cache efficiency
   - Implement memory pools for frequent allocations
   - Add NUMA-aware memory management if needed

#### Week 4: Preconditioning
7. **Basic Preconditioners**
   - Implement relaxation-based block preconditioner
   - Add simple spatial preconditioning (Jacobi-style)
   - Test convergence rates with different preconditioners

8. **Advanced Preconditioning**
   - Implement approximate Schur complement methods
   - Add physics-based preconditioner selection
   - Benchmark convergence vs computational cost

### Phase 3: Production Integration (Weeks 5-6)

#### Week 5: Performance Optimization
9. **Computational Efficiency**
   - Profile and optimize hot code paths
   - Implement vectorized operations where possible
   - Add OpenMP parallelization for Jacobian-vector products

10. **Method Selection Logic**
    - Implement automatic method selection based on problem size
    - Add runtime switching between methods
    - Create performance prediction models

#### Week 6: Integration and Documentation
11. **Production Features**
    - Add comprehensive error handling and recovery
    - Implement checkpointing for long-running simulations
    - Create user-friendly configuration interfaces

12. **Documentation and Examples**
    - Write comprehensive API documentation
    - Create tutorial examples for different problem sizes
    - Document performance characteristics and best practices

## Expected Outcomes

### Memory Improvements
- **Scalability**: Enable grids up to (32,32,32,32) within 6GB limit
- **Efficiency**: Reduce memory usage from O(n²) to O(n)
- **Flexibility**: Support adaptive grid refinement

### Performance Benefits
- **Convergence**: Maintain Newton quadratic convergence rates
- **Speed**: Comparable or better performance due to reduced memory pressure
- **Robustness**: Multiple fallback methods for challenging problems

### Scientific Impact
- **Larger Simulations**: Enable realistic 3D relativistic hydrodynamics
- **Better Physics**: Support full second-order Israel-Stewart coupling
- **Research Advancement**: Production-ready viscous relativistic fluid dynamics

## Risk Mitigation

### Technical Risks and Solutions

#### Convergence Issues
- **Risk**: Matrix-free methods may converge slowly without good preconditioning
- **Mitigation**: Implement multiple preconditioner options and automatic selection
- **Fallback**: Keep analytical Jacobian methods for critical cases

#### Performance Regression
- **Risk**: Function evaluations for Jacobian-vector products may be expensive
- **Mitigation**: Optimize RHS function evaluation, implement caching
- **Monitoring**: Continuous benchmarking against existing methods

#### Numerical Stability
- **Risk**: Finite difference approximations may introduce errors
- **Mitigation**: Adaptive perturbation parameter selection
- **Validation**: Extensive testing against analytical solutions

### Implementation Safety

#### Backward Compatibility
- Keep existing analytical Jacobian methods as default
- Add matrix-free methods as optional features
- Extensive regression testing with current test suite

#### Gradual Rollout
- Implement as feature flags with easy enable/disable
- Start with simple test problems before production use
- Comprehensive validation against known solutions

#### Memory Safety
- Maintain existing 6GB protection system
- Add additional memory monitoring for new methods
- Implement graceful degradation when memory is insufficient

## Success Metrics

### Performance Targets
- **Memory Usage**: Achieve O(n) scaling verified up to n=10⁶
- **Convergence**: Maintain Newton convergence rates within 10% of analytical methods
- **Speed**: Achieve competitive performance for large systems (n>10⁵)

### Scientific Goals
- **Grid Resolution**: Enable (32,32,32,32) Israel-Stewart simulations
- **Physics Fidelity**: Support full second-order viscous hydrodynamics
- **Production Readiness**: Robust enough for research publications

### Code Quality
- **Test Coverage**: >95% coverage for new Jacobian-free code
- **Documentation**: Complete API documentation and user guides
- **Maintainability**: Clean, modular code following project conventions

This plan transforms the current memory crisis into an opportunity to implement state-of-the-art numerical methods that will enable significantly larger and more realistic Israel-Stewart simulations while maintaining computational efficiency and scientific accuracy.
