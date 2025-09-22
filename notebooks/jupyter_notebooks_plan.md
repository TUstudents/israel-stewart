# Jupyter Notebook Series: Israel-Stewart Theory and Implementation

## Design Philosophy
- **Theory-Code Integration**: Mathematical concepts immediately followed by working code examples
- **Progressive Understanding**: Each section builds naturally on the previous
- **Analytical Validation**: Every implementation verified against exact solutions
- **Live Documentation**: Notebooks serve as both tutorials and API reference

---

## **Notebook 1: Relativistic Tensors and Four-Vectors**
**File:** `01_relativistic_tensors.ipynb`

### Section 1.1: Special Relativity Foundations
**Theory**: Spacetime as 4D manifold, Lorentz transformations, proper time
```python
# Four-vector basics: position, velocity, momentum
x_mu = np.array([t, x, y, z])  # Spacetime coordinates
u_mu = np.array([gamma, gamma*vx, gamma*vy, gamma*vz])  # Four-velocity

# Minkowski metric and normalization
eta = np.diag([-1, 1, 1, 1])  # Mostly-plus signature
u_squared = np.dot(u_mu, np.dot(eta, u_mu))
assert abs(u_squared + 1) < 1e-15  # u^μ u_μ = -1
```

### Section 1.2: Library Implementation
**Theory**: TensorField class design, index management, automatic contractions
```python
from israel_stewart.core import FourVector, MinkowskiMetric

# Create four-velocity at rest
metric = MinkowskiMetric()
u = FourVector(grid, metric, data=[1, 0, 0, 0])

# Verify normalization automatically
print(f"Four-velocity norm: {u.norm(metric)}")  # Should be -1
print(f"Spatial velocity: {u.spatial_magnitude()}")  # Should be 0
```

### Section 1.3: Stress-Energy Tensor
**Theory**: Energy-momentum conservation, perfect fluid form T^μν = (ρ+p)u^μu^ν + pg^μν
```python
from israel_stewart.core import StressEnergyTensor

# Perfect fluid stress-energy tensor
rho, p = 1.0, 1.0/3.0  # Energy density and pressure
T = StressEnergyTensor(grid, metric)
T.construct_perfect_fluid(rho, p, u)

# Verify conservation: ∇_μ T^μν = 0 (should be exact for static case)
divergence = T.compute_divergence(metric)
print(f"Conservation violation: {np.max(np.abs(divergence))}")  # ~machine precision
```

---

## **Notebook 2: Curved Spacetime and Covariant Derivatives**
**File:** `02_curved_spacetime.ipynb`

### Section 2.1: Why Curved Spacetime?
**Theory**: Equivalence principle, general covariance, connection to gravity
```python
# Accelerated observer sees fictitious forces
a = 9.8  # Acceleration in m/s²
# In accelerated frame, effective metric becomes time-dependent
g_effective = lambda t: np.diag([-(1 + a*t)**2, 1, 1, 1])
```

### Section 2.2: Christoffel Symbols
**Theory**: Connection coefficients, geodesic equation, parallel transport
**Mathematical form**: Γ^λ_μν = ½g^λσ(∂_μg_σν + ∂_νg_σμ - ∂_σg_μν)
```python
from israel_stewart.core import GeneralMetric

# Schwarzschild metric example
def schwarzschild_metric_components(r, M=1):
    """Schwarzschild metric in spherical coordinates"""
    rs = 2*M  # Schwarzschild radius
    g = np.zeros((4, 4))
    g[0,0] = -(1 - rs/r)  # g_tt
    g[1,1] = 1/(1 - rs/r)  # g_rr
    g[2,2] = r**2          # g_θθ
    g[3,3] = r**2 * np.sin(theta)**2  # g_φφ
    return g

metric = GeneralMetric(schwarzschild_metric_components)
christoffel = metric.compute_christoffel_symbols()

# Verify known analytical result for Γ^r_tr component
Gamma_r_tr_exact = rs/(2*r*(r-rs))
Gamma_r_tr_numerical = christoffel[1, 0, 1]
print(f"Analytical: {Gamma_r_tr_exact:.6f}")
print(f"Numerical:  {Gamma_r_tr_numerical:.6f}")
print(f"Error: {abs(Gamma_r_tr_exact - Gamma_r_tr_numerical):.2e}")
```

### Section 2.3: Covariant Derivatives in Action
**Theory**: Why ∂_μV^ν ≠ ∇_μV^ν, metric compatibility, torsion-free condition
```python
# Test covariant derivative of metric (should be zero)
metric_tensor = metric.get_metric_tensor()
covariant_deriv_metric = metric.covariant_derivative(metric_tensor)

print("Metric compatibility test:")
print(f"Max |∇_λ g_μν|: {np.max(np.abs(covariant_deriv_metric))}")
assert np.allclose(covariant_deriv_metric, 0, atol=1e-12)
```

---

## **Notebook 3: Perfect Fluids and Conservation Laws**
**File:** `03_perfect_fluid_conservation.ipynb`

### Section 3.1: Thermodynamics of Relativistic Fluids
**Theory**: Local thermodynamic equilibrium, equation of state, speed of sound
**Key relations**:
- p = p(ρ, s) - equation of state
- c_s² = (∂p/∂ρ)_s - adiabatic sound speed
- For radiation: p = ρ/3, c_s² = 1/3

```python
# Implement equation of state
class RadiationEOS:
    """Equation of state for relativistic ideal gas"""
    def pressure(self, rho):
        return rho / 3.0

    def sound_speed_squared(self, rho):
        return 1.0 / 3.0

    def temperature(self, rho):
        # Stefan-Boltzmann: ρ = aT⁴ where a = π²g*/30
        a = np.pi**2 * 37.5 / 30  # g* = 37.5 for QGP
        return (rho / a)**(1/4)

eos = RadiationEOS()

# Test thermodynamic consistency
rho = 1.0  # GeV/fm³
p = eos.pressure(rho)
cs2 = eos.sound_speed_squared(rho)
T = eos.temperature(rho)

print(f"Energy density: {rho:.3f} GeV/fm³")
print(f"Pressure: {p:.3f} GeV/fm³")
print(f"Sound speed: {np.sqrt(cs2):.3f} c")
print(f"Temperature: {T:.3f} GeV")
```

### Section 3.2: Conservation Laws Implementation
**Theory**: Energy-momentum conservation ∇_μT^μν = 0, continuity equation ∇_μN^μ = 0
```python
from israel_stewart.equations import ConservationLaws

# Setup conservation law checker
conservation = ConservationLaws(grid, metric)

# Create equilibrium perfect fluid
fields = ISFieldConfiguration(grid)
fields.rho.fill(1.0)
fields.pressure.fill(1.0/3.0)
fields.four_velocity.data[..., 0] = 1.0  # At rest
fields.four_velocity.data[..., 1:] = 0.0

# Check conservation (should be exact for static equilibrium)
violations = conservation.check_conservation(fields)
print("Conservation violations:")
print(f"Energy: {violations['energy']:.2e}")
print(f"Momentum x: {violations['momentum_x']:.2e}")
print(f"Momentum y: {violations['momentum_y']:.2e}")
print(f"Momentum z: {violations['momentum_z']:.2e}")
```

### Section 3.3: Simple Wave Solutions
**Theory**: Linear perturbations, sound wave equation ∂²φ/∂t² = c_s²∇²φ
```python
# Analytical sound wave: φ(x,t) = A cos(kx - ωt)
k = 2*np.pi / 5.0  # Wavelength λ = 5 fm
omega = np.sqrt(1/3) * k  # ω = c_s k for ideal fluid
A = 0.01  # Small amplitude

x = np.linspace(0, 10, 100)
t = 2.0
phi_analytical = A * np.cos(k*x - omega*t)

# Verify dispersion relation
print(f"k = {k:.3f} fm⁻¹")
print(f"ω_analytical = {omega:.3f} fm⁻¹")
print(f"ω_expected = c_s × k = {np.sqrt(1/3) * k:.3f} fm⁻¹")
print(f"Phase velocity = {omega/k:.3f} c")
```

---

## **Notebook 4: Bjorken Flow - The Fundamental Solution**
**File:** `04_bjorken_flow_complete.ipynb`

### Section 4.1: Heavy-Ion Physics Context
**Theory**: Relativistic nuclear collisions, quark-gluon plasma, boost invariance
**Physical picture**: Two nuclei collide at nearly light speed, creating hot dense matter

```python
# Heavy-ion collision kinematics
sqrt_s_NN = 5020  # GeV (LHC Pb-Pb)
gamma_lab = sqrt_s_NN / (2 * 0.938)  # Lorentz factor
print(f"Laboratory Lorentz factor: γ = {gamma_lab:.0f}")
print(f"Laboratory velocity: v = {np.sqrt(1 - 1/gamma_lab**2):.6f} c")

# Time scales
tau_form = 0.1  # fm/c - formation time
tau_hydro = 10  # fm/c - hydrodynamic evolution
tau_freeze = 15  # fm/c - freeze-out
print(f"Formation: {tau_form} fm/c")
print(f"Hydrodynamic evolution: {tau_hydro - tau_form} fm/c")
print(f"Total evolution: {tau_freeze} fm/c")
```

### Section 4.2: Milne Coordinates and Boost Invariance
**Theory**: Proper time τ = √(t²-z²), rapidity η = ½ln[(t+z)/(t-z)]
**Key insight**: Boost invariance ⟹ ∂/∂η = 0

```python
# Coordinate transformation: (t,z) ↔ (τ,η)
def cartesian_to_milne(t, z):
    tau = np.sqrt(t**2 - z**2)
    eta = 0.5 * np.log((t + z) / (t - z))
    return tau, eta

def milne_to_cartesian(tau, eta):
    t = tau * np.cosh(eta)
    z = tau * np.sinh(eta)
    return t, z

# Test coordinate transformation
t, z = 5.0, 3.0
tau, eta = cartesian_to_milne(t, z)
t_back, z_back = milne_to_cartesian(tau, eta)

print(f"Original: (t={t}, z={z})")
print(f"Milne: (τ={tau:.3f}, η={eta:.3f})")
print(f"Back: (t={t_back:.3f}, z={z_back:.3f})")
print(f"Round-trip error: {abs(t-t_back) + abs(z-z_back):.2e}")
```

### Section 4.3: Exact Solution and Validation
**Theory**: Scaling solution T(τ) = T₀(τ₀/τ)^(1/3), entropy conservation
```python
from israel_stewart.benchmarks import BjorkenFlowBenchmark

# Bjorken flow exact solution
T0 = 0.5  # GeV initial temperature
tau0 = 0.6  # fm/c initial time

def bjorken_temperature(tau):
    """Exact Bjorken flow temperature evolution"""
    return T0 * (tau0 / tau)**(1/3)

def bjorken_energy_density(tau):
    """Energy density for radiation: ρ = aT⁴"""
    a = np.pi**2 * 37.5 / 30
    return a * bjorken_temperature(tau)**4

# Create benchmark
benchmark = BjorkenFlowBenchmark(T0, tau0, eta_over_s=0.0)  # Ideal case

# Test at multiple times
test_times = np.array([1.0, 2.0, 5.0, 10.0])
for tau in test_times:
    numerical_T = benchmark.get_temperature(tau)
    analytical_T = bjorken_temperature(tau)
    error = abs(numerical_T - analytical_T) / analytical_T

    print(f"τ = {tau:4.1f} fm/c:")
    print(f"  Analytical T = {analytical_T:.4f} GeV")
    print(f"  Numerical T  = {numerical_T:.4f} GeV")
    print(f"  Relative error = {error:.2e}")
    assert error < 1e-10, f"Error too large at τ={tau}"
```

---

## **Notebook 5: First-Order Viscous Hydrodynamics**
**File:** `05_navier_stokes_causality.ipynb`

### Section 5.1: Transport Phenomena
**Theory**: Irreversible processes, linear response, Onsager relations
**Fundamental insight**: Gradients drive fluxes (Fourier, Fick, Newton laws)

```python
# Classical transport laws
def heat_flux_fourier(temperature_gradient, thermal_conductivity):
    """Fourier's law: q = -κ ∇T"""
    return -thermal_conductivity * temperature_gradient

def viscous_stress_newton(velocity_gradient, viscosity):
    """Newton's law: τ = η ∂v/∂y"""
    return viscosity * velocity_gradient

def diffusion_flux_fick(concentration_gradient, diffusivity):
    """Fick's law: j = -D ∇c"""
    return -diffusivity * concentration_gradient

# Example: Heat conduction
kappa = 0.1  # Thermal conductivity
dT_dx = -0.05  # Temperature gradient K/m
q_x = heat_flux_fourier(dT_dx, kappa)
print(f"Heat flux: {q_x:.4f} W/m² (negative = heat flows up gradient)")
```

### Section 5.2: Relativistic Navier-Stokes
**Theory**: First-order gradient expansion, frame dependence, constitutive relations
**Key equations**:
- π^μν = -η σ^μν (shear viscosity)
- Π = -ζ θ (bulk viscosity)
- q^μ = -κ ∇^μ(μ/T) (heat conduction)

```python
from israel_stewart.core import ProjectionOperator

# Compute kinematic quantities
def compute_shear_tensor(four_velocity, metric):
    """Shear tensor: σ^μν = ∇^⟨μu^ν⟩"""
    # Simplified implementation
    proj = ProjectionOperator(four_velocity, metric)
    grad_u = metric.covariant_derivative(four_velocity)
    return proj.traceless_symmetric_projection(grad_u)

def compute_expansion_rate(four_velocity, metric):
    """Expansion rate: θ = ∇_μ u^μ"""
    div_u = metric.covariant_derivative(four_velocity)
    return np.trace(div_u, axis1=-2, axis2=-1)

# Test with simple velocity field
u = FourVector(grid, metric, data=[1, 0.1*x, 0, 0])  # Linear velocity profile
sigma = compute_shear_tensor(u, metric)
theta = compute_expansion_rate(u, metric)

print(f"Shear tensor norm: {np.sqrt(np.sum(sigma**2)):.6f}")
print(f"Expansion rate: {np.mean(theta):.6f}")
```

### Section 5.3: Causality Crisis
**Theory**: Infinite propagation speed, instabilities, need for regularization
```python
# Demonstrate causality violation in first-order theory
# Consider heat equation: ∂T/∂t = D ∇²T (where D = κ/ρc_p)

import scipy.linalg as la

def solve_diffusion_equation(D, dx, dt, T_initial):
    """Solve 1D diffusion equation with finite differences"""
    nx = len(T_initial)

    # Build diffusion matrix
    A = np.zeros((nx, nx))
    r = D * dt / dx**2  # Diffusion number

    for i in range(1, nx-1):
        A[i, i-1] = r
        A[i, i] = 1 - 2*r
        A[i, i+1] = r

    # Boundary conditions (fixed ends)
    A[0, 0] = A[-1, -1] = 1

    return A @ T_initial

# Initial step function
x = np.linspace(0, 10, 100)
T_initial = np.where(x < 5, 1.0, 0.0)  # Step at x=5

# Evolve one timestep
D = 1.0  # Diffusion coefficient
dx = x[1] - x[0]
dt = 0.01
T_after = solve_diffusion_equation(D, dx, dt, T_initial)

# Check for instantaneous response everywhere
response = np.abs(T_after - T_initial)
print(f"Maximum change: {np.max(response):.6f}")
print(f"Change at x=0 (far from step): {response[0]:.2e}")
print("⚠️  Non-zero response everywhere = acausal behavior!")
```

---

## **Notebook 6: Israel-Stewart Second-Order Theory**
**File:** `06_israel_stewart_causal_viscosity.ipynb`

### Section 6.1: Extended Irreversible Thermodynamics
**Theory**: Memory effects, relaxation times, causal structure
**Key insight**: Dissipative fluxes are dynamic variables, not instantaneous responses

```python
# Relaxation-type evolution equations
def relaxation_equation_demo():
    """Demonstrate relaxation vs instantaneous response"""

    # Source function (driving force)
    def source(t):
        return np.exp(-t**2)  # Gaussian pulse

    # Compare instantaneous vs relaxation response
    t = np.linspace(0, 5, 100)

    # Instantaneous response (first-order theory)
    response_instantaneous = source(t)

    # Relaxation response: τ dy/dt + y = source
    tau = 0.5  # Relaxation time
    dt = t[1] - t[0]
    y_relaxation = np.zeros_like(t)

    for i in range(1, len(t)):
        dydt = (source(t[i]) - y_relaxation[i-1]) / tau
        y_relaxation[i] = y_relaxation[i-1] + dydt * dt

    # Plot comparison
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(t, source(t), 'k--', label='Source')
    plt.plot(t, response_instantaneous, 'r-', label='Instantaneous')
    plt.plot(t, y_relaxation, 'b-', label=f'Relaxation (τ={tau})')
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.legend()
    plt.title('Instantaneous vs Relaxation Response')
    plt.show()

    return t, source(t), response_instantaneous, y_relaxation

relaxation_equation_demo()
```

### Section 6.2: Complete Israel-Stewart Equations
**Theory**: Full second-order system, coupling terms, thermodynamic constraints
**Equations**:
```
Energy-momentum: ∇_μ T^μν = 0
Shear evolution: τ_π Dπ^μν + π^μν = 2η σ^μν + ...
Bulk evolution: τ_Π DΠ + Π = -ζ θ + ...
Heat evolution: τ_q Dq^μ + q^μ = κ ∇^μ(μ/T) + ...
```

```python
from israel_stewart.equations import ISRelaxationEquations
from israel_stewart.core import TransportCoefficients

# Define realistic transport coefficients for QGP
coeffs = TransportCoefficients(
    shear_viscosity=0.08,      # η/s = 0.08 (near KSS bound)
    bulk_viscosity=0.02,       # ζ/s = 0.02 (small for QGP)
    shear_relaxation_time=0.5, # τ_π ~ 5η/(sT)
    bulk_relaxation_time=0.3,  # τ_Π ~ ζ/(15(1/3-c_s²)sT)

    # Second-order coefficients
    lambda_pi_pi=0.1,          # π^μν-π^αβ coupling
    xi_1=0.2,                  # Bulk nonlinearity
    xi_2=0.1                   # Mixed bulk-shear coupling
)

# Initialize relaxation equations
relaxation = ISRelaxationEquations(grid, metric, coeffs)

print("Transport coefficient summary:")
print(f"η/s = {coeffs.shear_viscosity:.3f}")
print(f"ζ/s = {coeffs.bulk_viscosity:.3f}")
print(f"τ_π = {coeffs.shear_relaxation_time:.2f} fm/c")
print(f"τ_Π = {coeffs.bulk_relaxation_time:.2f} fm/c")
print(f"Second-order: λ_ππ={coeffs.lambda_pi_pi}, ξ₁={coeffs.xi_1}")
```

### Section 6.3: Relaxation Time Validation
**Theory**: Simple exponential decay, analytical solution for linearized case
```python
# Test simple exponential relaxation: dΠ/dt = -Π/τ
# Analytical solution: Π(t) = Π₀ exp(-t/τ)

Pi_0 = 0.1  # Initial bulk pressure
tau_Pi = 0.5  # Relaxation time

def analytical_bulk_relaxation(t, Pi_initial, tau):
    return Pi_initial * np.exp(-t / tau)

# Numerical evolution
fields = ISFieldConfiguration(grid)
fields.Pi.fill(Pi_0)  # Initial perturbation
fields.rho.fill(1.0)   # Background state
fields.pressure.fill(1.0/3.0)
fields.four_velocity.data[..., 0] = 1.0

# Evolve and compare
times = np.array([0.5, 1.0, 2.0, 5.0])
print("Bulk pressure relaxation validation:")
print("Time   Analytical  Numerical   Error")
print("-" * 40)

for t in times:
    # Reset and evolve
    fields_copy = fields.copy()
    relaxation.evolve_relaxation(fields_copy, t, method='implicit')

    Pi_numerical = np.mean(fields_copy.Pi.data)
    Pi_analytical = analytical_bulk_relaxation(t, Pi_0, tau_Pi)
    error = abs(Pi_numerical - Pi_analytical) / Pi_analytical

    print(f"{t:4.1f}   {Pi_analytical:.6f}   {Pi_numerical:.6f}   {error:.2e}")
    assert error < 1e-8, f"Relaxation error too large at t={t}"
```

---

## **Notebook 7: Numerical Methods and Spatial Discretization**
**File:** `07_numerical_methods_implementation.ipynb`

### Section 7.1: Finite Difference Theory
**Theory**: Taylor expansions, truncation error, numerical dispersion
**Central differences**: f'(x) ≈ [f(x+h) - f(x-h)]/(2h) + O(h²)

```python
# Demonstrate finite difference accuracy
def test_finite_difference_accuracy():
    """Test FD schemes on analytical functions"""

    # Test function and its exact derivative
    f = lambda x: np.sin(2*np.pi*x)
    f_prime_exact = lambda x: 2*np.pi*np.cos(2*np.pi*x)

    # Grid spacing study
    N_values = [10, 20, 40, 80, 160]
    errors_fd2 = []  # Second-order
    errors_fd4 = []  # Fourth-order

    for N in N_values:
        x = np.linspace(0, 1, N, endpoint=False)
        h = x[1] - x[0]

        # Periodic function values
        fx = f(x)

        # Second-order central difference
        fd2 = np.zeros_like(fx)
        for i in range(N):
            fd2[i] = (fx[(i+1)%N] - fx[(i-1)%N]) / (2*h)

        # Fourth-order central difference
        fd4 = np.zeros_like(fx)
        for i in range(N):
            fd4[i] = (-fx[(i+2)%N] + 8*fx[(i+1)%N] - 8*fx[(i-1)%N] + fx[(i-2)%N]) / (12*h)

        # Compute errors
        exact = f_prime_exact(x)
        errors_fd2.append(np.max(np.abs(fd2 - exact)))
        errors_fd4.append(np.max(np.abs(fd4 - exact)))

    # Check convergence rates
    h_values = 1.0 / np.array(N_values)

    print("Grid refinement study:")
    print("N      h        Error (2nd order)  Error (4th order)")
    print("-" * 55)
    for i, N in enumerate(N_values):
        print(f"{N:3d}  {h_values[i]:.4f}   {errors_fd2[i]:.6e}     {errors_fd4[i]:.6e}")

    # Verify convergence rates
    rate_fd2 = np.log(errors_fd2[-1]/errors_fd2[0]) / np.log(h_values[-1]/h_values[0])
    rate_fd4 = np.log(errors_fd4[-1]/errors_fd4[0]) / np.log(h_values[-1]/h_values[0])

    print(f"\nConvergence rates:")
    print(f"2nd order scheme: {rate_fd2:.2f} (expected: 2.0)")
    print(f"4th order scheme: {rate_fd4:.2f} (expected: 4.0)")

    return h_values, errors_fd2, errors_fd4

test_finite_difference_accuracy()
```

### Section 7.2: Conservative vs Non-Conservative Forms
**Theory**: Conservation laws, flux formulations, divergence theorem
```python
from israel_stewart.solvers import finite_difference

# Compare conservative and non-conservative schemes
grid = SpacetimeGrid(
    coordinate_system="cartesian",
    time_range=(0, 1),
    spatial_ranges=[(-5, 5)],
    grid_points=(10, 100)
)

conservative_fd = finite_difference.create_finite_difference_solver(
    'conservative', grid, metric, order=4
)

upwind_fd = finite_difference.create_finite_difference_solver(
    'upwind', grid, metric, order=2
)

# Test on conservation law: ∂ρ/∂t + ∇·(ρv) = 0
rho = np.exp(-(x**2)/2)  # Gaussian density
v = 0.5 * x              # Linear velocity field

# Conservative form: ∂ρ/∂t + ∂(ρv)/∂x = 0
drho_dt_conservative = -conservative_fd.compute_divergence(rho * v)

# Non-conservative form: ∂ρ/∂t + v·∇ρ + ρ∇·v = 0
grad_rho = conservative_fd.compute_gradient(rho)
div_v = conservative_fd.compute_divergence(v)
drho_dt_nonconservative = -(v * grad_rho + rho * div_v)

print("Conservation form comparison:")
print(f"Conservative max: {np.max(np.abs(drho_dt_conservative)):.6e}")
print(f"Non-conservative max: {np.max(np.abs(drho_dt_nonconservative)):.6e}")
print(f"Difference: {np.max(np.abs(drho_dt_conservative - drho_dt_nonconservative)):.6e}")
```

### Section 7.3: WENO Shock Capturing
**Theory**: High-order accuracy + shock capturing, smoothness indicators
```python
weno_fd = finite_difference.create_finite_difference_solver(
    'weno', grid, metric, weno_order=5
)

# Test on discontinuous function
x = np.linspace(-2, 2, 200)
# Step function + smooth part
f_discontinuous = np.where(x < 0, 1.0, 0.0) + 0.1*np.sin(5*x)

# Compare different schemes on discontinuous data
fd_conservative = conservative_fd.compute_derivative(f_discontinuous)
fd_weno = weno_fd.compute_derivative(f_discontinuous)

# WENO should have less oscillation near discontinuity
print("Shock capturing comparison:")
print(f"Standard FD oscillations: {np.std(fd_conservative):.6f}")
print(f"WENO oscillations: {np.std(fd_weno):.6f}")
print(f"WENO reduction factor: {np.std(fd_conservative)/np.std(fd_weno):.2f}")
```

---

## **Notebook 8: Sound Wave Analysis and Dispersion Relations**
**File:** `08_sound_waves_analytical_validation.ipynb`

### Section 8.1: Linear Wave Theory in Fluids
**Theory**: Small perturbations, normal modes, characteristic speeds
**Dispersion relation**: Connects frequency ω to wavenumber k

```python
# Ideal fluid sound waves: analytical dispersion ω = c_s k
c_s = 1/np.sqrt(3)  # Sound speed in radiation
k_range = np.linspace(0.1, 5.0, 50)
omega_ideal = c_s * k_range

# Phase velocity and group velocity
v_phase = omega_ideal / k_range  # Should be constant = c_s
v_group = np.gradient(omega_ideal, k_range)  # dω/dk

print("Ideal fluid wave properties:")
print(f"Sound speed c_s = {c_s:.6f}")
print(f"Phase velocity (const): {np.mean(v_phase):.6f} ± {np.std(v_phase):.2e}")
print(f"Group velocity (const): {np.mean(v_group):.6f} ± {np.std(v_group):.2e}")

# Verify they're equal for non-dispersive waves
assert np.allclose(v_phase, c_s, rtol=1e-12)
assert np.allclose(v_group, c_s, rtol=1e-12)
```

### Section 8.2: Viscous Corrections to Dispersion
**Theory**: First-order viscous damping ω = c_s k - i Γ k²
**Key insight**: Viscosity causes exponential damping

```python
from israel_stewart.benchmarks import SoundWaveAnalysis

# First-order viscous theory
eta = 0.1  # Shear viscosity
zeta = 0.02  # Bulk viscosity
rho_plus_p = 4.0/3.0  # For radiation

# Damping coefficient
Gamma = (4*eta/3 + zeta) / rho_plus_p

# Analytical first-order dispersion
omega_viscous_real = c_s * k_range
omega_viscous_imag = -Gamma * k_range**2  # Negative = damping

print("First-order viscous theory:")
print(f"Damping coefficient Γ = {Gamma:.6f}")
print(f"At k=1: ω = {omega_viscous_real[k_range==1.0][0]:.3f} - i{-omega_viscous_imag[k_range==1.0][0]:.3f}")

# Numerical validation
coeffs_viscous = TransportCoefficients(
    shear_viscosity=eta,
    bulk_viscosity=zeta,
    shear_relaxation_time=0.0,  # First-order limit
    bulk_relaxation_time=0.0
)

analysis = SoundWaveAnalysis(grid, metric, coeffs_viscous)
dispersion_numerical = analysis.analyze_dispersion_relation(k_range)

# Compare analytical vs numerical
omega_num = dispersion_numerical['omega']
damping_num = dispersion_numerical['attenuation']

print("\nAnalytical vs Numerical comparison:")
for i in [10, 20, 30, 40]:  # Sample points
    k = k_range[i]
    print(f"k={k:.2f}: ω_ana={omega_viscous_real[i]:.4f}, ω_num={omega_num[i]:.4f}")
    print(f"        γ_ana={-omega_viscous_imag[i]:.4f}, γ_num={damping_num[i]:.4f}")
```

### Section 8.3: Israel-Stewart Modifications
**Theory**: Second-order corrections, relaxation effects, causality restoration
```python
# Full Israel-Stewart dispersion (simplified)
tau_pi = 0.5  # Relaxation time

# Modified dispersion relation (approximate)
def israel_stewart_dispersion(k, c_s, Gamma, tau):
    """Approximate IS dispersion relation"""
    # This is simplified - full relation is more complex
    omega_squared = c_s**2 * k**2 * (1 + (Gamma * tau * k**2)**2) / (1 + (tau * c_s * k)**2)
    return np.sqrt(omega_squared)

omega_IS = israel_stewart_dispersion(k_range, c_s, Gamma, tau_pi)

# Compare different theories
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, omega_ideal, 'k-', label='Ideal', linewidth=2)
plt.plot(k_range, omega_viscous_real, 'r--', label='Viscous (real)', linewidth=2)
plt.plot(k_range, omega_IS, 'b:', label='Israel-Stewart', linewidth=2)
plt.xlabel('Wavenumber k')
plt.ylabel('Frequency ω')
plt.legend()
plt.title('Dispersion Relations')

plt.subplot(1, 2, 2)
plt.plot(k_range, -omega_viscous_imag, 'r--', label='Viscous damping', linewidth=2)
plt.xlabel('Wavenumber k')
plt.ylabel('Damping rate')
plt.legend()
plt.title('Wave Damping')

plt.tight_layout()
plt.show()

# Causality check: phase velocity should not exceed c
v_phase_IS = omega_IS / k_range
max_velocity = np.max(v_phase_IS)
print(f"\nCausality check:")
print(f"Maximum phase velocity: {max_velocity:.6f} c")
print(f"Causal bound (c = 1): {'✓' if max_velocity <= 1.0 else '✗'}")
```

This plan creates a seamless integration of theory and implementation, where each concept is immediately demonstrated with working code examples and validated against analytical results. Every calculation has a clear physical interpretation and mathematical foundation.
