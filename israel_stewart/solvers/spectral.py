"""
Fourier spectral method solver for Israel-Stewart hydrodynamics equations.

This module implements efficient spectral methods for solving relativistic
hydrodynamics with second-order viscous corrections using FFT-based operations.
"""

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
from scipy.optimize import newton_krylov

from ..core.performance import monitor_performance
from ..core.tensor_utils import optimized_einsum

if TYPE_CHECKING:
    from ..core.fields import ISFieldConfiguration, TransportCoefficients
    from ..core.spacetime_grid import SpacetimeGrid
    from ..equations.conservation import ConservationLaws
    from ..equations.relaxation import ISRelaxationEquations


class SpectralISolver:
    """
    Fourier spectral method for Israel-Stewart equations.

    Provides high-performance spectral differentiation and linear operator
    application for relativistic hydrodynamics with periodic boundaries.
    """

    def __init__(
        self,
        grid: "SpacetimeGrid",
        fields: "ISFieldConfiguration",
        coeffs: Optional["TransportCoefficients"] = None,
    ):
        """
        Initialize spectral solver.

        Args:
            grid: SpacetimeGrid defining computational domain
            fields: ISFieldConfiguration with hydrodynamic variables
            coeffs: Transport coefficients for viscous terms
        """
        self.grid = grid
        self.fields = fields
        self.coeffs = coeffs

        # Extract grid dimensions and spacing
        self.nt, self.nx, self.ny, self.nz = grid.grid_points
        self.dt = grid.dt

        # IMPORTANT: For spectral methods, we need proper periodic spacing: dx = L/N
        # SpacetimeGrid uses dx = L/(N-1) which breaks periodicity
        # Override with correct spectral spacing
        if hasattr(grid, 'spatial_ranges'):
            # Calculate proper spectral spacing: L/N instead of L/(N-1)
            spatial_extents = [r[1] - r[0] for r in grid.spatial_ranges]
            self.dx = spatial_extents[0] / self.nx
            self.dy = spatial_extents[1] / self.ny
            self.dz = spatial_extents[2] / self.nz
        else:
            # Fallback to grid spacing (may be incorrect)
            self.dx, self.dy, self.dz = grid.spatial_spacing

        # Precompute FFT plans for efficiency
        self.fft_plan = np.fft.fftn
        self.ifft_plan = np.fft.ifftn
        self.rfft_plan = np.fft.rfftn  # Real FFT for memory efficiency
        self.irfft_plan = np.fft.irfftn

        # Wave vectors for derivatives
        self.k_vectors = self._compute_wave_vectors()
        self.k_squared = self._compute_k_squared()

        # Cache for frequently used arrays
        self._fft_cache: dict[Any, Any] = {}
        self._derivative_cache: dict[Any, Any] = {}

    @monitor_performance("wave_vectors")
    def _compute_wave_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Setup wave vectors for spectral derivatives.

        Returns:
            Tuple of (kx, ky, kz) wave vector grids
        """
        # Compute frequency arrays
        kx = np.fft.fftfreq(self.nx, self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(self.ny, self.dy) * 2 * np.pi
        kz = np.fft.fftfreq(self.nz, self.dz) * 2 * np.pi

        # Create 3D meshgrids for vectorized operations
        kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing="ij")

        return kx_grid, ky_grid, kz_grid

    def _compute_k_squared(self) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """Compute k^2 = kx^2 + ky^2 + kz^2 for diffusion operators."""
        kx, ky, kz = self.k_vectors
        return cast(np.ndarray[Any, np.dtype[np.floating[Any]]], kx**2 + ky**2 + kz**2)

    @monitor_performance("spectral_derivative")
    def spatial_derivative(
        self, field: np.ndarray, direction: int, use_cache: bool = True
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """
        Compute spatial derivative using spectral method.

        ∂_i f = IFFT(ik_i * FFT(f))

        Args:
            field: Field to differentiate with shape (*spatial_shape,) or (*grid.shape,)
            direction: Spatial direction (0=x, 1=y, 2=z)
            use_cache: Whether to cache FFT results for repeated operations

        Returns:
            Spatial derivative ∂_i field
        """
        if direction not in [0, 1, 2]:
            raise ValueError(f"Direction must be 0, 1, or 2 (x, y, z), got {direction}")

        # Handle both 3D spatial fields and 4D spacetime fields
        if field.ndim == 3:
            spatial_field = field
        elif field.ndim == 4 and field.shape[0] == self.nt:
            # Take latest time slice for spatial derivative
            spatial_field = field[-1, :, :, :]
        else:
            raise ValueError(f"Field shape {field.shape} not compatible with grid")

        # Check cache first
        cache_key = (id(field), direction) if use_cache else None
        if cache_key and cache_key in self._derivative_cache:
            return cast(
                np.ndarray[Any, np.dtype[np.floating[Any]]], self._derivative_cache[cache_key]
            )

        # Forward FFT
        field_k = self.fft_plan(spatial_field)

        # Apply derivative operator ik_i
        k_direction = self.k_vectors[direction]
        deriv_k = 1j * k_direction * field_k

        # Inverse FFT to get real derivative
        result = self.ifft_plan(deriv_k).real

        # Cache result if requested
        if cache_key:
            self._derivative_cache[cache_key] = result

        return cast(np.ndarray[Any, np.dtype[np.floating[Any]]], result)

    @monitor_performance("gradient_computation")
    def spatial_gradient(self, field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute full spatial gradient ∇f = (∂_x f, ∂_y f, ∂_z f).

        Args:
            field: Scalar field to differentiate

        Returns:
            Tuple of (∂_x f, ∂_y f, ∂_z f)
        """
        return (
            self.spatial_derivative(field, 0),
            self.spatial_derivative(field, 1),
            self.spatial_derivative(field, 2),
        )

    @monitor_performance("divergence_computation")
    def spatial_divergence(self, vector_field: np.ndarray) -> np.ndarray:
        """
        Compute spatial divergence ∇·v = ∂_x v_x + ∂_y v_y + ∂_z v_z.

        Args:
            vector_field: Vector field with shape (*spatial_shape, 3)

        Returns:
            Divergence ∇·v
        """
        if vector_field.shape[-1] != 3:
            raise ValueError("Vector field must have 3 components")

        div_result = np.zeros(vector_field.shape[:-1])
        for i in range(3):
            div_result += self.spatial_derivative(vector_field[..., i], i)

        return div_result

    @monitor_performance("viscous_operator")
    def apply_viscous_operator(
        self, shear_mode: np.ndarray, viscosity: float, dt: float
    ) -> np.ndarray:
        """
        Apply the viscous operator in Fourier space.

        More efficient than real space for linear diffusion terms.
        Implements: exp(-ν k² Δt) for viscous damping

        Args:
            shear_mode: Field to apply viscous damping to
            viscosity: Kinematic viscosity coefficient
            dt: Time step

        Returns:
            Viscously damped field
        """
        # Forward FFT
        shear_k = self.fft_plan(shear_mode)

        # Viscous damping: exp(-nu k^2 dt)
        damping = np.exp(-viscosity * self.k_squared * dt)

        # Apply damping and inverse FFT
        return self.ifft_plan(damping * shear_k).real

    @monitor_performance("bulk_viscous_operator")
    def apply_bulk_viscous_operator(
        self, bulk_field: np.ndarray, bulk_viscosity: float, relaxation_time: float, dt: float
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """
        Apply bulk viscosity operator for Israel-Stewart evolution.

        Args:
            bulk_field: Bulk pressure field Π
            bulk_viscosity: Bulk viscosity ζ
            relaxation_time: Bulk relaxation time τ_Π
            dt: Time step

        Returns:
            Updated bulk pressure field
        """
        # Exponential relaxation: exp(-dt/tau_Pi)
        relaxation_factor = np.exp(-dt / relaxation_time)

        # Apply bulk viscous damping
        damped_field = self.apply_viscous_operator(bulk_field, bulk_viscosity / relaxation_time, dt)

        return cast(np.ndarray[Any, np.dtype[np.floating[Any]]], relaxation_factor * damped_field)

    @monitor_performance("laplacian")
    def laplacian(self, field: np.ndarray) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """
        Compute Laplacian ∇²f = ∂²_x f + ∂²_y f + ∂²_z f using spectral method.

        Args:
            field: Scalar field

        Returns:
            Laplacian ∇²f
        """
        # Forward FFT
        field_k = self.fft_plan(field)

        # Apply -k^2 operator
        laplacian_k = -self.k_squared * field_k

        # Inverse FFT
        return cast(np.ndarray[Any, np.dtype[np.floating[Any]]], self.ifft_plan(laplacian_k).real)

    def spectral_convolution(
        self, field1: np.ndarray, field2: np.ndarray, dealiasing: bool = True
    ) -> np.ndarray:
        """
        Compute convolution of two fields using FFT.

        Useful for nonlinear terms in IS equations.

        Args:
            field1: First field
            field2: Second field
            dealiasing: Apply 2/3 rule for dealiasing

        Returns:
            Convolution field1 * field2
        """
        # Forward FFTs
        field1_k = self.fft_plan(field1)
        field2_k = self.fft_plan(field2)

        # Pointwise multiplication in Fourier space
        conv_k = field1_k * field2_k

        # Apply dealiasing if requested
        if dealiasing:
            conv_k = self._apply_dealiasing(conv_k)

        # Inverse FFT
        return self.ifft_plan(conv_k).real

    def _apply_dealiasing(self, field_k: np.ndarray) -> np.ndarray:
        """Apply 2/3 rule dealiasing to prevent aliasing errors."""
        nx, ny, nz = field_k.shape

        # Zero out high-frequency modes
        kx_max = int(nx * 2 // 3)
        ky_max = int(ny * 2 // 3)
        kz_max = int(nz * 2 // 3)

        result = field_k.copy()
        result[kx_max:, :, :] = 0
        result[:, ky_max:, :] = 0
        result[:, :, kz_max:] = 0

        return result

    @monitor_performance("spectral_time_step")
    def advance_linear_terms(
        self, fields: "ISFieldConfiguration", dt: float, method: str = "exponential"
    ) -> None:
        """
        Advance linear terms in IS equations using spectral methods.

        Args:
            fields: Field configuration to evolve
            dt: Time step
            method: Integration method ('exponential', 'implicit')
        """
        if method == "exponential":
            self._exponential_advance(fields, dt)
        elif method == "implicit":
            self._implicit_spectral_advance(fields, dt)
        else:
            raise ValueError(f"Unknown spectral method: {method}")

    def _exponential_advance(self, fields: "ISFieldConfiguration", dt: float) -> None:
        """Advance using exponential integrators for stiff terms."""
        if self.coeffs is None:
            return

        # Bulk pressure evolution with relaxation
        if hasattr(self.coeffs, "bulk_relaxation_time") and self.coeffs.bulk_relaxation_time:
            tau_Pi = self.coeffs.bulk_relaxation_time
            fields.Pi = self.apply_bulk_viscous_operator(
                fields.Pi, self.coeffs.bulk_viscosity or 0.0, tau_Pi, dt
            )

        # Shear stress relaxation
        if hasattr(self.coeffs, "shear_relaxation_time") and self.coeffs.shear_relaxation_time:
            tau_pi = self.coeffs.shear_relaxation_time
            eta = self.coeffs.shear_viscosity or 0.0

            # Apply to each component of shear tensor
            for mu in range(4):
                for nu in range(4):
                    fields.pi_munu[..., mu, nu] = self.apply_viscous_operator(
                        fields.pi_munu[..., mu, nu], eta / tau_pi, dt
                    )

    def _implicit_spectral_advance(self, fields: "ISFieldConfiguration", dt: float) -> None:
        """Implicit spectral advance for very stiff problems."""
        # This would implement implicit spectral methods
        # For now, fall back to exponential method
        self._exponential_advance(fields, dt)

    def clear_cache(self) -> None:
        """Clear FFT and derivative caches to free memory."""
        self._fft_cache.clear()
        self._derivative_cache.clear()

    def __str__(self) -> str:
        return f"SpectralISolver(grid={self.nx}x{self.ny}x{self.nz})"

    def __repr__(self) -> str:
        return (
            f"SpectralISolver(grid_points={self.grid.grid_points}, "
            f"spacing=({self.dx:.3f}, {self.dy:.3f}, {self.dz:.3f}))"
        )


class SpectralISHydrodynamics:
    """
    Complete spectral hydrodynamics solver for Israel-Stewart equations.

    Integrates spectral methods with conservation laws and relaxation equations
    for efficient relativistic hydrodynamics simulations.
    """

    def __init__(
        self,
        grid: "SpacetimeGrid",
        fields: "ISFieldConfiguration",
        coeffs: Optional["TransportCoefficients"] = None,
    ):
        """
        Initialize integrated spectral hydrodynamics solver.

        Args:
            grid: SpacetimeGrid defining computational domain
            fields: ISFieldConfiguration with all hydrodynamic variables
            coeffs: Transport coefficients for viscous terms
        """
        self.grid = grid
        self.fields = fields
        self.coeffs = coeffs

        # Initialize spectral solver
        self.spectral = SpectralISolver(grid, fields, coeffs)

        # Initialize physics modules (may be None)
        self.conservation: ConservationLaws | None = None
        self.relaxation: ISRelaxationEquations | None = None
        self._init_physics_modules()

        # Time stepping parameters
        self.cfl_factor = 0.5
        self.max_dt = 0.01

    def _init_physics_modules(self) -> None:
        """Initialize conservation and relaxation equation modules."""
        try:
            from ..equations.conservation import ConservationLaws
            from ..equations.relaxation import ISRelaxationEquations

            self.conservation = ConservationLaws(self.fields, self.coeffs)

            if self.coeffs is not None:
                # Need metric for relaxation equations
                metric = self.grid.metric if hasattr(self.grid, "metric") else None
                if metric is None:
                    from ..core.metrics import MinkowskiMetric

                    metric = MinkowskiMetric()

                self.relaxation = ISRelaxationEquations(self.grid, metric, self.coeffs)
            else:
                self.relaxation = None

        except ImportError as e:
            warnings.warn(f"Could not initialize physics modules: {e}", stacklevel=2)
            self.conservation = None
            self.relaxation = None

    @monitor_performance("spectral_timestep")
    def time_step(self, dt: float, method: str = "split_step") -> None:
        """
        Advance hydrodynamics by one time step using spectral methods.

        Args:
            dt: Time step size
            method: Integration method ('split_step', 'spectral_imex')
        """
        if method == "split_step":
            self._split_step_advance(dt)
        elif method == "spectral_imex":
            self._spectral_imex_advance(dt)
        else:
            raise ValueError(f"Unknown time stepping method: {method}")

    def _split_step_advance(self, dt: float) -> None:
        """
        Split-step method: spectral linear terms + real-space nonlinear terms.
        """
        # Step 1: Advance linear diffusive terms spectrally
        self.spectral.advance_linear_terms(self.fields, dt / 2)

        # Step 2: Advance nonlinear conservation laws in real space
        if self.conservation is not None:
            self._advance_conservation_laws(dt)

        # Step 3: Advance Israel-Stewart relaxation terms
        if self.relaxation is not None:
            self._advance_relaxation_terms(dt)

        # Step 4: Final linear diffusive step
        self.spectral.advance_linear_terms(self.fields, dt / 2)

    def _spectral_imex_advance(self, dt: float) -> None:
        """
        IMEX method: implicit spectral linear terms + explicit nonlinear terms.
        """
        # Store initial state
        fields_old = self._copy_fields()

        # Explicit treatment of nonlinear terms
        if self.conservation is not None:
            conservation_rhs = self.conservation.evolution_equations()

        # Implicit spectral treatment of linear terms
        self.spectral.advance_linear_terms(self.fields, dt, method="implicit")

        # Add explicit nonlinear contributions
        if self.conservation is not None:
            self.fields.rho += dt * conservation_rhs["drho_dt"]
            # Update momentum (simplified for now)

    def _advance_conservation_laws(self, dt: float) -> None:
        """Advance conservation laws using spectral derivatives."""
        if self.conservation is None:
            return

        # Compute stress-energy tensor divergence using spectral methods
        T_munu = self.conservation.stress_energy_tensor()

        # Spectral computation of divergence T^mu_nu for each nu
        for nu in range(4):
            div_component = np.zeros_like(T_munu[..., 0, 0])

            # Sum over mu: partial_mu T^mu^nu (only spatial derivatives with spectral)
            for mu in range(1, 4):  # Skip time derivative for now
                spatial_dir = mu - 1  # Convert to 0,1,2 indexing
                div_component += self.spectral.spatial_derivative(T_munu[..., mu, nu], spatial_dir)

            # Update fields based on conservation equations
            if nu == 0:  # Energy conservation
                self.fields.rho -= dt * div_component
            else:  # Momentum conservation (simplified)
                pass  # Would update momentum density here

    def _advance_relaxation_terms(self, dt: float) -> None:
        """Advance Israel-Stewart relaxation equations."""
        if self.relaxation is None:
            return

        try:
            # Use existing relaxation equation solver
            self.relaxation.evolve_relaxation(self.fields, dt)
        except Exception as e:
            warnings.warn(f"Relaxation evolution failed: {e}", stacklevel=2)

    def _copy_fields(self) -> dict[str, np.ndarray]:
        """Create a copy of current field state."""
        return {
            "rho": self.fields.rho.copy(),
            "Pi": self.fields.Pi.copy(),
            "pi_munu": self.fields.pi_munu.copy(),
            "q_mu": self.fields.q_mu.copy(),
            "u_mu": self.fields.u_mu.copy(),
        }

    def adaptive_time_step(self) -> float:
        """
        Compute adaptive time step based on CFL condition and physics constraints.

        Returns:
            Optimal time step size
        """
        # CFL condition for spectral methods
        max_velocity = np.max(np.abs(self.fields.u_mu[..., 1:4]))
        if max_velocity == 0:
            max_velocity = 1e-10  # Avoid division by zero

        cfl_dt = (
            self.cfl_factor
            * min(self.spectral.dx, self.spectral.dy, self.spectral.dz)
            / max_velocity
        )

        # Viscous time step constraint
        if self.coeffs is not None:
            eta = getattr(self.coeffs, "shear_viscosity", 0.0) or 0.0
            if eta > 0:
                min_spacing_sq = min(self.spectral.dx**2, self.spectral.dy**2, self.spectral.dz**2)
                viscous_dt = 0.5 * min_spacing_sq / eta
                cfl_dt = min(cfl_dt, viscous_dt)

        # Relaxation time constraint
        if self.coeffs is not None:
            tau_pi = getattr(self.coeffs, "shear_relaxation_time", None)
            if tau_pi is not None and tau_pi > 0:
                relaxation_dt = 0.1 * tau_pi
                cfl_dt = min(cfl_dt, relaxation_dt)

        return float(min(cfl_dt, self.max_dt))

    @monitor_performance("spectral_simulation")
    def evolve(self, t_final: float, output_callback: Callable | None = None) -> None:
        """
        Evolve hydrodynamics from t=0 to t_final using adaptive time stepping.

        Args:
            t_final: Final simulation time
            output_callback: Optional callback for data output
        """
        t = 0.0
        step = 0

        while t < t_final:
            # Adaptive time step
            dt = self.adaptive_time_step()
            dt = min(dt, t_final - t)  # Don't overshoot

            # Advance one time step
            self.time_step(dt)

            t += dt
            step += 1

            # Output callback
            if output_callback is not None:
                output_callback(t, step, self.fields)

            # Progress reporting
            if step % 100 == 0:
                print(f"Step {step}: t = {t:.4f}, dt = {dt:.6f}")

    def __str__(self) -> str:
        return f"SpectralISHydrodynamics(grid={self.spectral.nx}x{self.spectral.ny}x{self.spectral.nz})"
