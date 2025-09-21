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
        if hasattr(grid, "spatial_ranges"):
            # Calculate proper spectral spacing: L/N instead of L/(N-1)
            spatial_extents = [r[1] - r[0] for r in grid.spatial_ranges]
            self.dx = spatial_extents[0] / self.nx
            self.dy = spatial_extents[1] / self.ny
            self.dz = spatial_extents[2] / self.nz
        else:
            # Fallback to grid spacing (may be incorrect for periodic domains)
            warnings.warn(
                "Using potentially incorrect grid spacing from SpacetimeGrid. "
                "For spectral methods with periodic boundaries, spacing should be L/N, not L/(N-1). "
                "Consider using grid.spatial_ranges to compute proper periodic spacing.",
                UserWarning,
                stacklevel=2,
            )
            self.dx, self.dy, self.dz = grid.spatial_spacing

        # Precompute FFT plans for efficiency
        self.fft_plan = np.fft.fftn
        self.ifft_plan = np.fft.ifftn
        self.rfft_plan = np.fft.rfftn  # Real FFT for memory efficiency
        self.irfft_plan = np.fft.irfftn

        # Adaptive FFT selection for optimal performance
        self.use_real_fft = True  # Enable real FFT optimization by default

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

        # Forward FFT with adaptive selection
        field_k = self.adaptive_fft(spatial_field)

        # Apply derivative operator ik_i with appropriate k_vector
        if field_k.shape != self.k_vectors[direction].shape:
            # For real FFT, compute appropriate k_vector
            nx, ny, nz_half = field_k.shape
            nz = (nz_half - 1) * 2

            if direction == 0:  # x-direction
                k_vec = np.fft.fftfreq(nx, self.dx) * 2 * np.pi
                k_direction = k_vec[:, np.newaxis, np.newaxis]
            elif direction == 1:  # y-direction
                k_vec = np.fft.fftfreq(ny, self.dy) * 2 * np.pi
                k_direction = k_vec[np.newaxis, :, np.newaxis]
            else:  # z-direction
                k_vec = np.fft.rfftfreq(nz, self.dz) * 2 * np.pi
                k_direction = k_vec[np.newaxis, np.newaxis, :]
        else:
            k_direction = self.k_vectors[direction]

        deriv_k = 1j * k_direction * field_k

        # Inverse FFT to get real derivative with adaptive selection
        result = self.adaptive_ifft(deriv_k, spatial_field.shape)

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

        Implements proper Israel-Stewart bulk pressure evolution:
        ∂Π/∂τ + Π/τ_Π = -ζ·θ + ξ₁·Π·θ + ξ₂·Π²/(ζ·τ_Π) + λ_Ππ·π^μν·θ

        Args:
            bulk_field: Bulk pressure field Π
            bulk_viscosity: Bulk viscosity ζ
            relaxation_time: Bulk relaxation time τ_Π
            dt: Time step

        Returns:
            Updated bulk pressure field
        """
        # Use proper Israel-Stewart physics if relaxation module available
        if hasattr(self, "relaxation") and self.relaxation is not None:
            try:
                # Compute expansion scalar θ = ∇_μ u^μ
                theta = self._compute_expansion_scalar()  # type: ignore[attr-defined]

                # Get shear tensor if available
                pi_munu = getattr(self.fields, "pi_munu", np.zeros((*bulk_field.shape, 4, 4)))

                # Use the proper Israel-Stewart evolution from relaxation module
                dPi_dt = self.relaxation._bulk_rhs(bulk_field, pi_munu, theta)

                # Apply explicit Euler update
                result = bulk_field + dt * dPi_dt

                return cast(np.ndarray[Any, np.dtype[np.floating[Any]]], result)

            except Exception as e:
                warnings.warn(
                    f"Failed to apply proper Israel-Stewart bulk evolution: {e}. "
                    f"Falling back to simplified operator.",
                    UserWarning,
                    stacklevel=2,
                )

        # Fallback: simplified exponential relaxation
        # ∂Π/∂τ ≈ -Π/τ_Π (linear relaxation only)
        relaxation_factor = np.exp(-dt / relaxation_time) if relaxation_time > 0 else 0.0

        return cast(np.ndarray[Any, np.dtype[np.floating[Any]]], relaxation_factor * bulk_field)

    @monitor_performance("laplacian")
    def laplacian(self, field: np.ndarray) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """
        Compute Laplacian ∇²f = ∂²_x f + ∂²_y f + ∂²_z f using spectral method.

        Args:
            field: Scalar field

        Returns:
            Laplacian ∇²f
        """
        # Forward FFT with adaptive selection
        field_k = self.adaptive_fft(field)

        # Apply -k^2 operator with appropriate k_squared
        k_squared = self.get_k_squared_for_field(field_k)
        laplacian_k = -k_squared * field_k

        # Inverse FFT with adaptive selection
        return cast(
            np.ndarray[Any, np.dtype[np.floating[Any]]],
            self.adaptive_ifft(laplacian_k, field.shape),
        )

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
        # Forward FFTs with adaptive selection
        field1_k = self.adaptive_fft(field1)
        field2_k = self.adaptive_fft(field2)

        # Pointwise multiplication in Fourier space
        conv_k = field1_k * field2_k

        # Apply dealiasing if requested
        if dealiasing:
            conv_k = self._apply_dealiasing(conv_k)

        # Inverse FFT with adaptive selection
        return self.adaptive_ifft(conv_k, field1.shape)

    def _apply_dealiasing(self, field_k: np.ndarray) -> np.ndarray:
        """
        Apply 2/3 rule dealiasing to prevent aliasing errors.

        Properly handles the fftfreq layout where high frequencies are split between
        the beginning and end of the array. Filters out the upper 1/3 of frequencies
        in each direction to prevent aliasing from nonlinear terms.
        """
        nx, ny, nz = field_k.shape
        result = field_k.copy()

        # 2/3 rule: keep only lower 2/3 of frequencies in each direction
        # For fftfreq layout: [0, 1, 2, ..., N/2-1, -N/2, -N/2+1, ..., -1]
        # We zero out modes where |k| > (2/3) * kmax

        kx_cutoff = int(nx // 3)  # Upper 1/3 to zero out
        ky_cutoff = int(ny // 3)
        kz_cutoff = int(nz // 3)

        # Zero out high positive frequencies (upper part of positive range)
        if kx_cutoff > 0:
            kx_start = nx // 2 - kx_cutoff
            result[kx_start : nx // 2, :, :] = 0
            # Zero out high negative frequencies (lower part of negative range)
            result[nx // 2 : nx // 2 + kx_cutoff, :, :] = 0

        if ky_cutoff > 0:
            ky_start = ny // 2 - ky_cutoff
            result[:, ky_start : ny // 2, :] = 0
            result[:, ny // 2 : ny // 2 + ky_cutoff, :] = 0

        if kz_cutoff > 0:
            kz_start = nz // 2 - kz_cutoff
            result[:, :, kz_start : nz // 2] = 0
            result[:, :, nz // 2 : nz // 2 + kz_cutoff] = 0

        return result

    def adaptive_fft(self, field: np.ndarray) -> np.ndarray:
        """
        Adaptive FFT that chooses between real and complex transforms for optimal performance.

        Uses real FFT for real-valued fields (50% memory savings, 30% speed improvement)
        and complex FFT for complex-valued fields.

        Args:
            field: Input field (real or complex)

        Returns:
            FFT of field in appropriate format
        """
        if self.use_real_fft and np.isrealobj(field):
            # Use real FFT for ~50% memory reduction and ~30% speed improvement
            return self.rfft_plan(field)
        else:
            # Use complex FFT for complex fields or when real FFT disabled
            return self.fft_plan(field)

    def get_k_squared_for_field(self, field_k: np.ndarray) -> np.ndarray:
        """
        Get appropriate k_squared array for the given FFT field shape.

        Args:
            field_k: FFT coefficients

        Returns:
            k_squared array with matching shape
        """
        if field_k.shape != self.k_squared.shape:
            # For real FFT, need to compute appropriate k_squared
            nx, ny, nz_half = field_k.shape
            nz = (nz_half - 1) * 2  # Original size

            kx = np.fft.fftfreq(nx, self.dx) * 2 * np.pi
            ky = np.fft.fftfreq(ny, self.dy) * 2 * np.pi
            kz = np.fft.rfftfreq(nz, self.dz) * 2 * np.pi  # Real FFT frequencies

            kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing="ij")
            return kx_grid**2 + ky_grid**2 + kz_grid**2
        else:
            return self.k_squared

    def adaptive_ifft(self, field_k: np.ndarray, original_shape: tuple | None = None) -> np.ndarray:
        """
        Adaptive inverse FFT that handles both real and complex transforms.

        Args:
            field_k: FFT coefficients
            original_shape: Original real field shape (needed for real IFFT)

        Returns:
            Inverse FFT result
        """
        if self.use_real_fft and field_k.dtype == np.complex128:
            # Check if this came from real FFT (reduced size in last dimension)
            if original_shape is not None:
                expected_rfft_shape = list(original_shape)
                expected_rfft_shape[-1] = original_shape[-1] // 2 + 1
                if field_k.shape == tuple(expected_rfft_shape):
                    return self.irfft_plan(field_k, s=original_shape, axes=(-3, -2, -1))

        # Default to complex IFFT
        return self.ifft_plan(field_k).real

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
            if hasattr(fields, 'pi_munu') and fields.pi_munu.ndim >= 2:
                # Ensure tensor has expected shape (..., 4, 4)
                tensor_shape = fields.pi_munu.shape
                if len(tensor_shape) >= 2 and tensor_shape[-2:] == (4, 4):
                    for mu in range(4):
                        for nu in range(4):
                            fields.pi_munu[..., mu, nu] = self.apply_viscous_operator(
                                fields.pi_munu[..., mu, nu], eta / tau_pi, dt
                            )
                else:
                    warnings.warn(
                        f"pi_munu tensor shape {tensor_shape} incompatible with 4x4 indices",
                        stacklevel=2
                    )

    def _implicit_spectral_advance(self, fields: "ISFieldConfiguration", dt: float) -> None:
        """
        Implicit spectral advance for very stiff problems.

        Solves linear systems in Fourier space for stiff diffusion and relaxation terms.
        Uses Newton-Krylov iteration for nonlinear implicit terms.
        """
        if self.coeffs is None:
            return

        # Transform fields to Fourier space for implicit treatment
        fields_k = self._transform_fields_to_fourier(fields)

        # Implicit treatment of diffusive terms
        self._solve_implicit_diffusion(fields_k, dt)

        # Implicit treatment of relaxation terms
        if hasattr(self.coeffs, "shear_relaxation_time") or hasattr(
            self.coeffs, "bulk_relaxation_time"
        ):
            self._solve_implicit_relaxation(fields_k, dt)

        # Transform back to real space
        self._transform_fields_from_fourier(fields, fields_k)

    def _transform_fields_to_fourier(self, fields: "ISFieldConfiguration") -> dict[str, np.ndarray]:
        """Transform field configuration to Fourier space."""
        fields_k = {}

        # Bulk pressure
        if hasattr(fields, "Pi"):
            fields_k["Pi"] = self.fft_plan(fields.Pi)

        # Shear stress tensor - transform each component
        if hasattr(fields, "pi_munu") and fields.pi_munu.ndim >= 2:
            tensor_shape = fields.pi_munu.shape
            if len(tensor_shape) >= 2 and tensor_shape[-2:] == (4, 4):
                fields_k["pi_munu"] = np.zeros_like(fields.pi_munu, dtype=complex)
                for mu in range(4):
                    for nu in range(4):
                        fields_k["pi_munu"][..., mu, nu] = self.fft_plan(fields.pi_munu[..., mu, nu])
            else:
                warnings.warn(
                    f"pi_munu tensor shape {tensor_shape} incompatible with 4x4 indices",
                    stacklevel=2
                )

        # Heat flux
        if hasattr(fields, "q_mu") and fields.q_mu.ndim >= 1:
            vector_shape = fields.q_mu.shape
            if len(vector_shape) >= 1 and vector_shape[-1] == 4:
                fields_k["q_mu"] = np.zeros_like(fields.q_mu, dtype=complex)
                for mu in range(4):
                    fields_k["q_mu"][..., mu] = self.fft_plan(fields.q_mu[..., mu])
            else:
                warnings.warn(
                    f"q_mu vector shape {vector_shape} incompatible with 4-component index",
                    stacklevel=2
                )

        return fields_k

    def _transform_fields_from_fourier(
        self, fields: "ISFieldConfiguration", fields_k: dict[str, np.ndarray]
    ) -> None:
        """Transform fields back from Fourier space to real space."""
        # Bulk pressure
        if "Pi" in fields_k and hasattr(fields, "Pi"):
            fields.Pi[:] = self.ifft_plan(fields_k["Pi"]).real

        # Shear stress tensor
        if "pi_munu" in fields_k and hasattr(fields, "pi_munu"):
            if fields.pi_munu.ndim >= 2 and fields.pi_munu.shape[-2:] == (4, 4):
                for mu in range(4):
                    for nu in range(4):
                        fields.pi_munu[..., mu, nu] = self.ifft_plan(
                            fields_k["pi_munu"][..., mu, nu]
                        ).real
            else:
                warnings.warn(
                    f"pi_munu tensor shape {fields.pi_munu.shape} incompatible with 4x4 indices",
                    stacklevel=2
                )

        # Heat flux
        if "q_mu" in fields_k and hasattr(fields, "q_mu"):
            if fields.q_mu.ndim >= 1 and fields.q_mu.shape[-1] == 4:
                for mu in range(4):
                    fields.q_mu[..., mu] = self.ifft_plan(fields_k["q_mu"][..., mu]).real
            else:
                warnings.warn(
                    f"q_mu vector shape {fields.q_mu.shape} incompatible with 4-component index",
                    stacklevel=2
                )

    def _solve_implicit_diffusion(self, fields_k: dict[str, np.ndarray], dt: float) -> None:
        """
        Solve implicit diffusion equation in Fourier space.

        For linear diffusion: ∂_t u = ν ∇²u
        Implicit solution: u^{n+1}_k = u^n_k / (1 + ν k² dt)
        """
        if (
            self.coeffs is None
            or not hasattr(self.coeffs, "shear_viscosity")
            or not self.coeffs.shear_viscosity
        ):
            return

        eta = self.coeffs.shear_viscosity
        diffusion_factor = 1.0 / (1.0 + eta * self.k_squared * dt)

        # Apply to bulk pressure
        if "Pi" in fields_k:
            bulk_visc = getattr(self.coeffs, "bulk_viscosity", 0.0) or 0.0
            if bulk_visc > 0:
                bulk_factor = 1.0 / (1.0 + bulk_visc * self.k_squared * dt)
                fields_k["Pi"] *= bulk_factor

        # Apply to shear stress components
        if "pi_munu" in fields_k:
            tensor_shape = fields_k["pi_munu"].shape
            if len(tensor_shape) >= 2 and tensor_shape[-2:] == (4, 4):
                for mu in range(4):
                    for nu in range(4):
                        fields_k["pi_munu"][..., mu, nu] *= diffusion_factor
            else:
                warnings.warn(
                    f"pi_munu Fourier tensor shape {tensor_shape} incompatible with 4x4 indices",
                    stacklevel=2
                )

        # Apply to heat flux
        if "q_mu" in fields_k:
            vector_shape = fields_k["q_mu"].shape
            if len(vector_shape) >= 1 and vector_shape[-1] == 4:
                thermal_diffusivity = eta  # Simplified assumption
                thermal_factor = 1.0 / (1.0 + thermal_diffusivity * self.k_squared * dt)
                for mu in range(4):
                    fields_k["q_mu"][..., mu] *= thermal_factor
            else:
                warnings.warn(
                    f"q_mu Fourier vector shape {vector_shape} incompatible with 4-component index",
                    stacklevel=2
                )

    def _solve_implicit_relaxation(self, fields_k: dict[str, np.ndarray], dt: float) -> None:
        """
        Solve implicit relaxation equations in Fourier space.

        For relaxation: ∂_t π = -π/τ + source
        Implicit solution requires solving linear system per mode.
        """
        # Bulk relaxation
        if (
            self.coeffs is not None
            and hasattr(self.coeffs, "bulk_relaxation_time")
            and self.coeffs.bulk_relaxation_time
        ):
            tau_Pi = self.coeffs.bulk_relaxation_time
            if tau_Pi > 0 and "Pi" in fields_k:
                # Implicit relaxation: (1 + dt/τ) π^{n+1} = π^n
                relaxation_factor = 1.0 / (1.0 + dt / tau_Pi)
                fields_k["Pi"] *= relaxation_factor

        # Shear relaxation
        if (
            self.coeffs is not None
            and hasattr(self.coeffs, "shear_relaxation_time")
            and self.coeffs.shear_relaxation_time
        ):
            tau_pi = self.coeffs.shear_relaxation_time
            if tau_pi > 0 and "pi_munu" in fields_k:
                tensor_shape = fields_k["pi_munu"].shape
                if len(tensor_shape) >= 2 and tensor_shape[-2:] == (4, 4):
                    relaxation_factor = 1.0 / (1.0 + dt / tau_pi)
                    for mu in range(4):
                        for nu in range(4):
                            fields_k["pi_munu"][..., mu, nu] *= relaxation_factor
                else:
                    warnings.warn(
                        f"pi_munu Fourier tensor shape {tensor_shape} incompatible with 4x4 indices",
                        stacklevel=2
                    )

        # Heat flux relaxation (if available)
        if hasattr(self.coeffs, "heat_relaxation_time"):
            tau_q = getattr(self.coeffs, "heat_relaxation_time", None)
            if tau_q and tau_q > 0 and "q_mu" in fields_k:
                vector_shape = fields_k["q_mu"].shape
                if len(vector_shape) >= 1 and vector_shape[-1] == 4:
                    relaxation_factor = 1.0 / (1.0 + dt / tau_q)
                    for mu in range(4):
                        fields_k["q_mu"][..., mu] *= relaxation_factor
                else:
                    warnings.warn(
                        f"q_mu Fourier vector shape {vector_shape} incompatible with 4-component index",
                        stacklevel=2
                    )

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

                    warnings.warn(
                        "No metric found in grid. Defaulting to flat Minkowski spacetime. "
                        "Spectral solver currently supports only flat spacetime problems.",
                        UserWarning,
                        stacklevel=2,
                    )
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
        IMEX Runge-Kutta method: implicit spectral linear terms + explicit nonlinear terms.

        Uses second-order IMEX scheme for optimal balance of stability and accuracy.
        """
        # Second-order IMEX scheme with two stages
        self._imex_rk2_step(dt)

    def _imex_rk2_step(self, dt: float) -> None:
        """
        Second-order IMEX Runge-Kutta scheme following standard Butcher tableau.

        Based on tableau (2.9):
        Stage 1: Y₁ = y^n (no update)
        Stage 2: Y₂ = y^n + dt/2 * E(Y₁) + dt/2 * I(Y₂)  [implicit midpoint for stiff]
        Stage 3: Y₃ = y^n + dt/2 * E(Y₂) + dt * I(Y₃)     [implicit Euler for stiff]
        Final:   y^{n+1} = y^n + dt/2 * E(Y₂) + dt * I(Y₃)
        """
        # Store initial state
        fields_initial = self._copy_fields()

        # === Stage 1: Y₁ = y^n (identity) ===
        # No update needed, Y₁ = y^n
        explicit_rhs_1 = self._compute_explicit_rhs()

        # === Stage 2: Y₂ = y^n + dt/2 * E(Y₁) + dt/2 * I(Y₂) ===
        # Apply explicit update from stage 1
        self._apply_explicit_update(explicit_rhs_1, dt / 2)

        # Apply implicit linear terms with midpoint rule (coefficient 1/2)
        self.spectral.advance_linear_terms(self.fields, dt / 2, method="implicit")

        # Compute explicit RHS at Y₂
        explicit_rhs_2 = self._compute_explicit_rhs()

        # === Stage 3: Y₃ = y^n + dt/2 * E(Y₂) + dt * I(Y₃) ===
        # Restore to initial state
        self._restore_fields(fields_initial)

        # Apply weighted explicit update: dt/2 * E(Y₂)
        self._apply_explicit_update(explicit_rhs_2, dt / 2)

        # Apply implicit linear terms with full timestep (coefficient 1)
        self.spectral.advance_linear_terms(self.fields, dt, method="implicit")

        # === Final Update: y^{n+1} = y^n + dt/2 * E(Y₂) + dt * I(Y₃) ===
        # The current state already contains: y^n + dt/2 * E(Y₂) + dt * I(Y₃)
        # This is exactly the final IMEX-RK2 update, so no additional step needed.
        # The implicit terms I(Y₃) are already applied via advance_linear_terms above.

    def _compute_explicit_rhs(self) -> dict[str, np.ndarray]:
        """
        Compute explicit (nonlinear) right-hand side terms.

        Returns dictionary with derivatives for each field.
        """
        explicit_rhs = {}

        # Conservation law terms (advection, pressure gradients)
        if self.conservation is not None:
            try:
                conservation_rhs = self.conservation.evolution_equations()
                explicit_rhs.update(conservation_rhs)
            except Exception as e:
                warnings.warn(f"Conservation RHS computation failed: {e}", stacklevel=2)

        # Nonlinear relaxation source terms
        if self.relaxation is not None:
            try:
                relaxation_rhs = self._compute_relaxation_sources()
                explicit_rhs.update(relaxation_rhs)
            except Exception as e:
                warnings.warn(f"Relaxation RHS computation failed: {e}", stacklevel=2)

        # Ensure all required fields have RHS terms
        self._ensure_complete_rhs(explicit_rhs)

        return explicit_rhs

    def _compute_relaxation_sources(self) -> dict[str, np.ndarray]:
        """
        Compute nonlinear source terms from Israel-Stewart relaxation equations.

        These are the terms that cannot be treated implicitly in spectral space.
        """
        relaxation_rhs = {}

        try:
            # Use relaxation equation module if available
            if self.relaxation is not None and hasattr(self.relaxation, "compute_source_terms"):
                source_terms = self.relaxation.compute_source_terms(self.fields)
                relaxation_rhs.update(source_terms)
            else:
                # Fallback: compute basic source terms manually
                relaxation_rhs = self._compute_basic_relaxation_sources()

        except Exception as e:
            warnings.warn(f"Relaxation source computation failed: {e}", stacklevel=2)
            relaxation_rhs = {}

        return relaxation_rhs

    def _compute_basic_relaxation_sources(self) -> dict[str, np.ndarray]:
        """
        Compute basic relaxation source terms manually.

        For nonlinear Israel-Stewart terms that appear in the relaxation equations.
        """
        sources = {}

        # Basic thermodynamic driving terms
        if hasattr(self.fields, "Pi") and hasattr(self.fields, "pressure"):
            # Bulk pressure source: ∇·u term
            if hasattr(self.fields, "u_mu"):
                velocity = self.fields.u_mu[..., 1:4]  # Spatial components
                div_u = self.spectral.spatial_divergence(velocity)
                sources["dPi_dt_source"] = -self.fields.pressure * div_u

        # Shear stress sources: velocity gradients
        if hasattr(self.fields, "pi_munu") and hasattr(self.fields, "u_mu"):
            # Simplified shear source (full implementation would require metric)
            sources["dpi_dt_source"] = np.zeros_like(self.fields.pi_munu)

        return sources

    def _apply_explicit_update(self, rhs_terms: dict[str, np.ndarray], dt: float) -> None:
        """
        Apply explicit update to fields using RHS terms.

        Updates field configuration in-place.
        """
        # Energy density update
        if "drho_dt" in rhs_terms and hasattr(self.fields, "rho"):
            self.fields.rho += dt * rhs_terms["drho_dt"]

        # Four-velocity update
        if "du_dt" in rhs_terms and hasattr(self.fields, "u_mu"):
            self.fields.u_mu += dt * rhs_terms["du_dt"]

        # Pressure update (if available)
        if "dp_dt" in rhs_terms and hasattr(self.fields, "pressure"):
            self.fields.pressure += dt * rhs_terms["dp_dt"]

        # Bulk pressure source terms
        if "dPi_dt_source" in rhs_terms and hasattr(self.fields, "Pi"):
            self.fields.Pi += dt * rhs_terms["dPi_dt_source"]

        # Shear stress source terms
        if "dpi_dt_source" in rhs_terms and hasattr(self.fields, "pi_munu"):
            self.fields.pi_munu += dt * rhs_terms["dpi_dt_source"]

    def _ensure_complete_rhs(self, rhs_terms: dict[str, np.ndarray]) -> None:
        """
        Ensure all required fields have RHS terms.

        Adds zero terms for fields without explicit RHS.
        """
        # Check required fields and add zero RHS if missing
        required_fields = ["drho_dt", "du_dt"]

        for field_name in required_fields:
            if field_name not in rhs_terms:
                if field_name == "drho_dt" and hasattr(self.fields, "rho"):
                    rhs_terms[field_name] = np.zeros_like(self.fields.rho)
                elif field_name == "du_dt" and hasattr(self.fields, "u_mu"):
                    rhs_terms[field_name] = np.zeros_like(self.fields.u_mu)

    def _restore_fields(self, field_backup: dict[str, np.ndarray]) -> None:
        """Restore field configuration from backup."""
        for field_name, field_data in field_backup.items():
            if hasattr(self.fields, field_name):
                field_attr = getattr(self.fields, field_name)
                if hasattr(field_attr, "shape") and field_attr.shape == field_data.shape:
                    field_attr[:] = field_data

    def _advance_conservation_laws(self, dt: float) -> None:
        """
        Advance conservation laws using physics-correct evolution equations.

        Uses the conservation equation interface for proper 4-divergence computation
        rather than incomplete spatial-only divergence.
        """
        if self.conservation is None:
            return

        # Get physics-correct evolution equations from conservation laws
        # This properly computes ∂_μ T^μν = 0 including time derivatives
        try:
            evolution_rhs = (
                self.conservation.evolution_equations() if self.conservation is not None else {}
            )

            # Second-order Runge-Kutta integration for accuracy
            self._rk2_conservation_step(evolution_rhs, dt)

        except Exception as e:
            warnings.warn(f"Conservation evolution failed, using fallback: {e}", stacklevel=2)
            self._fallback_conservation_advance(dt)

    def _rk2_conservation_step(self, evolution_rhs: dict[str, np.ndarray], dt: float) -> None:
        """
        Second-order Runge-Kutta time integration for conservation laws.

        Replaces first-order Euler for improved accuracy and stability.
        """
        # Stage 1: Compute k1 = f(t, y)
        k1_rho = evolution_rhs.get("drho_dt", np.zeros_like(self.fields.rho))
        k1_momentum = evolution_rhs.get("du_dt", np.zeros_like(self.fields.u_mu))

        # Store initial state
        rho_0 = self.fields.rho.copy()
        u_mu_0 = self.fields.u_mu.copy()

        # Intermediate step: y_1 = y_0 + (dt/2) * k1
        self.fields.rho[:] = rho_0 + (dt / 2) * k1_rho
        self.fields.u_mu[:] = u_mu_0 + (dt / 2) * k1_momentum

        # Stage 2: Compute k2 = f(t + dt/2, y_1)
        try:
            if self.conservation is not None:
                evolution_rhs_2 = self.conservation.evolution_equations()
            else:
                evolution_rhs_2 = {}
            k2_rho = evolution_rhs_2.get("drho_dt", k1_rho)
            k2_momentum = evolution_rhs_2.get("du_dt", k1_momentum)
        except Exception:
            # Fallback to k1 if second evaluation fails
            k2_rho = k1_rho
            k2_momentum = k1_momentum

        # Final update: y_n+1 = y_0 + dt * k2
        self.fields.rho = rho_0 + dt * k2_rho
        self.fields.u_mu = u_mu_0 + dt * k2_momentum

        # Update derived quantities
        self._update_derived_fields()

    def _fallback_conservation_advance(self, dt: float) -> None:
        """
        Fallback conservation advance using spectral spatial derivatives.

        Used when conservation.evolution_equations() is not available.
        Still attempts to compute proper divergence but with limitations.
        """
        try:
            # Compute stress-energy tensor
            T_munu = (
                self.conservation.stress_energy_tensor()
                if self.conservation is not None
                else np.zeros((4, 4))
            )

            # Energy conservation: ∂_t ρ = -∂_i T^i0 (correct conservation law)
            energy_flux_div = np.zeros_like(self.fields.rho)
            for i in range(3):  # Spatial directions
                energy_flux_div += self.spectral.spatial_derivative(T_munu[..., i + 1, 0], i)

            # Update energy density with proper sign
            self.fields.rho -= dt * energy_flux_div

            # Momentum conservation: ∂_t T^0i = -∂_j T^ji (simplified)
            if hasattr(self.fields, "momentum_density"):
                for i in range(3):
                    momentum_flux_div = np.zeros_like(self.fields.rho)
                    for j in range(3):
                        momentum_flux_div += self.spectral.spatial_derivative(
                            T_munu[..., j + 1, i + 1], j
                        )

                    # Update momentum density (if available)
                    if hasattr(self.fields.momentum_density, "__getitem__"):
                        self.fields.momentum_density[..., i] -= dt * momentum_flux_div

        except Exception as e:
            warnings.warn(f"Fallback conservation advance failed: {e}", stacklevel=2)

    def _update_derived_fields(self) -> None:
        """Update derived quantities after conservation law evolution."""
        try:
            # Update pressure from equation of state if available
            if hasattr(self.fields, "update_pressure"):
                self.fields.update_pressure()

            # Ensure four-velocity normalization
            if hasattr(self.fields, "normalize_four_velocity"):
                self.fields.normalize_four_velocity()

            # Update temperature and other thermodynamic quantities
            if hasattr(self.fields, "update_thermodynamics"):
                self.fields.update_thermodynamics()

        except Exception as e:
            warnings.warn(f"Derived field update failed: {e}", stacklevel=2)

    def _advance_relaxation_terms(self, dt: float) -> None:
        """Advance Israel-Stewart relaxation equations."""
        if self.relaxation is None:
            return

        try:
            # Use existing relaxation equation solver
            self.relaxation.evolve_relaxation(self.fields, dt)
        except Exception as e:
            warnings.warn(f"Relaxation evolution failed: {e}", stacklevel=2)

    def _compute_expansion_scalar(self) -> np.ndarray:
        """
        Compute expansion scalar θ = ∇_μ u^μ for Israel-Stewart equations.

        Returns:
            Expansion scalar field θ
        """
        try:
            # Get four-velocity
            u_mu = self.fields.u_mu

            # Compute covariant divergence ∇_μ u^μ
            # For flat spacetime: θ ≈ ∂_i u^i (spatial divergence)
            theta = np.zeros_like(self.fields.rho)

            for i in range(3):  # Spatial directions
                # Convert to contravariant components for flat spacetime
                u_contravariant_i = u_mu[..., i + 1]  # u^i = u_i in Minkowski
                theta += self.spectral.spatial_derivative(u_contravariant_i, i)

            return theta

        except Exception as e:
            warnings.warn(
                f"Failed to compute expansion scalar: {e}. Using zero.", UserWarning, stacklevel=2
            )
            return np.zeros_like(self.fields.rho)

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
