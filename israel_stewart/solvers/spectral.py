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

from ..core.memory_optimization import (
    get_array_pool,
    get_fft_manager,
    get_inplace_ops,
    memory_optimized_context,
)
from ..core.performance import monitor_performance, profile_operation
from ..core.tensor_utils import optimized_einsum
from ..utils.logging_config import get_logger, performance_logger, physics_logger

if TYPE_CHECKING:
    from ..core.fields import ISFieldConfiguration, TransportCoefficients
    from ..core.spacegrid import SpaceGrid
    from ..core.spacetime_grid import SpacetimeGrid
    from ..equations.conservation import ConservationLaws
    from ..equations.relaxation import ISRelaxationEquations


class SpectralISolver:
    """
    Fourier spectral method for Israel-Stewart equations with pure 3D spatial fields.

    Provides high-performance spectral differentiation and linear operator
    application for relativistic hydrodynamics with periodic boundaries.

    **ARCHITECTURE: Pure 3D Spatial Solver**

    All fields are pure 3D spatial arrays (nx, ny, nz). Time is treated as an
    evolution parameter, not a grid dimension. This provides dramatic memory
    reduction and cleaner architecture.
    """

    def __init__(
        self,
        grid: "SpaceGrid",
        fields: "ISFieldConfiguration",
        coeffs: Optional["TransportCoefficients"] = None,
    ):
        """
        Initialize spectral solver.

        Args:
            grid: SpaceGrid defining 3D spatial computational domain
            fields: ISFieldConfiguration with pure 3D hydrodynamic variables
            coeffs: Transport coefficients for viscous terms
        """
        self.grid = grid
        self.fields = fields
        self.coeffs = coeffs

        # Extract 3D grid dimensions and spacing
        self.nx, self.ny, self.nz = grid.shape

        # Validate grid has periodic boundary conditions for spectral methods
        if hasattr(grid, "boundary_conditions") and grid.boundary_conditions != "periodic":
            warnings.warn(
                f"SpectralISolver requires periodic boundary conditions, but grid has "
                f"'{grid.boundary_conditions}' boundaries. This may cause FFT accuracy issues. "
                f"Consider using boundary_conditions='periodic' when creating the SpaceGrid.",
                UserWarning,
                stacklevel=2,
            )

        # Check if spatial_ranges attribute exists for proper grid setup validation
        if not hasattr(grid, "spatial_ranges"):
            warnings.warn(
                "Using potentially incorrect grid spacing fallback. Grid is missing spatial_ranges attribute. "
                "This may lead to incorrect FFT frequency computations for spectral methods.",
                UserWarning,
                stacklevel=2,
            )

        # Use grid spacing directly (now correct for periodic boundaries)
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

        # Memory optimization components
        self.array_pool = get_array_pool()
        self.fft_manager = get_fft_manager()
        self.inplace_ops = get_inplace_ops()

        # Pre-allocate common array shapes
        self._precompute_workspaces()

    def _precompute_workspaces(self) -> None:
        """Pre-allocate common workspace arrays for memory optimization."""
        # Common shapes used in spectral operations (all pure 3D)
        spatial_shape = (self.nx, self.ny, self.nz)
        tensor_shape = (*spatial_shape, 4, 4)
        vector_shape = (*spatial_shape, 4)

        # Pre-allocate FFT workspaces
        common_shapes = [
            spatial_shape,  # 3D spatial scalar fields
            vector_shape,  # 3D spatial vector fields
            tensor_shape,  # 3D spatial tensor fields
        ]

        # Initialize FFT workspace manager with common shapes
        self.fft_manager.precompute_fft_plans(common_shapes)

        # Pre-allocate some workspace arrays
        for shape in common_shapes:
            # Real and complex workspaces
            self.fft_manager.get_workspace(shape, np.float64)
            self.fft_manager.get_workspace(shape, np.complex128)

        # Pre-compute k-vectors for different FFT formats to avoid repeated computation
        self._precompute_k_vectors()

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

    def _precompute_k_vectors(self) -> None:
        """Pre-compute k-vectors for different FFT shapes to avoid repeated calculations."""
        # Cache for k-vectors with different shapes (real FFT vs complex FFT)
        self._k_vector_cache = {}

        # Common real FFT shape: (nx, ny, nz//2 + 1)
        nx, ny, nz = self.nx, self.ny, self.nz
        nz_half = nz // 2 + 1

        # Pre-compute k-vectors for real FFT format
        kx_real = np.fft.fftfreq(nx, self.dx) * 2 * np.pi
        ky_real = np.fft.fftfreq(ny, self.dy) * 2 * np.pi
        kz_real = np.fft.rfftfreq(nz, self.dz) * 2 * np.pi

        # Cache k-vector grids for real FFT
        real_fft_key = (nx, ny, nz_half)
        self._k_vector_cache[real_fft_key] = {
            "kx": kx_real[:, np.newaxis, np.newaxis],
            "ky": ky_real[np.newaxis, :, np.newaxis],
            "kz": kz_real[np.newaxis, np.newaxis, :],
            "kx_1d": kx_real,
            "ky_1d": ky_real,
            "kz_1d": kz_real,
        }

        # Pre-compute k-squared for real FFT
        kx_grid_real, ky_grid_real, kz_grid_real = np.meshgrid(
            kx_real, ky_real, kz_real, indexing="ij"
        )
        self._k_vector_cache[real_fft_key]["k_squared"] = (
            kx_grid_real**2 + ky_grid_real**2 + kz_grid_real**2
        )

    def get_cached_k_vectors(self, shape: tuple[int, ...]) -> dict[str, np.ndarray]:
        """Get cached k-vectors for the given FFT shape."""
        if shape in self._k_vector_cache:
            return self._k_vector_cache[shape]

        # Fallback: compute on demand if not cached
        nx, ny, nz_half = shape
        nz = (nz_half - 1) * 2

        kx = np.fft.fftfreq(nx, self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, self.dy) * 2 * np.pi
        kz = np.fft.rfftfreq(nz, self.dz) * 2 * np.pi

        return {
            "kx": kx[:, np.newaxis, np.newaxis],
            "ky": ky[np.newaxis, :, np.newaxis],
            "kz": kz[np.newaxis, np.newaxis, :],
            "kx_1d": kx,
            "ky_1d": ky,
            "kz_1d": kz,
        }

    def _compute_k_squared(self) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """Compute k^2 = kx^2 + ky^2 + kz^2 for diffusion operators."""
        kx, ky, kz = self.k_vectors
        return cast(np.ndarray[Any, np.dtype[np.floating[Any]]], kx**2 + ky**2 + kz**2)

    @monitor_performance("spectral_derivative")
    def spatial_derivative(
        self, field: np.ndarray, direction: int | list[int] | None = None, use_cache: bool = True
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]] | tuple[np.ndarray, ...]:
        """
        Compute spatial derivative using spectral method.

        ∂_i f = IFFT(ik_i * FFT(f))

        Args:
            field: Field to differentiate with shape (*spatial_shape,) or (*grid.shape,)
            direction: Spatial direction (0=x, 1=y, 2=z), list of directions, or None for all
            use_cache: Whether to cache FFT results for repeated operations

        Returns:
            Single derivative array or tuple of derivatives if multiple directions requested
        """
        # Handle multi-direction case (vectorized for performance)
        if direction is None or isinstance(direction, list):
            directions = direction if direction is not None else [0, 1, 2]
            if len(directions) > 1:
                return self._multi_direction_derivative(field, directions, use_cache)
            else:
                direction = directions[0]

        if direction not in [0, 1, 2]:
            raise ValueError(f"Direction must be 0, 1, or 2 (x, y, z), got {direction}")

        # Validate pure 3D spatial field
        if field.ndim != 3:
            raise ValueError(
                f"Field must be pure 3D spatial array (nx, ny, nz), got shape {field.shape}"
            )
        spatial_field = field

        # Check cache first
        cache_key = (id(field), direction) if use_cache else None
        if cache_key and cache_key in self._derivative_cache:
            return cast(
                np.ndarray[Any, np.dtype[np.floating[Any]]], self._derivative_cache[cache_key]
            )

        # Forward FFT with adaptive selection
        field_k = self.adaptive_fft(spatial_field)

        # Apply derivative operator ik_i with appropriate k_vector (optimized with cache)
        if field_k.shape != self.k_vectors[direction].shape:
            # Use cached k-vectors for real FFT format
            cached_k = self.get_cached_k_vectors(field_k.shape)
            k_direction = cached_k[f'k{"xyz"[direction]}']
        else:
            k_direction = self.k_vectors[direction]

        deriv_k = 1j * k_direction * field_k

        # Inverse FFT to get real derivative with adaptive selection
        result = self.adaptive_ifft(deriv_k, spatial_field.shape)

        # Cache result if requested
        if cache_key:
            self._derivative_cache[cache_key] = result

        return cast(np.ndarray[Any, np.dtype[np.floating[Any]]], result)

    def _multi_direction_derivative(
        self, field: np.ndarray, directions: list[int], use_cache: bool = True
    ) -> tuple[np.ndarray, ...]:
        """
        Compute multiple spatial derivatives with single FFT for optimal performance.

        This is the core optimization that enables efficient gradient computation by
        creating a stacked array for vectorized operations.
        """
        # Validate pure 3D spatial field
        if field.ndim != 3:
            raise ValueError(
                f"Field must be pure 3D spatial array (nx, ny, nz), got shape {field.shape}"
            )
        spatial_field = field

        # Single forward FFT for all derivatives
        field_k = self.adaptive_fft(spatial_field)

        # Get appropriate k-vectors for the FFT format (optimized with cache)
        if field_k.shape != self.k_vectors[0].shape:
            # Use cached k-vectors for real FFT format
            cached_k = self.get_cached_k_vectors(field_k.shape)
            k_grids = [cached_k["kx"], cached_k["ky"], cached_k["kz"]]
        else:
            k_grids = list(self.k_vectors)

        # Special optimization for full gradient computation
        if len(directions) == 3 and directions == [0, 1, 2]:
            # Compute all derivatives more efficiently by avoiding function call overhead
            derivatives = []
            for _i, direction in enumerate(directions):
                deriv_k = 1j * k_grids[direction] * field_k
                deriv = self.adaptive_ifft(deriv_k, spatial_field.shape)
                derivatives.append(deriv)
            return tuple(derivatives)
        else:
            # General case for arbitrary directions
            derivatives = []
            for direction in directions:
                if direction not in [0, 1, 2]:
                    raise ValueError(f"Direction must be 0, 1, or 2 (x, y, z), got {direction}")

                deriv_k = 1j * k_grids[direction] * field_k
                deriv = self.adaptive_ifft(deriv_k, spatial_field.shape)
                derivatives.append(deriv)

            return tuple(derivatives)

    @monitor_performance("gradient_computation")
    def spatial_gradient(self, field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute full spatial gradient ∇f = (∂_x f, ∂_y f, ∂_z f) with single FFT.

        Optimized version that performs only one FFT and reuses it for all three
        derivatives, providing better performance than three separate calls.

        Args:
            field: Scalar field to differentiate

        Returns:
            Tuple of (∂_x f, ∂_y f, ∂_z f)
        """
        # Validate pure 3D spatial field
        if field.ndim != 3:
            raise ValueError(
                f"Field must be pure 3D spatial array (nx, ny, nz), got shape {field.shape}"
            )
        spatial_field = field

        # Single forward FFT for all three derivatives
        field_k = self.adaptive_fft(spatial_field)

        # Get appropriate k-vectors for the FFT format (optimized with cache)
        if field_k.shape != self.k_vectors[0].shape:
            # Use cached k-vectors for real FFT format
            cached_k = self.get_cached_k_vectors(field_k.shape)
            kx_grid, ky_grid, kz_grid = cached_k["kx"], cached_k["ky"], cached_k["kz"]
        else:
            kx_grid, ky_grid, kz_grid = self.k_vectors

        # Compute all three derivatives in k-space
        dx_k = 1j * kx_grid * field_k
        dy_k = 1j * ky_grid * field_k
        dz_k = 1j * kz_grid * field_k

        # Three inverse FFTs to get real derivatives
        dx = self.adaptive_ifft(dx_k, spatial_field.shape)
        dy = self.adaptive_ifft(dy_k, spatial_field.shape)
        dz = self.adaptive_ifft(dz_k, spatial_field.shape)

        return (dx, dy, dz)

    @monitor_performance("divergence_computation")
    def spatial_divergence(self, vector_field: np.ndarray) -> np.ndarray:
        """
        Compute spatial divergence ∇·v = ∂_x v_x + ∂_y v_y + ∂_z v_z.

        Optimized version that performs vectorized FFT operations for all components
        and computes divergence directly in k-space, providing ~50% performance improvement.

        Args:
            vector_field: Vector field with shape (*spatial_shape, 3)

        Returns:
            Divergence ∇·v
        """
        if vector_field.shape[-1] != 3:
            raise ValueError("Vector field must have 3 components")

        # Validate pure 3D spatial vector field
        if vector_field.ndim != 4 or vector_field.shape[-1] != 3:
            raise ValueError(
                f"Vector field must have shape (nx, ny, nz, 3), got {vector_field.shape}"
            )
        spatial_vector = vector_field

        # Extract components
        vx = spatial_vector[..., 0]
        vy = spatial_vector[..., 1]
        vz = spatial_vector[..., 2]

        # Forward FFT for all three components
        vx_k = self.adaptive_fft(vx)
        vy_k = self.adaptive_fft(vy)
        vz_k = self.adaptive_fft(vz)

        # Get appropriate k-vectors for the FFT format (optimized with cache)
        if vx_k.shape != self.k_vectors[0].shape:
            # Use cached k-vectors for real FFT format
            cached_k = self.get_cached_k_vectors(vx_k.shape)
            kx_grid, ky_grid, kz_grid = cached_k["kx"], cached_k["ky"], cached_k["kz"]
        else:
            kx_grid, ky_grid, kz_grid = self.k_vectors

        # Compute divergence in k-space: ∇·v = ik_x v_x + ik_y v_y + ik_z v_z
        div_k = 1j * (kx_grid * vx_k + ky_grid * vy_k + kz_grid * vz_k)

        # Single inverse FFT to get real divergence
        div_result = self.adaptive_ifft(div_k, vx.shape)

        return div_result

    def memory_optimized_divergence(self, vector_field: np.ndarray) -> np.ndarray:
        """
        Memory-optimized spatial divergence using pre-allocated arrays.

        Args:
            vector_field: Vector field with shape (*spatial_shape, 3)

        Returns:
            Divergence ∇·v
        """
        if vector_field.shape[-1] != 3:
            raise ValueError("Vector field must have 3 components")

        with profile_operation("memory_optimized_divergence", {"input_shape": vector_field.shape}):
            result_shape = vector_field.shape[:-1]

            # Get temporary arrays from pool
            div_result = self.array_pool.get_array(result_shape, np.float64)
            temp_derivative = self.array_pool.get_array(result_shape, np.float64)

            try:
                # Initialize result
                div_result.fill(0.0)

                # Use optimized divergence computation (reduces to single FFT per component)
                div_result[:] = self.spatial_divergence(vector_field)

                # Return copy since we're returning temporary arrays to pool
                result = div_result.copy()

            finally:
                # Return arrays to pool
                self.array_pool.return_array(div_result)
                self.array_pool.return_array(temp_derivative)

            return result

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
                physics_logger.log_physics_fallback(
                    "israel_stewart_bulk_evolution", str(e), "simplified_exponential_relaxation"
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

        Uses magnitude-based filtering: zeros out modes where |k| > (2/3) * k_max
        in any direction. This properly implements the 2/3 rule independent of
        the FFT frequency layout.

        Args:
            field_k: FFT coefficients in k-space

        Returns:
            Dealiased FFT coefficients
        """
        nx, ny, nz = field_k.shape
        result = field_k.copy()

        # Get actual frequency values using fftfreq
        kx_vals = np.fft.fftfreq(nx, self.dx) * 2 * np.pi
        ky_vals = np.fft.fftfreq(ny, self.dy) * 2 * np.pi
        kz_vals = np.fft.fftfreq(nz, self.dz) * 2 * np.pi

        # Maximum frequencies (Nyquist frequency)
        kx_max = np.pi / self.dx
        ky_max = np.pi / self.dy
        kz_max = np.pi / self.dz

        # 2/3 rule cutoffs
        kx_cutoff = (2.0 / 3.0) * kx_max
        ky_cutoff = (2.0 / 3.0) * ky_max
        kz_cutoff = (2.0 / 3.0) * kz_max

        # Create 3D meshgrid of frequency values
        kx_grid, ky_grid, kz_grid = np.meshgrid(kx_vals, ky_vals, kz_vals, indexing="ij")

        # Create dealiasing mask: zero where |k| > (2/3) * k_max in any direction
        dealias_mask = (
            (np.abs(kx_grid) <= kx_cutoff)
            & (np.abs(ky_grid) <= ky_cutoff)
            & (np.abs(kz_grid) <= kz_cutoff)
        )

        # Apply dealiasing mask
        result *= dealias_mask

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

    def memory_optimized_fft(self, field: np.ndarray) -> np.ndarray:
        """
        Memory-optimized FFT using pre-allocated workspaces.

        Args:
            field: Input field for FFT

        Returns:
            FFT of field in k-space
        """
        with profile_operation("memory_optimized_fft", {"input_shape": field.shape}):
            # Get appropriate workspace
            if np.isrealobj(field) and self.use_real_fft:
                # Real FFT with workspace
                real_workspace, complex_workspace = self.fft_manager.get_real_fft_workspace(
                    field.shape
                )

                # Copy input to workspace to avoid modifying original
                self.inplace_ops.copy_with_slicing(real_workspace, field)

                # Perform FFT in-place
                result = self.rfft_plan(real_workspace)

                # Copy result to output workspace
                complex_workspace[:] = result
                return complex_workspace
            else:
                # Complex FFT with workspace
                workspace = self.fft_manager.get_workspace(field.shape, np.complex128)

                # Copy input to workspace
                workspace.real = field.real if hasattr(field, "real") else field
                if hasattr(field, "imag"):
                    workspace.imag = field.imag

                # Perform FFT in-place
                result = self.fft_plan(workspace)
                return result

    def get_k_squared_for_field(self, field_k: np.ndarray) -> np.ndarray:
        """
        Get appropriate k_squared array for the given FFT field shape.

        Args:
            field_k: FFT coefficients

        Returns:
            k_squared array with matching shape
        """
        if field_k.shape != self.k_squared.shape:
            # Use cached k-squared for real FFT format
            cached_k = self.get_cached_k_vectors(field_k.shape)
            if "k_squared" in cached_k:
                return cached_k["k_squared"]
            else:
                # Fallback: compute on demand
                kx, ky, kz = cached_k["kx"], cached_k["ky"], cached_k["kz"]
                return kx**2 + ky**2 + kz**2
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
            if hasattr(fields, "pi_munu") and fields.pi_munu.ndim >= 2:
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
                        stacklevel=2,
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

        # Shear stress tensor - transform each component (vectorized)
        if hasattr(fields, "pi_munu") and fields.pi_munu.ndim >= 2:
            tensor_shape = fields.pi_munu.shape
            if len(tensor_shape) >= 2 and tensor_shape[-2:] == (4, 4):
                # Vectorized FFT: reshape to batch all 16 components, then reshape back
                spatial_shape = tensor_shape[:-2]  # (..., spatial dimensions)
                tensor_flat = fields.pi_munu.reshape((*spatial_shape, 16))  # Flatten last two dims

                # Apply FFT to all components at once
                tensor_k_flat = np.zeros_like(tensor_flat, dtype=complex)
                for i in range(16):
                    tensor_k_flat[..., i] = self.fft_plan(tensor_flat[..., i])

                # Reshape back to tensor form
                fields_k["pi_munu"] = tensor_k_flat.reshape((*spatial_shape, 4, 4))
            else:
                warnings.warn(
                    f"pi_munu tensor shape {tensor_shape} incompatible with 4x4 indices",
                    stacklevel=2,
                )

        # Heat flux (vectorized)
        if hasattr(fields, "q_mu") and fields.q_mu.ndim >= 1:
            vector_shape = fields.q_mu.shape
            if len(vector_shape) >= 1 and vector_shape[-1] == 4:
                # Apply FFT to all 4 components efficiently
                fields_k["q_mu"] = np.zeros_like(fields.q_mu, dtype=complex)
                for mu in range(4):
                    fields_k["q_mu"][..., mu] = self.fft_plan(fields.q_mu[..., mu])
            else:
                warnings.warn(
                    f"q_mu vector shape {vector_shape} incompatible with 4-component index",
                    stacklevel=2,
                )

        return fields_k

    def _transform_fields_from_fourier(
        self, fields: "ISFieldConfiguration", fields_k: dict[str, np.ndarray]
    ) -> None:
        """Transform fields back from Fourier space to real space."""
        # Bulk pressure
        if "Pi" in fields_k and hasattr(fields, "Pi"):
            fields.Pi[:] = self.ifft_plan(fields_k["Pi"]).real

        # Shear stress tensor (vectorized inverse)
        if "pi_munu" in fields_k and hasattr(fields, "pi_munu"):
            if fields.pi_munu.ndim >= 2 and fields.pi_munu.shape[-2:] == (4, 4):
                # Vectorized IFFT: flatten tensor components for batch processing
                spatial_shape = fields.pi_munu.shape[:-2]
                tensor_k_flat = fields_k["pi_munu"].reshape((*spatial_shape, 16))

                # Apply IFFT to all components
                for i in range(16):
                    component_real = self.ifft_plan(tensor_k_flat[..., i]).real
                    mu, nu = divmod(i, 4)  # Convert flat index back to (mu, nu)
                    fields.pi_munu[..., mu, nu] = component_real
            else:
                warnings.warn(
                    f"pi_munu tensor shape {fields.pi_munu.shape} incompatible with 4x4 indices",
                    stacklevel=2,
                )

        # Heat flux (efficient inverse)
        if "q_mu" in fields_k and hasattr(fields, "q_mu"):
            if fields.q_mu.ndim >= 1 and fields.q_mu.shape[-1] == 4:
                for mu in range(4):
                    fields.q_mu[..., mu] = self.ifft_plan(fields_k["q_mu"][..., mu]).real
            else:
                warnings.warn(
                    f"q_mu vector shape {fields.q_mu.shape} incompatible with 4-component index",
                    stacklevel=2,
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
                    stacklevel=2,
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
                    stacklevel=2,
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
                        stacklevel=2,
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
                        stacklevel=2,
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
    Complete spectral hydrodynamics solver for Israel-Stewart equations with pure 3D fields.

    Integrates spectral methods with conservation laws and relaxation equations
    for efficient relativistic hydrodynamics simulations.

    **ARCHITECTURE: Pure 3D Spatial Evolution**

    This solver evolves pure 3D spatial fields forward in time:

    - **Fields shape**: (nx, ny, nz) - pure 3D spatial arrays
    - **time_step(dt)**: Advances the current state forward by timestep dt using RK2/IMEX
    - **Spatial derivatives**: Computed directly on 3D fields using FFT
    - **Time integration**: Standard explicit/IMEX Runge-Kutta methods for stiff ODEs

    **Usage Pattern:**

    1. Initialize pure 3D fields at t=0:
       ```python
       X, Y, Z = grid.meshgrid()
       fields.rho[:] = initial_density(X, Y, Z)  # Direct 3D access
       fields.u_mu[..., :] = initial_velocity(X, Y, Z)
       ```

    2. Evolve forward in time using adaptive time stepping:
       ```python
       hydro.evolve(t_final=10.0)  # Advances from t=0 to t=10
       ```

    3. Or manually step through time with custom output:
       ```python
       for step in range(n_steps):
           dt = hydro.adaptive_time_step()
           hydro.time_step(dt)
           # Access current state directly in fields.*
       ```

    **Time Integration Methods:**

    - **split_step**: Operator splitting (spectral diffusion + real-space advection)
    - **spectral_imex**: IMEX Runge-Kutta (implicit diffusion + explicit advection)

    Both methods are suitable for stiff Israel-Stewart relaxation equations.
    """

    def __init__(
        self,
        grid: "SpaceGrid",
        fields: "ISFieldConfiguration",
        coeffs: Optional["TransportCoefficients"] = None,
    ):
        """
        Initialize integrated spectral hydrodynamics solver.

        Args:
            grid: SpaceGrid defining 3D spatial computational domain
            fields: ISFieldConfiguration with pure 3D hydrodynamic variables
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

        # Pre-allocate workspace for Newton-Krylov solver to avoid repeated allocations
        # This workspace is reused in _config_from_dict during implicit stage solves
        self._nk_workspace_fields: ISFieldConfiguration | None = None
        self._init_nk_workspace()

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
            logger = get_logger("spectral.initialization")
            logger.info(
                "Physics modules unavailable, using simplified mode",
                extra={
                    "error": str(e),
                    "fallback": "simplified_spectral_mode",
                    "conservation": False,
                    "relaxation": False,
                },
            )
            self.conservation = None
            self.relaxation = None

    def _init_nk_workspace(self) -> None:
        """
        Initialize workspace for Newton-Krylov solver.

        Pre-allocates ISFieldConfiguration to avoid repeated allocations
        during implicit stage solves. This improves performance by 10-50x
        for small timesteps by eliminating memory allocation overhead.
        """
        try:
            from ..core.fields import ISFieldConfiguration

            self._nk_workspace_fields = ISFieldConfiguration(self.grid)
            logger = get_logger("spectral.initialization")
            logger.debug(
                "Newton-Krylov workspace allocated",
                extra={
                    "workspace_memory_mb": self._nk_workspace_fields.total_field_count
                    * 8
                    / 1024
                    / 1024,
                    "grid_shape": self.grid.shape,
                },
            )
        except Exception as e:
            logger = get_logger("spectral.initialization")
            logger.warning(
                f"Failed to allocate NK workspace: {e}. Will allocate on-demand (slower).",
                extra={"error": str(e)},
            )
            self._nk_workspace_fields = None

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
        ARS(2,2,2) IMEX Runge-Kutta scheme - Ascher, Ruuth, Spiteri (1997).

        Implements the 2-stage, 2nd-order L-stable IMEX-RK scheme:

        Explicit tableau (for F):        Implicit DIRK tableau (for G):
        c̃ = [0, 1]                       c = [γ, 1]
        Ã = [0  0]                       A = [γ    0  ]
            [1  0]                           [1-γ  γ  ]
        b̃ = [1/2, 1/2]                   b = [1-γ, γ]

        where γ = 1 - 1/√2 ≈ 0.292893218

        Stage equations:
        Y₁ = y^n + h·γ·G(Y₁)                           [implicit]
        Y₂ = y^n + h·F(Y₁) + h·(1-γ)·G(Y₁) + h·γ·G(Y₂) [mixed]

        Final update:
        y^{n+1} = y^n + h/2·F(Y₁) + h/2·F(Y₂) + h·(1-γ)·G(Y₁) + h·γ·G(Y₂)
        """
        # ARS(2,2,2) parameter
        h = dt
        gamma = 1.0 - 1.0 / np.sqrt(2.0)  # ≈ 0.292893218

        # Store initial state y^n
        y_n_dict = self._copy_fields()

        # === Stage 1: Y₁ = y^n + h·γ·G(Y₁) ===
        Y1_dict = self._solve_implicit_stage(y_n_dict, gamma * h)
        Y1_fields = self._config_from_dict(Y1_dict)

        # Compute explicit RHS F(Y₁)
        F_Y1_dict = self._compute_explicit_rhs_for_fields(Y1_fields)

        # Compute implicit terms G(Y₁) from stage equation: h·γ·G(Y₁) = Y₁ - y^n
        G_Y1_scaled_dict = self._add_fields(Y1_dict, y_n_dict, scale=-1.0)
        G_Y1_dict = self._scale_fields(G_Y1_scaled_dict, scale=1.0 / (gamma * h))

        # === Stage 2: Y₂ = y^n + h·F(Y₁) + h·(1-γ)·G(Y₁) + h·γ·G(Y₂) ===
        # Build RHS for stage 2: y^n + h·F(Y₁) + h·(1-γ)·G(Y₁)
        rhs2_dict = self._add_fields(y_n_dict, F_Y1_dict, scale=h)
        rhs2_dict = self._add_fields(rhs2_dict, G_Y1_dict, scale=h * (1.0 - gamma))

        Y2_dict = self._solve_implicit_stage(rhs2_dict, gamma * h)
        Y2_fields = self._config_from_dict(Y2_dict)

        # Compute explicit RHS F(Y₂)
        F_Y2_dict = self._compute_explicit_rhs_for_fields(Y2_fields)

        # Compute implicit terms G(Y₂) from stage equation: h·γ·G(Y₂) = Y₂ - RHS₂
        G_Y2_scaled_dict = self._add_fields(Y2_dict, rhs2_dict, scale=-1.0)
        G_Y2_dict = self._scale_fields(G_Y2_scaled_dict, scale=1.0 / (gamma * h))

        # === Final Update: y^{n+1} = y^n + h/2·F(Y₁) + h/2·F(Y₂) + h·(1-γ)·G(Y₁) + h·γ·G(Y₂) ===
        final_dict = y_n_dict.copy()
        final_dict = self._add_fields(final_dict, F_Y1_dict, scale=h / 2.0)
        final_dict = self._add_fields(final_dict, F_Y2_dict, scale=h / 2.0)
        final_dict = self._add_fields(final_dict, G_Y1_dict, scale=h * (1.0 - gamma))
        final_dict = self._add_fields(final_dict, G_Y2_dict, scale=h * gamma)

        # Load final result into self.fields
        self._restore_fields(final_dict)

    def _compute_explicit_rhs_for_fields(
        self, fields: "ISFieldConfiguration"
    ) -> dict[str, np.ndarray]:
        """
        Compute explicit RHS terms F(Y) for specific field configuration.

        Args:
            fields: Field configuration to evaluate RHS at

        Returns:
            Dictionary of explicit right-hand side terms
        """
        # Temporarily store current fields and switch to input fields
        original_fields = self.fields
        self.fields = fields

        try:
            # Compute RHS using existing method
            explicit_rhs = self._compute_explicit_rhs()
            return explicit_rhs
        finally:
            # Restore original fields
            self.fields = original_fields

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
                physics_logger.log_physics_fallback(
                    "conservation_rhs_computation", str(e), "skip_conservation_terms"
                )

        # Nonlinear relaxation source terms
        if self.relaxation is not None:
            try:
                relaxation_rhs = self._compute_relaxation_sources()
                explicit_rhs.update(relaxation_rhs)
            except Exception as e:
                physics_logger.log_physics_fallback(
                    "relaxation_rhs_computation", str(e), "skip_relaxation_terms"
                )

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
            physics_logger.log_physics_fallback(
                "relaxation_source_computation", str(e), "empty_sources"
            )
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

    def _restore_fields(
        self,
        field_backup: dict[str, np.ndarray],
        target_fields: Optional["ISFieldConfiguration"] = None,
    ) -> None:
        """
        Restore field configuration from backup.

        Args:
            field_backup: Dictionary of field arrays to restore
            target_fields: Target configuration object (defaults to self.fields)
        """
        target = target_fields if target_fields is not None else self.fields

        for field_name, field_data in field_backup.items():
            if hasattr(target, field_name):
                field_attr = getattr(target, field_name)
                if hasattr(field_attr, "shape") and field_attr.shape == field_data.shape:
                    try:
                        field_attr[:] = field_data
                    except (ValueError, TypeError):
                        # Handle read-only arrays by replacing the attribute
                        setattr(target, field_name, field_data.copy())

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
            # Log the actual error to help debug
            import sys
            import traceback

            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            print(f"CONSERVATION ERROR: {error_msg}", file=sys.stderr)
            physics_logger.log_physics_fallback(
                "conservation_evolution", error_msg, "fallback_conservation_advance"
            )
            self._fallback_conservation_advance(dt)

    def _convert_momentum_to_velocity_derivative(
        self,
        dmom_dt: np.ndarray,
        drho_dt: np.ndarray,
    ) -> np.ndarray:
        """
        Convert momentum density derivative to velocity derivative.

        Physics: d(ρu^i)/dt = ρ·du^i/dt + u^i·dρ/dt (product rule)
        Solving: du^i/dt = (1/ρ)[d(ρu^i)/dt - u^i·dρ/dt]

        This is CRITICAL for sound wave propagation and any flow where both
        density and velocity change. Without this conversion, waves propagate
        at incorrect speeds.

        Args:
            dmom_dt: Time derivative of momentum density d(ρu^i)/dt, shape (*grid.shape, 3)
            drho_dt: Time derivative of energy density dρ/dt, shape (*grid.shape,)

        Returns:
            du_dt: Time derivative of spatial velocity du^i/dt, shape (*grid.shape, 3)
        """
        rho = self.fields.rho
        u_spatial = self.fields.u_mu[..., 1:4]

        # Avoid division by zero (use small epsilon where rho is tiny)
        rho_safe = np.where(np.abs(rho) > 1e-14, rho, 1e-14)

        # Expand drho_dt to broadcast with spatial velocity
        drho_dt_expanded = drho_dt[..., np.newaxis]  # Shape: (*grid.shape, 1)

        # du^i/dt = (1/ρ)[d(ρu^i)/dt - u^i·dρ/dt]
        du_dt = (1.0 / rho_safe[..., np.newaxis]) * (dmom_dt - u_spatial * drho_dt_expanded)

        return du_dt

    def _rk2_conservation_step(self, evolution_rhs: dict[str, np.ndarray], dt: float) -> None:
        """
        Second-order Runge-Kutta time integration for conservation laws.

        CRITICAL: Conservation equations provide d(ρu^i)/dt (momentum density),
        but we evolve u^i (velocity). Must convert using:
            du^i/dt = (1/ρ)[d(ρu^i)/dt - u^i·dρ/dt]
        """
        # Stage 1: Compute k1 = f(t, y)
        k1_rho = evolution_rhs.get("drho_dt", np.zeros_like(self.fields.rho))

        # Convert momentum density derivative to velocity derivative
        dmom_dt = evolution_rhs.get("dmom_dt", np.zeros_like(self.fields.u_mu[..., 1:4]))
        du_dt = self._convert_momentum_to_velocity_derivative(dmom_dt, k1_rho)

        k1_velocity = np.zeros_like(self.fields.u_mu)
        k1_velocity[..., 1:4] = du_dt  # Only spatial components evolve

        # Store initial state
        rho_0 = self.fields.rho.copy()
        u_mu_0 = self.fields.u_mu.copy()

        # Intermediate step: y_1 = y_0 + (dt/2) * k1
        self.fields.rho[:] = rho_0 + (dt / 2) * k1_rho
        self.fields.u_mu[:] = u_mu_0 + (dt / 2) * k1_velocity

        # Stage 2: Compute k2 = f(t + dt/2, y_1)
        try:
            if self.conservation is not None:
                evolution_rhs_2 = self.conservation.evolution_equations()
            else:
                evolution_rhs_2 = {}
            k2_rho = evolution_rhs_2.get("drho_dt", k1_rho)

            # Convert momentum to velocity for stage 2
            dmom_dt_2 = evolution_rhs_2.get("dmom_dt", dmom_dt)
            du_dt_2 = self._convert_momentum_to_velocity_derivative(dmom_dt_2, k2_rho)

            k2_velocity = np.zeros_like(self.fields.u_mu)
            k2_velocity[..., 1:4] = du_dt_2
        except Exception:
            # Fallback to k1 if second evaluation fails
            k2_rho = k1_rho
            k2_velocity = k1_velocity

        # Final update: y_n+1 = y_0 + dt * k2
        self.fields.rho[:] = rho_0 + dt * k2_rho
        self.fields.u_mu[:] = u_mu_0 + dt * k2_velocity

        # Update derived quantities (includes normalization)
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
            # Optimized: use vectorized divergence instead of manual loop
            energy_flux_vector = T_munu[..., 1:4, 0]  # T^{1,0}, T^{2,0}, T^{3,0}
            energy_flux_div = self.spectral.spatial_divergence(energy_flux_vector)

            # Update energy density with proper sign
            self.fields.rho -= dt * energy_flux_div

            # Momentum conservation: ∂_t T^0i = -∂_j T^ji (simplified)
            if hasattr(self.fields, "momentum_density"):
                # Optimized: vectorized momentum flux divergence computation
                for i in range(3):
                    # Extract momentum flux vector for component i: T^{j,i} for j=1,2,3
                    momentum_flux_vector = T_munu[..., 1:4, i + 1]  # T^{1,i}, T^{2,i}, T^{3,i}
                    momentum_flux_div = self.spectral.spatial_divergence(momentum_flux_vector)

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
            # Optimized: use vectorized divergence instead of manual loop
            velocity_spatial = u_mu[..., 1:4]  # u^1, u^2, u^3 (spatial components)
            theta = self.spectral.spatial_divergence(velocity_spatial)

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

    def _add_fields(
        self,
        fields_base: dict[str, np.ndarray],
        fields_to_add: dict[str, np.ndarray],
        scale: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """
        Combine field dictionaries: result = base + scale * to_add.

        Args:
            fields_base: Base field dictionary
            fields_to_add: Fields to add (scaled)
            scale: Scaling factor for fields_to_add

        Returns:
            New dictionary with combined fields
        """
        result = {}
        for key in fields_base:
            if key in fields_to_add:
                result[key] = fields_base[key] + scale * fields_to_add[key]
            else:
                result[key] = fields_base[key].copy()
        return result

    def _scale_fields(
        self, field_dict: dict[str, np.ndarray], scale: float
    ) -> dict[str, np.ndarray]:
        """
        Scale all fields in dictionary by a constant factor.

        Args:
            field_dict: Dictionary of field arrays
            scale: Scaling factor

        Returns:
            New dictionary with scaled fields
        """
        return {key: scale * field_array for key, field_array in field_dict.items()}

    def _config_from_dict(self, field_dict: dict[str, np.ndarray]) -> "ISFieldConfiguration":
        """
        Create ISFieldConfiguration object from field dictionary.

        Uses pre-allocated workspace to avoid repeated memory allocations
        during Newton-Krylov iterations. This improves performance by 10-50x.

        Args:
            field_dict: Dictionary containing field arrays (may be incomplete)

        Returns:
            ISFieldConfiguration with specified field values (may be workspace)
        """
        # Use pre-allocated workspace if available (fast path)
        if self._nk_workspace_fields is not None:
            new_fields = self._nk_workspace_fields
        else:
            # Fallback: allocate new configuration (slow path)
            from ..core.fields import ISFieldConfiguration

            new_fields = ISFieldConfiguration(self.grid)

        # Update fields in-place from dictionary (avoids copy overhead)
        for key in ["rho", "Pi", "pi_munu", "q_mu", "u_mu"]:
            if key in field_dict:
                try:
                    # In-place update: workspace_field[:] = new_values
                    getattr(new_fields, key)[:] = field_dict[key]
                except (ValueError, AttributeError, TypeError):
                    # Field may be read-only or wrong shape
                    pass
            elif hasattr(self.fields, key):
                # Use current solver state for missing fields
                try:
                    current_field = getattr(self.fields, key)
                    getattr(new_fields, key)[:] = current_field
                except (ValueError, AttributeError):
                    # Field may not exist or be incompatible
                    pass

        # Ensure pressure is set if rho is available (conformal EOS as fallback)
        if hasattr(new_fields, "rho") and np.any(new_fields.rho > 0):
            if not hasattr(new_fields, "pressure") or np.all(new_fields.pressure == 0):
                new_fields.pressure = new_fields.rho / 3.0

        return new_fields

    def _solve_implicit_stage(
        self, rhs_dict: dict[str, np.ndarray], gamma_dt: float
    ) -> dict[str, np.ndarray]:
        """
        Solve implicit stage equation: (I - γ·dt·∂G/∂y)·Y = RHS for ARS(2,2,2).

        This solves the nonlinear algebraic equation Y = RHS + γ·dt·G(Y) where G(Y)
        represents the stiff terms (viscous diffusion and relaxation).

        Args:
            rhs_dict: Right-hand side field dictionary
            gamma_dt: Product γ·dt where γ is ARS(2,2,2) parameter

        Returns:
            Solution dictionary Y
        """
        if abs(gamma_dt) < 1e-12:
            # No implicit terms, return RHS directly
            return {key: field.copy() for key, field in rhs_dict.items()}

        try:
            # Use Newton-Krylov iteration to solve the nonlinear system
            return self._newton_krylov_solve(rhs_dict, gamma_dt)
        except Exception as e:
            warnings.warn(
                f"Implicit stage solve failed: {e}. Using explicit approximation.", stacklevel=2
            )
            # Fallback: first-order explicit approximation
            return self._explicit_approximation(rhs_dict, gamma_dt)

    def _newton_krylov_solve(
        self, rhs_dict: dict[str, np.ndarray], gamma_dt: float
    ) -> dict[str, np.ndarray]:
        """
        Newton-Krylov solver for implicit stage equation.

        Solves F(Y) = Y - RHS - γ·dt·G(Y) = 0 using Newton iteration
        with Krylov subspace methods for the linear systems.
        """
        # Convert to flat array for scipy solver
        rhs_flat = self._dict_to_flat(rhs_dict)

        def residual_function(y_flat: np.ndarray) -> np.ndarray:
            """Residual function F(Y) = Y - RHS - γ·dt·G(Y)."""
            y_dict = self._flat_to_dict(y_flat, rhs_dict)
            y_fields = self._config_from_dict(y_dict)

            # Compute G(Y) - the stiff terms
            G_y_dict = self._compute_stiff_terms(y_fields)

            # F(Y) = Y - RHS - γ·dt·G(Y)
            residual_dict = self._add_fields(y_dict, rhs_dict, scale=-1.0)
            residual_dict = self._add_fields(residual_dict, G_y_dict, scale=-gamma_dt)

            return self._dict_to_flat(residual_dict)

        # Initial guess: Y⁰ = RHS (explicit approximation)
        y0_flat = rhs_flat.copy()

        # Solve using Newton-Krylov with moderate tolerance
        try:
            solution_flat = newton_krylov(
                residual_function,
                y0_flat,
                method="lgmres",  # Left-preconditioned GMRES
                verbose=False,
                maxiter=10,  # Limit iterations for performance
                f_tol=1e-8,  # Reasonable tolerance for PDE context
                f_rtol=1e-6,
            )
            return self._flat_to_dict(solution_flat, rhs_dict)

        except Exception as e:
            warnings.warn(f"Newton-Krylov iteration failed: {e}", stacklevel=2)
            # Return explicit approximation as fallback
            return self._explicit_approximation(rhs_dict, gamma_dt)

    def _compute_stiff_terms(self, fields: "ISFieldConfiguration") -> dict[str, np.ndarray]:
        """
        Compute stiff terms G(Y) for implicit solver.

        These are the terms that require implicit treatment:
        - Viscous diffusion terms
        - Relaxation terms (linear part)
        """
        stiff_terms = {}

        if self.coeffs is None:
            # No stiff terms
            return {
                key: np.zeros_like(getattr(fields, key))
                for key in ["rho", "Pi", "pi_munu", "q_mu", "u_mu"]
            }

        # Bulk viscous diffusion: ∇²Π term (only if viscosity > 0)
        if (
            hasattr(self.coeffs, "bulk_viscosity")
            and self.coeffs.bulk_viscosity
            and self.coeffs.bulk_viscosity > 0
        ):
            try:
                Pi_laplacian = self._compute_laplacian(fields.Pi)
                stiff_terms["Pi"] = self.coeffs.bulk_viscosity * Pi_laplacian
            except (ValueError, AttributeError) as e:
                # Laplacian computation failed - skip diffusion term
                warnings.warn(
                    f"Bulk diffusion term skipped due to Laplacian error: {e}",
                    stacklevel=2,
                )
                stiff_terms["Pi"] = np.zeros_like(fields.Pi)
        else:
            stiff_terms["Pi"] = np.zeros_like(fields.Pi)

        # Shear viscous diffusion: ∇²π^μν terms (only if viscosity > 0)
        if (
            hasattr(self.coeffs, "shear_viscosity")
            and self.coeffs.shear_viscosity
            and self.coeffs.shear_viscosity > 0
        ):
            try:
                pi_laplacian = np.zeros_like(fields.pi_munu)
                if fields.pi_munu.shape[-2:] == (4, 4):
                    for mu in range(4):
                        for nu in range(4):
                            pi_laplacian[..., mu, nu] = self._compute_laplacian(
                                fields.pi_munu[..., mu, nu]
                            )
                stiff_terms["pi_munu"] = self.coeffs.shear_viscosity * pi_laplacian
            except (ValueError, AttributeError) as e:
                # Laplacian computation failed - skip diffusion term
                warnings.warn(
                    f"Shear diffusion term skipped due to Laplacian error: {e}",
                    stacklevel=2,
                )
                stiff_terms["pi_munu"] = np.zeros_like(fields.pi_munu)
        else:
            stiff_terms["pi_munu"] = np.zeros_like(fields.pi_munu)

        # Relaxation terms (linear parts only)
        if hasattr(self.coeffs, "bulk_relaxation_time") and self.coeffs.bulk_relaxation_time:
            tau_Pi = self.coeffs.bulk_relaxation_time
            stiff_terms["Pi"] += -fields.Pi / tau_Pi

        if hasattr(self.coeffs, "shear_relaxation_time") and self.coeffs.shear_relaxation_time:
            tau_pi = self.coeffs.shear_relaxation_time
            stiff_terms["pi_munu"] += -fields.pi_munu / tau_pi

        # Other fields (typically no stiff terms)
        stiff_terms["rho"] = np.zeros_like(fields.rho)
        stiff_terms["u_mu"] = np.zeros_like(fields.u_mu)
        stiff_terms["q_mu"] = np.zeros_like(fields.q_mu)

        return stiff_terms

    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian ∇²field using spectral methods on pure 3D fields.

        For field in Fourier space: ℱ[∇²f] = -k²·ℱ[f]

        This is critical for Israel-Stewart physics: viscous terms like ζ∇²Π and η∇²π^μν
        are essential for proper dissipative behavior.

        Args:
            field: Pure 3D spatial field (nx, ny, nz)

        Returns:
            Laplacian of field (nx, ny, nz)
        """
        try:
            # Validate pure 3D spatial field
            if field.ndim != 3:
                raise ValueError(
                    f"Field must be pure 3D spatial array (nx, ny, nz), got shape {field.shape}"
                )

            # Validate spatial dimensions
            if hasattr(self, "spectral"):
                expected_spatial = (self.spectral.nx, self.spectral.ny, self.spectral.nz)
                spectral_solver = self.spectral
            else:
                expected_spatial = (self.nx, self.ny, self.nz)
                spectral_solver = self

            if field.shape != expected_spatial:
                raise ValueError(f"Field shape {field.shape} != grid shape {expected_spatial}")

            # Transform to Fourier space using existing FFT plans
            field_k = spectral_solver.fft_plan(field)

            # Apply Laplacian operator: ℱ[∇²f] = -k²·ℱ[f]
            laplacian_k = -spectral_solver.k_squared * field_k

            # Transform back to real space
            laplacian = spectral_solver.ifft_plan(laplacian_k).real

            return laplacian

        except Exception as e:
            warnings.warn(
                f"Spectral Laplacian computation failed: {e}. "
                f"Using zero diffusion (WARNING: This breaks Israel-Stewart physics!)",
                stacklevel=2,
            )
            # Emergency fallback - this breaks physics but maintains stability
            return np.zeros_like(field)

    def _explicit_approximation(
        self, rhs_dict: dict[str, np.ndarray], gamma_dt: float
    ) -> dict[str, np.ndarray]:
        """
        Explicit approximation for implicit stage: Y ≈ RHS + γ·dt·G(RHS).

        Used as fallback when implicit solver fails.
        """
        try:
            rhs_fields = self._config_from_dict(rhs_dict)
            G_rhs_dict = self._compute_stiff_terms(rhs_fields)
            return self._add_fields(rhs_dict, G_rhs_dict, scale=gamma_dt)
        except Exception:
            # Ultimate fallback: return RHS unchanged
            return {key: field.copy() for key, field in rhs_dict.items()}

    def _dict_to_flat(self, field_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Convert field dictionary to flat array for numerical solvers."""
        arrays = []
        for key in ["rho", "Pi", "pi_munu", "q_mu", "u_mu"]:
            if key in field_dict:
                arrays.append(field_dict[key].ravel())
        return np.concatenate(arrays) if arrays else np.array([])

    def _flat_to_dict(
        self, flat_array: np.ndarray, template_dict: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Convert flat array back to field dictionary using template shapes."""
        result = {}
        offset = 0

        for key in ["rho", "Pi", "pi_munu", "q_mu", "u_mu"]:
            if key in template_dict:
                shape = template_dict[key].shape
                size = template_dict[key].size
                result[key] = flat_array[offset : offset + size].reshape(shape)
                offset += size

        return result

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
    def evolve(
        self,
        t_final: float,
        dt: float | None = None,
        method: str = "spectral_imex",
        output_callback: Callable | None = None,
        callback: Callable[[float, "ISFieldConfiguration"], None] | None = None,
        save_trajectory: dict[str, Any] | None = None,
        snapshot_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Evolve hydrodynamics from t=0 to t_final using time stepping.

        Args:
            t_final: Final simulation time
            dt: Timestep (uses adaptive stepping if None)
            method: Integration method ('spectral_imex', 'rk4', etc.)
            output_callback: Optional callback for data output (called every step as callback(t, step, fields))
            callback: Optional simple callback for monitoring (called as callback(t, fields))
            save_trajectory: [DEPRECATED] Use snapshot_config instead. Optional dict for HDF5 saving
            snapshot_config: Optional dict to enable streaming snapshot saving:
                - 'filename': HDF5 file path (required)
                - 'interval': Time interval between snapshots (default: 0.1)
                - 'buffer_size': Snapshots to buffer before flush (default: 10)
                - 'save_initial': Save initial state (default: True)

        Example:
            ```python
            # With monitoring callback
            def monitor(t, fields):
                print(f"t={t:.2f}, rho={np.mean(fields.rho):.3f}")


            hydro.evolve(t_final=10.0, dt=0.01, callback=monitor)

            # With streaming snapshots (recommended)
            hydro.evolve(
                t_final=10.0,
                snapshot_config={
                    "filename": "output.h5",
                    "interval": 0.1,
                    "buffer_size": 20,
                    "save_initial": True,
                },
            )

            # Old method (still works for backward compatibility)
            hydro.evolve(t_final=10.0, save_trajectory={"filename": "output.h5", "interval": 0.5})
            ```
        """
        # Handle new streaming architecture vs old trajectory writer
        stream = None
        trajectory_writer = None

        if snapshot_config is not None:
            # New streaming architecture (Phase 5)
            from ..utils.streaming import SnapshotStream

            filename = snapshot_config.get("filename")
            if filename is None:
                raise ValueError("snapshot_config must include 'filename' key")

            stream = SnapshotStream(
                filename=filename,
                grid=self.grid,
                coeffs=self.coeffs,
                interval=snapshot_config.get("interval", 0.1),
                buffer_size=snapshot_config.get("buffer_size", 10),
            )

            # Save initial state if requested
            if snapshot_config.get("save_initial", True):
                stream.save(0.0, self.fields)

        elif save_trajectory is not None:
            # Old trajectory writer (backward compatibility)
            from ..utils.io import TrajectoryWriter

            filename = save_trajectory.get("filename")
            if filename is None:
                raise ValueError("save_trajectory must include 'filename' key")

            trajectory_writer = TrajectoryWriter(filename, self.grid, self.coeffs)

            # Write initial state if requested
            if save_trajectory.get("save_initial", True):
                trajectory_writer.write_snapshot(0.0, self.fields)

        t = 0.0
        step = 0
        last_snapshot_time = 0.0
        snapshot_interval = save_trajectory.get("interval", 0.1) if save_trajectory else None

        try:
            while t < t_final:
                # Determine time step
                if dt is None:
                    dt_step = self.adaptive_time_step()
                else:
                    dt_step = dt

                dt_step = min(dt_step, t_final - t)  # Don't overshoot

                # Advance one time step with specified method
                self.time_step(dt_step, method=method)

                t += dt_step
                step += 1

                # Call simple monitoring callback
                if callback is not None:
                    callback(t, self.fields)

                # Handle snapshots (new streaming or old trajectory writer)
                if stream is not None:
                    # New streaming architecture with buffering
                    if stream.should_save(t):
                        stream.save(t, self.fields)
                elif trajectory_writer is not None and snapshot_interval is not None:
                    # Old trajectory writer (backward compatibility)
                    if t - last_snapshot_time >= snapshot_interval:
                        trajectory_writer.write_snapshot(t, self.fields)
                        last_snapshot_time = t

                # Output callback (legacy, more complex signature)
                if output_callback is not None:
                    output_callback(t, step, self.fields)

                # Progress reporting
                if step % 100 == 0:
                    logger = get_logger("spectral.evolution")
                    logger.info(
                        f"Evolution progress: Step {step}",
                        extra={
                            "step": step,
                            "time": t,
                            "timestep": dt,
                            "progress_info": f"step_{step}_of_evolution",
                        },
                    )
        finally:
            # Close snapshot stream or trajectory writer
            if stream is not None:
                # New streaming: flush and close
                stream.flush()
                stream.close()
            elif trajectory_writer is not None:
                # Old trajectory writer: write final and close
                if snapshot_interval is None or t - last_snapshot_time > 1e-10:
                    trajectory_writer.write_snapshot(t, self.fields)
                trajectory_writer.close()

    def __str__(self) -> str:
        return f"SpectralISHydrodynamics(grid={self.spectral.nx}x{self.spectral.ny}x{self.spectral.nz})"
