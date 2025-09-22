"""
Memory optimization utilities for Israel-Stewart hydrodynamics solver.

This module provides memory management tools to reduce allocation overhead
and improve performance for large-scale relativistic hydrodynamics simulations.
"""

import warnings
from typing import Any, Dict, Tuple, Optional, Union
import numpy as np
from contextlib import contextmanager


class ArrayPool:
    """
    Memory pool for reusing numpy arrays to reduce allocation overhead.

    Manages a pool of pre-allocated arrays of different shapes and dtypes
    to avoid repeated memory allocation during computations.
    """

    def __init__(self, max_pool_size: int = 100):
        """
        Initialize array pool.

        Args:
            max_pool_size: Maximum number of arrays to keep in pool
        """
        self.max_pool_size = max_pool_size
        self.pool: Dict[Tuple[Tuple[int, ...], str], list[np.ndarray]] = {}
        self.allocation_count = 0
        self.reuse_count = 0

    def get_array(self, shape: Tuple[int, ...], dtype: Union[str, np.dtype] = np.float64) -> np.ndarray:
        """
        Get array from pool or create new one.

        Args:
            shape: Shape of array needed
            dtype: Data type of array

        Returns:
            Array from pool or newly allocated
        """
        dtype_str = str(np.dtype(dtype))
        key = (shape, dtype_str)

        if key in self.pool and self.pool[key]:
            # Reuse existing array
            array = self.pool[key].pop()
            array.fill(0)  # Clear previous data
            self.reuse_count += 1
            return array
        else:
            # Create new array
            self.allocation_count += 1
            return np.zeros(shape, dtype=dtype)

    def return_array(self, array: np.ndarray) -> None:
        """
        Return array to pool for reuse.

        Args:
            array: Array to return to pool
        """
        key = (array.shape, str(array.dtype))

        if key not in self.pool:
            self.pool[key] = []

        # Only store if pool not full
        if len(self.pool[key]) < self.max_pool_size:
            self.pool[key].append(array)

    def get_efficiency_stats(self) -> Dict[str, Any]:
        """Get pool efficiency statistics."""
        total_requests = self.allocation_count + self.reuse_count
        reuse_rate = self.reuse_count / total_requests if total_requests > 0 else 0

        return {
            "total_requests": total_requests,
            "allocations": self.allocation_count,
            "reuses": self.reuse_count,
            "reuse_rate": reuse_rate,
            "pool_sizes": {str(key): len(arrays) for key, arrays in self.pool.items()},
            "total_pooled_arrays": sum(len(arrays) for arrays in self.pool.values())
        }

    def clear_pool(self) -> None:
        """Clear all arrays from pool."""
        self.pool.clear()
        self.allocation_count = 0
        self.reuse_count = 0


class FFTWorkspaceManager:
    """
    Manager for FFT workspaces and plans to optimize spectral operations.

    Pre-allocates FFT workspaces and manages FFT plans for repeated operations
    to minimize memory allocation overhead in spectral methods.
    """

    def __init__(self):
        """Initialize FFT workspace manager."""
        self.workspaces: Dict[Tuple[Tuple[int, ...], str], np.ndarray] = {}
        self.fft_plans: Dict[Tuple[Tuple[int, ...], str], Any] = {}
        self.usage_stats: Dict[str, int] = {"hits": 0, "misses": 0}

    def get_workspace(self, shape: Tuple[int, ...], dtype: Union[str, np.dtype] = np.complex128) -> np.ndarray:
        """
        Get pre-allocated workspace for FFT operations.

        Args:
            shape: Shape of workspace needed
            dtype: Data type of workspace

        Returns:
            Pre-allocated workspace array
        """
        dtype_str = str(np.dtype(dtype))
        key = (shape, dtype_str)

        if key in self.workspaces:
            self.usage_stats["hits"] += 1
            workspace = self.workspaces[key]
            workspace.fill(0)  # Clear previous data
            return workspace
        else:
            self.usage_stats["misses"] += 1
            # Create and cache new workspace
            workspace = np.zeros(shape, dtype=dtype)
            self.workspaces[key] = workspace
            return workspace

    def get_real_fft_workspace(self, real_shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get workspaces for real FFT operations.

        Args:
            real_shape: Shape of real input array

        Returns:
            Tuple of (real_workspace, complex_workspace)
        """
        # Real workspace
        real_workspace = self.get_workspace(real_shape, np.float64)

        # Complex workspace (reduced size for real FFT)
        complex_shape = list(real_shape)
        complex_shape[-1] = real_shape[-1] // 2 + 1
        complex_workspace = self.get_workspace(tuple(complex_shape), np.complex128)

        return real_workspace, complex_workspace

    def precompute_fft_plans(self, shapes: list[Tuple[int, ...]], use_scipy: bool = True) -> None:
        """
        Precompute FFT plans for given shapes.

        Args:
            shapes: List of array shapes to precompute plans for
            use_scipy: Whether to use scipy.fft for better performance
        """
        if use_scipy:
            try:
                import scipy.fft
                fft_module = scipy.fft
            except ImportError:
                warnings.warn("scipy not available, falling back to numpy.fft")
                fft_module = np.fft
        else:
            fft_module = np.fft

        for shape in shapes:
            # Create dummy array for plan computation
            dummy_array = np.zeros(shape, dtype=np.float64)

            # Store plan information (in real implementation, would cache actual plans)
            plan_key = (shape, "forward")
            self.fft_plans[plan_key] = {
                "module": fft_module,
                "shape": shape,
                "type": "forward"
            }

            # Also store inverse plan
            plan_key = (shape, "inverse")
            self.fft_plans[plan_key] = {
                "module": fft_module,
                "shape": shape,
                "type": "inverse"
            }

    def get_efficiency_stats(self) -> Dict[str, Any]:
        """Get workspace efficiency statistics."""
        total_requests = self.usage_stats["hits"] + self.usage_stats["misses"]
        hit_rate = self.usage_stats["hits"] / total_requests if total_requests > 0 else 0

        return {
            "total_requests": total_requests,
            "cache_hits": self.usage_stats["hits"],
            "cache_misses": self.usage_stats["misses"],
            "hit_rate": hit_rate,
            "workspaces_cached": len(self.workspaces),
            "fft_plans_cached": len(self.fft_plans),
            "memory_usage_mb": sum(arr.nbytes for arr in self.workspaces.values()) / (1024 * 1024)
        }

    def clear_cache(self) -> None:
        """Clear all cached workspaces and plans."""
        self.workspaces.clear()
        self.fft_plans.clear()
        self.usage_stats = {"hits": 0, "misses": 0}


class InPlaceOperations:
    """
    Utilities for in-place array operations to minimize memory allocation.

    Provides methods for performing common tensor operations in-place
    to reduce memory usage and improve cache performance.
    """

    @staticmethod
    def add_inplace(target: np.ndarray, source: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        Perform in-place addition: target += alpha * source.

        Args:
            target: Target array to modify
            source: Source array to add
            alpha: Scaling factor

        Returns:
            Modified target array
        """
        if alpha == 1.0:
            target += source
        else:
            target += alpha * source
        return target

    @staticmethod
    def multiply_inplace(target: np.ndarray, factor: Union[float, np.ndarray]) -> np.ndarray:
        """
        Perform in-place multiplication: target *= factor.

        Args:
            target: Target array to modify
            factor: Scaling factor or array

        Returns:
            Modified target array
        """
        target *= factor
        return target

    @staticmethod
    def apply_operator_inplace(target: np.ndarray, operator: np.ndarray,
                              workspace: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply operator to array in-place using optional workspace.

        Args:
            target: Target array to modify
            operator: Operator array
            workspace: Optional workspace array

        Returns:
            Modified target array
        """
        if workspace is not None:
            # Use workspace to avoid temporary allocation
            np.multiply(target, operator, out=workspace)
            target[:] = workspace
        else:
            target *= operator
        return target

    @staticmethod
    def copy_with_slicing(target: np.ndarray, source: np.ndarray,
                         target_slice: Any = None, source_slice: Any = None) -> None:
        """
        Copy data between arrays with slicing to avoid full array allocation.

        Args:
            target: Target array
            source: Source array
            target_slice: Slice for target array
            source_slice: Slice for source array
        """
        if target_slice is None:
            target_slice = slice(None)
        if source_slice is None:
            source_slice = slice(None)

        target[target_slice] = source[source_slice]


class MemoryOptimizedContext:
    """
    Context manager for memory-optimized operations.

    Provides access to array pools and workspace managers within a context
    for automatic cleanup and resource management.
    """

    def __init__(self, array_pool_size: int = 100, enable_fft_workspace: bool = True):
        """
        Initialize memory optimization context.

        Args:
            array_pool_size: Size of array pool
            enable_fft_workspace: Whether to enable FFT workspace management
        """
        self.array_pool = ArrayPool(array_pool_size)
        self.fft_manager = FFTWorkspaceManager() if enable_fft_workspace else None
        self.in_place_ops = InPlaceOperations()

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and cleanup resources."""
        # Get efficiency statistics before cleanup
        stats = self.get_memory_stats()

        # Issue warnings for poor efficiency
        if stats["array_pool"]["reuse_rate"] < 0.5 and stats["array_pool"]["total_requests"] > 100:
            warnings.warn(
                f"Low array pool efficiency: {stats['array_pool']['reuse_rate']:.1%} reuse rate"
            )

        if self.fft_manager and stats["fft_workspace"]["hit_rate"] < 0.7:
            warnings.warn(
                f"Low FFT workspace efficiency: {stats['fft_workspace']['hit_rate']:.1%} hit rate"
            )

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory optimization statistics."""
        stats = {
            "array_pool": self.array_pool.get_efficiency_stats(),
            "fft_workspace": self.fft_manager.get_efficiency_stats() if self.fft_manager else {}
        }
        return stats

    @contextmanager
    def temporary_array(self, shape: Tuple[int, ...], dtype: Union[str, np.dtype] = np.float64):
        """
        Context manager for temporary arrays that are automatically returned to pool.

        Args:
            shape: Shape of temporary array
            dtype: Data type of array
        """
        array = self.array_pool.get_array(shape, dtype)
        try:
            yield array
        finally:
            self.array_pool.return_array(array)


# Global instances for convenience
_global_array_pool = ArrayPool()
_global_fft_manager = FFTWorkspaceManager()
_global_inplace_ops = InPlaceOperations()


def get_array_pool() -> ArrayPool:
    """Get global array pool instance."""
    return _global_array_pool


def get_fft_manager() -> FFTWorkspaceManager:
    """Get global FFT workspace manager."""
    return _global_fft_manager


def get_inplace_ops() -> InPlaceOperations:
    """Get global in-place operations utility."""
    return _global_inplace_ops


@contextmanager
def memory_optimized_context(array_pool_size: int = 100, enable_fft_workspace: bool = True):
    """
    Global context manager for memory optimization.

    Args:
        array_pool_size: Size of array pool
        enable_fft_workspace: Whether to enable FFT workspace management
    """
    context = MemoryOptimizedContext(array_pool_size, enable_fft_workspace)
    with context:
        yield context


def memory_usage_report() -> Dict[str, Any]:
    """Generate comprehensive memory usage report."""
    return {
        "array_pool": _global_array_pool.get_efficiency_stats(),
        "fft_workspace": _global_fft_manager.get_efficiency_stats()
    }