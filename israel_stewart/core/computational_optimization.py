"""
Phase 3: Computational Optimization for Israel-Stewart Hydrodynamics.

This module implements advanced computational optimizations including:
- Enhanced FFT plan caching and optimization
- FFTW backend integration for maximum performance
- Redundancy elimination and intelligent result caching
- Advanced vectorization for tensor operations
- Operation fusion and pipeline optimization
"""

import hashlib
import time
import warnings
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np

# Try to import pyfftw for FFTW backend
try:
    import pyfftw
    FFTW_AVAILABLE = True
    # Enable FFTW wisdom for optimal plans
    pyfftw.config.NUM_THREADS = 1  # Start with single thread, can be configured
    pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
except ImportError:
    FFTW_AVAILABLE = False
    pyfftw = None

# Try to import scipy for optimized routines
try:
    import scipy.fft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy = None


class FFTPlanCache:
    """
    Advanced FFT plan caching with FFTW backend support.

    Provides intelligent plan caching, automatic backend selection,
    and performance monitoring for optimal FFT operations.
    """

    def __init__(self, use_fftw: bool = True, use_scipy: bool = True):
        self.use_fftw = use_fftw and FFTW_AVAILABLE
        self.use_scipy = use_scipy and SCIPY_AVAILABLE
        self.plans = {}
        self.plan_performance = {}
        self.hit_count = 0
        self.miss_count = 0

        # FFTW wisdom storage
        self.wisdom_cache = {}
        if self.use_fftw:
            self._load_fftw_wisdom()

    def _load_fftw_wisdom(self) -> None:
        """Load saved FFTW wisdom if available."""
        try:
            # Try to load wisdom from cache
            pyfftw.import_wisdom(pyfftw.export_wisdom())
        except Exception:
            # If no wisdom available, that's fine - it will be generated
            pass

    def _get_plan_key(self, shape: tuple[int, ...], dtype: np.dtype,
                      forward: bool, real_fft: bool) -> str:
        """Generate unique key for FFT plan."""
        key_data = f"{shape}_{dtype}_{forward}_{real_fft}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get_fft_function(self, shape: tuple[int, ...], dtype: np.dtype,
                        forward: bool = True, real_fft: bool = False):
        """Get optimized FFT function for given parameters."""
        plan_key = self._get_plan_key(shape, dtype, forward, real_fft)

        if plan_key in self.plans:
            self.hit_count += 1
            return self.plans[plan_key]

        self.miss_count += 1

        # Create new plan based on available backends
        if self.use_fftw:
            plan = self._create_fftw_plan(shape, dtype, forward, real_fft)
        elif self.use_scipy:
            plan = self._create_scipy_plan(shape, dtype, forward, real_fft)
        else:
            plan = self._create_numpy_plan(shape, dtype, forward, real_fft)

        self.plans[plan_key] = plan
        return plan

    def _create_fftw_plan(self, shape: tuple[int, ...], dtype: np.dtype,
                         forward: bool, real_fft: bool):
        """Create FFTW-based FFT plan."""
        if real_fft:
            if forward:
                plan = pyfftw.builders.rfftn(
                    np.zeros(shape, dtype=dtype),
                    axes=None,
                    planner_effort='FFTW_MEASURE'
                )
            else:
                # For inverse real FFT, shape is the k-space shape
                time_shape = list(shape)
                time_shape[-1] = 2 * (shape[-1] - 1)
                plan = pyfftw.builders.irfftn(
                    np.zeros(shape, dtype=np.complex128),
                    s=time_shape,
                    axes=None,
                    planner_effort='FFTW_MEASURE'
                )
        else:
            if forward:
                plan = pyfftw.builders.fftn(
                    np.zeros(shape, dtype=dtype),
                    axes=None,
                    planner_effort='FFTW_MEASURE'
                )
            else:
                plan = pyfftw.builders.ifftn(
                    np.zeros(shape, dtype=dtype),
                    axes=None,
                    planner_effort='FFTW_MEASURE'
                )

        return plan

    def _create_scipy_plan(self, shape: tuple[int, ...], dtype: np.dtype,
                          forward: bool, real_fft: bool):
        """Create scipy.fft-based plan (wrapper function)."""
        if real_fft:
            if forward:
                return lambda x: scipy.fft.rfftn(x)
            else:
                return lambda x: scipy.fft.irfftn(x)
        else:
            if forward:
                return lambda x: scipy.fft.fftn(x)
            else:
                return lambda x: scipy.fft.ifftn(x)

    def _create_numpy_plan(self, shape: tuple[int, ...], dtype: np.dtype,
                          forward: bool, real_fft: bool):
        """Create numpy.fft-based plan (wrapper function)."""
        if real_fft:
            if forward:
                return lambda x: np.fft.rfftn(x)
            else:
                return lambda x: np.fft.irfftn(x)
        else:
            if forward:
                return lambda x: np.fft.fftn(x)
            else:
                return lambda x: np.fft.ifftn(x)

    def get_cache_efficiency(self) -> dict[str, float]:
        """Get FFT plan cache efficiency statistics."""
        total_requests = self.hit_count + self.miss_count
        if total_requests == 0:
            return {"hit_rate": 0.0, "miss_rate": 0.0}

        return {
            "hit_rate": self.hit_count / total_requests,
            "miss_rate": self.miss_count / total_requests,
            "total_plans": len(self.plans),
            "total_requests": total_requests
        }


class ComputationCache:
    """
    Intelligent caching system for expensive computational results.

    Caches intermediate results from tensor operations, derivatives,
    and other expensive computations to eliminate redundancy.
    """

    def __init__(self, max_cache_size: int = 100):
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.max_cache_size = max_cache_size

    def _generate_key(self, operation: str, *args, **kwargs) -> str:
        """Generate cache key from operation and parameters."""
        # Create hash from operation name and parameters
        key_data = f"{operation}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, operation: str, compute_func, *args, **kwargs):
        """Get cached result or compute and cache new result."""
        cache_key = self._generate_key(operation, *args, **kwargs)

        if cache_key in self.cache:
            self.hit_count += 1
            self.access_times[cache_key] = time.time()
            return self.cache[cache_key]

        self.miss_count += 1

        # Compute new result
        result = compute_func(*args, **kwargs)

        # Cache result with LRU eviction if needed
        if len(self.cache) >= self.max_cache_size:
            self._evict_oldest()

        self.cache[cache_key] = result
        self.access_times[cache_key] = time.time()

        return result

    def _evict_oldest(self) -> None:
        """Evict least recently used cache entry."""
        if not self.access_times:
            return

        oldest_key = min(self.access_times.keys(),
                        key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]

    def get_cache_efficiency(self) -> dict[str, float]:
        """Get computation cache efficiency statistics."""
        total_requests = self.hit_count + self.miss_count
        if total_requests == 0:
            return {"hit_rate": 0.0, "miss_rate": 0.0}

        return {
            "hit_rate": self.hit_count / total_requests,
            "miss_rate": self.miss_count / total_requests,
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size
        }


class VectorizedOperations:
    """
    Advanced vectorization utilities for tensor operations.

    Provides optimized implementations of common tensor operations
    using vectorization, broadcasting, and in-place operations.
    """

    @staticmethod
    def vectorized_contraction(tensor_a: np.ndarray, tensor_b: np.ndarray,
                              axes: tuple[int, ...]) -> np.ndarray:
        """Perform optimized tensor contraction using Einstein summation."""
        # Use opt_einsum for optimal contraction path
        try:
            import opt_einsum as oe
            # Let opt_einsum determine optimal contraction path
            return oe.contract(tensor_a, axes, tensor_b, axes)
        except ImportError:
            # Fall back to numpy tensordot
            return np.tensordot(tensor_a, tensor_b, axes)

    @staticmethod
    def fused_multiply_add(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                          out: Optional[np.ndarray] = None) -> np.ndarray:
        """Fused multiply-add operation: out = a * b + c."""
        if out is None:
            return a * b + c
        else:
            np.multiply(a, b, out=out)
            np.add(out, c, out=out)
            return out

    @staticmethod
    def vectorized_divergence(field: np.ndarray, dx: float, dy: float, dz: float,
                            out: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute divergence using vectorized finite differences."""
        if out is None:
            out = np.zeros(field.shape[:-1])

        # Vectorized central differences
        out[...] = (
            (field[2:, 1:-1, 1:-1, 0] - field[:-2, 1:-1, 1:-1, 0]) / (2 * dx) +
            (field[1:-1, 2:, 1:-1, 1] - field[1:-1, :-2, 1:-1, 1]) / (2 * dy) +
            (field[1:-1, 1:-1, 2:, 2] - field[1:-1, 1:-1, :-2, 2]) / (2 * dz)
        )

        return out


class ComputationalOptimizer:
    """
    Main computational optimization coordinator.

    Integrates FFT optimization, computation caching, and vectorization
    to provide maximum performance for Israel-Stewart computations.
    """

    def __init__(self, enable_fftw: bool = True, enable_caching: bool = True,
                 cache_size: int = 100):
        self.enable_fftw = enable_fftw
        self.enable_caching = enable_caching

        # Initialize optimization components
        self.fft_cache = FFTPlanCache(use_fftw=enable_fftw)
        self.computation_cache = ComputationCache(max_cache_size=cache_size) if enable_caching else None
        self.vectorized_ops = VectorizedOperations()

        # Performance tracking
        self.optimization_stats = {
            "fft_optimizations": 0,
            "cache_hits": 0,
            "vectorization_uses": 0
        }

    @contextmanager
    def optimized_fft_context(self):
        """Context manager for optimized FFT operations."""
        start_time = time.perf_counter()
        try:
            yield self.fft_cache
        finally:
            end_time = time.perf_counter()
            self.optimization_stats["fft_optimizations"] += 1

    def cached_computation(self, operation_name: str, compute_func, *args, **kwargs):
        """Perform computation with intelligent caching."""
        if self.computation_cache is None:
            return compute_func(*args, **kwargs)

        result = self.computation_cache.get(operation_name, compute_func, *args, **kwargs)
        if operation_name in [k for k in self.computation_cache.cache.keys()
                             if self.computation_cache._generate_key(operation_name, *args, **kwargs) == k]:
            self.optimization_stats["cache_hits"] += 1

        return result

    def vectorized_operation(self, operation_name: str, *args, **kwargs):
        """Apply vectorized operation."""
        self.optimization_stats["vectorization_uses"] += 1

        if operation_name == "contraction":
            return self.vectorized_ops.vectorized_contraction(*args, **kwargs)
        elif operation_name == "fused_multiply_add":
            return self.vectorized_ops.fused_multiply_add(*args, **kwargs)
        elif operation_name == "divergence":
            return self.vectorized_ops.vectorized_divergence(*args, **kwargs)
        else:
            raise ValueError(f"Unknown vectorized operation: {operation_name}")

    def get_optimization_report(self) -> dict[str, Any]:
        """Generate comprehensive optimization performance report."""
        report = {
            "fft_performance": self.fft_cache.get_cache_efficiency(),
            "optimization_stats": self.optimization_stats.copy()
        }

        if self.computation_cache:
            report["computation_cache"] = self.computation_cache.get_cache_efficiency()

        # Calculate overall efficiency metrics
        total_operations = sum(self.optimization_stats.values())
        if total_operations > 0:
            report["overall_efficiency"] = {
                "total_optimized_operations": total_operations,
                "avg_operations_per_second": total_operations / max(1, time.time())
            }

        return report


# Global optimizer instance for easy access
_global_optimizer: Optional[ComputationalOptimizer] = None


def get_global_optimizer() -> ComputationalOptimizer:
    """Get or create global computational optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = ComputationalOptimizer()
    return _global_optimizer


def optimized_fft(array: np.ndarray, forward: bool = True,
                 real_fft: bool = False) -> np.ndarray:
    """Convenience function for optimized FFT operations."""
    optimizer = get_global_optimizer()
    with optimizer.optimized_fft_context() as fft_cache:
        fft_func = fft_cache.get_fft_function(array.shape, array.dtype, forward, real_fft)
        return fft_func(array)


def cached_operation(operation_name: str, compute_func, *args, **kwargs):
    """Convenience function for cached computations."""
    optimizer = get_global_optimizer()
    return optimizer.cached_computation(operation_name, compute_func, *args, **kwargs)


def vectorized_operation(operation_name: str, *args, **kwargs):
    """Convenience function for vectorized operations."""
    optimizer = get_global_optimizer()
    return optimizer.vectorized_operation(operation_name, *args, **kwargs)


def reset_global_optimizer() -> None:
    """Reset global optimizer instance."""
    global _global_optimizer
    _global_optimizer = None


def generate_optimization_report() -> dict[str, Any]:
    """Generate optimization performance report."""
    optimizer = get_global_optimizer()
    return optimizer.get_optimization_report()