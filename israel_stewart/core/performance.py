"""
Performance monitoring utilities for tensor operations.

This module provides performance tracking and optimization utilities
for relativistic hydrodynamics tensor computations.
"""

import functools
import time
import tracemalloc
import warnings
from collections import defaultdict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import numpy as np




class DetailedProfiler:
    """
    Enhanced profiler with hierarchical timing and memory tracking.

    Provides detailed profiling capabilities for complex operations with
    nested function calls, FFT-specific tracking, and memory allocation analysis.
    """

    def __init__(self) -> None:
        """Initialize detailed profiler."""
        # Hierarchical timing data
        self.call_stack: list[str] = []
        self.nested_timings: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.call_hierarchy: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Memory tracking
        self.memory_snapshots: dict[str, list[tuple[float, float]]] = defaultdict(
            list
        )  # (current, peak)
        self.large_allocations: list[dict[str, Any]] = []  # Track allocations > 10MB

        # FFT-specific tracking
        self.fft_operations: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # Cache performance
        self.cache_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"hits": 0, "misses": 0})

        # Operation metadata
        self.operation_metadata: dict[str, dict[str, Any]] = defaultdict(dict)

        # Timing data for individual operations
        self.operation_times: dict[str, list[float]] = defaultdict(list)
        self.operation_counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def profile_operation(
        self, operation_name: str, metadata: dict[str, Any] | None = None
    ) -> Generator[None, None, None]:
        """
        Context manager for profiling operations with hierarchical tracking.

        Args:
            operation_name: Name of the operation to profile
            metadata: Optional metadata about the operation (grid size, etc.)
        """
        # Record start state
        start_time = time.perf_counter()

        # Memory tracking
        memory_started_here = False
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            memory_started_here = True

        start_memory = tracemalloc.get_traced_memory()

        # Track call hierarchy
        parent_operation = self.call_stack[-1] if self.call_stack else "root"
        self.call_stack.append(operation_name)
        self.call_hierarchy[parent_operation][operation_name] += 1

        # Store metadata
        if metadata:
            self.operation_metadata[operation_name].update(metadata)

        try:
            yield
        finally:
            # Record end state
            end_time = time.perf_counter()
            end_memory = tracemalloc.get_traced_memory()

            # Calculate metrics
            elapsed = end_time - start_time
            memory_delta = end_memory[0] - start_memory[0]  # Current memory change
            peak_memory = end_memory[1]  # Peak memory during operation

            # Record timing data
            self.operation_times[operation_name].append(elapsed)
            self.operation_counts[operation_name] += 1

            # Record hierarchical timing
            if parent_operation != "root":
                self.nested_timings[parent_operation][operation_name].append(elapsed)

            # Record memory snapshots
            self.memory_snapshots[operation_name].append(
                (end_memory[0] / (1024 * 1024), peak_memory / (1024 * 1024))
            )

            # Track large allocations
            if memory_delta > 10 * 1024 * 1024:  # > 10MB
                self.large_allocations.append(
                    {
                        "operation": operation_name,
                        "size_mb": memory_delta / (1024 * 1024),
                        "timestamp": time.time(),
                        "parent": parent_operation,
                    }
                )

            # Clean up
            self.call_stack.pop()
            if memory_started_here:
                tracemalloc.stop()

    def profile_fft_operation(
        self,
        operation_type: str,
        array_shape: tuple[int, ...],
        elapsed_time: float,
        axes: tuple[int, ...] | None = None,
    ) -> None:
        """
        Record FFT operation performance.

        Args:
            operation_type: Type of FFT operation (forward, inverse, etc.)
            array_shape: Shape of array being transformed
            elapsed_time: Time taken for the operation
            axes: FFT axes (optional)
        """
        self.fft_operations[operation_type].append(
            {
                "shape": array_shape,
                "elapsed_time": elapsed_time,
                "axes": axes,
                "size_elements": np.prod(array_shape),
                "timestamp": time.time(),
            }
        )

    def record_cache_hit(self, cache_name: str) -> None:
        """Record a cache hit."""
        self.cache_stats[cache_name]["hits"] += 1

    def record_cache_miss(self, cache_name: str) -> None:
        """Record a cache miss."""
        self.cache_stats[cache_name]["misses"] += 1

    def get_hierarchical_report(self) -> dict[str, Any]:
        """
        Generate comprehensive hierarchical performance report.

        Returns:
            Detailed performance analysis with recommendations
        """
        report = {
            "summary": self._generate_summary(),
            "hierarchical_timing": self._analyze_hierarchical_timing(),
            "memory_analysis": self._analyze_memory_usage(),
            "fft_analysis": self._analyze_fft_performance(),
            "cache_analysis": self._analyze_cache_performance(),
            "optimization_targets": self._identify_optimization_targets(),
            "scaling_analysis": self._analyze_scaling_behavior(),
        }

        return report

    def _generate_summary(self) -> dict[str, Any]:
        """Generate high-level summary statistics."""
        total_operations = sum(self.operation_counts.values())
        total_time = sum(sum(times) for times in self.operation_times.values())

        # Find slowest operations
        slowest_ops = sorted(
            [(op, sum(times)) for op, times in self.operation_times.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        # Find most frequent operations
        frequent_ops = sorted(self.operation_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_operations": total_operations,
            "total_time": total_time,
            "unique_operations": len(self.operation_times),
            "slowest_operations": slowest_ops,
            "most_frequent_operations": frequent_ops,
            "large_allocations_count": len(self.large_allocations),
        }

    def _analyze_hierarchical_timing(self) -> dict[str, Any]:
        """Analyze timing breakdown by operation hierarchy."""
        hierarchy_analysis = {}

        for parent, children in self.nested_timings.items():
            parent_total_time = sum(self.operation_times.get(parent, [0]))

            child_breakdown = {}
            for child, times in children.items():
                child_total_time = sum(times)
                child_percentage = (
                    (child_total_time / parent_total_time * 100) if parent_total_time > 0 else 0
                )
                child_breakdown[child] = {
                    "total_time": child_total_time,
                    "percentage": child_percentage,
                    "call_count": len(times),
                    "avg_time": np.mean(times) if times else 0,
                }

            # Sort children by total time
            sorted_children = dict(
                sorted(child_breakdown.items(), key=lambda x: x[1]["total_time"], reverse=True)
            )

            hierarchy_analysis[parent] = {
                "parent_total_time": parent_total_time,
                "children": sorted_children,
                "call_count": self.operation_counts.get(parent, 0),
            }

        return hierarchy_analysis

    def _analyze_memory_usage(self) -> dict[str, Any]:
        """Analyze memory usage patterns."""
        memory_analysis = {
            "large_allocations": self.large_allocations[-10:],  # Last 10 large allocations
            "memory_hotspots": {},
            "memory_trends": {},
        }

        # Analyze memory hotspots
        for operation, snapshots in self.memory_snapshots.items():
            if snapshots:
                current_memories = [s[0] for s in snapshots]
                peak_memories = [s[1] for s in snapshots]

                memory_analysis["memory_hotspots"][operation] = {
                    "avg_current_mb": np.mean(current_memories),
                    "max_current_mb": np.max(current_memories),
                    "avg_peak_mb": np.mean(peak_memories),
                    "max_peak_mb": np.max(peak_memories),
                    "memory_variance": np.var(current_memories),
                    "sample_count": len(snapshots),
                }

        # Sort by maximum peak memory
        hotspots_sorted = dict(
            sorted(
                memory_analysis["memory_hotspots"].items(),
                key=lambda x: x[1]["max_peak_mb"],
                reverse=True,
            )
        )
        memory_analysis["memory_hotspots"] = hotspots_sorted

        return memory_analysis

    def _analyze_fft_performance(self) -> dict[str, Any]:
        """Analyze FFT operation performance."""
        fft_analysis = {}

        for fft_type, operations in self.fft_operations.items():
            if operations:
                times = [op["elapsed_time"] for op in operations]
                sizes = [op["size_elements"] for op in operations]

                fft_analysis[fft_type] = {
                    "total_operations": len(operations),
                    "total_time": sum(times),
                    "avg_time": np.mean(times),
                    "time_per_element": np.mean(times) / np.mean(sizes) if sizes else 0,
                    "largest_transform": max(sizes) if sizes else 0,
                    "smallest_transform": min(sizes) if sizes else 0,
                }

        return fft_analysis

    def _analyze_cache_performance(self) -> dict[str, Any]:
        """Analyze cache hit rates and efficiency."""
        cache_analysis = {}

        for cache_name, stats in self.cache_stats.items():
            total_requests = stats["hits"] + stats["misses"]
            hit_rate = stats["hits"] / total_requests if total_requests > 0 else 0

            cache_analysis[cache_name] = {
                "hits": stats["hits"],
                "misses": stats["misses"],
                "total_requests": total_requests,
                "hit_rate": hit_rate,
                "efficiency": "good" if hit_rate > 0.8 else "fair" if hit_rate > 0.5 else "poor",
            }

        return cache_analysis

    def _identify_optimization_targets(self) -> list[dict[str, Any]]:
        """Identify top optimization targets based on impact."""
        targets = []

        # Time-based targets
        for operation, times in self.operation_times.items():
            if times:
                total_time = sum(times)
                avg_time = np.mean(times)
                call_count = len(times)

                if total_time > 1.0:  # Operations taking more than 1 second total
                    targets.append(
                        {
                            "operation": operation,
                            "type": "time_bottleneck",
                            "total_time": total_time,
                            "avg_time": avg_time,
                            "call_count": call_count,
                            "impact_score": total_time * call_count,
                            "recommendation": f"High time impact: {total_time:.2f}s total, {avg_time:.3f}s average",
                        }
                    )

        # Memory-based targets
        for allocation in self.large_allocations:
            if allocation["size_mb"] > 50:  # > 50MB allocations
                targets.append(
                    {
                        "operation": allocation["operation"],
                        "type": "memory_bottleneck",
                        "size_mb": allocation["size_mb"],
                        "parent": allocation["parent"],
                        "impact_score": allocation["size_mb"],
                        "recommendation": f"Large allocation: {allocation['size_mb']:.1f}MB",
                    }
                )

        # Frequency-based targets
        for operation, count in self.operation_counts.items():
            if count > 1000:  # Very frequent operations
                avg_time = np.mean(self.operation_times.get(operation, [0]))
                targets.append(
                    {
                        "operation": operation,
                        "type": "frequency_bottleneck",
                        "call_count": count,
                        "avg_time": avg_time,
                        "impact_score": count * avg_time,
                        "recommendation": f"Very frequent: {count} calls, consider caching",
                    }
                )

        # Sort by impact score
        return sorted(targets, key=lambda x: x["impact_score"], reverse=True)

    def _analyze_scaling_behavior(self) -> dict[str, Any]:
        """Analyze how operations scale with problem size."""
        scaling_analysis = {}

        # Group operations by grid size metadata if available
        grid_size_groups: dict[str, dict[tuple[int, ...], list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for operation, metadata in self.operation_metadata.items():
            if "grid_size" in metadata and operation in self.operation_times:
                grid_size = tuple(metadata["grid_size"])
                times = self.operation_times[operation]
                grid_size_groups[operation][grid_size].extend(times)

        # Analyze scaling for operations with multiple grid sizes
        for operation, size_groups in grid_size_groups.items():
            if len(size_groups) > 1:
                sizes_and_times = []
                for grid_size, times in size_groups.items():
                    total_elements = np.prod(grid_size)
                    avg_time = np.mean(times)
                    sizes_and_times.append((total_elements, avg_time))

                sizes_and_times.sort()

                # Estimate scaling exponent (rough approximation)
                if len(sizes_and_times) >= 2:
                    size_ratio = sizes_and_times[-1][0] / sizes_and_times[0][0]
                    time_ratio = sizes_and_times[-1][1] / sizes_and_times[0][1]
                    scaling_exponent = (
                        np.log(time_ratio) / np.log(size_ratio) if size_ratio > 1 else 0
                    )

                    scaling_analysis[operation] = {
                        "scaling_exponent": scaling_exponent,
                        "scaling_behavior": "linear"
                        if 0.8 <= scaling_exponent <= 1.2
                        else "superlinear"
                        if scaling_exponent > 1.2
                        else "sublinear",
                        "data_points": sizes_and_times,
                    }

        return scaling_analysis

    def reset_stats(self) -> None:
        """Reset all profiling statistics."""
        self.call_stack.clear()
        self.nested_timings.clear()
        self.call_hierarchy.clear()
        self.memory_snapshots.clear()
        self.large_allocations.clear()
        self.fft_operations.clear()
        self.cache_stats.clear()
        self.operation_metadata.clear()
        self.operation_times.clear()
        self.operation_counts.clear()


# Global detailed profiler instance
_detailed_profiler = DetailedProfiler()


def get_detailed_profiler() -> DetailedProfiler:
    """Get the global detailed profiler instance."""
    return _detailed_profiler


def monitor_performance(operation_name: str) -> Callable[[Callable], Callable]:
    """
    Decorator for monitoring performance of tensor operations.

    Args:
        operation_name: Name to use for tracking this operation

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with _detailed_profiler.profile_operation(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def profile_operation(operation_name: str, metadata: dict[str, Any] | None = None):
    """
    Context manager for detailed profiling with hierarchical tracking.

    Args:
        operation_name: Name of the operation to profile
        metadata: Optional metadata about the operation

    Returns:
        Context manager for profiling
    """
    return _detailed_profiler.profile_operation(operation_name, metadata)


def performance_report() -> dict[str, Any]:
    """
    Get current performance report.

    Returns:
        Performance statistics and recommendations
    """
    return _detailed_profiler.get_hierarchical_report()


def detailed_performance_report() -> dict[str, Any]:
    """
    Get detailed hierarchical performance report.

    Returns:
        Comprehensive performance analysis
    """
    return _detailed_profiler.get_hierarchical_report()


def reset_performance_stats() -> None:
    """Reset all performance monitoring statistics."""
    _detailed_profiler.reset_stats()


# Optimization utilities
def suggest_einsum_optimization(einsum_string: str, *tensor_shapes: tuple[int, ...]) -> str | None:
    """
    Suggest optimized Einstein summation path.

    Args:
        einsum_string: Einstein summation string
        tensor_shapes: Shapes of input tensors

    Returns:
        Optimization suggestion or None
    """
    try:
        # Try to use opt_einsum if available
        import opt_einsum

        # Create dummy arrays with given shapes
        dummy_arrays = [np.zeros(shape) for shape in tensor_shapes]

        # Get optimized path
        path_info = opt_einsum.contract_path(einsum_string, *dummy_arrays, optimize="optimal")

        if len(path_info) > 1 and hasattr(path_info[1], "largest_intermediate"):
            return f"Use opt_einsum for {einsum_string} - reduces memory by factor of {path_info[1].largest_intermediate:.1e}"

    except ImportError:
        # opt_einsum not available - this is expected in some environments
        pass
    except (ValueError, TypeError) as e:
        # Invalid einsum string or tensor shapes
        warnings.warn(f"Invalid einsum optimization request: {e}", stacklevel=2)
    except AttributeError:
        # path_info doesn't have expected attributes - opt_einsum version mismatch
        pass

    return None




def fft_efficiency_report() -> dict[str, Any]:
    """
    Get FFT efficiency report.

    Returns:
        FFT performance analysis and recommendations
    """
    return _fft_profiler.get_fft_efficiency_report()


def check_tensor_cache_efficiency(
    cache_hits: int, cache_misses: int, cache_name: str = "tensor_cache"
) -> None:
    """
    Check cache efficiency and issue warnings if needed.

    Args:
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        cache_name: Name of cache for warning messages
    """
    total_requests = cache_hits + cache_misses
    if total_requests > 100:  # Only check efficiency after reasonable sample size
        hit_rate = cache_hits / total_requests
        if hit_rate < 0.5:  # Less than 50% hit rate
            warnings.warn(
                f"Low cache efficiency for {cache_name}: {hit_rate:.1%} hit rate "
                f"({cache_hits} hits, {cache_misses} misses)",
                stacklevel=2,
            )
