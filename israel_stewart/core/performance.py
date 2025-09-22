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


class PerformanceMonitor:
    """
    Monitor performance of tensor operations and provide optimization hints.

    Tracks computation time, memory usage, and operation counts to help
    identify bottlenecks in relativistic hydrodynamics calculations.
    """

    def __init__(self) -> None:
        """Initialize performance monitoring."""
        self.operation_counts: dict[str, int] = {}
        self.timing_data: dict[str, list[float]] = {}
        self.memory_usage: dict[str, list[float]] = {}
        self.warnings_issued: dict[str, bool] = {}

    def time_operation(self, operation_name: str) -> Callable[[Callable], Callable]:
        """
        Decorator to time tensor operations.

        Args:
            operation_name: Name of operation to track

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Start timing and memory tracking
                start_time = time.perf_counter()

                # Start memory tracking if tracemalloc is not already started
                memory_started_here = False
                if not tracemalloc.is_tracing():
                    tracemalloc.start()
                    memory_started_here = True

                start_memory = tracemalloc.get_traced_memory()[0]  # current memory

                # Execute function
                result = func(*args, **kwargs)

                # Stop timing and memory tracking
                end_time = time.perf_counter()
                end_memory = tracemalloc.get_traced_memory()[0]  # current memory

                # Clean up memory tracing if we started it
                if memory_started_here:
                    tracemalloc.stop()

                # Record timing
                elapsed = end_time - start_time
                if operation_name not in self.timing_data:
                    self.timing_data[operation_name] = []
                self.timing_data[operation_name].append(elapsed)

                # Record memory usage delta
                memory_delta = end_memory - start_memory
                if operation_name not in self.memory_usage:
                    self.memory_usage[operation_name] = []
                self.memory_usage[operation_name].append(memory_delta)

                # Update operation count
                self.operation_counts[operation_name] = (
                    self.operation_counts.get(operation_name, 0) + 1
                )

                # Issue performance warnings if needed
                self._check_performance_warnings(operation_name, elapsed, memory_delta)

                return result

            return wrapper

        return decorator

    def _check_performance_warnings(
        self, operation_name: str, elapsed_time: float, memory_delta: float = 0.0
    ) -> None:
        """Check if performance warnings should be issued."""
        # Warn about slow operations (>1 second)
        if elapsed_time > 1.0 and not self.warnings_issued.get(f"{operation_name}_slow", False):
            warnings.warn(
                f"Operation {operation_name} took {elapsed_time:.2f}s - consider optimization",
                stacklevel=2,
            )
            self.warnings_issued[f"{operation_name}_slow"] = True

        # Warn about high memory usage operations (>100MB)
        memory_mb = memory_delta / (1024 * 1024)
        if memory_mb > 100 and not self.warnings_issued.get(f"{operation_name}_memory", False):
            warnings.warn(
                f"Operation {operation_name} used {memory_mb:.1f}MB - consider memory optimization",
                stacklevel=2,
            )
            self.warnings_issued[f"{operation_name}_memory"] = True

        # Warn about very frequent operations (>1000 calls)
        call_count = self.operation_counts.get(operation_name, 0)
        if call_count > 1000 and not self.warnings_issued.get(f"{operation_name}_frequent", False):
            warnings.warn(
                f"Operation {operation_name} called {call_count} times - consider caching",
                stacklevel=2,
            )
            self.warnings_issued[f"{operation_name}_frequent"] = True

    def get_performance_report(self) -> dict[str, Any]:
        """
        Generate performance report.

        Returns:
            Dictionary with performance statistics
        """
        report: dict[str, Any] = {
            "operation_counts": self.operation_counts.copy(),
            "timing_statistics": {},
            "memory_statistics": {},
            "recommendations": [],
        }

        # Compute timing statistics
        for op_name, times in self.timing_data.items():
            if times:
                times_array = np.array(times)
                report["timing_statistics"][op_name] = {
                    "mean_time": float(np.mean(times_array)),
                    "std_time": float(np.std(times_array)),
                    "min_time": float(np.min(times_array)),
                    "max_time": float(np.max(times_array)),
                    "total_time": float(np.sum(times_array)),
                    "call_count": len(times),
                }

        # Compute memory statistics
        for op_name, memory_deltas in self.memory_usage.items():
            if memory_deltas:
                memory_array = np.array(memory_deltas)
                # Convert bytes to MB for readability
                memory_mb = memory_array / (1024 * 1024)
                report["memory_statistics"][op_name] = {
                    "mean_memory_mb": float(np.mean(memory_mb)),
                    "std_memory_mb": float(np.std(memory_mb)),
                    "min_memory_mb": float(np.min(memory_mb)),
                    "max_memory_mb": float(np.max(memory_mb)),
                    "total_memory_mb": float(np.sum(memory_mb)),
                    "call_count": len(memory_deltas),
                }

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations()

        return report

    def _generate_recommendations(self) -> list:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Check for frequently called operations
        for op_name, count in self.operation_counts.items():
            if count > 100:
                recommendations.append(
                    f"Consider caching results for frequently called operation: {op_name} ({count} calls)"
                )

        # Check for slow operations
        for op_name, times in self.timing_data.items():
            if times:
                mean_time = np.mean(times)
                if mean_time > 0.1:
                    recommendations.append(
                        f"Slow operation detected: {op_name} (avg {mean_time:.3f}s)"
                    )

        # Check for memory-intensive operations
        for op_name, memory_deltas in self.memory_usage.items():
            if memory_deltas:
                mean_memory_mb = np.mean(memory_deltas) / (1024 * 1024)
                if mean_memory_mb > 50:  # More than 50MB average
                    recommendations.append(
                        f"Memory-intensive operation: {op_name} (avg {mean_memory_mb:.1f}MB per call)"
                    )

                # Check for operations with high memory variance (potential leaks)
                std_memory_mb = np.std(memory_deltas) / (1024 * 1024)
                if std_memory_mb > mean_memory_mb * 2:  # High variance
                    recommendations.append(
                        f"Unstable memory usage in {op_name} - check for memory leaks or inefficient allocation"
                    )

        return recommendations

    def reset_stats(self) -> None:
        """Reset all performance statistics."""
        self.operation_counts.clear()
        self.timing_data.clear()
        self.memory_usage.clear()
        self.warnings_issued.clear()


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


# Global performance monitor instances
_global_monitor = PerformanceMonitor()
_detailed_profiler = DetailedProfiler()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _global_monitor


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
    return _global_monitor.time_operation(operation_name)


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
    return _global_monitor.get_performance_report()


def detailed_performance_report() -> dict[str, Any]:
    """
    Get detailed hierarchical performance report.

    Returns:
        Comprehensive performance analysis
    """
    return _detailed_profiler.get_hierarchical_report()


def reset_performance_stats() -> None:
    """Reset all performance monitoring statistics."""
    _global_monitor.reset_stats()
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


class FFTProfiler:
    """
    Specialized profiler for FFT operations with performance analysis.

    Wraps numpy.fft operations to provide detailed timing and efficiency analysis
    for spectral methods in Israel-Stewart hydrodynamics.
    """

    def __init__(self, detailed_profiler: DetailedProfiler | None = None) -> None:
        """
        Initialize FFT profiler.

        Args:
            detailed_profiler: Optional detailed profiler instance to use
        """
        self.detailed_profiler = detailed_profiler or _detailed_profiler
        self.fft_cache: dict[tuple[int, ...], Any] = {}  # Cache FFT plans/wisdom
        self.cache_enabled = True

    def profile_fft(
        self, array: np.ndarray, axes: tuple[int, ...] | None = None, norm: str | None = None
    ) -> np.ndarray:
        """
        Profile forward FFT operation.

        Args:
            array: Input array for FFT
            axes: Axes over which to compute FFT
            norm: Normalization mode

        Returns:
            FFT result
        """
        start_time = time.perf_counter()

        # Check cache
        cache_key = (array.shape, axes, norm)
        if self.cache_enabled and cache_key in self.fft_cache:
            self.detailed_profiler.record_cache_hit("fft_plan_cache")
        else:
            self.detailed_profiler.record_cache_miss("fft_plan_cache")

        # Perform FFT
        result = np.fft.fftn(array, axes=axes, norm=norm)

        # Record performance
        elapsed_time = time.perf_counter() - start_time
        self.detailed_profiler.profile_fft_operation("forward_fft", array.shape, elapsed_time, axes)

        return result

    def profile_ifft(
        self, array: np.ndarray, axes: tuple[int, ...] | None = None, norm: str | None = None
    ) -> np.ndarray:
        """
        Profile inverse FFT operation.

        Args:
            array: Input array for inverse FFT
            axes: Axes over which to compute inverse FFT
            norm: Normalization mode

        Returns:
            Inverse FFT result
        """
        start_time = time.perf_counter()

        # Check cache
        cache_key = (array.shape, axes, norm)
        if self.cache_enabled and cache_key in self.fft_cache:
            self.detailed_profiler.record_cache_hit("ifft_plan_cache")
        else:
            self.detailed_profiler.record_cache_miss("ifft_plan_cache")

        # Perform inverse FFT
        result = np.fft.ifftn(array, axes=axes, norm=norm)

        # Record performance
        elapsed_time = time.perf_counter() - start_time
        self.detailed_profiler.profile_fft_operation("inverse_fft", array.shape, elapsed_time, axes)

        return result

    def profile_rfft(
        self, array: np.ndarray, axis: int = -1, norm: str | None = None
    ) -> np.ndarray:
        """
        Profile real-valued forward FFT operation.

        Args:
            array: Input real array for FFT
            axis: Axis over which to compute FFT
            norm: Normalization mode

        Returns:
            Real FFT result
        """
        start_time = time.perf_counter()

        # Perform real FFT
        result = np.fft.rfft(array, axis=axis, norm=norm)

        # Record performance
        elapsed_time = time.perf_counter() - start_time
        self.detailed_profiler.profile_fft_operation("real_fft", array.shape, elapsed_time, (axis,))

        return result

    def profile_irfft(
        self, array: np.ndarray, n: int | None = None, axis: int = -1, norm: str | None = None
    ) -> np.ndarray:
        """
        Profile real-valued inverse FFT operation.

        Args:
            array: Input complex array for inverse real FFT
            n: Length of transformed axis
            axis: Axis over which to compute inverse FFT
            norm: Normalization mode

        Returns:
            Inverse real FFT result
        """
        start_time = time.perf_counter()

        # Perform inverse real FFT
        result = np.fft.irfft(array, n=n, axis=axis, norm=norm)

        # Record performance
        elapsed_time = time.perf_counter() - start_time
        self.detailed_profiler.profile_fft_operation(
            "inverse_real_fft", array.shape, elapsed_time, (axis,)
        )

        return result

    def get_fft_efficiency_report(self) -> dict[str, Any]:
        """
        Generate FFT-specific efficiency report.

        Returns:
            FFT performance analysis and recommendations
        """
        fft_analysis = self.detailed_profiler._analyze_fft_performance()
        cache_analysis = self.detailed_profiler._analyze_cache_performance()

        # Compute efficiency metrics
        efficiency_report = {
            "fft_operations": fft_analysis,
            "cache_performance": {
                k: v for k, v in cache_analysis.items() if "fft" in k.lower() or "plan" in k.lower()
            },
            "recommendations": [],
        }

        # Generate FFT-specific recommendations
        recommendations = []

        # Check for inefficient FFT sizes
        for fft_type, data in fft_analysis.items():
            if data.get("time_per_element", 0) > 1e-6:  # > 1 μs per element
                recommendations.append(
                    f"Slow {fft_type} detected: {data['time_per_element']:.2e}s per element. "
                    "Consider using power-of-2 sizes or FFTW backend."
                )

        # Check cache efficiency
        for cache_name, stats in cache_analysis.items():
            if "fft" in cache_name.lower() and stats.get("hit_rate", 0) < 0.7:
                recommendations.append(
                    f"Low FFT cache efficiency: {stats['hit_rate']:.1%} hit rate. "
                    "Consider pre-computing FFT plans or enabling wisdom."
                )

        efficiency_report["recommendations"] = recommendations

        return efficiency_report

    def enable_caching(self) -> None:
        """Enable FFT plan caching."""
        self.cache_enabled = True

    def disable_caching(self) -> None:
        """Disable FFT plan caching."""
        self.cache_enabled = False

    def clear_cache(self) -> None:
        """Clear FFT plan cache."""
        self.fft_cache.clear()


# Global FFT profiler instance
_fft_profiler = FFTProfiler()


def get_fft_profiler() -> FFTProfiler:
    """Get the global FFT profiler instance."""
    return _fft_profiler


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
