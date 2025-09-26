"""
Centralized logging configuration for Israel-Stewart hydrodynamics.

This module provides structured logging setup with support for both
development and production environments, performance monitoring,
and scientific computing specific features.
"""

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Any, Optional

try:
    import structlog

    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


class ISLoggerMixin:
    """Mixin class to add logger to any class."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically module or class name)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"israel_stewart.{name}")


def configure_logging(
    level: str = "INFO",
    format_type: str = "console",
    log_file: Path | None = None,
    enable_performance: bool = False,
    enable_memory_tracking: bool = False,
    enable_solver_logging: bool = False,
    enable_physics_validation: bool = False,
    enable_debug_mode: bool = False,
) -> None:
    """
    Configure logging for the Israel-Stewart package.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ("console", "json", "structured")
        log_file: Optional file path for file logging
        enable_performance: Enable detailed performance logging
        enable_memory_tracking: Enable memory usage tracking
        enable_solver_logging: Enable detailed solver operation logging
        enable_physics_validation: Enable physics validation and conservation logging
        enable_debug_mode: Enable comprehensive debug logging
    """
    level = level.upper()

    # Base logging configuration
    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "format": "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": (
                    "%(asctime)s | %(name)-20s | %(levelname)-8s | "
                    "%(filename)s:%(lineno)d | %(funcName)s | %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S.%f",
            },
            "json": {
                "format": "%(message)s",
                "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
                if "pythonjsonlogger" in sys.modules
                else "logging.Formatter",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "console" if format_type == "console" else "detailed",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "israel_stewart": {
                "level": level,
                "handlers": ["console"],
                "propagate": False,
            },
            # Silence noisy third-party loggers in production
            "matplotlib": {"level": "WARNING"},
            "numba": {"level": "WARNING"},
            "scipy": {"level": "WARNING"},
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"],
        },
    }

    # Add file handler if requested
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": "detailed",
            "filename": str(log_file),
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
        }
        config["loggers"]["israel_stewart"]["handlers"].append("file")

    # Performance logging configuration
    if enable_performance:
        config["loggers"]["israel_stewart.performance"] = {
            "level": "DEBUG",
            "handlers": config["loggers"]["israel_stewart"]["handlers"].copy(),
            "propagate": False,
        }

    # Memory tracking configuration
    if enable_memory_tracking:
        config["loggers"]["israel_stewart.memory"] = {
            "level": "DEBUG",
            "handlers": config["loggers"]["israel_stewart"]["handlers"].copy(),
            "propagate": False,
        }

    # Solver-specific logging configuration
    if enable_solver_logging:
        solver_loggers = [
            "israel_stewart.solvers",
            "israel_stewart.solvers.finite_difference",
            "israel_stewart.solvers.spectral",
            "israel_stewart.solvers.implicit",
            "israel_stewart.solvers.splitting",
        ]
        for solver_logger in solver_loggers:
            config["loggers"][solver_logger] = {
                "level": "DEBUG" if enable_debug_mode else "INFO",
                "handlers": config["loggers"]["israel_stewart"]["handlers"].copy(),
                "propagate": False,
            }

    # Physics validation logging configuration
    if enable_physics_validation:
        physics_loggers = [
            "israel_stewart.physics",
            "israel_stewart.equations",
            "israel_stewart.equations.conservation",
            "israel_stewart.equations.relaxation",
        ]
        for physics_logger in physics_loggers:
            config["loggers"][physics_logger] = {
                "level": "DEBUG" if enable_debug_mode else "INFO",
                "handlers": config["loggers"]["israel_stewart"]["handlers"].copy(),
                "propagate": False,
            }

    # Debug mode: Enable all subsystems
    if enable_debug_mode:
        config["loggers"]["israel_stewart"]["level"] = "DEBUG"
        # Add specialized debug handlers for different subsystems
        debug_subsystems = ["core", "benchmarks", "linearization", "stochastic"]
        for subsystem in debug_subsystems:
            config["loggers"][f"israel_stewart.{subsystem}"] = {
                "level": "DEBUG",
                "handlers": config["loggers"]["israel_stewart"]["handlers"].copy(),
                "propagate": False,
            }

    # Apply configuration
    logging.config.dictConfig(config)

    # Configure structlog if available
    if HAS_STRUCTLOG and format_type == "structured":
        _configure_structlog(level)


def _configure_structlog(level: str) -> None:
    """Configure structlog for structured logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
            if level == "DEBUG"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        logger_factory=structlog.PrintLoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def setup_from_environment() -> None:
    """
    Setup logging configuration from environment variables.

    Environment Variables:
        ISRAEL_STEWART_LOG_LEVEL: Log level (default: INFO)
        ISRAEL_STEWART_LOG_FORMAT: Format type (default: console)
        ISRAEL_STEWART_LOG_FILE: Optional log file path
        ISRAEL_STEWART_LOG_PERFORMANCE: Enable performance logging (default: False)
        ISRAEL_STEWART_LOG_MEMORY: Enable memory tracking (default: False)
        ISRAEL_STEWART_LOG_SOLVERS: Enable solver logging (default: False)
        ISRAEL_STEWART_LOG_PHYSICS: Enable physics validation logging (default: False)
        ISRAEL_STEWART_LOG_DEBUG: Enable comprehensive debug mode (default: False)
    """
    level = os.getenv("ISRAEL_STEWART_LOG_LEVEL", "INFO")
    format_type = os.getenv("ISRAEL_STEWART_LOG_FORMAT", "console")
    log_file_str = os.getenv("ISRAEL_STEWART_LOG_FILE")
    enable_performance = os.getenv("ISRAEL_STEWART_LOG_PERFORMANCE", "false").lower() == "true"
    enable_memory = os.getenv("ISRAEL_STEWART_LOG_MEMORY", "false").lower() == "true"
    enable_solver_logging = os.getenv("ISRAEL_STEWART_LOG_SOLVERS", "false").lower() == "true"
    enable_physics_validation = os.getenv("ISRAEL_STEWART_LOG_PHYSICS", "false").lower() == "true"
    enable_debug_mode = os.getenv("ISRAEL_STEWART_LOG_DEBUG", "false").lower() == "true"

    log_file = Path(log_file_str) if log_file_str else None

    configure_logging(
        level=level,
        format_type=format_type,
        log_file=log_file,
        enable_performance=enable_performance,
        enable_memory_tracking=enable_memory,
        enable_solver_logging=enable_solver_logging,
        enable_physics_validation=enable_physics_validation,
        enable_debug_mode=enable_debug_mode,
    )


# Performance and memory logging utilities
class PerformanceLogger:
    """Logger for performance metrics and timing."""

    def __init__(self, name: str = "performance"):
        self.logger = get_logger(name)

    def log_operation(self, operation: str, duration: float, **kwargs: Any) -> None:
        """Log a completed operation with timing."""
        self.logger.info(
            f"Operation completed: {operation}",
            extra={
                "operation": operation,
                "duration_seconds": duration,
                "performance_data": kwargs,
            },
        )

    def log_memory_usage(self, operation: str, memory_mb: float, **kwargs: Any) -> None:
        """Log memory usage for an operation."""
        self.logger.debug(
            f"Memory usage: {operation}",
            extra={
                "operation": operation,
                "memory_mb": memory_mb,
                "memory_data": kwargs,
            },
        )


class PhysicsLogger:
    """Logger for physics validation and scientific computing."""

    def __init__(self, name: str = "physics"):
        self.logger = get_logger(name)

    def log_conservation_check(self, quantity: str, error: float, tolerance: float) -> None:
        """Log conservation law validation results."""
        status = "PASSED" if error < tolerance else "FAILED"
        level = logging.INFO if error < tolerance else logging.WARNING

        self.logger.log(
            level,
            f"Conservation check {status}: {quantity}",
            extra={
                "quantity": quantity,
                "error": error,
                "tolerance": tolerance,
                "status": status,
            },
        )

    def log_convergence(
        self, solver: str, iterations: int, residual: float, converged: bool
    ) -> None:
        """Log numerical convergence information."""
        status = "CONVERGED" if converged else "FAILED"
        level = logging.INFO if converged else logging.WARNING

        self.logger.log(
            level,
            f"Solver {status}: {solver}",
            extra={
                "solver": solver,
                "iterations": iterations,
                "residual": residual,
                "converged": converged,
            },
        )

    def log_physics_fallback(self, operation: str, reason: str, fallback: str) -> None:
        """Log when physics computations fall back to simpler methods."""
        self.logger.warning(
            f"Physics fallback: {operation}",
            extra={
                "operation": operation,
                "reason": reason,
                "fallback_method": fallback,
            },
        )

    def log_error_recovery(
        self, operation: str, error: str, recovery_action: str, success: bool
    ) -> None:
        """Log error recovery attempts and their outcomes."""
        level = logging.INFO if success else logging.WARNING
        status = "SUCCESS" if success else "FAILED"

        self.logger.log(
            level,
            f"Error recovery {status}: {operation}",
            extra={
                "operation": operation,
                "original_error": error,
                "recovery_action": recovery_action,
                "recovery_success": success,
            },
        )


# Runtime logging control functions
def set_log_level(level: str) -> None:
    """Adjust log level at runtime for all Israel-Stewart loggers."""
    level = level.upper()
    israel_stewart_logger = logging.getLogger("israel_stewart")
    israel_stewart_logger.setLevel(getattr(logging, level))

    # Also update all child loggers
    for name in logging.getLogger().manager.loggerDict:
        if name.startswith("israel_stewart."):
            logger = logging.getLogger(name)
            if logger.handlers:  # Only update loggers that have been configured
                logger.setLevel(getattr(logging, level))


def enable_performance_logging(enabled: bool = True) -> None:
    """Enable or disable performance logging at runtime."""
    perf_logger = logging.getLogger("israel_stewart.performance")
    if enabled:
        if not perf_logger.handlers:
            # Configure performance logger if not already configured
            perf_logger.setLevel(logging.DEBUG)
            # Use the same handlers as the main logger
            main_logger = logging.getLogger("israel_stewart")
            for handler in main_logger.handlers:
                perf_logger.addHandler(handler)
            perf_logger.propagate = False
        perf_logger.disabled = False
    else:
        perf_logger.disabled = True


def enable_memory_debugging(enabled: bool = True) -> None:
    """Enable or disable memory debugging logging at runtime."""
    mem_logger = logging.getLogger("israel_stewart.memory")
    if enabled:
        if not mem_logger.handlers:
            # Configure memory logger if not already configured
            mem_logger.setLevel(logging.DEBUG)
            # Use the same handlers as the main logger
            main_logger = logging.getLogger("israel_stewart")
            for handler in main_logger.handlers:
                mem_logger.addHandler(handler)
            mem_logger.propagate = False
        mem_logger.disabled = False
    else:
        mem_logger.disabled = True


def get_logging_status() -> dict[str, Any]:
    """Get current logging configuration status."""
    israel_stewart_logger = logging.getLogger("israel_stewart")
    perf_logger = logging.getLogger("israel_stewart.performance")
    mem_logger = logging.getLogger("israel_stewart.memory")

    return {
        "main_level": logging.getLevelName(israel_stewart_logger.level),
        "performance_enabled": not perf_logger.disabled and bool(perf_logger.handlers),
        "memory_debugging_enabled": not mem_logger.disabled and bool(mem_logger.handlers),
        "handlers_count": len(israel_stewart_logger.handlers),
        "active_loggers": [
            name
            for name in logging.getLogger().manager.loggerDict
            if name.startswith("israel_stewart.")
            and logging.getLogger(name).handlers
            and not logging.getLogger(name).disabled
        ],
    }


# Global logger instances for convenience
performance_logger = PerformanceLogger()
physics_logger = PhysicsLogger()

# Initialize logging with environment settings by default
if not logging.getLogger("israel_stewart").handlers:
    setup_from_environment()
