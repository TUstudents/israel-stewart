"""Diagnose what the benchmark actually does."""
import numpy as np

k = 8.0
omega = 5.457140  # From benchmark output
period = 2 * np.pi / omega
three_periods = 3 * period

print(f"Wave number k: {k}")
print(f"Analytical ω: {omega:.6f}")
print(f"Period T = 2π/ω: {period:.6f}")
print(f"3 periods = 3T: {three_periods:.6f}")
print()

# From benchmark output:
timesteps = 692
dt_estimate = three_periods / timesteps

print(f"Benchmark ran {timesteps} timesteps")
print(f"Estimated dt ≈ {dt_estimate:.6f}")
print(f"Total simulation time ≈ {three_periods:.6f}")
print()
print("The benchmark runs for 3 WAVE PERIODS, not t=0.3!")
print(f"For k=8, this means evolving to t ≈ {three_periods:.2f}")
print()
print("Over this long time:")
print("- Eigenmode structure degrades (verified in earlier scripts)")
print("- Measured freq/damping become inaccurate")
