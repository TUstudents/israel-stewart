#!/usr/bin/env python3
"""Check grid spacing."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from israel_stewart.core.spacegrid import SpaceGrid

grid = SpaceGrid(
    coordinate_system="cartesian",
    spatial_ranges=[(0.0, 2*np.pi)] * 3,
    grid_points=(32, 32, 16),
    boundary_conditions="periodic",
)

print(f"Domain size: {2*np.pi:.6f}")
print(f"Grid points: {grid.grid_points}")
print(f"Grid spacing: {grid.spatial_spacing}")
print()

dx, dy, dz = grid.spatial_spacing
print(f"dx = {dx:.6f}")
print(f"dy = {dy:.6f}")
print(f"dz = {dz:.6f}")
print()

# For k=1 wave, gradient without spacing gives k_numerical = k * dx
k = 1.0
k_numerical = k * dx
print(f"True wave number: k = {k:.6f}")
print(f"Numerical k (if dx=1 assumed): k_num = k/dx = {k/dx:.6f}")
print(f"Error factor: {1/dx:.6f}")
