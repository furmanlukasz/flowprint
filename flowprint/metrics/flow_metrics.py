"""
Flow field metrics for latent trajectory analysis.

Computes trajectory-based metrics including:
- Mean speed and speed variability (CV)
- Path tortuosity
- Explored variance
- Flow field divergence and curl
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.signal import savgol_filter


def compute_velocity(
    trajectory: np.ndarray,
    method: str = "savgol",
    savgol_window: int = 5,
    savgol_order: int = 2,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Compute velocity from trajectory.

    Args:
        trajectory: (n_samples, n_dims) array
        method: "savgol" or "finite_diff"
        savgol_window: Window size for Savitzky-Golay filter
        savgol_order: Polynomial order for Savitzky-Golay
        dt: Time step (for finite difference)

    Returns:
        Velocity array (n_samples, n_dims)
    """
    if method == "savgol":
        velocity = savgol_filter(
            trajectory, savgol_window, savgol_order, deriv=1, axis=0, mode="interp"
        )
    elif method == "finite_diff":
        velocity = np.gradient(trajectory, dt, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")
    return velocity


def compute_flow_metrics(
    trajectory: np.ndarray,
    smooth_window: int = 5,
) -> Dict[str, float]:
    """
    Compute flow metrics from a latent trajectory.

    Args:
        trajectory: (n_samples, n_dims) array
        smooth_window: Window for Savitzky-Golay smoothing

    Returns:
        Dict with metrics: speed, speed_cv, tortuosity, explored_variance
    """
    # Smooth for tortuosity
    smoothed = savgol_filter(trajectory, smooth_window, 2, axis=0, mode="interp")

    # Velocity from smoothed trajectory
    velocity = compute_velocity(smoothed, method="savgol", savgol_window=smooth_window)

    # Speed statistics
    speed = np.linalg.norm(velocity, axis=1)
    mean_speed = float(np.mean(speed))
    speed_cv = float(np.std(speed) / (mean_speed + 1e-10))

    # Tortuosity: ratio of path length to displacement
    eps = 1e-8
    step_lengths = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
    path_length = np.sum(step_lengths) + eps
    displacement = np.linalg.norm(smoothed[-1] - smoothed[0]) + eps
    tortuosity = float(path_length / displacement)

    # Explored variance
    explored_variance = float(np.sum(np.var(trajectory, axis=0)))

    return {
        "speed": mean_speed,
        "speed_cv": speed_cv,
        "tortuosity": tortuosity,
        "explored_variance": explored_variance,
    }


def compute_flow_field(
    positions: np.ndarray,
    velocity: np.ndarray,
    bounds: Tuple[float, float, float, float],
    grid_size: int = 15,
    min_samples: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Compute spatially binned flow field from positions and velocities.

    Args:
        positions: (n_samples, 2) 2D positions
        velocity: (n_samples, 2) 2D velocities
        bounds: (xmin, xmax, ymin, ymax) grid bounds
        grid_size: Number of bins per dimension
        min_samples: Minimum samples per bin for valid estimate

    Returns:
        Dict with X, Y grid coordinates, flow_x, flow_y, counts,
        divergence, and curl arrays
    """
    xmin, xmax, ymin, ymax = bounds
    x_edges = np.linspace(xmin, xmax, grid_size + 1)
    y_edges = np.linspace(ymin, ymax, grid_size + 1)

    dx = (xmax - xmin) / grid_size
    dy = (ymax - ymin) / grid_size

    X, Y = np.meshgrid(
        (x_edges[:-1] + x_edges[1:]) / 2,
        (y_edges[:-1] + y_edges[1:]) / 2,
    )

    flow_x = np.zeros((grid_size, grid_size))
    flow_y = np.zeros((grid_size, grid_size))
    counts = np.zeros((grid_size, grid_size))

    x_idx = np.clip(np.digitize(positions[:, 0], x_edges) - 1, 0, grid_size - 1)
    y_idx = np.clip(np.digitize(positions[:, 1], y_edges) - 1, 0, grid_size - 1)

    for i in range(len(positions)):
        xi, yi = x_idx[i], y_idx[i]
        flow_x[yi, xi] += velocity[i, 0]
        flow_y[yi, xi] += velocity[i, 1]
        counts[yi, xi] += 1

    # Average velocities
    mask = counts > 0
    flow_x[mask] /= counts[mask]
    flow_y[mask] /= counts[mask]

    # Compute divergence: div = dvx/dx + dvy/dy
    dvx_dx = np.zeros_like(flow_x)
    dvy_dy = np.zeros_like(flow_y)
    dvx_dx[:, 1:-1] = (flow_x[:, 2:] - flow_x[:, :-2]) / (2 * dx)
    dvy_dy[1:-1, :] = (flow_y[2:, :] - flow_y[:-2, :]) / (2 * dy)
    divergence = dvx_dx + dvy_dy

    # Compute curl: curl = dvy/dx - dvx/dy
    dvy_dx = np.zeros_like(flow_y)
    dvx_dy = np.zeros_like(flow_x)
    dvy_dx[:, 1:-1] = (flow_y[:, 2:] - flow_y[:, :-2]) / (2 * dx)
    dvx_dy[1:-1, :] = (flow_x[2:, :] - flow_x[:-2, :]) / (2 * dy)
    curl = dvy_dx - dvx_dy

    return {
        "X": X,
        "Y": Y,
        "flow_x": flow_x,
        "flow_y": flow_y,
        "counts": counts,
        "divergence": divergence,
        "curl": curl,
        "valid_mask": counts >= min_samples,
    }


def compute_field_metrics(
    flow_field: Dict[str, np.ndarray],
    min_samples: int = 3,
) -> Dict[str, float]:
    """
    Compute summary metrics from a flow field.

    Args:
        flow_field: Output from compute_flow_field
        min_samples: Minimum samples for valid cells

    Returns:
        Dict with divergence and curl statistics
    """
    valid_mask = flow_field["counts"] >= min_samples

    if valid_mask.sum() == 0:
        return {
            "mean_divergence": 0.0,
            "std_divergence": 0.0,
            "mean_curl": 0.0,
            "mean_abs_curl": 0.0,
            "curl_circulation": 0.0,
            "n_valid_cells": 0,
        }

    div_valid = flow_field["divergence"][valid_mask]
    curl_valid = flow_field["curl"][valid_mask]

    # Compute cell area for circulation
    dx = flow_field["X"][0, 1] - flow_field["X"][0, 0] if flow_field["X"].shape[1] > 1 else 1.0
    dy = flow_field["Y"][1, 0] - flow_field["Y"][0, 0] if flow_field["Y"].shape[0] > 1 else 1.0

    return {
        "mean_divergence": float(np.mean(div_valid)),
        "std_divergence": float(np.std(div_valid)),
        "mean_curl": float(np.mean(curl_valid)),
        "mean_abs_curl": float(np.mean(np.abs(curl_valid))),
        "curl_circulation": float(np.sum(curl_valid) * dx * dy),
        "n_valid_cells": int(valid_mask.sum()),
    }
