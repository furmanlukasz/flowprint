"""
Kinetic energy proxy computation for latent trajectories.

The kinetic energy proxy E_k(t) = ||v(t)||^2 emphasizes intermittency
and burst-like dynamics, complementing first-order speed measures.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import kurtosis, skew


def compute_kinetic_energy(
    trajectory: np.ndarray,
    method: str = "savgol",
    savgol_window: int = 5,
    savgol_order: int = 2,
    trim_edges: bool = True,
) -> np.ndarray:
    """
    Compute kinetic energy proxy E_k(t) = ||v(t)||^2.

    Args:
        trajectory: (n_samples, n_dims) latent trajectory
        method: Velocity estimation method ("savgol" or "finite_diff")
        savgol_window: Window for Savitzky-Golay filter
        savgol_order: Polynomial order
        trim_edges: Remove edge artifacts from filtering

    Returns:
        Kinetic energy time series (n_samples,) or (n_samples - 2*trim,)
    """
    if method == "savgol":
        velocity = savgol_filter(
            trajectory, savgol_window, savgol_order, deriv=1, axis=0, mode="interp"
        )
        if trim_edges:
            trim_n = savgol_window * 2
            velocity = velocity[trim_n:-trim_n]
    elif method == "finite_diff":
        velocity = np.gradient(trajectory, axis=0)
        if trim_edges:
            velocity = velocity[2:-2]
    else:
        raise ValueError(f"Unknown method: {method}")

    # Kinetic energy = squared speed
    energy = np.sum(velocity**2, axis=1)
    return energy


def compute_kinetic_energy_metrics(
    energy: np.ndarray,
) -> dict[str, float]:
    """
    Compute summary statistics for kinetic energy time series.

    Args:
        energy: Kinetic energy time series

    Returns:
        Dict with mean, std, CV, kurtosis, skewness, Q95/Q50 ratio
    """
    mean_energy = float(np.mean(energy))
    std_energy = float(np.std(energy))
    cv = std_energy / (mean_energy + 1e-10)

    # Tail heaviness: Q95/Q50 ratio (interpretable measure)
    q50 = np.percentile(energy, 50)
    q95 = np.percentile(energy, 95)
    tail_ratio = q95 / (q50 + 1e-10)

    return {
        "mean_energy": mean_energy,
        "std_energy": std_energy,
        "cv": float(cv),
        "kurtosis": float(kurtosis(energy)),
        "skewness": float(skew(energy)),
        "q95_q50_ratio": float(tail_ratio),
    }


def compute_energy_landscape(
    energy: np.ndarray,
    positions_2d: np.ndarray,
    bounds: tuple,
    grid_size: int = 20,
) -> dict[str, np.ndarray]:
    """
    Compute spatial distribution of kinetic energy on 2D embedding.

    Args:
        energy: Kinetic energy time series (n_samples,)
        positions_2d: 2D embedding positions (n_samples, 2)
        bounds: (xmin, xmax, ymin, ymax)
        grid_size: Number of bins per dimension

    Returns:
        Dict with X, Y grid coordinates and mean_energy, counts arrays
    """
    # Ensure matching lengths
    n = min(len(energy), len(positions_2d))
    energy = energy[:n]
    positions_2d = positions_2d[:n]

    xmin, xmax, ymin, ymax = bounds
    x_edges = np.linspace(xmin, xmax, grid_size + 1)
    y_edges = np.linspace(ymin, ymax, grid_size + 1)

    X, Y = np.meshgrid(
        (x_edges[:-1] + x_edges[1:]) / 2,
        (y_edges[:-1] + y_edges[1:]) / 2,
    )

    energy_sum = np.zeros((grid_size, grid_size))
    counts = np.zeros((grid_size, grid_size))

    x_idx = np.clip(np.digitize(positions_2d[:, 0], x_edges) - 1, 0, grid_size - 1)
    y_idx = np.clip(np.digitize(positions_2d[:, 1], y_edges) - 1, 0, grid_size - 1)

    for i in range(n):
        xi, yi = x_idx[i], y_idx[i]
        energy_sum[yi, xi] += energy[i]
        counts[yi, xi] += 1

    # Mean energy per cell
    mean_energy = np.zeros_like(energy_sum)
    mask = counts > 0
    mean_energy[mask] = energy_sum[mask] / counts[mask]

    return {
        "X": X,
        "Y": Y,
        "mean_energy": mean_energy,
        "counts": counts,
    }
