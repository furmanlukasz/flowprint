"""
Discriminability analysis for regime comparison.

Computes ANOVA-based effect sizes (eta-squared) to quantify
how well metrics discriminate between regimes.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats


def compute_window_metrics(
    trajectory: np.ndarray,
    window_size: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Compute per-window metrics for discriminability analysis.

    Args:
        trajectory: (n_samples, n_dims) latent trajectory
        window_size: Number of samples per window

    Returns:
        Dict with arrays of per-window speed, variance, tortuosity
    """
    n_samples = len(trajectory)
    n_windows = n_samples // window_size

    speeds = []
    variances = []
    tortuosities = []

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window = trajectory[start:end]

        # Speed: mean velocity magnitude
        velocity = np.diff(window, axis=0)
        speed = np.mean(np.linalg.norm(velocity, axis=1))
        speeds.append(speed)

        # Variance: total variance in window
        var = np.sum(np.var(window, axis=0))
        variances.append(var)

        # Tortuosity: path length / displacement
        step_lengths = np.linalg.norm(np.diff(window, axis=0), axis=1)
        path_length = np.sum(step_lengths) + 1e-8
        displacement = np.linalg.norm(window[-1] - window[0]) + 1e-8
        tort = path_length / displacement
        tortuosities.append(tort)

    return {
        "speed": np.array(speeds),
        "variance": np.array(variances),
        "tortuosity": np.array(tortuosities),
    }


def compute_discriminability(
    metric_values: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute ANOVA discriminability with eta-squared effect size.

    Args:
        metric_values: Array of metric values
        labels: Array of regime labels (same length)

    Returns:
        Dict with F-statistic, p-value, and eta-squared
    """
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return {
            "f_statistic": 0.0,
            "p_value": 1.0,
            "eta_squared": 0.0,
            "effect_size": "undefined",
        }

    # Group values by label
    groups = [metric_values[labels == lbl] for lbl in unique_labels]

    # Filter empty groups
    groups = [g for g in groups if len(g) > 0]

    if len(groups) < 2:
        return {
            "f_statistic": 0.0,
            "p_value": 1.0,
            "eta_squared": 0.0,
            "effect_size": "undefined",
        }

    # One-way ANOVA
    f_stat, p_val = stats.f_oneway(*groups)

    # Eta-squared: SS_between / SS_total
    grand_mean = np.mean(metric_values)
    ss_total = np.sum((metric_values - grand_mean) ** 2)

    ss_between = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups
    )

    eta_sq = ss_between / (ss_total + 1e-10)

    # Effect size interpretation (Cohen, 1988)
    if eta_sq >= 0.14:
        effect = "large"
    elif eta_sq >= 0.06:
        effect = "medium"
    else:
        effect = "small"

    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_val),
        "eta_squared": float(eta_sq),
        "effect_size": effect,
    }


def compute_regime_discriminability(
    trajectory: np.ndarray,
    regime_labels: np.ndarray,
    window_size: int = 50,
) -> Dict[str, Dict[str, float]]:
    """
    Compute discriminability for multiple metrics across regimes.

    Args:
        trajectory: (n_samples, n_dims) latent trajectory
        regime_labels: (n_samples,) regime label per sample
        window_size: Window size for metric computation

    Returns:
        Dict mapping metric name to discriminability results
    """
    # Compute per-window metrics
    window_metrics = compute_window_metrics(trajectory, window_size)

    # Downsample labels to window level
    n_windows = len(window_metrics["speed"])
    window_labels = np.array([
        regime_labels[i * window_size] for i in range(n_windows)
    ])

    results = {}
    for metric_name, values in window_metrics.items():
        results[metric_name] = compute_discriminability(values, window_labels)

    return results
