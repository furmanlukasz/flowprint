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


def compute_per_regime_window_metrics(
    trajectory: np.ndarray,
    regime_labels: np.ndarray,
    regime_names: List[str],
    window_size: int = 50,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute per-window metrics WITHIN each regime separately.

    This is the correct approach: extract contiguous segments for each regime,
    then compute windows within those segments. This avoids artifacts from
    windows spanning regime boundaries.

    Args:
        trajectory: (n_samples, n_dims) latent trajectory
        regime_labels: (n_samples,) regime index per sample
        regime_names: List of regime names (may have duplicates for cycles)
        window_size: Samples per window

    Returns:
        Dict mapping regime_name -> {metric_name: array of values}
    """
    unique_names = list(dict.fromkeys(regime_names))

    # Initialize per-regime metrics
    regime_metrics = {
        name: {"speed": [], "variance": [], "tortuosity": []}
        for name in unique_names
    }

    for name in unique_names:
        # Find all regime indices that correspond to this name
        matching_ids = [i for i, n in enumerate(regime_names) if n == name]

        # Get trajectory samples for this regime
        mask = np.isin(regime_labels, matching_ids)
        regime_traj = trajectory[mask]

        # Compute metrics on non-overlapping windows WITHIN this regime
        n_windows = len(regime_traj) // window_size
        for w in range(n_windows):
            start = w * window_size
            end = start + window_size
            window = regime_traj[start:end]

            if len(window) < window_size:
                continue

            # Speed: mean velocity magnitude
            velocity = np.diff(window, axis=0)
            speeds = np.linalg.norm(velocity, axis=1)
            regime_metrics[name]["speed"].append(float(np.mean(speeds)))

            # Variance: total variance in window
            var = float(np.sum(np.var(window, axis=0)))
            regime_metrics[name]["variance"].append(var)

            # Tortuosity: path_length / displacement
            path_len = np.sum(speeds)
            disp = np.linalg.norm(window[-1] - window[0])
            tort = path_len / (disp + 1e-8)
            regime_metrics[name]["tortuosity"].append(float(tort))

    # Convert to arrays
    for name in unique_names:
        for metric in regime_metrics[name]:
            regime_metrics[name][metric] = np.array(regime_metrics[name][metric])

    return regime_metrics


def compute_regime_discriminability(
    trajectory: np.ndarray,
    regime_labels: np.ndarray,
    window_size: int = 50,
    regime_names: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute discriminability for multiple metrics across regimes.

    Uses per-regime window computation to avoid boundary artifacts.

    Args:
        trajectory: (n_samples, n_dims) latent trajectory
        regime_labels: (n_samples,) regime label per sample
        window_size: Window size for metric computation
        regime_names: Optional list of regime names (for cycle handling)

    Returns:
        Dict mapping metric name to discriminability results
    """
    # If regime_names not provided, create from unique labels
    if regime_names is None:
        unique_labels = np.unique(regime_labels)
        regime_names = [f"regime_{i}" for i in unique_labels]

    unique_names = list(dict.fromkeys(regime_names))

    # Compute per-regime window metrics (correct approach)
    regime_metrics = compute_per_regime_window_metrics(
        trajectory, regime_labels, regime_names, window_size
    )

    results = {}
    for metric in ["speed", "variance", "tortuosity"]:
        # Collect groups for ANOVA
        groups = [regime_metrics[name][metric] for name in unique_names]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2 or not all(len(g) > 1 for g in groups):
            results[metric] = {
                "f_statistic": float("nan"),
                "p_value": 1.0,
                "eta_squared": 0.0,
                "effect_size": "undefined",
                "n_windows": [len(g) for g in groups],
            }
            continue

        # One-way ANOVA
        from scipy.stats import f_oneway
        f_stat, p_val = f_oneway(*groups)

        # Eta-squared
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        ss_total = np.sum((all_data - grand_mean) ** 2)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        eta_sq = ss_between / (ss_total + 1e-10)

        # Effect size interpretation
        if eta_sq >= 0.14:
            effect = "large"
        elif eta_sq >= 0.06:
            effect = "medium"
        else:
            effect = "small"

        results[metric] = {
            "f_statistic": float(f_stat),
            "p_value": float(p_val),
            "eta_squared": float(eta_sq),
            "effect_size": effect,
            "n_windows": [len(g) for g in groups],
        }

    return results
