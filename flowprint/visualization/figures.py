"""
Figure generation for FlowPrint analysis.

Reproduces all figures from the paper:
- Electrode time series
- Main analysis (4-panel)
- Flow fields per regime
- Discriminability violin plots
- Kinetic energy analysis
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import hilbert


def plot_electrode_timeseries(
    observations: np.ndarray,
    time: np.ndarray,
    switch_times: List[float],
    sfreq: float,
    channels: List[int] = [0, 5, 10, 20, 28],
    time_window: Tuple[float, float] = (0.0, 60.0),
    nfft: int = 2048,
    figsize: Tuple[float, float] = (12, 9),
) -> plt.Figure:
    """
    Multi-panel visualization of raw observations.

    Panel A: Raw time series with regime switches
    Panel B: Power spectral density
    Panel C: Hilbert amplitude envelope

    Args:
        observations: (n_channels, n_samples) array
        time: (n_samples,) time vector in seconds
        switch_times: Regime switch times
        sfreq: Sampling frequency
        channels: Channel indices to plot
        time_window: (start, end) in seconds
        nfft: FFT size for PSD
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    t0, t1 = time_window
    i0 = int(max(0, np.floor(t0 * sfreq)))
    i1 = int(min(observations.shape[1], np.ceil(t1 * sfreq)))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.2, 1.2], hspace=0.45)

    # Panel A: Raw time series
    ax1 = fig.add_subplot(gs[0, 0])
    offset = 0.0
    for ch in channels:
        sig = observations[ch, i0:i1]
        ax1.plot(time[i0:i1], sig + offset, lw=1.0)
        ax1.text(t0, offset, f"Ch{ch}", va="bottom", fontsize=9)
        offset += 2.5 * np.std(sig) + 1e-6

    for st in switch_times:
        if t0 <= st <= t1:
            ax1.axvline(st, linestyle="--", linewidth=1, color="blue", alpha=0.5)

    ax1.set_title("Raw time series (selected channels) with regime switches")
    ax1.set_xlabel("Time (s)")
    ax1.set_yticks([])

    # Panel B: PSD
    ax2 = fig.add_subplot(gs[1, 0])
    seg = observations[:, i0:i1]
    freqs = np.fft.rfftfreq(nfft, d=1 / sfreq)
    Y = np.fft.rfft(seg - seg.mean(axis=1, keepdims=True), n=nfft, axis=1)
    psd = (np.abs(Y) ** 2).mean(axis=0)
    ax2.plot(freqs, psd)
    ax2.set_xlim(0, 60)
    ax2.set_title("Power spectral density (simple periodogram, mean across channels)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power (a.u.)")

    # Panel C: Hilbert amplitude
    ax3 = fig.add_subplot(gs[2, 0])
    ch0 = channels[0]
    analytic = hilbert(observations[ch0, i0:i1])
    amp = np.abs(analytic)
    ax3.plot(time[i0:i1], amp, lw=1.0)
    for st in switch_times:
        if t0 <= st <= t1:
            ax3.axvline(st, linestyle="--", linewidth=1, color="blue", alpha=0.5)
    ax3.set_title(f"Hilbert amplitude envelope (channel {ch0})")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude (a.u.)")

    return fig


def plot_main_analysis(
    embedded: np.ndarray,
    regime_labels: np.ndarray,
    regime_names: List[str],
    switch_times: List[float],
    total_duration: float,
    flow_field: Dict[str, np.ndarray],
    flow_metrics: Dict[str, Dict[str, float]],
    regime_colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (16, 12),
) -> plt.Figure:
    """
    Main 4-panel analysis figure.

    Panel A: Ground-truth regime timeline
    Panel B: Embedded trajectories colored by regime
    Panel C: Density + flow field
    Panel D: Normalized flow metrics by regime

    Args:
        embedded: (n_samples, 2) embedded trajectory
        regime_labels: (n_samples,) regime indices
        regime_names: List of regime names
        switch_times: Regime switch times
        total_duration: Total duration in seconds
        flow_field: Output from compute_flow_field
        flow_metrics: Dict mapping regime -> metrics dict
        regime_colors: Optional color mapping
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if regime_colors is None:
        regime_colors = {
            "global": "#1f77b4",
            "cluster": "#ff7f0e",
            "sparse": "#2ca02c",
            "ring": "#d62728",
        }

    unique_names = list(dict.fromkeys(regime_names))

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Panel A: Timeline
    ax_a = fig.add_subplot(gs[0, 0])
    for i, name in enumerate(regime_names):
        start = switch_times[i]
        end = switch_times[i + 1] if i + 1 < len(switch_times) else total_duration
        color = regime_colors.get(name, "#888888")
        ax_a.axvspan(start, end, color=color, alpha=0.7)
    ax_a.set_xlim(0, total_duration)
    ax_a.set_ylim(0, 1)
    ax_a.set_xlabel("Time (s)")
    ax_a.set_title("A) Ground-Truth Regime Sequence", fontweight="bold")
    ax_a.set_yticks([])

    # Panel B: Embedded trajectories
    ax_b = fig.add_subplot(gs[0, 1])
    step = max(1, len(embedded) // 5000)
    embedded_ds = embedded[::step]
    labels_ds = regime_labels[::step][: len(embedded_ds)]

    for name in unique_names:
        matching_ids = [i for i, n in enumerate(regime_names) if n == name]
        mask = np.isin(labels_ds, matching_ids)
        color = regime_colors.get(name, "#888888")
        ax_b.scatter(
            embedded_ds[mask, 0],
            embedded_ds[mask, 1],
            c=color,
            s=2,
            alpha=0.4,
            label=name,
        )
    ax_b.set_xlabel("Dim 1")
    ax_b.set_ylabel("Dim 2")
    ax_b.set_title("B) Embedded Trajectories (colored by regime)", fontweight="bold")
    ax_b.legend(markerscale=3)
    ax_b.set_aspect("equal")

    # Panel C: Density + Flow field
    ax_c = fig.add_subplot(gs[1, 0])

    # Compute density
    xmin, xmax = embedded[:, 0].min(), embedded[:, 0].max()
    ymin, ymax = embedded[:, 1].min(), embedded[:, 1].max()

    H, xedges, yedges = np.histogram2d(
        embedded[:, 0], embedded[:, 1], bins=50,
        range=[[xmin, xmax], [ymin, ymax]]
    )

    ax_c.imshow(
        H.T, origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        cmap="Blues", alpha=0.7, aspect="auto"
    )

    # Overlay flow field
    X = flow_field["X"]
    Y = flow_field["Y"]
    flow_x = flow_field["flow_x"]
    flow_y = flow_field["flow_y"]
    counts = flow_field["counts"]

    mask = counts > 5
    if mask.any():
        mag = np.sqrt(flow_x[mask] ** 2 + flow_y[mask] ** 2)
        norm_fx = np.where(mag > 0, flow_x[mask] / mag, 0)
        norm_fy = np.where(mag > 0, flow_y[mask] / mag, 0)
        ax_c.quiver(
            X[mask], Y[mask], norm_fx, norm_fy, mag,
            cmap="inferno", alpha=0.85,
            scale=25, width=0.004, headwidth=4, headlength=5,
        )
    ax_c.set_xlabel("Dim 1")
    ax_c.set_ylabel("Dim 2")
    ax_c.set_title("C) Density + Flow Field", fontweight="bold")

    # Panel D: Flow metrics bar chart
    ax_d = fig.add_subplot(gs[1, 1])
    metric_names = ["speed", "speed_cv", "tortuosity", "explored_variance"]
    x = np.arange(len(unique_names))
    width = 0.2

    for i, metric in enumerate(metric_names):
        values = [flow_metrics.get(name, {}).get(metric, 0) for name in unique_names]
        # Normalize for visualization
        max_val = max(values) if values else 1
        values_norm = [v / max_val for v in values]
        ax_d.bar(x + i * width, values_norm, width, label=metric)

    ax_d.set_xticks(x + width * 1.5)
    ax_d.set_xticklabels(unique_names)
    ax_d.set_ylabel("Normalized Value")
    ax_d.set_title("D) Flow Metrics by Regime", fontweight="bold")
    ax_d.legend(loc="upper right")

    return fig


def plot_flow_fields(
    regime_data: Dict[str, Dict],
    regime_colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (16, 4),
) -> plt.Figure:
    """
    Regime-specific flow field visualization.

    Args:
        regime_data: Dict mapping regime name to dict with:
            - embedded: (n, 2) positions
            - flow_field: output from compute_flow_field
            - field_metrics: output from compute_field_metrics
        regime_colors: Optional color mapping
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if regime_colors is None:
        regime_colors = {
            "global": "#1f77b4",
            "cluster": "#ff7f0e",
            "sparse": "#2ca02c",
            "ring": "#d62728",
        }

    regime_names = list(regime_data.keys())
    n_regimes = len(regime_names)

    fig, axes = plt.subplots(1, n_regimes, figsize=figsize)
    if n_regimes == 1:
        axes = [axes]

    for ax, name in zip(axes, regime_names):
        data = regime_data[name]
        embedded = data["embedded"]
        ff = data["flow_field"]
        metrics = data.get("field_metrics", {})

        # Density
        xmin, xmax = embedded[:, 0].min(), embedded[:, 0].max()
        ymin, ymax = embedded[:, 1].min(), embedded[:, 1].max()

        H, _, _ = np.histogram2d(
            embedded[:, 0], embedded[:, 1], bins=30,
            range=[[xmin, xmax], [ymin, ymax]]
        )
        ax.imshow(
            H.T, origin="lower",
            extent=[xmin, xmax, ymin, ymax],
            cmap="Blues", alpha=0.6, aspect="auto"
        )

        # Flow field
        mask = ff["counts"] > 3
        if mask.any():
            mag = np.sqrt(ff["flow_x"][mask] ** 2 + ff["flow_y"][mask] ** 2)
            norm_fx = np.where(mag > 0, ff["flow_x"][mask] / mag, 0)
            norm_fy = np.where(mag > 0, ff["flow_y"][mask] / mag, 0)
            ax.quiver(
                ff["X"][mask], ff["Y"][mask], norm_fx, norm_fy, mag,
                cmap="inferno", alpha=0.8, scale=30,
            )

        # Annotations
        div = metrics.get("mean_divergence", 0)
        curl = metrics.get("mean_abs_curl", 0)
        ax.set_title(f"{name.capitalize()}\ndiv={div:.2f}, curl={curl:.2f}")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    plt.tight_layout()
    return fig


def plot_discriminability_per_regime(
    regime_metrics: Dict[str, Dict[str, np.ndarray]],
    regime_names: List[str],
    discriminability: Dict[str, Dict[str, float]],
    regime_colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (14, 5),
) -> plt.Figure:
    """
    Violin plots of per-window metric distributions by regime.

    Uses pre-computed per-regime metrics (correct approach that avoids
    boundary artifacts from windows spanning multiple regimes).

    Args:
        regime_metrics: Dict mapping regime_name -> {metric: array of values}
        regime_names: List of unique regime names
        discriminability: Dict mapping metric -> discriminability results
        regime_colors: Optional color mapping
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if regime_colors is None:
        regime_colors = {
            "global": "#1f77b4",
            "cluster": "#ff7f0e",
            "sparse": "#2ca02c",
            "ring": "#d62728",
        }

    metric_titles = {
        "speed": "Speed (latent units/step)",
        "variance": "Explored Variance",
        "tortuosity": "Path Tortuosity",
    }

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, metric in zip(axes, ["speed", "variance", "tortuosity"]):
        # Collect data per regime
        data_for_plot = []
        positions = []
        colors_for_plot = []

        for i, name in enumerate(regime_names):
            if name in regime_metrics and metric in regime_metrics[name]:
                vals = regime_metrics[name][metric]
                if len(vals) > 0:
                    data_for_plot.append(vals)
                    positions.append(i)
                    colors_for_plot.append(regime_colors.get(name, "#888888"))

        if data_for_plot:
            # Violin plot with means and medians
            parts = ax.violinplot(
                data_for_plot,
                positions=positions,
                showmeans=True,
                showmedians=True,
            )

            # Color the violins
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors_for_plot[i])
                pc.set_alpha(0.7)

            # Style the lines
            for partname in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
                if partname in parts:
                    parts[partname].set_color("black")
                    parts[partname].set_linewidth(1)

        ax.set_xticks(range(len(regime_names)))
        ax.set_xticklabels(regime_names)
        ax.set_title(metric_titles.get(metric, metric.capitalize()), fontweight="bold")
        ax.set_ylabel("Value")

        # Add effect size annotation
        disc = discriminability.get(metric, {})
        eta = disc.get("eta_squared", 0)
        f_stat = disc.get("f_statistic", float("nan"))
        p_val = disc.get("p_value", 1.0)

        effect_label = "large" if eta > 0.14 else "medium" if eta > 0.06 else "small"

        # Significance stars
        if np.isnan(p_val):
            sig_str = ""
        elif p_val < 0.001:
            sig_str = "***"
        elif p_val < 0.01:
            sig_str = "**"
        elif p_val < 0.05:
            sig_str = "*"
        else:
            sig_str = "ns"

        if not np.isnan(f_stat):
            annotation = f"η²={eta:.3f} ({effect_label})\nF={f_stat:.1f} {sig_str}"
        else:
            annotation = f"η²={eta:.3f} ({effect_label})"

        ax.text(
            0.02, 0.98, annotation,
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    fig.suptitle("Regime Discriminability: Per-Window Metric Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_discriminability(
    window_metrics: Dict[str, np.ndarray],
    window_labels: np.ndarray,
    regime_names: List[str],
    discriminability: Dict[str, Dict[str, float]],
    regime_colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (14, 5),
) -> plt.Figure:
    """
    Violin plots of per-window metric distributions by regime.

    Args:
        window_metrics: Dict with arrays of per-window metrics
        window_labels: (n_windows,) regime labels (indices into regime_names)
        regime_names: List of regime names (may contain duplicates for cycles)
        discriminability: Dict mapping metric -> discriminability results
        regime_colors: Optional color mapping
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if regime_colors is None:
        regime_colors = {
            "global": "#1f77b4",
            "cluster": "#ff7f0e",
            "sparse": "#2ca02c",
            "ring": "#d62728",
        }

    # Get unique regime names (handling cycles)
    unique_names = list(dict.fromkeys(regime_names))

    # Map labels (regime indices) to regime names
    # Build mapping: regime_id -> regime_name
    label_to_name = {i: regime_names[i] for i in range(len(regime_names))}

    metric_titles = {
        "speed": "Speed (latent units/step)",
        "variance": "Explored Variance",
        "tortuosity": "Path Tortuosity",
    }

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, metric in zip(axes, ["speed", "variance", "tortuosity"]):
        if metric not in window_metrics:
            continue

        values = window_metrics[metric]

        # Group values by regime NAME (not index) to handle multiple cycles
        data_for_plot = []
        positions = []
        colors_for_plot = []

        for i, name in enumerate(unique_names):
            # Find all regime indices that correspond to this name
            matching_ids = [idx for idx, n in enumerate(regime_names) if n == name]
            # Get values for windows belonging to any of these regime indices
            mask = np.isin(window_labels, matching_ids)
            vals = values[mask]

            if len(vals) > 0:
                data_for_plot.append(vals)
                positions.append(i)
                colors_for_plot.append(regime_colors.get(name, "#888888"))

        if data_for_plot:
            # Violin plot with means and medians
            parts = ax.violinplot(
                data_for_plot,
                positions=positions,
                showmeans=True,
                showmedians=True,
            )

            # Color the violins
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors_for_plot[i])
                pc.set_alpha(0.7)

            # Style the lines
            for partname in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
                if partname in parts:
                    parts[partname].set_color("black")
                    parts[partname].set_linewidth(1)

        ax.set_xticks(range(len(unique_names)))
        ax.set_xticklabels(unique_names)
        ax.set_title(metric_titles.get(metric, metric.capitalize()), fontweight="bold")
        ax.set_ylabel("Value")

        # Add effect size annotation with F-statistic and significance
        disc = discriminability.get(metric, {})
        eta = disc.get("eta_squared", 0)
        f_stat = disc.get("f_statistic", float("nan"))
        p_val = disc.get("p_value", 1.0)

        effect_label = "large" if eta > 0.14 else "medium" if eta > 0.06 else "small"

        # Significance stars
        if np.isnan(p_val):
            sig_str = ""
        elif p_val < 0.001:
            sig_str = "***"
        elif p_val < 0.01:
            sig_str = "**"
        elif p_val < 0.05:
            sig_str = "*"
        else:
            sig_str = "ns"

        if not np.isnan(f_stat):
            annotation = f"η²={eta:.3f} ({effect_label})\nF={f_stat:.1f} {sig_str}"
        else:
            annotation = f"η²={eta:.3f} ({effect_label})"

        ax.text(
            0.02, 0.98, annotation,
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    fig.suptitle("Regime Discriminability: Per-Window Metric Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_kinetic_energy(
    energy: np.ndarray,
    time: np.ndarray,
    regime_labels: np.ndarray,
    regime_names: List[str],
    energy_landscape: Dict[str, np.ndarray],
    per_regime_metrics: Dict[str, Dict[str, float]],
    regime_colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (16, 10),
) -> plt.Figure:
    """
    Kinetic energy analysis figure (4 panels).

    Panel A: Energy time series
    Panel B: Per-regime energy distributions
    Panel C: Energy landscape
    Panel D: Summary metrics

    Args:
        energy: (n_samples,) kinetic energy time series
        time: (n_samples,) time vector
        regime_labels: (n_samples,) regime indices
        regime_names: List of regime names
        energy_landscape: Output from compute_energy_landscape
        per_regime_metrics: Dict mapping regime -> energy metrics
        regime_colors: Optional color mapping
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if regime_colors is None:
        regime_colors = {
            "global": "#1f77b4",
            "cluster": "#ff7f0e",
            "sparse": "#2ca02c",
            "ring": "#d62728",
        }

    # Match lengths
    n = min(len(energy), len(time), len(regime_labels))
    energy = energy[:n]
    time = time[:n]
    regime_labels = regime_labels[:n]

    unique_names = list(dict.fromkeys(regime_names))

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: Time series
    ax_a = fig.add_subplot(gs[0, 0])
    for name in unique_names:
        matching_ids = [i for i, n in enumerate(regime_names) if n == name]
        mask = np.isin(regime_labels, matching_ids)
        color = regime_colors.get(name, "#888888")
        ax_a.scatter(time[mask], energy[mask], c=color, s=1, alpha=0.3, label=name)

    # Smoothed trend
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(energy, size=500)
    ax_a.plot(time, smoothed, "k-", lw=2, label="Smoothed")

    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("$E_k(t) = ||v(t)||^2$")
    ax_a.set_title("A) Kinetic Energy Time Series", fontweight="bold")
    ax_a.legend(markerscale=5)

    # Panel B: Distributions
    ax_b = fig.add_subplot(gs[0, 1])
    for name in unique_names:
        matching_ids = [i for i, n in enumerate(regime_names) if n == name]
        mask = np.isin(regime_labels, matching_ids)
        color = regime_colors.get(name, "#888888")
        ax_b.hist(energy[mask], bins=50, alpha=0.5, color=color, label=name, density=True)

    ax_b.set_xlabel("$E_k$")
    ax_b.set_ylabel("Density")
    ax_b.set_title("B) Energy Distributions", fontweight="bold")
    ax_b.legend()

    # Panel C: Energy landscape
    ax_c = fig.add_subplot(gs[1, 0])
    im = ax_c.imshow(
        energy_landscape["mean_energy"],
        origin="lower",
        extent=[
            energy_landscape["X"].min(),
            energy_landscape["X"].max(),
            energy_landscape["Y"].min(),
            energy_landscape["Y"].max(),
        ],
        cmap="hot",
        aspect="auto",
    )
    plt.colorbar(im, ax=ax_c, label="Mean $E_k$")
    ax_c.set_xlabel("Dim 1")
    ax_c.set_ylabel("Dim 2")
    ax_c.set_title("C) Energy Landscape", fontweight="bold")

    # Panel D: Summary metrics
    ax_d = fig.add_subplot(gs[1, 1])
    x = np.arange(len(unique_names))
    width = 0.35

    means = [per_regime_metrics.get(name, {}).get("mean_energy", 0) for name in unique_names]
    cvs = [per_regime_metrics.get(name, {}).get("cv", 0) for name in unique_names]

    ax_d.bar(x - width / 2, means, width, label="Mean $E_k$", color="steelblue")
    ax_d2 = ax_d.twinx()
    ax_d2.bar(x + width / 2, cvs, width, label="CV", color="coral")

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(unique_names)
    ax_d.set_ylabel("Mean $E_k$", color="steelblue")
    ax_d2.set_ylabel("CV", color="coral")
    ax_d.set_title("D) Summary Metrics", fontweight="bold")

    # Combined legend
    lines1, labels1 = ax_d.get_legend_handles_labels()
    lines2, labels2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    return fig
