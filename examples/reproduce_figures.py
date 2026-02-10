#!/usr/bin/env python
"""
Reproduce all figures from the FlowPrint paper.

Usage:
    python reproduce_figures.py --output-dir figures/

This script generates:
- fig_electrode_timeseries.png/pdf
- fig_analysis_main.png/pdf
- fig_flow_fields.png/pdf
- fig_discriminability.png/pdf
- fig_kinetic_energy.png/pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap

# FlowPrint imports
from flowprint.simulation import CoupledStuartLandauNetwork
from flowprint.metrics import (
    compute_flow_metrics,
    compute_flow_field,
    compute_field_metrics,
    compute_kinetic_energy,
    compute_kinetic_energy_metrics,
    compute_energy_landscape,
    compute_regime_discriminability,
    compute_window_metrics,
)
from flowprint.visualization import (
    plot_electrode_timeseries,
    plot_main_analysis,
    plot_flow_fields,
    plot_discriminability,
    plot_kinetic_energy,
)


def extract_phase_representation(
    observations: np.ndarray,
    sfreq: float,
    lowcut: float = 2.0,
    highcut: float = 48.0,
) -> np.ndarray:
    """
    Extract circular phase-amplitude representation from observations.

    Args:
        observations: (n_channels, n_samples) raw signals
        sfreq: Sampling frequency
        lowcut, highcut: Bandpass filter cutoffs

    Returns:
        (3 * n_channels, n_samples) phase representation
    """
    from scipy.signal import butter, filtfilt, hilbert

    n_channels, n_samples = observations.shape

    # Bandpass filter
    nyq = sfreq / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, observations, axis=1)

    # Hilbert transform
    analytic = hilbert(filtered, axis=1)
    phase = np.angle(analytic)
    amplitude = np.abs(analytic)

    # Circular representation: [cos(phi), sin(phi), log(1+amp)]
    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)
    log_amp = np.log1p(amplitude)

    # Stack: (3*n_channels, n_samples)
    representation = np.vstack([cos_phase, sin_phase, log_amp])

    return representation


def create_simple_autoencoder(input_dim: int, latent_dim: int = 32):
    """Create a simple convolutional autoencoder."""
    import torch
    import torch.nn as nn

    class ConvAutoencoder(nn.Module):
        def __init__(self, in_features, latent_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, in_features),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

        def encode(self, x):
            return self.encoder(x)

    return ConvAutoencoder(input_dim, latent_dim)


def train_autoencoder(model, data, n_epochs=50, batch_size=64, lr=1e-3):
    """Train the autoencoder."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = TensorDataset(torch.FloatTensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss = {total_loss/len(loader):.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Reproduce FlowPrint paper figures")
    parser.add_argument("--output-dir", type=str, default="figures",
                        help="Output directory for figures")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--duration", type=float, default=160.0,
                        help="Simulation duration in seconds")
    parser.add_argument("--n-epochs", type=int, default=50,
                        help="Autoencoder training epochs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FlowPrint: Reproducing Paper Figures")
    print("=" * 70)

    # =========================================================================
    # Step 1: Generate Simulation
    # =========================================================================
    print("\n[Step 1] Generating coupled Stuart-Landau simulation...")

    net = CoupledStuartLandauNetwork(
        n_oscillators=30,
        n_channels=30,
        sfreq=250.0,
        seed=args.seed,
    )
    net.default_topologies(seed=args.seed)

    # Regime schedule: 10s per regime, 4 cycles
    regime_order = ["global", "cluster", "sparse", "ring"]
    n_cycles = int(args.duration / (4 * 10))
    schedule = [(name, 10.0) for _ in range(n_cycles) for name in regime_order]

    result = net.generate(
        total_duration_s=args.duration,
        regime_schedule=schedule,
        coupling_strength=5.0,
        noise_std=0.1,
        obs_noise_std=0.05,
        obs_noise_color=1.0,
        transition_s=0.3,
    )

    sfreq = result.params["sfreq"]
    print(f"  Duration: {result.t[-1]:.1f}s ({len(result.t)} samples)")
    print(f"  Regimes: {list(dict.fromkeys(result.regime_names))}")

    # =========================================================================
    # Step 2: Extract Phase Representation
    # =========================================================================
    print("\n[Step 2] Extracting phase representation...")

    phase_data = extract_phase_representation(result.y, sfreq)
    print(f"  Phase shape: {phase_data.shape}")

    # Chunk into windows for autoencoder
    chunk_samples = int(5.0 * sfreq)  # 5 second chunks
    n_chunks = phase_data.shape[1] // chunk_samples
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        chunk = phase_data[:, start:end].T  # (time, features)
        chunks.append(chunk.flatten())

    chunks = np.array(chunks)
    print(f"  Chunks: {chunks.shape}")

    # =========================================================================
    # Step 3: Train Autoencoder
    # =========================================================================
    print("\n[Step 3] Training autoencoder...")

    import torch
    model = create_simple_autoencoder(chunks.shape[1], latent_dim=32)
    model = train_autoencoder(model, chunks, n_epochs=args.n_epochs)

    # Compute latent representation
    model.set_to_eval_mode = lambda: model.eval()  # Workaround
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        latent = model.encode(torch.FloatTensor(chunks).to(device)).cpu().numpy()
    print(f"  Latent shape: {latent.shape}")

    # =========================================================================
    # Step 4: Embed and Compute Metrics
    # =========================================================================
    print("\n[Step 4] Embedding and computing metrics...")

    # Standardize latent
    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent)

    # UMAP embedding
    embedder = umap.UMAP(n_components=2, random_state=args.seed, n_jobs=1)
    embedded = embedder.fit_transform(latent_scaled)
    print(f"  Embedded shape: {embedded.shape}")

    # Align regime labels with latent (downsampled)
    compression = len(result.regime_id) / len(latent)
    labels_aligned = np.array([
        result.regime_id[min(int(i * compression), len(result.regime_id) - 1)]
        for i in range(len(latent))
    ])

    # Compute flow metrics per regime
    unique_names = list(dict.fromkeys(result.regime_names))
    flow_metrics = {}
    regime_flow_data = {}

    for name in unique_names:
        matching_ids = [i for i, n in enumerate(result.regime_names) if n == name]
        mask = np.isin(labels_aligned, matching_ids)
        regime_latent = latent_scaled[mask]
        regime_embedded = embedded[mask]

        # Flow metrics on full latent
        metrics = compute_flow_metrics(regime_latent)
        flow_metrics[name] = metrics

        # Flow field on 2D
        from flowprint.metrics.flow_metrics import compute_velocity
        velocity = compute_velocity(regime_embedded)

        bounds = (embedded[:, 0].min(), embedded[:, 0].max(),
                  embedded[:, 1].min(), embedded[:, 1].max())

        ff = compute_flow_field(regime_embedded[:-1], velocity[:-1], bounds)
        field_metrics = compute_field_metrics(ff)

        regime_flow_data[name] = {
            "embedded": regime_embedded,
            "flow_field": ff,
            "field_metrics": field_metrics,
        }

        print(f"  {name}: speed={metrics['speed']:.3f}, var={metrics['explored_variance']:.2f}")

    # Discriminability
    print("\n[Step 5] Computing discriminability...")
    disc = compute_regime_discriminability(latent_scaled, labels_aligned)
    for metric, results in disc.items():
        print(f"  {metric}: eta_sq={results['eta_squared']:.3f} ({results['effect_size']})")

    # =========================================================================
    # Step 6: Generate Figures
    # =========================================================================
    print("\n[Step 6] Generating figures...")

    # Figure 1: Electrode time series
    fig1 = plot_electrode_timeseries(
        result.y, result.t, result.switch_times, sfreq,
        time_window=(0, 60),
    )
    fig1.savefig(output_dir / "fig_electrode_timeseries.png", dpi=150, bbox_inches="tight")
    fig1.savefig(output_dir / "fig_electrode_timeseries.pdf", dpi=300, bbox_inches="tight")
    print("  Saved: fig_electrode_timeseries")

    # Figure 2: Main analysis
    # Compute combined flow field
    from flowprint.metrics.flow_metrics import compute_velocity
    velocity_all = compute_velocity(embedded)
    bounds = (embedded[:, 0].min(), embedded[:, 0].max(),
              embedded[:, 1].min(), embedded[:, 1].max())
    ff_all = compute_flow_field(embedded[:-1], velocity_all[:-1], bounds)

    fig2 = plot_main_analysis(
        embedded, labels_aligned, result.regime_names,
        result.switch_times, args.duration, ff_all, flow_metrics,
    )
    fig2.savefig(output_dir / "fig_analysis_main.png", dpi=150, bbox_inches="tight")
    fig2.savefig(output_dir / "fig_analysis_main.pdf", dpi=300, bbox_inches="tight")
    print("  Saved: fig_analysis_main")

    # Figure 3: Flow fields per regime
    fig3 = plot_flow_fields(regime_flow_data)
    fig3.savefig(output_dir / "fig_flow_fields.png", dpi=150, bbox_inches="tight")
    fig3.savefig(output_dir / "fig_flow_fields.pdf", dpi=300, bbox_inches="tight")
    print("  Saved: fig_flow_fields")

    # Figure 4: Discriminability
    window_metrics = compute_window_metrics(latent_scaled)
    n_windows = len(window_metrics["speed"])
    window_labels = np.array([
        labels_aligned[i * 50] for i in range(n_windows)
        if i * 50 < len(labels_aligned)
    ])[:n_windows]

    fig4 = plot_discriminability(window_metrics, window_labels, result.regime_names, disc)
    fig4.savefig(output_dir / "fig_discriminability.png", dpi=150, bbox_inches="tight")
    fig4.savefig(output_dir / "fig_discriminability.pdf", dpi=300, bbox_inches="tight")
    print("  Saved: fig_discriminability")

    # Figure 5: Kinetic energy
    energy = compute_kinetic_energy(latent_scaled, trim_edges=True)
    trim = 10  # Match trimming
    time_trimmed = np.linspace(0, args.duration, len(energy))
    labels_trimmed = labels_aligned[trim:-trim][:len(energy)]

    energy_landscape = compute_energy_landscape(
        energy, embedded[trim:-trim][:len(energy)], bounds
    )

    per_regime_energy = {}
    for name in unique_names:
        matching_ids = [i for i, n in enumerate(result.regime_names) if n == name]
        mask = np.isin(labels_trimmed, matching_ids)
        regime_energy = energy[mask]
        per_regime_energy[name] = compute_kinetic_energy_metrics(regime_energy)

    fig5 = plot_kinetic_energy(
        energy, time_trimmed, labels_trimmed, result.regime_names,
        energy_landscape, per_regime_energy,
    )
    fig5.savefig(output_dir / "fig_kinetic_energy.png", dpi=150, bbox_inches="tight")
    fig5.savefig(output_dir / "fig_kinetic_energy.pdf", dpi=300, bbox_inches="tight")
    print("  Saved: fig_kinetic_energy")

    # Save results summary
    results_summary = {
        "simulation": {
            "duration_s": args.duration,
            "n_channels": 30,
            "n_oscillators": 30,
            "sfreq": sfreq,
            "regimes": unique_names,
        },
        "flow_metrics": {name: {k: float(v) for k, v in m.items()}
                        for name, m in flow_metrics.items()},
        "discriminability": {
            metric: {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                     for k, v in r.items()}
            for metric, r in disc.items()
        },
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("DONE! All figures saved to:", output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
