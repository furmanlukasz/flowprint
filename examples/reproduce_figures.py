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

IMPORTANT: Uses sliding-window convolutional encoder for proper temporal
resolution (~2800 latent points from 160s, not 32).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from sklearn.preprocessing import StandardScaler
import umap

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
)
from flowprint.metrics.discriminability import compute_per_regime_window_metrics
from flowprint.visualization import (
    plot_electrode_timeseries,
    plot_main_analysis,
    plot_flow_fields,
    plot_kinetic_energy,
)
from flowprint.visualization.figures import plot_discriminability_per_regime


def extract_phase_representation(
    observations: np.ndarray,
    sfreq: float,
    lowcut: float = 1.0,
    highcut: float = 30.0,
) -> np.ndarray:
    """
    Extract circular phase-amplitude representation from observations.

    Args:
        observations: (n_channels, n_samples) raw signals
        sfreq: Sampling frequency
        lowcut, highcut: Bandpass filter cutoffs

    Returns:
        (3 * n_channels, n_samples) phase representation [cos, sin, log_amp]
    """
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


# =============================================================================
# CONVOLUTIONAL AUTOENCODER (sliding window, preserves temporal resolution)
# =============================================================================

class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder that preserves temporal resolution.

    Unlike chunk-flatten-MLP approaches, this uses strided convolutions
    that compress time by ~4x, giving ~2800 latent points from 160s.

    Input: (batch, n_features, time) where n_features = 3 * n_channels
    Output: reconstruction, latent of shape (batch, time', hidden_size)
    """

    def __init__(self, n_channels: int, hidden_size: int = 32, phase_channels: int = 3):
        super().__init__()
        input_size = n_channels * phase_channels

        # Encoder: Conv1d with stride=2 for 4x compression
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )

        # LSTM for sequential encoding
        self.encoder_lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_size, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass: returns (reconstruction, latent)."""
        # Encode
        h = self.encoder_conv(x)  # (batch, 128, time')
        h = h.permute(0, 2, 1)    # (batch, time', 128)
        latent, _ = self.encoder_lstm(h)  # (batch, time', hidden_size)

        # Decode
        h_dec, _ = self.decoder_lstm(latent)  # (batch, time', 128)
        h_dec = h_dec.permute(0, 2, 1)        # (batch, 128, time')
        reconstruction = self.decoder_conv(h_dec)  # (batch, input_size, time)

        # Handle size mismatch
        if reconstruction.shape[2] != x.shape[2]:
            reconstruction = nn.functional.interpolate(
                reconstruction, size=x.shape[2], mode='linear', align_corners=False
            )

        return reconstruction, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space."""
        h = self.encoder_conv(x)
        h = h.permute(0, 2, 1)
        latent, _ = self.encoder_lstm(h)
        return latent


def chunk_phase_data(phase_data: np.ndarray, chunk_samples: int) -> list:
    """Chunk phase data into overlapping windows for training."""
    n_features, n_samples = phase_data.shape
    chunks = []
    stride = chunk_samples // 2  # 50% overlap for training

    for start in range(0, n_samples - chunk_samples + 1, stride):
        chunk = phase_data[:, start:start + chunk_samples]
        chunks.append(chunk)

    return chunks


def train_autoencoder(
    phase_data: np.ndarray,
    n_channels: int,
    hidden_size: int = 32,
    n_epochs: int = 50,
    chunk_duration: float = 5.0,
    sfreq: float = 250.0,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> ConvAutoencoder:
    """
    Train convolutional autoencoder on phase data.

    Uses overlapping chunks for training but encodes full signal for inference.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    # Chunk data for training
    chunk_samples = int(chunk_duration * sfreq)
    chunks = chunk_phase_data(phase_data, chunk_samples)
    print(f"  Training chunks: {len(chunks)} (from {phase_data.shape[1]} samples)")

    # Create dataset
    data = torch.stack([torch.from_numpy(c) for c in chunks])
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = ConvAutoencoder(n_channels=n_channels, hidden_size=hidden_size)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Train
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for (batch,) in loader:
            batch = batch.float().to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss = {total_loss/len(loader):.4f}")

    model.eval()
    return model


def compute_latent_trajectory(
    model: ConvAutoencoder,
    phase_data: np.ndarray,
) -> np.ndarray:
    """
    Compute latent trajectory by encoding the FULL signal.

    This is the key difference from chunk-flatten approaches:
    we get one latent vector per ~4 samples, not per 5-second chunk.

    Returns:
        (T', hidden_size) latent trajectory where T' ≈ T/4
    """
    device = next(model.parameters()).device
    model.eval()

    # Add batch dimension and encode full signal
    x = torch.from_numpy(phase_data).float().unsqueeze(0).to(device)

    with torch.no_grad():
        latent = model.encode(x)

    # Remove batch dimension: (1, T', hidden) -> (T', hidden)
    return latent.squeeze(0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Reproduce FlowPrint paper figures")
    parser.add_argument("--output-dir", type=str, default="figures",
                        help="Output directory for figures")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--duration", type=float, default=None,
                        help="Simulation duration (default: auto from n_cycles * 4 * regime_duration)")
    parser.add_argument("--n-epochs", type=int, default=50,
                        help="Autoencoder training epochs")
    parser.add_argument("--n-cycles", type=int, default=4,
                        help="Number of regime cycles (default: 4)")
    parser.add_argument("--regime-duration", type=float, default=10.0,
                        help="Duration per regime in seconds (default: 10s)")
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

    # Regime schedule: default 10s per regime, 4 cycles = 160s of regime data
    regime_order = ["global", "cluster", "sparse", "ring"]
    schedule = [(name, args.regime_duration) for _ in range(args.n_cycles) for name in regime_order]

    # Calculate actual duration from schedule (overrides --duration if not set)
    scheduled_duration = args.n_cycles * 4 * args.regime_duration
    if args.duration is not None and abs(args.duration - scheduled_duration) > 0.1:
        print(f"  Note: --duration={args.duration}s overridden by schedule={scheduled_duration}s")
    total_duration = scheduled_duration
    print(f"  Schedule: {args.n_cycles} cycles × 4 regimes × {args.regime_duration}s = {total_duration}s")

    result = net.generate(
        total_duration_s=scheduled_duration,
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

    # =========================================================================
    # Step 3: Train Convolutional Autoencoder
    # =========================================================================
    print("\n[Step 3] Training convolutional autoencoder...")

    model = train_autoencoder(
        phase_data,
        n_channels=30,
        hidden_size=32,
        n_epochs=args.n_epochs,
    )

    # =========================================================================
    # Step 4: Compute Full Latent Trajectory
    # =========================================================================
    print("\n[Step 4] Computing latent trajectory (full resolution)...")

    latent = compute_latent_trajectory(model, phase_data)
    print(f"  Latent shape: {latent.shape}")
    print(f"  Compression ratio: {phase_data.shape[1] / latent.shape[0]:.1f}x")

    # =========================================================================
    # Step 5: Embed and Compute Metrics
    # =========================================================================
    print("\n[Step 5] Embedding and computing metrics...")

    # Standardize latent
    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent)
    print(f"  Latent variance after scaling: {latent_scaled.var(axis=0).mean():.4f}")

    # Clip outliers (matches original pipeline)
    p99 = np.percentile(np.abs(latent_scaled), 99)
    clip_threshold = max(3.0, p99)
    latent_clipped = np.clip(latent_scaled, -clip_threshold, clip_threshold)
    n_clipped = (np.abs(latent_scaled) > clip_threshold).sum()
    if n_clipped > 0:
        print(f"  Clipped {n_clipped} extreme values (>{clip_threshold:.1f} std)")

    # UMAP embedding (on clipped latent) - match original paper settings
    embedder = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        random_state=args.seed,
        n_jobs=1
    )
    embedded = embedder.fit_transform(latent_clipped)
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
        regime_latent = latent_clipped[mask]
        regime_embedded = embedded[mask]

        # Flow metrics on full latent
        metrics = compute_flow_metrics(regime_latent)
        flow_metrics[name] = metrics

        # Flow field on 2D
        from flowprint.metrics.flow_metrics import compute_velocity
        velocity = compute_velocity(regime_embedded)

        bounds = (embedded[:, 0].min(), embedded[:, 0].max(),
                  embedded[:, 1].min(), embedded[:, 1].max())

        if len(regime_embedded) > 10:
            ff = compute_flow_field(regime_embedded[:-1], velocity[:-1], bounds)
            field_metrics = compute_field_metrics(ff)

            regime_flow_data[name] = {
                "embedded": regime_embedded,
                "flow_field": ff,
                "field_metrics": field_metrics,
            }

        print(f"  {name}: speed={metrics['speed']:.4f}, var={metrics['explored_variance']:.2f}, n={mask.sum()}")

    # Discriminability (using per-regime window computation)
    print("\n[Step 6] Computing discriminability...")

    # Window size: 50 samples (matches original code, ~0.2s at 250Hz after 4x compression)
    window_size = 50
    print(f"  Window size: {window_size} samples (~{len(latent_clipped) // window_size} total windows)")

    # Compute discriminability with proper per-regime windowing
    disc = compute_regime_discriminability(
        latent_clipped, labels_aligned,
        window_size=window_size,
        regime_names=result.regime_names
    )

    # Also compute per-regime window metrics for violin plots
    regime_window_metrics = compute_per_regime_window_metrics(
        latent_clipped, labels_aligned, result.regime_names, window_size
    )

    for metric, results in disc.items():
        eta = results['eta_squared']
        f_stat = results.get('f_statistic', float('nan'))
        n_win = results.get('n_windows', [])
        if not np.isnan(f_stat):
            print(f"  {metric}: F={f_stat:.1f}, η²={eta:.3f} ({results['effect_size']}) [n={n_win}]")
        else:
            print(f"  {metric}: η²={eta:.3f} ({results['effect_size']})")

    # =========================================================================
    # Step 7: Generate Figures
    # =========================================================================
    print("\n[Step 7] Generating figures...")

    # Figure 1: Electrode time series
    fig1 = plot_electrode_timeseries(
        result.y, result.t, result.switch_times, sfreq,
        time_window=(0, 60),
    )
    fig1.savefig(output_dir / "fig_electrode_timeseries.png", dpi=150, bbox_inches="tight")
    fig1.savefig(output_dir / "fig_electrode_timeseries.pdf", dpi=300, bbox_inches="tight")
    print("  Saved: fig_electrode_timeseries")

    # Figure 2: Main analysis
    from flowprint.metrics.flow_metrics import compute_velocity
    velocity_all = compute_velocity(embedded)
    bounds = (embedded[:, 0].min(), embedded[:, 0].max(),
              embedded[:, 1].min(), embedded[:, 1].max())
    ff_all = compute_flow_field(embedded[:-1], velocity_all[:-1], bounds)

    fig2 = plot_main_analysis(
        embedded, labels_aligned, result.regime_names,
        result.switch_times, total_duration, ff_all, flow_metrics,
    )
    fig2.savefig(output_dir / "fig_analysis_main.png", dpi=150, bbox_inches="tight")
    fig2.savefig(output_dir / "fig_analysis_main.pdf", dpi=300, bbox_inches="tight")
    print("  Saved: fig_analysis_main")

    # Figure 3: Flow fields per regime
    fig3 = plot_flow_fields(regime_flow_data)
    fig3.savefig(output_dir / "fig_flow_fields.png", dpi=150, bbox_inches="tight")
    fig3.savefig(output_dir / "fig_flow_fields.pdf", dpi=300, bbox_inches="tight")
    print("  Saved: fig_flow_fields")

    # Figure 4: Discriminability (using per-regime window metrics)
    fig4 = plot_discriminability_per_regime(
        regime_window_metrics, unique_names, disc
    )
    fig4.savefig(output_dir / "fig_discriminability.png", dpi=150, bbox_inches="tight")
    fig4.savefig(output_dir / "fig_discriminability.pdf", dpi=300, bbox_inches="tight")
    print("  Saved: fig_discriminability")

    # Figure 5: Kinetic energy
    energy = compute_kinetic_energy(latent_clipped, trim_edges=True)
    trim = 10
    time_trimmed = np.linspace(0, total_duration, len(energy))
    labels_trimmed = labels_aligned[trim:-trim][:len(energy)]

    energy_landscape = compute_energy_landscape(
        energy, embedded[trim:-trim][:len(energy)], bounds
    )

    per_regime_energy = {}
    for name in unique_names:
        matching_ids = [i for i, n in enumerate(result.regime_names) if n == name]
        mask = np.isin(labels_trimmed, matching_ids)
        if mask.sum() > 0:
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
            "duration_s": total_duration,
            "n_channels": 30,
            "n_oscillators": 30,
            "sfreq": sfreq,
            "regimes": unique_names,
        },
        "latent_shape": list(latent.shape),
        "compression_ratio": float(phase_data.shape[1] / latent.shape[0]),
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
