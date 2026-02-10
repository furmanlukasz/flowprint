"""
Noise generation utilities for realistic observation models.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


def generate_colored_noise(
    n_samples: int,
    n_channels: int,
    sfreq: float,
    alpha: float = 1.0,
    scale: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate 1/f^alpha colored noise.

    Args:
        n_samples: Number of time samples
        n_channels: Number of channels
        sfreq: Sampling frequency (Hz)
        alpha: Spectral exponent (0=white, 1=pink, 2=brown)
        scale: Output amplitude scale
        seed: Random seed

    Returns:
        Colored noise array (n_channels, n_samples)
    """
    rng = np.random.default_rng(seed)

    # Generate white noise in frequency domain
    freqs = np.fft.rfftfreq(n_samples, d=1 / sfreq)

    # Avoid division by zero at DC
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0

    # 1/f^(alpha/2) filter (alpha/2 because we work with amplitude, not power)
    filt = 1.0 / (freqs ** (alpha / 2))
    filt[0] = 0  # Remove DC

    noise = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        # Random phases
        white = rng.normal(size=n_samples)
        white_fft = np.fft.rfft(white)
        colored_fft = white_fft * filt
        colored = np.fft.irfft(colored_fft, n=n_samples)
        # Normalize to unit variance then scale
        colored = colored / (np.std(colored) + 1e-10) * scale
        noise[ch, :] = colored

    return noise
