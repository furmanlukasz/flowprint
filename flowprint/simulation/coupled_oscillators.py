"""
Coupled Stuart-Landau oscillator network simulator with topology switching.

The Stuart-Landau equation describes the normal form of a Hopf bifurcation
and provides a canonical model for limit-cycle oscillators.

Key features:
- Multivariate oscillatory dynamics with explicit phase-amplitude structure
- Regime switching via coupling topology (adjacency/Laplacian)
- Ground-truth regime labels and transition times
- Linear observation model (mixing to channels)
- Euler-Maruyama integration (SDE with additive Gaussian noise)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from flowprint.simulation.topologies import (
    adjacency_global,
    adjacency_clusters,
    adjacency_sparse,
    adjacency_ring,
    laplacian_from_adjacency,
    _normalize_adjacency,
)
from flowprint.simulation.noise import generate_colored_noise


@dataclass
class SimulationResult:
    """Container for simulated data and ground truth."""

    y: np.ndarray  # (n_channels, n_samples) observations
    z: np.ndarray  # (n_oscillators, n_samples) complex oscillator states
    t: np.ndarray  # (n_samples,) time vector in seconds
    regime_names: List[str]  # regime names in order of appearance
    regime_id: np.ndarray  # (n_samples,) integer regime index per sample
    switch_times: List[float]  # switch times in seconds
    params: Dict[str, object]  # simulation parameters


def euler_maruyama_step(
    z: np.ndarray,
    mu: np.ndarray,
    omega: np.ndarray,
    dt: float,
    noise_std: float,
    L: Optional[np.ndarray] = None,
    coupling_strength: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    One Euler-Maruyama step for coupled Stuart-Landau oscillators.

    dz = [(mu + i*omega) z - |z|^2 z - coupling * L @ z] dt + noise * sqrt(dt) dW

    Args:
        z: Complex oscillator states (n,)
        mu: Limit cycle amplitude parameters (n,)
        omega: Natural frequencies in rad/s (n,)
        dt: Timestep in seconds
        noise_std: Additive complex noise std
        L: Laplacian matrix for coupling (n x n)
        coupling_strength: Global coupling scaling
        rng: Random generator

    Returns:
        Updated complex states (n,)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Intrinsic drift: Stuart-Landau dynamics
    drift = (mu + 1j * omega) * z - (np.abs(z) ** 2) * z

    # Coupling drift via Laplacian
    if L is not None and coupling_strength != 0.0:
        drift = drift - coupling_strength * (L @ z)

    # Additive complex noise
    if noise_std > 0:
        dW = (rng.normal(size=z.shape) + 1j * rng.normal(size=z.shape)) * np.sqrt(dt)
        z = z + drift * dt + noise_std * dW
    else:
        z = z + drift * dt

    return z


class CoupledStuartLandauNetwork:
    """
    Coupled Stuart-Landau oscillator network with switching coupling topology.

    Regime switching is implemented by switching the coupling Laplacian L(t),
    optionally with smooth transitions.

    Example:
        >>> net = CoupledStuartLandauNetwork(n_oscillators=30, n_channels=30)
        >>> net.default_topologies()
        >>> result = net.generate(total_duration_s=160.0)
    """

    def __init__(
        self,
        n_oscillators: int = 30,
        n_channels: int = 30,
        sfreq: float = 250.0,
        seed: Optional[int] = 0,
        mixing: str = "random",
        mixing_scale: float = 1.0,
    ) -> None:
        """
        Initialize the network.

        Args:
            n_oscillators: Number of oscillators
            n_channels: Number of observation channels
            sfreq: Sampling frequency (Hz)
            seed: Random seed
            mixing: Mixing type ("random" or "identity")
            mixing_scale: Scale for mixing matrix
        """
        self.n_osc = int(n_oscillators)
        self.n_ch = int(n_channels)
        self.sfreq = float(sfreq)
        self.dt = 1.0 / self.sfreq
        self.rng = np.random.default_rng(seed)

        # Mixing matrix W: (n_channels, n_oscillators)
        if mixing == "identity":
            if self.n_ch != self.n_osc:
                raise ValueError("mixing='identity' requires n_channels == n_oscillators")
            W = np.eye(self.n_ch, dtype=float)
        elif mixing == "random":
            W = self.rng.normal(size=(self.n_ch, self.n_osc))
            col_norm = np.linalg.norm(W, axis=0, keepdims=True)
            col_norm[col_norm == 0] = 1.0
            W = W / col_norm
        else:
            raise ValueError("mixing must be 'random' or 'identity'")
        self.W = mixing_scale * W

        self._topologies: Dict[str, np.ndarray] = {}
        self._laplacians: Dict[str, np.ndarray] = {}

    def set_topologies(
        self, topologies: Dict[str, np.ndarray], normalize: str = "mean_degree"
    ) -> None:
        """
        Register coupling topologies by name.

        Args:
            topologies: Dict mapping name -> adjacency matrix
            normalize: Normalization mode ("mean_degree", "max_eig", "row", "none")
        """
        self._topologies = {}
        self._laplacians = {}
        for name, A in topologies.items():
            A = np.asarray(A, dtype=float)
            if A.shape != (self.n_osc, self.n_osc):
                raise ValueError(
                    f"Topology '{name}' has shape {A.shape}, expected {(self.n_osc, self.n_osc)}"
                )
            A = _normalize_adjacency(A, mode=normalize)
            L = laplacian_from_adjacency(A)
            self._topologies[name] = A
            self._laplacians[name] = L

    def default_topologies(self, seed: Optional[int] = None) -> None:
        """
        Create standard set of four named topologies with high contrast.

        - global: all-to-all (promotes full synchronization)
        - cluster: strongly modular (3 clusters)
        - sparse: very sparse random (promotes desynchronization)
        - ring: directed ring (promotes traveling waves)
        """
        if seed is None:
            seed = int(self.rng.integers(0, 10_000_000))
        tops = {
            "global": adjacency_global(self.n_osc, self_loops=False),
            "cluster": adjacency_clusters(
                self.n_osc, n_clusters=3, p_in=1.0, p_out=0.01, seed=seed
            ),
            "sparse": adjacency_sparse(
                self.n_osc, density=0.03, directed=False, seed=seed + 1
            ),
            "ring": adjacency_ring(self.n_osc, k_neighbors=2, directed=True),
        }
        self.set_topologies(tops, normalize="mean_degree")

    @property
    def topologies(self) -> Dict[str, np.ndarray]:
        """Get registered topology adjacency matrices."""
        return self._topologies.copy()

    @property
    def laplacians(self) -> Dict[str, np.ndarray]:
        """Get registered topology Laplacians."""
        return self._laplacians.copy()

    def generate(
        self,
        total_duration_s: float = 160.0,
        regime_schedule: Optional[List[Tuple[str, float]]] = None,
        mu_mean: float = 1.0,
        mu_std: float = 0.2,
        omega_mean_hz: float = 10.0,
        omega_std_hz: float = 2.0,
        omega_gradient_hz: float = 2.0,
        coupling_strength: float = 5.0,
        noise_std: float = 0.1,
        obs_noise_std: float = 0.05,
        obs_noise_color: float = 1.0,
        transition_s: float = 0.3,
        z0: Optional[np.ndarray] = None,
    ) -> SimulationResult:
        """
        Generate time series with scheduled regime switches.

        Args:
            total_duration_s: Total duration in seconds
            regime_schedule: List of (regime_name, duration_s). If None, cycles
                through all topologies with equal duration.
            mu_mean, mu_std: Oscillator mu distribution (limit cycle radius)
            omega_mean_hz, omega_std_hz: Frequency distribution in Hz
            omega_gradient_hz: Frequency gradient for ring topology (traveling waves)
            coupling_strength: Laplacian coupling strength
            noise_std: Oscillator noise std
            obs_noise_std: Observation noise std
            obs_noise_color: Noise color exponent (0=white, 1=pink, 2=brown)
            transition_s: Smooth transition duration at switches
            z0: Initial complex state (optional)

        Returns:
            SimulationResult with observations, states, and metadata
        """
        if not self._laplacians:
            raise RuntimeError("No topologies set. Call default_topologies() first.")

        n_steps = int(np.round(total_duration_s * self.sfreq))
        t = np.arange(n_steps) / self.sfreq

        # Regime schedule
        if regime_schedule is None:
            names = list(self._laplacians.keys())
            per = total_duration_s / len(names)
            regime_schedule = [(nm, per) for nm in names]

        # Expand schedule to per-sample regime id
        regime_names: List[str] = []
        regime_id = np.zeros(n_steps, dtype=int)
        switch_times: List[float] = [0.0]
        cursor = 0
        for nm, dur_s in regime_schedule:
            if nm not in self._laplacians:
                raise ValueError(f"Regime '{nm}' not found in topologies.")
            n = int(np.round(dur_s * self.sfreq))
            if n <= 0:
                continue
            end = min(n_steps, cursor + n)
            if cursor >= n_steps:
                break
            if (not regime_names) or (regime_names[-1] != nm):
                regime_names.append(nm)
            rid = len(regime_names) - 1
            regime_id[cursor:end] = rid
            cursor = end
            if cursor < n_steps:
                switch_times.append(cursor / self.sfreq)

        if cursor < n_steps:
            regime_id[cursor:] = regime_id[cursor - 1] if cursor > 0 else 0

        # Per-oscillator parameters
        mu = self.rng.normal(loc=mu_mean, scale=mu_std, size=self.n_osc)
        omega_base = 2 * np.pi * self.rng.normal(
            loc=omega_mean_hz, scale=omega_std_hz, size=self.n_osc
        )

        # Frequency gradient for ring topology
        omega_gradient = 2 * np.pi * np.linspace(
            -omega_gradient_hz / 2, omega_gradient_hz / 2, self.n_osc
        )

        omega_per_regime = {}
        for name in regime_names:
            if name == "ring":
                omega_per_regime[name] = 2 * np.pi * omega_mean_hz + omega_gradient
            else:
                omega_per_regime[name] = omega_base

        # Initialize near limit cycle
        if z0 is None:
            r0 = np.sqrt(np.maximum(mu, 0.05))
            phase0 = self.rng.uniform(0, 2 * np.pi, size=self.n_osc)
            z = r0 * np.exp(1j * phase0)
        else:
            z = np.asarray(z0, dtype=complex).copy()

        Z = np.zeros((self.n_osc, n_steps), dtype=np.complex128)
        unique_L = [self._laplacians[nm] for nm in regime_names]
        trans_steps = int(np.round(transition_s * self.sfreq)) if transition_s > 0 else 0

        in_transition = False
        transition_counter = 0
        prev_L = None
        target_L = None
        prev_omega = None
        target_omega = None

        for i in range(n_steps):
            rid = regime_id[i]
            regime_name = regime_names[rid]
            L_current = unique_L[rid]
            omega_current = omega_per_regime[regime_name]

            # Smooth transitions
            if trans_steps > 0 and i > 0:
                prev_rid = regime_id[i - 1]
                if prev_rid != rid and not in_transition:
                    in_transition = True
                    transition_counter = 0
                    prev_L = unique_L[prev_rid]
                    target_L = L_current
                    prev_omega = omega_per_regime[regime_names[prev_rid]]
                    target_omega = omega_current

                if in_transition:
                    transition_counter += 1
                    alpha = min(1.0, transition_counter / trans_steps)
                    alpha = 0.5 * (1 + np.tanh(4 * (alpha - 0.5)))
                    L = (1 - alpha) * prev_L + alpha * target_L
                    omega = (1 - alpha) * prev_omega + alpha * target_omega
                    if transition_counter >= trans_steps:
                        in_transition = False
                else:
                    L = L_current
                    omega = omega_current
            else:
                L = L_current
                omega = omega_current

            Z[:, i] = z
            z = euler_maruyama_step(
                z=z,
                mu=mu,
                omega=omega,
                dt=self.dt,
                noise_std=noise_std,
                L=L,
                coupling_strength=coupling_strength,
                rng=self.rng,
            )

        # Observation model: y = W @ Re(z) + noise
        y = self.W @ np.real(Z)
        if obs_noise_std > 0:
            if obs_noise_color > 0:
                colored = generate_colored_noise(
                    n_samples=n_steps,
                    n_channels=self.n_ch,
                    sfreq=self.sfreq,
                    alpha=obs_noise_color,
                    scale=obs_noise_std,
                    seed=int(self.rng.integers(0, 10_000_000)),
                )
                y = y + colored
            else:
                y = y + self.rng.normal(scale=obs_noise_std, size=y.shape)

        params = dict(
            n_oscillators=self.n_osc,
            n_channels=self.n_ch,
            sfreq=self.sfreq,
            dt=self.dt,
            mu_mean=mu_mean,
            mu_std=mu_std,
            omega_mean_hz=omega_mean_hz,
            omega_std_hz=omega_std_hz,
            coupling_strength=coupling_strength,
            noise_std=noise_std,
            obs_noise_std=obs_noise_std,
            obs_noise_color=obs_noise_color,
            transition_s=transition_s,
            topologies=list(self._laplacians.keys()),
            regime_schedule=regime_schedule,
        )

        return SimulationResult(
            y=y,
            z=Z,
            t=t,
            regime_names=regime_names,
            regime_id=regime_id,
            switch_times=switch_times,
            params=params,
        )
