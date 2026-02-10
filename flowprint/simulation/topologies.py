"""
Graph topology generators and spectral analysis for coupled oscillator networks.

Provides adjacency matrix generators for different coupling topologies:
- Global (all-to-all): promotes full synchronization
- Cluster (modular): multi-cluster synchrony patterns
- Sparse (random): promotes desynchronization
- Ring (directional): promotes traveling wave patterns
"""

from __future__ import annotations

import numpy as np


def _normalize_adjacency(A: np.ndarray, mode: str = "mean_degree") -> np.ndarray:
    """
    Normalize adjacency to keep coupling stable across topologies.

    IMPORTANT: "max_eig" normalization equalizes different topologies, erasing
    the dynamical impact of topology structure. Use "mean_degree" instead to
    preserve topology-dependent dynamics while keeping coupling stable.

    Args:
        A: Adjacency matrix
        mode: Normalization mode
            - "mean_degree": divide by mean degree (RECOMMENDED)
            - "max_eig": divide by largest eigenvalue magnitude
            - "row": row-stochastic (rows sum to 1)
            - "none": no normalization

    Returns:
        Normalized adjacency matrix
    """
    A = np.asarray(A, dtype=float)
    if mode == "none":
        return A
    if mode == "row":
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return A / row_sums
    if mode == "mean_degree":
        mean_degree = A.sum() / A.shape[0]
        if mean_degree == 0:
            mean_degree = 1.0
        return A / mean_degree
    if mode == "max_eig":
        w = np.linalg.eigvals(A)
        lam = np.max(np.abs(w)) if w.size else 1.0
        if lam == 0:
            lam = 1.0
        return A / lam
    raise ValueError(f"Unknown normalization mode: {mode}")


def adjacency_global(n: int, self_loops: bool = False) -> np.ndarray:
    """
    All-to-all adjacency matrix.

    Creates complete graph coupling that promotes global synchronization.

    Args:
        n: Number of nodes
        self_loops: Whether to include self-connections

    Returns:
        Adjacency matrix (n x n)
    """
    A = np.ones((n, n), dtype=float)
    if not self_loops:
        np.fill_diagonal(A, 0.0)
    return A


def adjacency_clusters(
    n: int,
    n_clusters: int = 3,
    p_in: float = 1.0,
    p_out: float = 0.01,
    seed: int | None = None,
) -> np.ndarray:
    """
    Block-structured adjacency: dense within clusters, sparse between.

    Creates modular connectivity that produces multi-cluster synchrony patterns.

    Args:
        n: Number of nodes
        n_clusters: Number of clusters
        p_in: Within-cluster connection probability (default 1.0 = fully connected)
        p_out: Between-cluster connection probability (default 0.01 = very sparse)
        seed: Random seed

    Returns:
        Adjacency matrix (n x n)
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=float)
    labels = np.repeat(np.arange(n_clusters), np.ceil(n / n_clusters).astype(int))[:n]
    rng.shuffle(labels)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if labels[i] == labels[j]:
                if rng.random() < p_in:
                    A[i, j] = 1.0
            else:
                if rng.random() < p_out:
                    A[i, j] = 1.0
    return A


def adjacency_sparse(
    n: int,
    density: float = 0.05,
    directed: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """
    Random sparse adjacency (Erdos-Renyi).

    Creates sparse random connectivity that promotes desynchronization.

    Args:
        n: Number of nodes
        density: Edge probability
        directed: Whether graph is directed
        seed: Random seed

    Returns:
        Adjacency matrix (n x n)
    """
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < density).astype(float)
    np.fill_diagonal(A, 0.0)
    if not directed:
        A = np.maximum(A, A.T)
    return A


def adjacency_ring(
    n: int,
    k_neighbors: int = 2,
    directed: bool = True,
) -> np.ndarray:
    """
    Ring lattice adjacency.

    Creates ring connectivity that promotes traveling wave dynamics.
    Use directed=True (default) for asymmetric coupling that promotes
    directional wave propagation.

    Args:
        n: Number of nodes
        k_neighbors: Number of neighbors on each side
        directed: If True, creates asymmetric ring for traveling waves

    Returns:
        Adjacency matrix (n x n)
    """
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for d in range(1, k_neighbors + 1):
            j1 = (i + d) % n
            j2 = (i - d) % n
            A[i, j1] = 1.0
            A[i, j2] = 1.0
    if not directed:
        A = np.maximum(A, A.T)
    return A


def laplacian_from_adjacency(A: np.ndarray) -> np.ndarray:
    """
    Compute graph Laplacian L = D - A for diffusive coupling.

    Args:
        A: Adjacency matrix

    Returns:
        Laplacian matrix
    """
    A = np.asarray(A, dtype=float)
    deg = np.sum(A, axis=1)
    return np.diag(deg) - A


def compute_laplacian_spectrum(L: np.ndarray) -> dict[str, float]:
    """
    Compute spectral properties of the graph Laplacian.

    Key properties:
    - lambda_2 (algebraic connectivity): Larger = more connected/synchronizable
    - lambda_max: Largest eigenvalue, affects stability
    - spectral_gap: lambda_2 / lambda_max, measures synchronization efficiency

    For the 4 topologies:
    - Global: High lambda_2, small spectral gap
    - Cluster: Moderate lambda_2 with gap structure
    - Sparse: Low lambda_2 (poor connectivity)
    - Ring: Low lambda_2 but structured spectrum

    Args:
        L: Graph Laplacian matrix (n x n)

    Returns:
        Dict with spectral properties
    """
    L_sym = (L + L.T) / 2
    eigenvalues = np.linalg.eigvalsh(L_sym)
    eigenvalues = np.sort(np.real(eigenvalues))
    eigenvalues = np.where(np.abs(eigenvalues) < 1e-10, 0, eigenvalues)

    lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
    lambda_max = eigenvalues[-1] if len(eigenvalues) > 0 else 1.0
    spectral_gap = lambda_2 / lambda_max if lambda_max > 0 else 0.0

    nonzero_eigs = eigenvalues[eigenvalues > 1e-10]
    spectral_width = float(np.std(nonzero_eigs)) if len(nonzero_eigs) > 0 else 0.0
    n_components = int(np.sum(eigenvalues < 1e-10))

    return {
        "lambda_2": float(lambda_2),
        "lambda_max": float(lambda_max),
        "spectral_gap": float(spectral_gap),
        "spectral_width": float(spectral_width),
        "n_components": n_components,
        "eigenvalues": eigenvalues.tolist(),
    }
