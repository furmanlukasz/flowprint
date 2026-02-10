"""
Simulation module for coupled oscillator networks.
"""

from flowprint.simulation.coupled_oscillators import (
    CoupledStuartLandauNetwork,
    SimulationResult,
    euler_maruyama_step,
)
from flowprint.simulation.topologies import (
    adjacency_global,
    adjacency_clusters,
    adjacency_sparse,
    adjacency_ring,
    laplacian_from_adjacency,
    compute_laplacian_spectrum,
)
from flowprint.simulation.noise import generate_colored_noise

__all__ = [
    "CoupledStuartLandauNetwork",
    "SimulationResult",
    "euler_maruyama_step",
    "adjacency_global",
    "adjacency_clusters",
    "adjacency_sparse",
    "adjacency_ring",
    "laplacian_from_adjacency",
    "compute_laplacian_spectrum",
    "generate_colored_noise",
]
