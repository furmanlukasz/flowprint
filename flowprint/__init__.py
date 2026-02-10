"""
FlowPrint: Flow-based fingerprints of dynamical regimes on shared manifolds.

A framework for characterizing dynamical regimes in multivariate time series using
latent trajectories, flow fields, and geometric fingerprints.
"""

__version__ = "0.1.0"

from flowprint.metrics import (
    compute_discriminability,
    compute_flow_metrics,
    compute_kinetic_energy,
)
from flowprint.simulation import (
    CoupledStuartLandauNetwork,
    SimulationResult,
)
from flowprint.visualization import (
    plot_discriminability,
    plot_electrode_timeseries,
    plot_flow_fields,
    plot_kinetic_energy,
)

__all__ = [
    # Simulation
    "CoupledStuartLandauNetwork",
    "SimulationResult",
    # Metrics
    "compute_flow_metrics",
    "compute_kinetic_energy",
    "compute_discriminability",
    # Visualization
    "plot_electrode_timeseries",
    "plot_flow_fields",
    "plot_discriminability",
    "plot_kinetic_energy",
]
