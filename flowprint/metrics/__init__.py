"""
Metrics module for flow-based trajectory analysis.
"""

from flowprint.metrics.flow_metrics import (
    compute_flow_metrics,
    compute_velocity,
    compute_flow_field,
    compute_field_metrics,
)
from flowprint.metrics.kinetic_energy import (
    compute_kinetic_energy,
    compute_kinetic_energy_metrics,
    compute_energy_landscape,
)
from flowprint.metrics.discriminability import (
    compute_discriminability,
    compute_window_metrics,
    compute_regime_discriminability,
)

__all__ = [
    "compute_flow_metrics",
    "compute_velocity",
    "compute_flow_field",
    "compute_field_metrics",
    "compute_kinetic_energy",
    "compute_kinetic_energy_metrics",
    "compute_energy_landscape",
    "compute_discriminability",
    "compute_window_metrics",
    "compute_regime_discriminability",
]
