# FlowPrint

**Flow-based fingerprints of dynamical regimes on shared manifolds.**

FlowPrint is a framework for characterizing dynamical regimes in multivariate time series using latent trajectories, flow fields, and geometric fingerprints—rather than discrete state labels.

## Overview

Traditional methods for analyzing multivariate oscillatory signals often reduce dynamics to discrete states (microstates, HMM states) and compare occupancy statistics. FlowPrint takes a complementary approach: it treats the signal as a continuous trajectory evolving on a learned manifold and characterizes regimes by *how* the system moves, not just *where* it visits.

Key features:
- **Coupled Stuart-Landau oscillator simulation** with topology-switching for ground-truth validation
- **Flow field estimation** on learned latent representations
- **Trajectory-based metrics**: speed, tortuosity, explored variance, kinetic energy
- **Discriminability analysis** using ANOVA with effect sizes (η²)

## Installation

```bash
# From source
git clone https://github.com/username/flowprint.git
cd flowprint
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from flowprint import CoupledStuartLandauNetwork
from flowprint.metrics import compute_flow_metrics, compute_discriminability
from flowprint.visualization import plot_electrode_timeseries

# Generate simulation with topology-switching
net = CoupledStuartLandauNetwork(n_oscillators=30, n_channels=30, seed=42)
net.default_topologies()

result = net.generate(
    total_duration_s=160.0,
    coupling_strength=5.0,
    transition_s=0.3,
)

# Visualize raw observations
fig = plot_electrode_timeseries(
    result.y, result.t, result.switch_times,
    sfreq=result.params["sfreq"],
    time_window=(0, 60),
)
fig.savefig("electrode_timeseries.png")

print(f"Simulated {result.y.shape[1]/result.params['sfreq']:.0f}s of data")
print(f"Regimes: {list(dict.fromkeys(result.regime_names))}")
```

## Reproducing Paper Figures

The `examples/reproduce_figures.py` script regenerates all figures from the paper:

```bash
python examples/reproduce_figures.py --output-dir figures/
```

This will generate:
- `fig_electrode_timeseries.png` - Raw observations with regime switches
- `fig_analysis_main.png` - Main 4-panel analysis figure
- `fig_flow_fields.png` - Regime-specific flow fields
- `fig_discriminability.png` - Violin plots with effect sizes
- `fig_kinetic_energy.png` - Kinetic energy analysis

## Module Structure

```
flowprint/
├── simulation/           # Coupled oscillator network
│   ├── coupled_oscillators.py
│   ├── topologies.py
│   └── noise.py
├── metrics/             # Flow and trajectory metrics
│   ├── flow_metrics.py
│   ├── kinetic_energy.py
│   └── discriminability.py
└── visualization/       # Figure generation
    └── figures.py
```

## Key Concepts

### Simulation

The `CoupledStuartLandauNetwork` implements a network of coupled limit-cycle oscillators with switchable coupling topology:

- **Global**: All-to-all coupling → promotes full synchronization
- **Cluster**: Modular coupling → multi-cluster synchrony
- **Sparse**: Random sparse coupling → promotes desynchronization
- **Ring**: Directional ring → traveling wave patterns

### Flow Metrics

Computed from latent trajectories:

| Metric | Description | Discriminability |
|--------|-------------|------------------|
| Speed | Mean velocity magnitude | High (η² > 0.5) |
| Explored Variance | State-space coverage | High (η² > 0.5) |
| Tortuosity | Path curvature | Low for topology contrast |
| Kinetic Energy | ||v||² intermittency proxy | Medium (η² ~ 0.2) |

### Discriminability

Uses one-way ANOVA with eta-squared (η²) effect size:
- η² > 0.14: Large effect
- 0.06 < η² < 0.14: Medium effect
- η² < 0.06: Small effect

## Citation

If you use FlowPrint in your research, please cite:

```bibtex
@article{furman2026flowprint,
  title={A Dynamical Microscope for Multivariate Oscillatory Signals:
         Validating Regime Recovery on Shared Manifolds},
  author={Furman, Łukasz and Minati, Ludovico and Duch, Włodzisław},
  journal={NeuroImage},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
