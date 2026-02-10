"""Tests for the coupled Stuart-Landau oscillator simulation."""

import numpy as np

from flowprint.simulation.coupled_oscillators import (
    CoupledStuartLandauNetwork,
    SimulationResult,
)


class TestCoupledStuartLandauNetwork:
    """Tests for the CoupledStuartLandauNetwork class."""

    def test_basic_initialization(self):
        """Test basic network initialization."""
        net = CoupledStuartLandauNetwork(n_oscillators=10, n_channels=10, seed=42)
        assert net.n_osc == 10
        assert net.n_ch == 10

    def test_default_topologies(self):
        """Test that default topologies are added correctly."""
        net = CoupledStuartLandauNetwork(n_oscillators=10, n_channels=10, seed=42)
        net.default_topologies()

        # Should have 4 topologies: global, cluster, sparse, ring
        assert len(net.topologies) == 4
        assert "global" in net.topologies
        assert "cluster" in net.topologies
        assert "sparse" in net.topologies
        assert "ring" in net.topologies

    def test_generate_simulation(self):
        """Test that simulation generates correct output shape."""
        net = CoupledStuartLandauNetwork(n_oscillators=10, n_channels=10, seed=42)
        net.default_topologies()

        result = net.generate(
            total_duration_s=10.0,
            coupling_strength=5.0,
            transition_s=0.3,
        )

        # Check result is SimulationResult
        assert isinstance(result, SimulationResult)

        # Check shapes
        # 10 seconds at 250 Hz = 2500 samples
        expected_samples = int(10.0 * 250)
        assert result.y.shape == (10, expected_samples)  # (n_channels, n_samples)
        assert result.z.shape == (10, expected_samples)  # (n_oscillators, n_samples)
        assert len(result.regime_id) == expected_samples

    def test_regime_labels_coverage(self):
        """Test that all regimes are represented in labels."""
        net = CoupledStuartLandauNetwork(n_oscillators=10, n_channels=10, seed=42)
        net.default_topologies()

        # With 40s and 10s per regime, we should cycle through all 4 topologies
        result = net.generate(
            total_duration_s=40.0,
            coupling_strength=5.0,
            transition_s=0.3,
        )

        unique_regimes = set(result.regime_id)
        assert len(unique_regimes) == 4  # All 4 topologies should appear

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        net1 = CoupledStuartLandauNetwork(n_oscillators=10, n_channels=10, seed=42)
        net1.default_topologies()
        result1 = net1.generate(total_duration_s=5.0, coupling_strength=5.0)

        net2 = CoupledStuartLandauNetwork(n_oscillators=10, n_channels=10, seed=42)
        net2.default_topologies()
        result2 = net2.generate(total_duration_s=5.0, coupling_strength=5.0)

        np.testing.assert_array_almost_equal(result1.y, result2.y)
        np.testing.assert_array_almost_equal(result1.z, result2.z)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        net1 = CoupledStuartLandauNetwork(n_oscillators=10, n_channels=10, seed=42)
        net1.default_topologies()
        result1 = net1.generate(total_duration_s=5.0, coupling_strength=5.0)

        net2 = CoupledStuartLandauNetwork(n_oscillators=10, n_channels=10, seed=123)
        net2.default_topologies()
        result2 = net2.generate(total_duration_s=5.0, coupling_strength=5.0)

        # Results should be different
        assert not np.allclose(result1.y, result2.y)


class TestSimulationResult:
    """Tests for the SimulationResult dataclass."""

    def test_simulation_result_fields(self):
        """Test that SimulationResult has all required fields."""
        net = CoupledStuartLandauNetwork(n_oscillators=10, n_channels=10, seed=42)
        net.default_topologies()
        result = net.generate(total_duration_s=5.0, coupling_strength=5.0)

        # Check all required fields exist
        assert hasattr(result, 'y')
        assert hasattr(result, 'z')
        assert hasattr(result, 'regime_id')
        assert hasattr(result, 'regime_names')
        assert hasattr(result, 'switch_times')
        assert hasattr(result, 't')
