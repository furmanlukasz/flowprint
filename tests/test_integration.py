"""Integration tests for the full FlowPrint pipeline."""

import numpy as np
import pytest

from flowprint.simulation.coupled_oscillators import CoupledStuartLandauNetwork


class TestFullPipeline:
    """Integration tests for the complete analysis pipeline."""

    @pytest.fixture
    def simulation_result(self):
        """Create a short simulation for testing."""
        net = CoupledStuartLandauNetwork(n_oscillators=10, n_channels=10, seed=42)
        net.default_topologies()
        return net.generate(
            total_duration_s=40.0,  # 4 regimes x 10s
            coupling_strength=5.0,
            transition_s=0.3,
        )

    def test_simulation_produces_oscillatory_signal(self, simulation_result):
        """Test that simulation produces oscillatory signals."""
        y = simulation_result.y

        # Check that signal is not constant
        assert np.std(y) > 0

        # Check that signal has reasonable range (not exploding)
        assert np.max(np.abs(y)) < 100

    def test_regime_labels_match_samples(self, simulation_result):
        """Test that regime labels have correct length."""
        n_samples = simulation_result.y.shape[1]
        assert len(simulation_result.regime_id) == n_samples

    def test_all_four_regimes_present(self, simulation_result):
        """Test that all four topology regimes are present."""
        unique_regimes = set(simulation_result.regime_id)
        assert len(unique_regimes) == 4

    def test_switch_times_are_ordered(self, simulation_result):
        """Test that switch times are in ascending order."""
        switch_times = simulation_result.switch_times
        assert all(switch_times[i] <= switch_times[i + 1] for i in range(len(switch_times) - 1))

    def test_time_vector_matches_samples(self, simulation_result):
        """Test that time vector has correct length."""
        n_samples = simulation_result.y.shape[1]
        assert len(simulation_result.t) == n_samples

    def test_latent_states_are_complex(self, simulation_result):
        """Test that latent oscillator states are complex-valued."""
        z = simulation_result.z
        assert np.iscomplexobj(z)


class TestPaperReproducibility:
    """Tests specifically for paper figure reproducibility."""

    def test_paper_configuration_runs(self):
        """Test that the exact paper configuration runs without error."""
        # Paper parameters from main-sim.tex
        net = CoupledStuartLandauNetwork(
            n_oscillators=30,
            n_channels=30,
            seed=42,
        )
        net.default_topologies()

        result = net.generate(
            total_duration_s=160.0,  # Paper: 160s
            coupling_strength=5.0,   # Paper: kappa = 5.0
            transition_s=0.3,        # Paper: 0.3s smoothing
        )

        # Check expected dimensions
        # 160s at 250Hz = 40000 samples
        expected_samples = int(160.0 * 250)
        assert result.y.shape == (30, expected_samples)
        assert result.z.shape == (30, expected_samples)

    def test_four_cycles_of_four_regimes(self):
        """Test that 160s produces 4 cycles of 4 regimes (16 segments)."""
        net = CoupledStuartLandauNetwork(n_oscillators=10, n_channels=10, seed=42)
        net.default_topologies()

        result = net.generate(
            total_duration_s=160.0,
            coupling_strength=5.0,
            transition_s=0.3,
        )

        # Should have all 4 regimes present
        unique_regimes = set(result.regime_id)
        assert len(unique_regimes) == 4

        # Check roughly equal time in each regime
        regime_counts = {}
        for r in result.regime_id:
            regime_counts[r] = regime_counts.get(r, 0) + 1

        # Each regime should have ~25% of samples (with some tolerance for transitions)
        total_samples = len(result.regime_id)
        for count in regime_counts.values():
            proportion = count / total_samples
            assert 0.20 < proportion < 0.30, f"Regime proportion {proportion} outside expected range"
