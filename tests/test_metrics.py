"""Tests for flow metrics and discriminability computation."""

import numpy as np

from flowprint.metrics.flow_metrics import compute_flow_metrics, compute_velocity


class TestFlowMetrics:
    """Tests for flow metric computation."""

    def test_compute_flow_metrics_shape(self):
        """Test that flow metrics returns correct structure."""
        # Create dummy trajectory: (n_samples, n_dims)
        n_samples, n_dims = 1000, 32
        trajectory = np.random.randn(n_samples, n_dims)

        metrics = compute_flow_metrics(trajectory, smooth_window=5)

        # Check that all expected metrics are present
        assert "speed" in metrics
        assert "speed_cv" in metrics
        assert "tortuosity" in metrics
        assert "explored_variance" in metrics

    def test_speed_is_positive(self):
        """Test that speed is always non-negative."""
        trajectory = np.random.randn(500, 10)
        metrics = compute_flow_metrics(trajectory, smooth_window=5)

        assert metrics["speed"] >= 0

    def test_explored_variance_is_positive(self):
        """Test that explored variance is non-negative."""
        trajectory = np.random.randn(500, 10)
        metrics = compute_flow_metrics(trajectory, smooth_window=5)

        # Explored variance is sum of variances, always non-negative
        assert metrics["explored_variance"] >= 0

    def test_constant_trajectory_has_zero_speed(self):
        """Test that a constant trajectory has zero speed."""
        # Constant trajectory
        trajectory = np.ones((500, 10))
        metrics = compute_flow_metrics(trajectory, smooth_window=5)

        # Speed should be zero (or very close to zero)
        assert metrics["speed"] < 1e-10


class TestVelocityComputation:
    """Tests for velocity computation."""

    def test_savgol_velocity_shape(self):
        """Test that Savitzky-Golay velocity has correct shape."""
        trajectory = np.random.randn(100, 10)
        velocity = compute_velocity(trajectory, method="savgol")

        assert velocity.shape == trajectory.shape

    def test_finite_diff_velocity_shape(self):
        """Test that finite difference velocity has correct shape."""
        trajectory = np.random.randn(100, 10)
        velocity = compute_velocity(trajectory, method="finite_diff")

        assert velocity.shape == trajectory.shape


class TestDiscriminability:
    """Tests for discriminability analysis."""

    def test_eta_squared_bounds(self):
        """Test that eta-squared is bounded [0, 1]."""
        # Create data with clear group differences
        n_per_group = 100
        groups = {
            "A": np.random.randn(n_per_group) + 0,
            "B": np.random.randn(n_per_group) + 2,
            "C": np.random.randn(n_per_group) + 4,
        }

        # Compute eta-squared manually
        all_data = np.concatenate(list(groups.values()))
        grand_mean = np.mean(all_data)

        ss_total = np.sum((all_data - grand_mean) ** 2)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups.values())
        eta_sq = ss_between / ss_total

        assert 0 <= eta_sq <= 1

    def test_identical_groups_have_low_eta_squared(self):
        """Test that similar groups have low eta-squared."""
        # All groups from same distribution
        np.random.seed(42)
        data = np.random.randn(99)
        groups = {
            "A": data[:33].copy(),
            "B": data[33:66].copy(),
            "C": data[66:99].copy(),
        }

        all_data = np.concatenate(list(groups.values()))
        grand_mean = np.mean(all_data)

        ss_total = np.sum((all_data - grand_mean) ** 2)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups.values())
        eta_sq = ss_between / (ss_total + 1e-10)

        # eta-squared should be small (not exactly 0 due to sampling)
        assert eta_sq < 0.1

    def test_well_separated_groups_have_high_eta_squared(self):
        """Test that well-separated groups have high eta-squared."""
        # Groups with very different means
        np.random.seed(42)
        groups = {
            "A": np.random.randn(100) * 0.1 + 0,
            "B": np.random.randn(100) * 0.1 + 10,
            "C": np.random.randn(100) * 0.1 + 20,
        }

        all_data = np.concatenate(list(groups.values()))
        grand_mean = np.mean(all_data)

        ss_total = np.sum((all_data - grand_mean) ** 2)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups.values())
        eta_sq = ss_between / ss_total

        # eta-squared should be very high (close to 1)
        assert eta_sq > 0.9
