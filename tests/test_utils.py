#! /usr/bin/env python3

"""Tests for utility functions."""

import unittest

import numpy as np

from history_matching import features_from_observations
from history_matching import mean_and_variance_for_observations


class UtilityTests(unittest.TestCase):

    """Tests for utility functions."""

    def test_mean_and_variance_for_observations(self):
        """Test the mean_and_variance_for_observations function."""
        # "Happy Path" only at this time
        raw_observations = {"height": [175, 175, 173, 163, 61], "weight": [97, 100, 63, 54, 11]}
        mean_and_variance = mean_and_variance_for_observations(raw_observations)
        heights = np.array([175, 175, 173, 163, 61])
        weights = np.array([97, 100, 63, 54, 11])
        assert np.float64(mean_and_variance.mean["height"]) == heights.mean()
        assert np.float64(mean_and_variance.mean["weight"]) == weights.mean()
        assert np.float64(mean_and_variance.variance["height"]) == heights.var(ddof=1)  # Use N-1 for variance
        assert np.float64(mean_and_variance.variance["weight"]) == weights.var(ddof=1)  # Use N-1 for variance

        return

    def test_features_from_observations(self):
        """Test the features_from_observations function."""
        # "Happy Path" only at this time
        raw_observations = {"height": [175, 175, 173, 163, 61], "weight": [97, 100, 63, 54, 11]}
        mean_and_variance = mean_and_variance_for_observations(raw_observations)
        assert set(features_from_observations(mean_and_variance)) == set(raw_observations.keys())

        return


if __name__ == "__main__":
    unittest.main()
