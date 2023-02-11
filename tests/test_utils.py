#! /usr/bin/env python3

import unittest

import numpy as np

from history_matching import features_from_observations, mean_and_variance_for_observations


class UtilityTests(unittest.TestCase):

    def test_mean_and_variance_for_observations(self):

        # "Happy Path" only at this time
        raw_observations = {
            "height": [175, 175, 173, 163,  61],
            "weight": [ 97, 100,  63,  54,  11]
        }
        mean_and_variance = mean_and_variance_for_observations(raw_observations)
        heights = np.array([175, 175, 173, 163, 61])
        weights = np.array([ 97, 100,  63,  54, 11])
        self.assertEqual(np.float64(mean_and_variance.means["height"])    , heights.mean())
        self.assertEqual(np.float64(mean_and_variance.means["weight"])    , weights.mean())
        self.assertEqual(np.float64(mean_and_variance.variances["height"]), heights.var(ddof=1))  # Use N-1 for variance
        self.assertEqual(np.float64(mean_and_variance.variances["weight"]), weights.var(ddof=1))  # Use N-1 for variance

        return

    def test_features_from_observations(self):

        # "Happy Path" only at this time
        raw_observations = {
            "height": [175, 175, 173, 163,  61],
            "weight": [ 97, 100,  63,  54,  11]
        }
        mean_and_variance = mean_and_variance_for_observations(raw_observations)
        self.assertSetEqual(set(features_from_observations(mean_and_variance)), set(raw_observations.keys()))

        return


if __name__ == "__main__":
    unittest.main()
