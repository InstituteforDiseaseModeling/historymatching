#! /usr/bin/env python3

import math
from pathlib import Path
import unittest

import pandas as pd

import history_matching.features as hmf

WORK_DIR = Path(__file__).parent.absolute() / "data"

class DerivedFeaturesTests(unittest.TestCase):

    """Check that current code generates same features and stats as previous code."""

    def test_getFeatures(self):

        """Check that current code generates same features and stats as previous code."""

        GET_DIR = WORK_DIR / "getFeatures"

        # NOTE:
        # These inputs and outputs were captured from an example in the phylomodels repository

        getFeatures_x                 = pd.read_feather(GET_DIR / "in-x.ftr")                                       # simulation results
        getFeatures_xref              = pd.read_feather(GET_DIR / "in-xref.ftr")                                    # observations
        getFeatures_ySim_features     = pd.read_feather(GET_DIR / "out-ySim_features.ftr")                          # simulation results + derived statistics
        getFeatures_ySim_featureStats = pd.read_hdf(GET_DIR / "out-ySim_featureStats.hdf", "ySim_featureStats")     # features w/stats (RSD, skew, variance, stddev, fano, QCD, mean)

        active_features = set(["diff_Linf", "series", "diff_L1", "derivative2_cauchyFit", "diff_L2", "sum"])
        # active_features = set(["passthrough"])
        active_statistics = None    # None means use all set([])

        computed_features, computed_stats = hmf.getFeatures( getFeatures_x, getFeatures_xref, active_features, active_statistics )

        self.assertEqual(set(computed_features.columns), set(getFeatures_ySim_features))
        for column in computed_features.columns:
            # self.assertTrue((computed_features[column] == getFeatures_ySim_features[column]).all())
            for computed, saved in zip(computed_features[column], getFeatures_ySim_features[column]):
                self.assertAlmostEqual(computed, saved, delta=saved/1e6)

        self.assertEqual(set(computed_stats.columns), set(getFeatures_ySim_featureStats))
        for column in computed_stats.columns:
            for row in getFeatures_ySim_featureStats.index:
                test = computed_stats[column][row]
                expected = getFeatures_ySim_featureStats[column][row]
                if not (math.isnan(test) and math.isnan(expected)):
                    self.assertAlmostEqual(test, expected, delta=expected/1e6)

        return


class ClortonTests(unittest.TestCase):

    def test_selectModelFeatures(self):

        GET_DIR = WORK_DIR / "getFeatures"

        modelOutputs = pd.read_feather(GET_DIR / "in-x.ftr")        # simulation results
        observations = pd.read_feather(GET_DIR / "in-xref.ftr")     # observations

        featureStats = hmf.getFeatureStatistics(modelOutputs)

        selected, target, simulation = hmf.select_features(modelOutputs, observations, featureStats, "fano", 1, [])

        return


class SelectFeaturesTests(unittest.TestCase):

    """Check that current code selects the same feature[s] as the previous code."""

    def test_select_features(self):

        """Check that current code selects the same feature[s] as the previous code."""

        SEL_DIR = WORK_DIR / "selectFeatures"

        # NOTE:
        # These inputs and outputs were captured from an example in the phylomodels repository

        select_features_f          = pd.read_feather(SEL_DIR / "in-f.ftr")                                               # simulation results + derived statistics (see ySim_features)
        select_features_fref       = pd.read_feather(SEL_DIR / "in-fref.ftr")                                            # observations + derived statistics
        select_features_fStats     = pd.read_hdf(SEL_DIR / "in-fStats.hdf")                                              # features w/start (see ySim_featureStats)
        select_features_iteration  = int(Path(SEL_DIR / "in-iteration.txt").read_text(encoding="utf-8").strip())         # integer
        select_features_metric     = Path(SEL_DIR / "in-metric.txt").read_text(encoding="utf-8").strip()                 # selection metric string ("fano")
        select_features_feature    = Path(SEL_DIR / "out-feature.txt").read_text(encoding="utf-8").strip()               # selected feature string ("sum_x")
        select_features_frefTarget = float(Path(SEL_DIR / "out-frefTarget.txt").read_text(encoding="utf-8").strip())     # selected feature target (observation) value
        select_features_fTarget    = pd.read_hdf(SEL_DIR / "out-fTarget.hdf", "fTarget")                                 # simulation values for selected feature

        feature_history = []
        computed_feature, reference_target, simulated_targets = hmf.select_features(select_features_f, select_features_fref, select_features_fStats, select_features_metric, select_features_iteration, feature_history)

        self.assertEqual(computed_feature, select_features_feature)
        self.assertEqual(reference_target, select_features_frefTarget)
        self.assertTrue((simulated_targets == select_features_fTarget).all())

        return

if __name__ == "__main__":
    unittest.main()
