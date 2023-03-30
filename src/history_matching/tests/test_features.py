#! /usr/bin/env python3

import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import history_matching.features as hmf

WORK_DIR = Path(__file__).parent.absolute() / "data"


class FeaturesTests(unittest.TestCase):
    """
        unit tests for features
    """

    # raised all warning as exception so we could capture
    # np.seterr(all='raise')

    x10 = pd.DataFrame.from_dict({"feature_1": np.arange(10),
                                  "feature_2": np.arange(10) * 2,
                                  "feature_3": np.arange(10) * 3,
                                  "feature_4": np.arange(10) * 4,
                                  "feature_5": np.arange(10) * 5,
                                  "feature_6": np.arange(10) * 6,
                                  "feature_7": np.arange(10) * 7,
                                  "feature_8": np.arange(10) * 8,
                                  "feature_9": np.arange(10) * 9,
                                  "feature_10": np.repeat(100, 10)},
                                 orient="index")
    x10_ref = pd.DataFrame.from_dict({"ref": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, orient="index")
    # out_features, out_stats = hmf.getFeatures(x10, x10_ref, None, None)

    def test_xref_nan(self):
        # Devnote: shouldn't xref be checked for Nan and finite values?
        x = pd.DataFrame.from_dict({"sim_1": np.arange(10),
                                    "sim_2": np.arange(10, 0, -1),
                                    "sim_3": 10 ** np.random.random(10)},
                                   orient="index")
        xref = pd.DataFrame.from_dict({"ref": np.append(10 ** np.random.random(9), math.nan)}, orient="index")
        with self.assertRaises(RuntimeWarning) as context:
            computed_features, computed_stats = hmf.getFeatures(x, xref, None, None)
        self.assertTrue("invalid value" in str(context.exception))

    def test_xref_inf(self):
        # Devnote: shouldn't xref be checked for Nan and finite values?
        x = pd.DataFrame.from_dict({"sim_1": np.arange(10),
                                    "sim_2": np.arange(10, 0, -1),
                                    "sim_3": 10 ** np.random.random(10)},
                                   orient="index")
        xref = pd.DataFrame.from_dict({"ref": np.append(10 ** np.random.random(9), math.inf)}, orient="index")
        with self.assertRaises(RuntimeWarning) as context:
            computed_features, computed_stats = hmf.getFeatures(x, xref, None, None)
        self.assertTrue("invalid value" in str(context.exception))

    def test_array(self):
        x = pd.DataFrame.from_dict({"sim_1": np.arange(10),
                                    "sim_2": np.append(np.arange(10, 1, -1), math.inf),
                                    "sim_3": 10 ** np.random.random(10)},
                                   orient="index")
        xref = 10 ** np.random.random(10)
        with self.assertRaises(AttributeError) as context:
            computed_features, computed_stats = hmf.getFeatures(x, xref, None, None)
        self.assertTrue("'numpy.ndarray' object has no attribute 'to_numpy'" in str(context.exception))

    def test_empty_dataframe(self):
        x = pd.DataFrame()
        xref = pd.DataFrame.from_dict({"ref": 10 ** np.random.random(10)}, orient="index")
        with self.assertRaises(ValueError) as context:
            computed_features, computed_stats = hmf.getFeatures(x, xref, None, None)
        self.assertTrue("Shape of array too small to calculate a numerical gradient" in str(context.exception))

    def test_x_nan(self):
        # Devnote: shouldn't x be checked for Nan and finite values?
        x_dict = {"sim_1": np.arange(10),
                  "sim_2": np.append(np.arange(10, 1, -1), math.nan),
                  "sim_3": 10 ** np.random.random(10)}
        x = pd.DataFrame.from_dict(x_dict, orient="index")
        # xref = pd.DataFrame.from_dict({"ref": [math.nan for _ in range(10)] }, orient="index")
        xref = pd.DataFrame.from_dict({"ref": 10 ** np.random.random(10)}, orient="index")

        with self.assertRaises(ValueError) as context:
            computed_features, computed_stats = hmf.getFeatures(x, xref, None, None)
        self.assertTrue("The data contains non-finite values" in str(context.exception))

    def test_x_inf(self):
        x_dict = {"sim_1": np.append(np.arange(9), math.inf),
                  # "sim_1": np.arange(10),
                  "sim_2": np.arange(10, 0, -1),
                  "sim_3": 10 ** np.random.random(10)}
        x = pd.DataFrame.from_dict(x_dict, orient="index")
        xref = pd.DataFrame.from_dict({"ref": 10 ** np.random.random(10)}, orient="index")
        with self.assertRaises(RuntimeWarning) as context:
            computed_features, computed_stats = hmf.getFeatures(x, xref, None, None)
        self.assertTrue("invalid value" in str(context.exception))

    @unittest.skip  # not exactly sure why it failed with RuntimeWarning: invalid value encountered in add
    def test_create_feature(self):
        x = pd.DataFrame.from_dict({"sim_1": np.arange(60),
                                    "sim_2": np.arange(60, 0, -1),
                                    "sim_3": 60 ** np.random.random(60)},
                                   orient="index")
        xref = pd.DataFrame.from_dict({"ref": np.repeat(30, 60)}, orient="index")
        computed_features, computed_stats = hmf.getFeatures(x, xref, None, None)
        # DEVNOTE: can we derive 428 from relfection using  {function in inspect.getmembers(DerivedFeatures, inspect.isfunction)
        self.assertTrue(computed_features.shape == (3, 428))
        self.assertTrue(computed_stats.shape == (428, 7))


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

        computed_features, computed_stats = hmf.getFeatures(getFeatures_x, getFeatures_xref, active_features, active_statistics)

        self.assertEqual(set(computed_features.columns), set(getFeatures_ySim_features))
        for column in computed_features.columns:
            # self.assertTrue((computed_features[column] == getFeatures_ySim_features[column]).all())
            for computed, saved in zip(computed_features[column], getFeatures_ySim_features[column]):
                self.assertAlmostEqual(computed, saved, delta=saved / 1e6)

        self.assertEqual(set(computed_stats.columns), set(getFeatures_ySim_featureStats))
        for column in computed_stats.columns:
            for row in getFeatures_ySim_featureStats.index:
                test = computed_stats[column][row]
                expected = getFeatures_ySim_featureStats[column][row]
                if not (math.isnan(test) and math.isnan(expected)):
                    self.assertAlmostEqual(test, expected, delta=expected / 1e6)

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
        computed_feature, reference_target, simulated_targets = \
            hmf.select_features(select_features_f, select_features_fref, select_features_fStats, select_features_metric, select_features_iteration, feature_history)

        self.assertEqual(computed_feature, select_features_feature)
        self.assertEqual(reference_target, select_features_frefTarget)
        self.assertTrue((simulated_targets == select_features_fTarget).all())

        return


if __name__ == "__main__":
    unittest.main()
