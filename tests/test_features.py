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

    x10 = pd.DataFrame.from_dict(
        {
            "feature_1": np.arange(10),
            "feature_2": np.arange(10) * 2,
            "feature_3": np.arange(10) * 3,
            "feature_4": np.arange(10) * 4,
            "feature_5": np.arange(10) * 5,
            "feature_6": np.arange(10) * 6,
            "feature_7": np.arange(10) * 7,
            "feature_8": np.arange(10) * 8,
            "feature_9": np.arange(10) * 9,
            "feature_10": np.repeat(100, 10),
        },
        orient="index",
    )
    x10_ref = pd.DataFrame.from_dict({"ref": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, orient="index")
    # out_features, out_stats = hmf.getFeatures(x10, x10_ref, None, None)

    @unittest.skip  # current package version(s) aren't hiccuping on nan values
    def test_xref_nan(self):
        # Devnote: shouldn't xref be checked for Nan and finite values?
        x = pd.DataFrame.from_dict({"sim_1": np.arange(10), "sim_2": np.arange(10, 0, -1), "sim_3": 10 ** np.random.random(10)}, orient="index")
        xref = pd.DataFrame.from_dict({"ref": np.append(10 ** np.random.random(9), math.nan)}, orient="index")
        with self.assertRaises(RuntimeWarning) as context:
            computed_features, computed_stats = hmf.getFeatures(x, xref, None, None)
        assert "invalid value" in str(context.exception)

    @unittest.skip  # current package version(s) aren't hiccuping on inf values
    def test_xref_inf(self):
        # Devnote: shouldn't xref be checked for Nan and finite values?
        x = pd.DataFrame.from_dict({"sim_1": np.arange(10), "sim_2": np.arange(10, 0, -1), "sim_3": 10 ** np.random.random(10)}, orient="index")
        xref = pd.DataFrame.from_dict({"ref": np.append(10 ** np.random.random(9), math.inf)}, orient="index")
        with self.assertRaises(RuntimeWarning) as context:
            computed_features, computed_stats = hmf.getFeatures(x, xref, None, None)
        assert "invalid value" in str(context.exception)

    def test_array(self):
        x = pd.DataFrame.from_dict({"sim_1": np.arange(10), "sim_2": np.append(np.arange(10, 1, -1), math.inf), "sim_3": 10 ** np.random.random(10)}, orient="index")
        xref = 10 ** np.random.random(10)
        with self.assertRaises(AttributeError) as context:
            computed_features, computed_stats = hmf.getFeatures(x, xref, None, None)
        assert "'numpy.ndarray' object has no attribute 'to_numpy'" in str(context.exception)

    def test_empty_dataframe(self):
        x = pd.DataFrame()
        xref = pd.DataFrame.from_dict({"ref": 10 ** np.random.random(10)}, orient="index")
        with self.assertRaises(ValueError) as context:
            computed_features, computed_stats = hmf.getFeatures(x, xref, None, None)
        assert "Shape of array too small to calculate a numerical gradient" in str(context.exception)

    @unittest.skip  # current package version(s) aren't hiccuping on nan values
    def test_x_nan(self):
        # Devnote: shouldn't x be checked for Nan and finite values?
        x_dict = {"sim_1": np.arange(10), "sim_2": np.append(np.arange(10, 1, -1), math.nan), "sim_3": 10 ** np.random.random(10)}
        x = pd.DataFrame.from_dict(x_dict, orient="index")
        # xref = pd.DataFrame.from_dict({"ref": [math.nan for _ in range(10)] }, orient="index")
        xref = pd.DataFrame.from_dict({"ref": 10 ** np.random.random(10)}, orient="index")

        with self.assertRaises(ValueError) as context:
            computed_features, computed_stats = hmf.getFeatures(x, xref, None, None)
        assert "The data contains non-finite values" in str(context.exception)

    @unittest.skip  # current package version(s) aren't hiccuping on inf values
    def test_x_inf(self):
        x_dict = {
            "sim_1": np.append(np.arange(9), math.inf),
            # "sim_1": np.arange(10),
            "sim_2": np.arange(10, 0, -1),
            "sim_3": 10 ** np.random.random(10),
        }
        x = pd.DataFrame.from_dict(x_dict, orient="index")
        xref = pd.DataFrame.from_dict({"ref": 10 ** np.random.random(10)}, orient="index")
        with self.assertRaises(RuntimeWarning) as context:
            computed_features, computed_stats = hmf.getFeatures(x, xref, None, None)
        assert "invalid value" in str(context.exception)

    @unittest.skip  # not exactly sure why it failed with RuntimeWarning: invalid value encountered in add
    def test_create_feature(self):
        x = pd.DataFrame.from_dict({"sim_1": np.arange(60), "sim_2": np.arange(60, 0, -1), "sim_3": 60 ** np.random.random(60)}, orient="index")
        xref = pd.DataFrame.from_dict({"ref": np.repeat(30, 60)}, orient="index")
        computed_features, computed_stats = hmf.getFeatures(x, xref, None, None)
        # DEVNOTE: can we derive 428 from relfection using  {function in inspect.getmembers(DerivedFeatures, inspect.isfunction)
        assert computed_features.shape == (3, 428)
        assert computed_stats.shape == (428, 7)


if __name__ == "__main__":
    unittest.main()
