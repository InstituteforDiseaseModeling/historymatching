import pandas as pd
import unittest

from hm2.error import *
import hm2.data_validation as dv


class TestFramesForValidity(unittest.TestCase):
    def test_non_frames(self):
        frame = "I am not a DataFrame"
        with self.assertRaises(HMNotADataFrame) as cm:
            dv.ValidateParameterInfoFrame(frame)
        self.assertTrue(cm.exception.df_name == "ParameterInfoFrame")
        with self.assertRaises(HMNotADataFrame) as cm:
            dv.ValidateParameterSamplesFrame(frame)
        self.assertTrue(cm.exception.df_name == "ParameterSamplesFrame")
        with self.assertRaises(HMNotADataFrame) as cm:
            dv.ValidateObservationsFrame(frame)
        self.assertTrue(cm.exception.df_name == "ObservationsFrame")
        with self.assertRaises(HMNotADataFrame) as cm:
            dv.ValidateSimFrame(frame)
        self.assertTrue(cm.exception.df_name == "SimFrame")
        with self.assertRaises(HMNotADataFrame) as cm:
            dv.ValidateMatchedFrame(frame)
        self.assertTrue(cm.exception.df_name == "MatchedFrame")

    def test_extra_columns(self):
        param_info = pd.DataFrame(
            {
                "name": ["beta", "gamma"],
                "min": [1e-6, 1e-6],
                "max": [0.01, 0.5],
                "extra": [0, 0],
            }
        )
        with self.assertRaises(HMExtraColumns) as cm:
            dv.ValidateParameterInfoFrame(param_info)
        self.assertTrue(cm.exception.df_name == "ParameterInfoFrame")

    def test_time_increases(self):
        observations = pd.DataFrame(
            {
                "observation_id": [0, 1],
                "time": [15.0, 1.0],
                "observation": ["prevalence", "prevalence"],
                "value": [15, 40],
                "stdev": [4, 2.3],
            }
        )

        self.assertRaises(
            HMTimeIsNotMonotonic, dv.ValidateObservationsFrame, observations
        )

    def test_unique_observation(self):
        observations = pd.DataFrame(
            {
                "observation_id": [0, 1],
                "time": [15.0, 15.0],
                "observation": ["prevalence", "prevalence"],
                "value": [15, 40],
                "stdev": [4, 2.3],
            }
        )

        self.assertRaises(
            HMTwoObservationsAtOneTime, dv.ValidateObservationsFrame, observations
        )


if __name__ == "__main__":
    unittest.main()
