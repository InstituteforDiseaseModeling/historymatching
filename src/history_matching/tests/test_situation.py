#! /usr/bin/env python3

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from history_matching import Situation, latin_hypercube_sampler
from history_matching.emulators import BaseEmulator

valid_parameter_space = pd.DataFrame(data=[["x", 0, 10], ["y", 0, 100], ["z", 0, 1000]], columns=["parameter", "minimum", "maximum"])
valid_observations = pd.DataFrame(data=[["height", 1.75, 0.01], ["weight", 98.87, 1.13]], columns=["features", "means", "variances"])
valid_sample_points = latin_hypercube_sampler(valid_parameter_space, 10)
valid_sample_points["iteration"] = 0
valid_simulator_results = pd.DataFrame(data=[[0, 5, 50, 500, 1.76, 98.5]], columns=["replicate", "x", "y", "z", "height", "weight"])


class SituationValidationTests(unittest.TestCase):

    def test_valid_iteration(self):

        Situation.validate_iteration(0)
        Situation.validate_iteration(1)
        Situation.validate_iteration(1 << 30)   # ~1 million

        return

    def test_invalid_iteration(self):

        with self.assertRaises(TypeError):
            Situation.validate_iteration("42")

        with self.assertRaises(ValueError):
            Situation.validate_iteration(3.14159265)

        with self.assertRaises(ValueError):
            Situation.validate_iteration(-1)

        return

    def test_valid_parameter_space(self):

        Situation.validate_parameter_space(valid_parameter_space)
        # Extra columns are acceptable.
        Situation.validate_parameter_space(pd.DataFrame(data=[["x", 0, 10, "first axis"],
                                                              ["y", 0, 100, "second axis"],
                                                              ["z", 0, 1000, "third axis"]],
                                                        columns=["parameter", "minimum", "maximum", "note"]))

        return

    def test_invalid_parameter_space(self):

        parameter_space = pd.DataFrame(data=[["x", 0, 10], ["y", 0, 100], ["z", 0, 1000]], columns=["parameter", "minimum", "maximum"])
        Situation.validate_parameter_space(parameter_space)

        # Must be pd.DataFrame, not list or Numpy array (or anything other than pd.DataFrame)
        with self.assertRaises(TypeError):
            Situation.validate_parameter_space(["k", 0, 10])

        with self.assertRaises(TypeError):
            Situation.validate_parameter_space(np.arange(3))

        # Must have "parameter", "minimum", and "maximum" columns
        with self.assertRaises(RuntimeError):
            Situation.validate_parameter_space(pd.DataFrame(data=[[0, 10], [0, 100], [0, 1000]], columns=["minimum", "maximum"]))

        with self.assertRaises(RuntimeError):
            Situation.validate_parameter_space(pd.DataFrame(data=[["x", 10], ["y", 100], ["z", 1000]], columns=["parameter", "maximum"]))

        with self.assertRaises(RuntimeError):
            Situation.validate_parameter_space(pd.DataFrame(data=[["x", 0], ["y", 0], ["z", 0]], columns=["parameter", "minimum"]))

        # Must have at least one parameter
        with self.assertRaises(RuntimeError):
            Situation.validate_parameter_space(pd.DataFrame(columns=["parameter", "minimum", "maximum"]))

        # Each minimum must be <= corresponding maximum
        with self.assertRaises(ValueError):
            Situation.validate_parameter_space(pd.DataFrame(data=[["x", 10, 0], ["y", 0, 100], ["z", 0, 1000]], columns=["parameter", "minimum", "maximum"]))

        with self.assertRaises(ValueError):
            Situation.validate_parameter_space(pd.DataFrame(data=[["x", 0, 10], ["y", 100, 0], ["z", 0, 1000]], columns=["parameter", "minimum", "maximum"]))

        with self.assertRaises(ValueError):
            Situation.validate_parameter_space(pd.DataFrame(data=[["x", 0, 10], ["y", 0, 100], ["z", 1000, 0]], columns=["parameter", "minimum", "maximum"]))

        return

    # Situation.validate_sample_points(self.sample_points, self.parameter_space)
    def test_valid_sample_points(self):

        Situation.validate_sample_points(valid_sample_points, valid_parameter_space)

        return

    def test_invalid_sample_points(self):

        parameter_space = pd.DataFrame(data=[["x", 0, 10], ["y", 0, 100], ["z", 0, 1000]], columns=["parameter", "minimum", "maximum"])

        # Must be pd.DataFrame -=> TypeError
        with self.assertRaises(TypeError):
            Situation.validate_sample_points([5, 50, 500], parameter_space)

        # Must contain "iteration" and all parameter space parameters -=> RuntimeError
        sample_points = latin_hypercube_sampler(parameter_space, 10)
        sample_points["iteration"] = 0

        missing_iteration = sample_points.drop(columns=["iteration"])
        with self.assertRaises(RuntimeError):
            Situation.validate_sample_points(missing_iteration, parameter_space)

        missing_y = sample_points.drop(columns="y")
        with self.assertRaises(RuntimeError):
            Situation.validate_sample_points(missing_y, parameter_space)

        # Must contain at least one sample point -=> RuntimeError
        no_samples = pd.DataFrame(columns=sample_points.columns)
        with self.assertRaises(RuntimeError):
            Situation.validate_sample_points(no_samples, parameter_space)

        # Sample points must be within the parameter space -=> ValueError
        bad_mins = sample_points.copy()
        bad_mins.x -= 1_000_000
        with self.assertRaises(ValueError):
            Situation.validate_sample_points(bad_mins, parameter_space)

        bad_maxes = sample_points.copy()
        bad_maxes.y += 1_000_000
        with self.assertRaises(ValueError):
            Situation.validate_sample_points(bad_maxes, parameter_space)

        return

    # Situation.validate_observations(self.observations)
    def test_valid_observations(self):

        Situation.validate_observations(valid_observations)

        return

    def test_invalid_observations(self):

        # Must be pd.DataFrame -=> TypeError
        with self.assertRaises(TypeError):
            Situation.validate_observations([0, 1, 1, 2])

        # Must have "statistic" column -=> RuntimeError
        observations = pd.DataFrame(data=[["mean", 1.75, 98.87], ["variance", 0.01, 1.13]], columns=["statistic", "height", "weight"])
        observations.drop(columns="statistic", inplace=True)
        with self.assertRaises(RuntimeError):
            Situation.validate_observations(observations)

        # Must have at least one feature column -=> RuntimeError
        observations = pd.DataFrame(data=[["mean", 1.75, 98.87], ["variance", 0.01, 1.13]], columns=["statistic", "height", "weight"])
        observations.drop(columns=["height", "weight"], inplace=True)
        with self.assertRaises(RuntimeError):
            Situation.validate_observations(observations)

        # Must have a row for mean
        observations = pd.DataFrame(data=[["variance", 0.01, 1.13]], columns=["statistic", "height", "weight"])
        with self.assertRaises(RuntimeError):
            Situation.validate_observations(observations)

        # Must have a row for variance
        observations = pd.DataFrame(data=[["mean", 1.75, 98.87]], columns=["statistic", "height", "weight"])
        with self.assertRaises(RuntimeError):
            Situation.validate_observations(observations)

        return

    # Situation.validate_simulator_results(self.simulator_results, self.parameter_space, self.observations)
    def test_valid_simulator_results(self):

        # Must be pd.DataFrame
        # Must have "replicate" column
        # Must have same columns as parameter space parameters
        # Must have same columns as features in observations

        Situation.validate_simulator_results(valid_simulator_results, valid_parameter_space, valid_observations)

        return

    def test_invalid_simulator_results(self):

        parameter_space = pd.DataFrame(data=[["x", 0, 10], ["y", 0, 100], ["z", 0, 1000]], columns=["parameter", "minimum", "maximum"])
        observations = pd.DataFrame(data=[["height", 1.75, 0.01], ["weight", 98.87, 1.13]], columns=["features", "means", "variances"])

        # Must be pd.DataFrame -=> TypeError
        with self.assertRaises(TypeError):
            Situation.validate_simulator_results([[0, 5, 50, 500, 1.76, 98.5]], parameter_space, observations)

        # Must have "replicate" column -=> RuntimeError
        results = pd.DataFrame(data=[[5, 50, 500, 1.76, 98.5]], columns=["x", "y", "z", "height", "weight"])
        with self.assertRaises(RuntimeError):
            Situation.validate_simulator_results(results, parameter_space, observations)

        # Must have same columns as parameter space parameters -=> RuntimeError
        results = pd.DataFrame(data=[[0, 5, 500, 1.76, 98.5]], columns=["replicate", "x", "z", "height", "weight"])
        with self.assertRaises(RuntimeError):
            Situation.validate_simulator_results(results, parameter_space, observations)

        # Must have same columns as features in observations -=> RuntimeError
        results = pd.DataFrame(data=[[0, 5, 50, 500, 98.5]], columns=["replicate", "x", "y", "z", "weight"])
        with self.assertRaises(RuntimeError):
            Situation.validate_simulator_results(results, parameter_space, observations)

        return

    @staticmethod
    def dummyEmulator():

        x = pd.DataFrame(data=[[0, 0], [0, 1], [1, 0], [1, 1]], columns=["slope", "intercept"])
        y = pd.DataFrame(data=[[0, 0], [1, 1], [0, 1], [1, 2]], columns=["x0", "x1"])

        return BaseEmulator(x, y)

    def test_valid_emulator_bank(self):

        # Must be a dictionary mapping int:dict
        # Keys must be >= 0
        # Must be a dictionary mapping int:dict
        # Values must be dictionaries mapping str:BaseEmulator
        # Keys must be features from observations
        # Values must be dictionaries mapping str:BaseEmulator

        emulator_bank = {
            0: {"height": SituationValidationTests.dummyEmulator()},
            1: {"height": SituationValidationTests.dummyEmulator(), "weight": SituationValidationTests.dummyEmulator()}
        }

        Situation.validate_emulator_bank(emulator_bank, valid_observations)

        return

    def test_invalid_emulator_bank(self):

        # Must be a dictionary mapping int:dict -=> TypeError
        # Keys here are strings.
        emulator_bank = {
            "0": {"height": SituationValidationTests.dummyEmulator()},
            "1": {"height": SituationValidationTests.dummyEmulator(), "weight": SituationValidationTests.dummyEmulator()}
        }

        with self.assertRaises(TypeError):
            Situation.validate_emulator_bank(emulator_bank, valid_observations)

        # Keys must be >= 0
        # -1 is not a valid iteration.
        emulator_bank = {
            0: {"height": SituationValidationTests.dummyEmulator()},
            -1: {"height": SituationValidationTests.dummyEmulator(), "weight": SituationValidationTests.dummyEmulator()}
        }

        with self.assertRaises(TypeError):
            Situation.validate_emulator_bank([0, SituationValidationTests.dummyEmulator()], valid_observations)

        # Must be a dictionary mapping int:dict
        # Value in iteration 0 is a list.
        emulator_bank = {
            0: ["height", SituationValidationTests.dummyEmulator()],
            1: {"height": SituationValidationTests.dummyEmulator(), "weight": SituationValidationTests.dummyEmulator()}
        }

        with self.assertRaises(TypeError):
            Situation.validate_emulator_bank([0, SituationValidationTests.dummyEmulator()], valid_observations)

        # Values must be dictionaries mapping str:BaseEmulator
        # Iteration dictionary keys are integers.
        emulator_bank = {
            0: {0: SituationValidationTests.dummyEmulator()},
            1: {0: SituationValidationTests.dummyEmulator(), 1: SituationValidationTests.dummyEmulator()}
        }

        with self.assertRaises(TypeError):
            Situation.validate_emulator_bank(emulator_bank, valid_observations)

        # Keys must be features from observations
        # Iteration 1 has key "mass" rather than "weight".
        emulator_bank = {
            0: {"height": SituationValidationTests.dummyEmulator()},
            1: {"height": SituationValidationTests.dummyEmulator(), "mass": SituationValidationTests.dummyEmulator()}
        }

        with self.assertRaises(ValueError):
            Situation.validate_emulator_bank(emulator_bank, valid_observations)

        # Values must be dictionaries mapping str:BaseEmulator
        # Iteration 0 maps "height" to a lambda, not BaseEmulator.
        emulator_bank = {
            0: {"height": lambda x: 3.14159265},
            1: {"height": SituationValidationTests.dummyEmulator(), "weight": SituationValidationTests.dummyEmulator()}
        }

        with self.assertRaises(TypeError):
            Situation.validate_emulator_bank(emulator_bank, valid_observations)

        return


class SituationSaveReadTests(unittest.TestCase):

    def test_roundtrip(self):

        situation = Situation(valid_parameter_space, valid_observations, valid_sample_points, iteration=42)
        situation.simulator_results = valid_simulator_results

        try:
            handle, filename = tempfile.mkstemp()
            os.close(handle)
            situation.save(filename)

            copy = Situation.read(filename)

            # TODO - consider using Pandas DataFrame.equal(), but it isn't exact

            self.assertListEqual(list(copy.parameter_space.columns),   list(situation.parameter_space.columns))
            self.assertListEqual(list(copy.observations.columns),      list(situation.observations.columns))
            self.assertListEqual(list(copy.sample_points.columns),     list(situation.sample_points.columns))
            self.assertListEqual(list(copy.simulator_results.columns), list(situation.simulator_results.columns))

            self.assertTrue(all([actual == expected for (actual, expected) in zip(copy.parameter_space.itertuples(),   situation.parameter_space.itertuples())]))
            self.assertTrue(all([actual == expected for (actual, expected) in zip(copy.observations.itertuples(),      situation.observations.itertuples())]))
            self.assertTrue(all([actual == expected for (actual, expected) in zip(copy.sample_points.itertuples(),     situation.sample_points.itertuples())]))
            self.assertTrue(all([actual == expected for (actual, expected) in zip(copy.simulator_results.itertuples(), situation.simulator_results.itertuples())]))
            # TODO - verify emulator_bank, currently empty

        finally:
            os.remove(filename)

        return


if __name__ == "__main__":
    unittest.main()
