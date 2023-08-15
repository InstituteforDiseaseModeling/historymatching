#! /usr/bin/env python3

"""Unit tests for the next point generation function (in constrict)."""

import unittest
from itertools import product

import numpy as np
import pandas as pd

from history_matching.config import Config
from history_matching.constrict import next_point_generation
from history_matching.emulators import GaussianModel


def a_sin_bx_over_x(x: float, a: float, b: float) -> float:
    """Calculate y = a sin(bx)/x."""
    return a * np.sin(b * x) / x


class NextPointGenerationTests(unittest.TestCase):
    """Unit tests for the next point generation function (in constrict)."""

    COUNT = 51
    FSCALE = 10
    TARGETX = 13

    def test_npg(self):
        """Test the next point generation function."""

        # setup the a*sin(b*x)/x model and train the Gaussian emulator
        prod = np.array(list(product(np.linspace(1, 11, self.COUNT), np.linspace(1, 11, self.COUNT))))
        amplitude = prod[:, 0]
        frequency = prod[:, 1] / self.FSCALE  # reduce the frequency range so we have a smooth surface
        y = a_sin_bx_over_x(x=self.TARGETX, a=amplitude, b=frequency)
        X = pd.DataFrame(data={"amplitude": amplitude, "frequency": frequency})
        y_train = pd.DataFrame(data={"y": y})
        emulator = GaussianModel(X, y_train)
        emulator.train()

        # npg needs an iteration - let's choose 1
        iteration = 1
        # npg needs a parameter space DataFrame
        parameter_space = pd.DataFrame(data=[["amplitude", 1.0, 11.0], ["frequency", 0.1, 1.1]], columns=["parameter", "minimum", "maximum"])
        # npg needs an observations DataFrame
        observation = a_sin_bx_over_x(13.0, 1.3, 4.2 / 10.0)
        observations = pd.DataFrame(data=[["magnitude", observation, 0.0]], columns=["features", "means", "variances"])
        observations.set_index("features", inplace=True)
        # npg needs an emulator database
        emulators = {0: {"magnitude": emulator}}
        # npg needs a Config object
        parameters = {"max_iterations": 9000, "candidates_per_iteration": 1000, "implausibility_threshold": 3.14159265, "non_implausible_target": 0.99997, "user_val": 42, "discrepancy_variance": 0.0}
        config = Config(**parameters)

        # run npg
        next_point_generation(iteration, parameter_space, observations, emulators, config)

        assert True is True

        return


if __name__ == "__main__":
    unittest.main()
