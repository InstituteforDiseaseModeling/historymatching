#! /usr/bin/env python3

"""Unit tests for GPFlowGPR class."""

import unittest
from itertools import product

import numpy as np
import pandas as pd

from history_matching.emulators import GPFlowGPR


def a_sin_bx_over_x(x: float, a: float, b: float) -> float:
    """Calculate y = a sin(bx)/x."""
    return a * np.sin(b * x) / x


class GPFlowGPRTests(unittest.TestCase):

    """Unit tests for the GPFlowGPR class."""

    COUNT = 51
    FSCALE = 10
    TARGETX = 13

    @classmethod
    def setUpClass(cls) -> None:
        """Create and train a GPFlowGPR emulator."""
        prod = np.array(list(product(np.linspace(1, 11, cls.COUNT), np.linspace(1, 11, cls.COUNT))))
        amplitude = prod[:, 0]
        frequency = prod[:, 1] / cls.FSCALE  # reduce the frequency range so we have a smooth surface
        # calculate y = a sin(bx)/x for each a and b at x = 13
        y = a_sin_bx_over_x(x=cls.TARGETX, a=amplitude, b=frequency)
        # create a dataframe called X with columns amplitude and frequency
        X = pd.DataFrame(data={"amplitude": amplitude, "frequency": frequency})
        # create a dataframe called y_train with column y
        y_train = pd.DataFrame(data={"y": y})
        # create a Gaussian emulator with X and y
        emulator = GPFlowGPR(X, y_train)
        # train the emulator
        emulator.train()
        cls._model = emulator

        return

    def test_gpflowgpr_emulator(self):
        """Test the trained GPFlowGPR emulator."""

        # predict the value of y at a = 1.3 and b = 4.2/FSCALE
        amplitude = 1.3
        frequency = 4.2 / self.FSCALE
        point = pd.DataFrame(data={"amplitude": [amplitude], "frequency": [frequency]})
        y_hat = self._model.predict(point)
        y_expect = a_sin_bx_over_x(self.TARGETX, amplitude, frequency)
        print(f"{y_hat=}")
        print(f"{y_expect=}")

        assert np.allclose(y_hat["value"].to_numpy(), y_expect, atol=y_hat["variance"].to_numpy()), f"{y_hat['value']=} != {y_expect=}"

        # predict the value of y at a = 2.71828183 and b = 3.14159265/FSCALE
        amplitude = 2.71828183
        frequency = 3.14159265 / self.FSCALE
        point = pd.DataFrame(data={"amplitude": [amplitude], "frequency": [frequency]})
        y_hat = self._model.predict(point)
        y_expect = a_sin_bx_over_x(self.TARGETX, amplitude, frequency)
        print(f"{y_hat=}")
        print(f"{y_expect=}")

        assert np.allclose(y_hat["value"].to_numpy(), y_expect, atol=y_hat["variance"].to_numpy()), f"{y_hat['value']=} != {y_expect=}"

        return


if __name__ == "__main__":
    unittest.main()
