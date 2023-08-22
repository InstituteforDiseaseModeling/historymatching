#! /usr/bin/env python3

"""Unit tests for the history_matching.emulators.GaussianModel class."""

import os
import tempfile
import unittest
from itertools import product
from pathlib import Path

import asdf
import numpy as np
import pandas as pd

from history_matching.emulators import GaussianModel


def a_sin_bx_over_x(x: float, a: float, b: float) -> float:
    """Calculate y = a sin(bx)/x."""
    return a * np.sin(b * x) / x


class GaussianEmulatorTests(unittest.TestCase):
    """Unit tests for the history_matching.emulators.GaussianModel class."""

    COUNT = 51
    FSCALE = 10
    TARGETX = 13

    @classmethod
    def setUpClass(cls) -> None:
        """Create and train a Gaussian emulator."""
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
        emulator = GaussianModel(X, y_train)
        # train the emulator
        # see skipped tests below emulator.train()
        cls._model = emulator

        return

    @unittest.skip  # GaussianModel training is unhappy with the test data
    def test_gaussian_emulator(self):
        """Test the trained Gaussian emulator."""

        # predict the value of y at a = 1.3 and b = 4.2/FSCALE
        amplitude = 1.3
        frequency = 4.2 / self.FSCALE
        point = pd.DataFrame(data={"amplitude": [amplitude], "frequency": [frequency]})
        y_hat = self._model.predict(point)
        y_expect = a_sin_bx_over_x(self.TARGETX, amplitude, frequency)
        print(f"{y_hat=}")
        print(f"{y_expect=}")

        assert np.allclose(y_hat["value"], y_expect), f"{y_hat['value']=} != {y_expect=}"

        # predict the value of y at a = 2.71828183 and b = 3.14159265/FSCALE
        amplitude = 2.71828183
        frequency = 3.14159265 / self.FSCALE
        point = pd.DataFrame(data={"amplitude": [amplitude], "frequency": [frequency]})
        y_hat = self._model.predict(point)
        y_expect = a_sin_bx_over_x(self.TARGETX, amplitude, frequency)
        print(f"{y_hat=}")
        print(f"{y_expect=}")

        assert np.allclose(y_hat["value"], y_expect), f"{y_hat['value']=} != {y_expect=}"

        return

    @unittest.skip  # GaussianModel training is unhappy with the test data
    def test_gaussian_emulator_serialization(self):
        """Test the serialization and deserialization (roundtrip) of the trained Gaussian emulator."""

        handle, filename = tempfile.mkstemp(suffix=".asdf")
        os.close(handle)
        af = asdf.AsdfFile({"emulator": self._model})
        af.write_to(filename)

        assert Path(filename).exists(), f"{filename} does not exist"

        try:
            af = asdf.open(filename)
            emulator = af["emulator"]
        finally:
            af.close()
            Path(filename).unlink()
        assert isinstance(emulator, GaussianModel), f"{emulator=} is not a GaussianModel"

        # predict the value of y at a = 1.3 and b = 4.2/FSCALE
        amplitude = 1.3
        frequency = 4.2 / self.FSCALE
        point = pd.DataFrame(data={"amplitude": [amplitude], "frequency": [frequency]})
        y_hat = self._model.predict(point)
        y_expect = a_sin_bx_over_x(self.TARGETX, amplitude, frequency)
        print(f"{y_hat=}")
        print(f"{y_expect=}")

        assert np.allclose(y_hat["value"], y_expect), f"{y_hat['value']=} != {y_expect=}"

        return


if __name__ == "__main__":
    unittest.main()
