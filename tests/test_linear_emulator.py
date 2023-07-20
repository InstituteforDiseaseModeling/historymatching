#! /usr/bin/env python3

"""Unit tests for the history_matching.emulators.LinearModel class."""

import os
import unittest
from itertools import product
from pathlib import Path
from tempfile import mkstemp

import asdf
import numpy as np
import pandas as pd

from history_matching.emulators import LinearModel


def mx_plus_b(x: float, m: float, b: float) -> float:
    """Calculate y = mx + b."""
    return m * x + b


class LinearEmulatorTests(unittest.TestCase):
    """Unit tests for the history_matching.emulators.LinearModel class."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create and train a linear emulator."""
        prod = np.array(list(product(np.linspace(0, 10, 11), np.linspace(0, 10, 11))))
        slope = prod[:, 0]
        intercept = prod[:, 1]
        # calculate y = mx + b for each m and b at x = 42
        y = mx_plus_b(x=42, m=slope, b=intercept)
        # create a dataframe called X with columns slope and intercept
        X = pd.DataFrame(data={"slope": slope, "intercept": intercept})
        # create a dataframe called y with column y
        y = pd.DataFrame(data={"y": y})
        # create a linear emulator with X and y
        emulator = LinearModel(X, y)
        # train the emulator
        emulator.train()
        cls._model = emulator

        return

    def test_linear_emulator(self):
        """Test the trained linear emulator."""

        # predict the value of y at m = 1.3 and b = 4.2
        y_hat = self._model.predict(pd.DataFrame(data={"slope": [1.3], "intercept": [4.2]}))
        y_expect = mx_plus_b(42, 1.3, 4.2)
        print(f"{y_hat=}")
        print(f"{y_expect=}")

        assert np.allclose(y_hat["value"], y_expect), f"{y_hat['value']=} != {y_expect=}"

        # predict the value of y at m = 2.71828183 and b = 3.14159265
        y_hat = self._model.predict(pd.DataFrame(data={"slope": [2.71828183], "intercept": [3.14159265]}))
        y_expect = mx_plus_b(42, 2.71828183, 3.14159265)
        print(f"{y_hat=}")
        print(f"{y_expect=}")

        assert np.allclose(y_hat["value"], y_expect), f"{y_hat['value']=} != {y_expect=}"

        return

    def test_linear_emulator_serialization(self):
        """Test the serialization and deserialization (roundtrip) of the trained linear emulator."""

        handle, filename = mkstemp(suffix=".asdf")
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
        assert isinstance(emulator, LinearModel), f"{emulator=} is not a LinearModel"

        # predict the value of y at m = 1.3 and b = 4.2
        y_hat = emulator.predict(pd.DataFrame(data={"slope": [1.3], "intercept": [4.2]}))
        y_expect = mx_plus_b(42, 1.3, 4.2)
        print(f"{y_hat=}")
        print(f"{y_expect=}")

        assert np.allclose(y_hat["value"], y_expect), f"{y_hat['value']=} != {y_expect=}"

        return


if __name__ == "__main__":
    unittest.main()
