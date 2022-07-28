import unittest
import pandas as pd

from hm2.error import HistoryMatchingError
from hm2.utility import Scaler, drop_key


class ScalerTest(unittest.TestCase):
    def test_not_a_dataframe(self):
        data = [[1, 2, 3, 4, 5], [10, 11, 12, 13, 14], [-1, -2, -3, -4, -5]]
        self.assertRaises(TypeError, Scaler, data)

    def test_columns_mismatch(self):
        data = pd.DataFrame(
            {"x": [1, 2, 3, 4, 5], "y": [10, 11, 12, 13, 14], "z": [-1, -2, -3, -4, -5]}
        )
        s = Scaler(data)
        data = pd.DataFrame(
            {"m": [1, 2, 3, 4, 5], "y": [10, 11, 12, 13, 14], "z": [-1, -2, -3, -4, -5]}
        )
        self.assertRaises(HistoryMatchingError, s.transform, data)

    def test_scales_correctly(self):
        data = pd.DataFrame(
            {"x": [1, 2, 3, 4, 5], "y": [10, 11, 12, 13, 14], "z": [-5, -4, -3, -2, -1]}
        )
        s = Scaler(data)
        data = pd.DataFrame({"x": [3], "y": [10], "z": [-1]})
        trans = s.transform(data)
        self.assertTrue((trans["x"] == 0.5).all())
        self.assertTrue((trans["y"] == 0).all())
        self.assertTrue((trans["z"] == 1).all())

    def test_repr(self):
        data = pd.DataFrame(
            {"x": [1, 2, 3, 4, 5], "y": [10, 11, 12, 13, 14], "z": [-5, -4, -3, -2, -1]}
        )
        s = Scaler(data)
        self.assertTrue(
            str(s)
            == """  feature   min   max  range
0       x   1.0   5.0    4.0
1       y  10.0  14.0    4.0
2       z  -5.0  -1.0    4.0"""
        )


class DropKeyTest(unittest.TestCase):
    def test_drop_key(self):
        a = {"hi": 1, "bye": 2}
        b = drop_key(a, "bye")
        self.assertTrue("bye" in a)
        self.assertTrue("bye" not in b)

    def test_drop_missing_without_ignore_raises(self):
        a = {"hi": 1, "bye": 2}
        self.assertRaises(HistoryMatchingError, drop_key, a, "ted")

    def test_drop_missing_with_ignore_is_fine(self):
        a = {"hi": 1, "bye": 2}
        b = drop_key(a, "ted", ignore_missing=True)
        self.assertTrue(a == b)


if __name__ == "__main__":
    unittest.main()
