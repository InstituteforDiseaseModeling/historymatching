#! /usr/bin/env python3

import unittest

import pandas as pd

from history_matching import grid_sampler
from history_matching import latin_hypercube_sampler
from history_matching import random_sampler


class SamplerTests(unittest.TestCase):
    parameter_space = pd.DataFrame(data=[["x", 0, 10], ["y", 0, 100], ["z", 0, 1000]], columns=["parameter", "minimum", "maximum"])

    def test_lhs(self):
        points = latin_hypercube_sampler(SamplerTests.parameter_space, 10)
        assert set(points.x) == {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5}
        # Chances these are equal = 1:10! (1/3628800)
        assert not list(points.x) == [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        assert set(points.y) == {5, 15, 25, 35, 45, 55, 65, 75, 85, 95}
        # Chances these are equal = 1:10! (1/3628800)
        assert not list(points.y) == [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        assert set(points.z) == {50, 150, 250, 350, 450, 550, 650, 750, 850, 950}
        # Chances these are equal = 1:10! (1/3628800)
        assert not list(points.z) == [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]

        return

    def test_grid(self):
        points = grid_sampler(SamplerTests.parameter_space, 10)
        assert set(points.x) == {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5}
        assert set(points.y) == {5, 15, 25, 35, 45, 55, 65, 75, 85, 95}
        assert set(points.z) == {50, 150, 250, 350, 450, 550, 650, 750, 850, 950}

        return

    def test_random(self):
        points = random_sampler(SamplerTests.parameter_space, 10)
        assert len(points.x) == 10
        assert all(p >= 0 and p <= 10 for p in points.x)
        assert len(points.y) == 10
        assert all(p >= 0 and p <= 100 for p in points.y)
        assert len(points.z) == 10
        assert all(p >= 0 and p <= 1000 for p in points.z)

        return


if __name__ == "__main__":
    unittest.main()
