#! /usr/bin/env python3

import unittest

import pandas as pd

from history_matching import grid_sampler, latin_hypercube_sampler, random_sampler

class SamplerTests(unittest.TestCase):

    parameter_space = pd.DataFrame(data=[["x", 0, 10], ["y", 0, 100], ["z", 0, 1000]], columns=["parameter", "minimum", "maximum"])

    def test_lhs(self):

        points = latin_hypercube_sampler(SamplerTests.parameter_space, 10)
        self.assertSetEqual(set(points.x), set([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]))
        # Chances these are equal = 1:10! (1/3628800)
        self.assertFalse(list(points.x) == list([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]))
        self.assertSetEqual(set(points.y), set([5, 15, 25, 35, 45, 55, 65, 75, 85, 95]))
        # Chances these are equal = 1:10! (1/3628800)
        self.assertFalse(list(points.y) == list([5, 15, 25, 35, 45, 55, 65, 75, 85, 95]))
        self.assertSetEqual(set(points.z), set([50, 150, 250, 350, 450, 550, 650, 750, 850, 950]))
        # Chances these are equal = 1:10! (1/3628800)
        self.assertFalse(list(points.z) == list([50, 150, 250, 350, 450, 550, 650, 750, 850, 950]))

        return

    def test_grid(self):

        points = grid_sampler(SamplerTests.parameter_space, 10)
        self.assertSetEqual(set(points.x), set([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]))
        self.assertSetEqual(set(points.y), set([5, 15, 25, 35, 45, 55, 65, 75, 85, 95]))
        self.assertSetEqual(set(points.z), set([50, 150, 250, 350, 450, 550, 650, 750, 850, 950]))

        return

    def test_random(self):

        points = random_sampler(SamplerTests.parameter_space, 10)
        self.assertEqual(len(points.x), 10)
        self.assertTrue(all(map(lambda p: (p >= 0) and (p <= 10), points.x)))
        self.assertEqual(len(points.y), 10)
        self.assertTrue(all(map(lambda p: (p >= 0) and (p <= 100), points.y)))
        self.assertEqual(len(points.z), 10)
        self.assertTrue(all(map(lambda p: (p >= 0) and (p <= 1000), points.z)))

        return


if __name__ == "__main__":
    unittest.main()
