#! /usr/bin/env python3

import unittest

from history_matching.config import Config

class ConfigTests(unittest.TestCase):

    def test_constructor(self):

        parameters = {
            "max_iterations": 9000,
            "implausibility_threshold": 3.14159265,
            "non_implausible_target": .99997,
            "user_val": 42
        }
        config = Config(**parameters)

        self.assertEqual(config.max_iterations, parameters["max_iterations"])
        self.assertEqual(config.implausibility_threshold, parameters["implausibility_threshold"])
        self.assertEqual(config.non_implausible_target, parameters["non_implausible_target"])
        self.assertEqual(config.user["user_val"], parameters["user_val"])

        return


if __name__ == "__main__":
    unittest.main()
