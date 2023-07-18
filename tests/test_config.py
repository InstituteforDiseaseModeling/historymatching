#! /usr/bin/env python3

import unittest

from history_matching.config import Config


class ConfigTests(unittest.TestCase):

    """Test Config object"""

    def test_constructor(self):
        """Test Config constructor"""

        parameters = {"max_iterations": 9000, "candidates_per_iteration": 1000, "implausibility_threshold": 3.14159265, "non_implausible_target": 0.99997, "user_val": 42}
        config = Config(**parameters)

        assert config.max_iterations == parameters["max_iterations"]
        assert config.candidates_per_iteration == parameters["candidates_per_iteration"]
        assert config.implausibility_threshold == parameters["implausibility_threshold"]
        assert config.non_implausible_target == parameters["non_implausible_target"]
        assert config.user["user_val"] == parameters["user_val"]

        return


if __name__ == "__main__":
    unittest.main()
