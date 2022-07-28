import glob
import subprocess
import unittest

import pandas as pd


class TestFramesForValidity(unittest.TestCase):
    def test_examples_run(self):
        # Find examples and assert that we found some (implicit path check)
        examples_script_list = glob.glob("examples/*.py")
        self.assertTrue(len(examples_script_list) > 0)

        # Ensure each example runs
        for example_script in examples_script_list:
            ret = subprocess.call(
                ["python3", example_script],
                shell=False,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
            self.assertTrue(
                ret == 0,
                msg=f"{example_script} did not run cleanly. Probably an exception!",
            )


if __name__ == "__main__":
    unittest.main()
