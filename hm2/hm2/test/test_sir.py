import unittest

from hm2.examples.sir import SIR



class TestSIR(unittest.TestCase):
    def test_seed_works(self):
      sir1 = SIR(seed=123456)
      sir2 = SIR(seed=123456)
      results1 = sir1.sim()
      results2 = sir2.sim()
      self.assertTrue(results1.equals(results2))