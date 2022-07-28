import unittest

from hm2.models import SIR



class TestSIR(unittest.TestCase):
    def test_sir_determinism(self):
      sir1 = SIR(seed=123456)
      sir2 = SIR(seed=123456)
      results1 = sir1.run()
      results2 = sir2.run()
      self.assertTrue(results1.equals(results2))
      sir1.__repr__()
