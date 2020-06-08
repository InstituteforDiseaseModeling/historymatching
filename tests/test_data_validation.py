import pandas as pd
import unittest

from hm2.error import HistoryMatchingError
import hm2.data_validation as dv

class TestFramesForValidity(unittest.TestCase):
    def test_non_frames(self):
      frame = "I am not a DataFrame"
      self.assertRaises(TypeError, dv.ValidateParameterSamplesFrame, frame)
      self.assertRaises(TypeError, dv.ValidateObservationsFrame,     frame)
      self.assertRaises(TypeError, dv.ValidateSimFrame,              frame)
      self.assertRaises(TypeError, dv.ValidateMatchedFrame,          frame)