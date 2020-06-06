import pandas as pd
import unittest

from hm2.error import HistoryMatchingError
import hm2.data_validation as dv

class TestFramesForValidity(unittest.TestCase):
    def test_non_frames(self):
      frame = "I am not a DataFrame"
      self.assertRaises(TypeError, dv.ValidateParameterSamplesFrame,    frame)
      self.assertRaises(TypeError, dv.ValidateParameterSamplesFrame,    frame)
      self.assertRaises(TypeError, dv.ValidateSummaryObservationsFrame, frame)
      self.assertRaises(TypeError, dv.ValidateSummarySimFrame,          frame)
      self.assertRaises(TypeError, dv.ValidateTimeObservationsFrame,    frame)
      self.assertRaises(TypeError, dv.ValidateTimeSimFrame,             frame)
      self.assertRaises(TypeError, dv.ValidateMatchedFrame,             frame)

    def test_none_returns_none(self):
      frame = None
      self.assertTrue(dv.ValidateSimFrame(frame) is None)
      self.assertTrue(dv.ValidateSummaryObservationsFrame(frame) is None)
      self.assertTrue(dv.ValidateSummarySimFrame(frame) is None)
      self.assertTrue(dv.ValidateTimeObservationsFrame(frame) is None)
      self.assertTrue(dv.ValidateTimeSimFrame(frame) is None)
