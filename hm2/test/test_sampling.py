import numpy as np
import pandas as pd
import unittest

from hm2.sampling import latin_hypercube, latin_hypercube_within
from hm2.error import HistoryMatchingError

class SamplingTest(unittest.TestCase):
    def test_missing_name(self):
        param_info = pd.DataFrame({
            'name2': ['Beta', 'Gamma'],
            'min':  [  1e-6,    1e-6],
            'max':  [  0.01,     0.5]
        })
        self.assertRaises(HistoryMatchingError, latin_hypercube, param_info, 30)

    def test_missing_min(self):
        param_info = pd.DataFrame({
            'name': ['Beta', 'Gamma'],
            'min2': [  1e-6,    1e-6],
            'max':  [  0.01,     0.5]
        })
        self.assertRaises(HistoryMatchingError, latin_hypercube, param_info, 30)

    def test_missing_max(self):
        param_info = pd.DataFrame({
            'name':  ['Beta', 'Gamma'],
            'min':   [  1e-6,    1e-6],
            'max2':  [  0.01,     0.5]
        })
        self.assertRaises(HistoryMatchingError, latin_hypercube, param_info, 30)

    def test_missing_misordered(self):
        param_info = pd.DataFrame({
            'name':  ['Beta', 'Gamma'],
            'min':   [     3,       4],
            'max':   [   100,    -100]
        })
        self.assertRaises(HistoryMatchingError, latin_hypercube, param_info, 30)

    def test_scaling(self):
        param_info = pd.DataFrame({
            'name':  ['Beta', 'Gamma'],
            'min':   [   -100,     10],
            'max':   [    100,    100]
        })

        cube = latin_hypercube(param_info, 10000)

        #TODO: Better statistical tests?
        self.assertTrue( (cube['Beta' ] >  98).any() )
        self.assertTrue( (cube['Beta' ] < -98).any() )
        self.assertTrue( (cube['Gamma'] >  98).any() )
        self.assertTrue( (cube['Gamma'] <  12).any() )

        self.assertTrue( not (cube['Beta' ] >  100).any() )
        self.assertTrue( not (cube['Beta' ] < -100).any() )
        self.assertTrue( not (cube['Gamma'] >  100).any() )
        self.assertTrue( not (cube['Gamma'] <   10).any() )

    def test_has_sample_id(self):
        param_info = pd.DataFrame({
            'name':  ['Beta', 'Gamma'],
            'min':   [   -100,     10],
            'max':   [    100,    100]
        })

        cube = latin_hypercube(param_info, 10000)

        self.assertEqual(cube['param_id'].tolist(), list(range(len(cube))))



class SampleWithinTest(unittest.TestCase):
    def test_invalid_frame(self):
        param_info = pd.DataFrame({
            'name2': ['Beta', 'Gamma'],
            'min':  [  1e-6,    1e-6],
            'max':  [  0.01,     0.5]
        })
        self.assertRaises(HistoryMatchingError, latin_hypercube_within, param_info, 30)

    def test_scaling(self):
        param_info = pd.DataFrame({
            'name':  ['Beta', 'Gamma'],
            'min':   [   -100,     10],
            'max':   [    100,    100]
        })

        cube = latin_hypercube(param_info, 10000)

        latin_hypercube_within(cube, 10000)

        #TODO: Better statistical tests?
        self.assertTrue( (cube['Beta' ] >  98).any() )
        self.assertTrue( (cube['Beta' ] < -98).any() )
        self.assertTrue( (cube['Gamma'] >  98).any() )
        self.assertTrue( (cube['Gamma'] <  12).any() )

        self.assertTrue( not (cube['Beta' ] >  100).any() )
        self.assertTrue( not (cube['Beta' ] < -100).any() )
        self.assertTrue( not (cube['Gamma'] >  100).any() )
        self.assertTrue( not (cube['Gamma'] <   10).any() )
