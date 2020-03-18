import unittest
import numpy as np
import pandas as pd
from hm2.sampling import latin_hypercube

class SamplingTest(unittest.TestCase):
    def test_missing_name(self):
        param_info = pd.DataFrame({
            'name2': ['Beta', 'Gamma'],
            'min':  [  1e-6,    1e-6],
            'max':  [  0.01,     0.5]
        })
        self.assertRaises(AssertionError, latin_hypercube, param_info, 30)

    def test_missing_min(self):
        param_info = pd.DataFrame({
            'name': ['Beta', 'Gamma'],
            'min2': [  1e-6,    1e-6],
            'max':  [  0.01,     0.5]
        })
        self.assertRaises(AssertionError, latin_hypercube, param_info, 30)

    def test_missing_max(self):
        param_info = pd.DataFrame({
            'name':  ['Beta', 'Gamma'],
            'min':   [  1e-6,    1e-6],
            'max2':  [  0.01,     0.5]
        })
        self.assertRaises(AssertionError, latin_hypercube, param_info, 30)

    def test_missing_misordered(self):
        param_info = pd.DataFrame({
            'name':  ['Beta', 'Gamma'],
            'min':   [     3,       4],
            'max':   [   100,    -100]
        })
        self.assertRaises(AssertionError, latin_hypercube, param_info, 30)
        
    def test_scaling(self):
        param_info = pd.DataFrame({
            'name':  ['Beta', 'Gamma'],
            'min':   [   -100,     10],
            'max':   [    100,    100]
        })

        cube = latin_hypercube(param_info, 10000)

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

        self.assertEqual(cube['sample_id'].tolist(), list(range(len(cube))))