import numpy as np
import pandas as pd
import unittest

from hm2.sampling import latin_hypercube, latin_hypercube_within
from hm2.error import *

class SamplingTest(unittest.TestCase):
    def test_missing_name(self):
        param_info = pd.DataFrame({
            'name2': ['Beta', 'Gamma'],
            'min':  [  1e-6,    1e-6],
            'max':  [  0.01,     0.5]
        })
        with self.assertRaises(HMMissingColumn) as cm:
            latin_hypercube(param_info, 30)
        self.assertTrue(cm.exception.missing_column=='name')
        self.assertTrue(cm.exception.df_name=='ParameterInfoFrame')

    def test_missing_min(self):
        param_info = pd.DataFrame({
            'name': ['Beta', 'Gamma'],
            'min2': [  1e-6,    1e-6],
            'max':  [  0.01,     0.5]
        })
        with self.assertRaises(HMMissingColumn) as cm:
            latin_hypercube(param_info, 30)
        self.assertTrue(cm.exception.missing_column=='min')
        self.assertTrue(cm.exception.df_name=='ParameterInfoFrame')

    def test_missing_max(self):
        param_info = pd.DataFrame({
            'name':  ['Beta', 'Gamma'],
            'min':   [  1e-6,    1e-6],
            'max2':  [  0.01,     0.5]
        })
        with self.assertRaises(HMMissingColumn) as cm:
            latin_hypercube(param_info, 30)
        self.assertTrue(cm.exception.missing_column=='max')
        self.assertTrue(cm.exception.df_name=='ParameterInfoFrame')

    def test_max_smaller_than_min(self):
        param_info = pd.DataFrame({
            'name':  ['Beta', 'Gamma'],
            'min':   [     3,       4],
            'max':   [   100,    -100]
        })
        self.assertRaises(HMMaxLessThanMin, latin_hypercube, param_info, 30)

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

    def test_set_random_stays_internal(self):
        #Ensure that setting the random_state for reproducibility doesn't affect outside random state
        param_info = pd.DataFrame({
            'name':  ['Beta', 'Gamma'],
            'min':   [   -100,     10],
            'max':   [    100,    100]
        })

        #We test by seeing whether a full sequence is equal to a sequence generated from the same seed
        #where the generation of the second sequence is split by a call to the latin hypercube

        #Generate the full sequence
        np.random.seed(123456)
        full = np.random.randint(low=0, high=100, size=20)

        #Generate half the sequence
        np.random.seed(123456)
        first_half = np.random.randint(low=0, high=100, size=10)

        #Generate hypercube. This will change the random state if we fail to preserve it
        cube = latin_hypercube(param_info, 10000, random_state=654321)

        second_half = np.random.randint(low=0, high=100, size=10)

        self.assertTrue(np.array_equal(full,np.hstack((first_half,second_half))))

    def test_set_random_reproduces(self):
        #Ensure that setting the random_state for reproducibility doesn't affect outside random state
        param_info = pd.DataFrame({
            'name':  ['Beta', 'Gamma'],
            'min':   [   -100,     10],
            'max':   [    100,    100]
        })

        #Generate hypercube twice. If random_state doesn't work we'll probably get different numbers.
        cube1 = latin_hypercube(param_info, 10000, random_state=654321)
        cube2 = latin_hypercube(param_info, 10000, random_state=654321)

        self.assertTrue(cube1.equals(cube2))


class SampleWithinTest(unittest.TestCase):
    def test_invalid_frame(self):
        param_info = pd.DataFrame({
            'name2': ['Beta', 'Gamma'],
            'min':  [  1e-6,    1e-6],
            'max':  [  0.01,     0.5]
        })
        with self.assertRaises(HMMissingColumn) as cm:
            latin_hypercube_within(param_info, 30)
        self.assertTrue(cm.exception.missing_column=='param_id')
        self.assertTrue(cm.exception.df_name=='ParameterSamplesFrame')

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

    def test_empty(self):
        param_info = pd.DataFrame({'name':[],'min':[],'max':[]})
        self.assertRaises(HMParameterSamplesEmpty, latin_hypercube_within, param_info, 30)