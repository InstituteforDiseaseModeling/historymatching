import unittest
import numpy as np
import pandas as pd

from hm2.examples.sir import SIR
import hm2.sampling
import hm2.boilerplate



class SamplingTest(unittest.TestCase):
    @staticmethod
    def run_model_correctly(model):
        results = model.sim()
        results['prevalence'] = results['per_infected']
        results['Stdev'] = 1 #Junk value TODO
        return results

    def test_observations_missing_time(self):
        #Observations is missing `time`
        observations = pd.DataFrame({
            'observation_id': [ 0,   1],
            'prevalence':     [15,  40],
            'Stdev':          [ 4, 2.3],
        })

        param_info = pd.DataFrame({
            'name': ['beta', 'gamma'],
            'min':  [  1e-6,    1e-6],
            'max':  [  0.01,     0.5]
        })

        self.assertRaises(AssertionError,
          hm2.boilerplate.time_analysis,
          parameter_samples = hm2.sampling.latin_hypercube(param_info, 100),
          observations = observations,
          init_func=SIR,
          run_func=self.run_model_correctly,
          replicates=1
        )

    def test_observations_missing_observation_id(self):
        #Observations is missing `observation_id`
        observations = pd.DataFrame({
            'time':           [ 3,  15],
            'prevalence':     [15,  40],
            'Stdev':          [ 4, 2.3],
        })

        param_info = pd.DataFrame({
            'name': ['beta', 'gamma'],
            'min':  [  1e-6,    1e-6],
            'max':  [  0.01,     0.5]
        })

        self.assertRaises(AssertionError,
          hm2.boilerplate.time_analysis,
          parameter_samples = hm2.sampling.latin_hypercube(param_info, 100),
          observations = observations,
          init_func=SIR,
          run_func=self.run_model_correctly,
          replicates=1
        )

    def test_model_results_missing_time(self):
        def run_model_without_time(model):
            results = model.sim()
            results['prevalence'] = results['per_infected']
            results['Stdev'] = 1 #Junk value TODO
            del results['time']
            return results

        observations = pd.DataFrame({
            'time':           [ 3,  15],
            'observation_id': [ 0,   1],
            'prevalence':     [15,  40],
            'Stdev':          [ 4, 2.3],
        })

        param_info = pd.DataFrame({
            'name': ['beta', 'gamma'],
            'min':  [  1e-6,    1e-6],
            'max':  [  0.01,     0.5]
        })

        self.assertRaises(AssertionError,
          hm2.boilerplate.time_analysis,
          parameter_samples = hm2.sampling.latin_hypercube(param_info, 100),
          observations = observations,
          init_func=SIR,
          run_func=run_model_without_time,
          replicates=1
        )

    def test_model_results_missing_observation_name(self):
        def run_model_without_time(model):
            results = model.sim()
            # Oh no, we forgot to include prevalence in the results!
            # results['prevalence'] = results['per_infected']
            results['Stdev'] = 1 #Junk value TODO
            return results

        observations = pd.DataFrame({
            'time':           [ 3,  15],
            'observation_id': [ 0,   1],
            'prevalence':     [15,  40],
            'Stdev':          [ 4, 2.3],
        })

        param_info = pd.DataFrame({
            'name': ['beta', 'gamma'],
            'min':  [  1e-6,    1e-6],
            'max':  [  0.01,     0.5]
        })

        self.assertRaises(Exception,
          hm2.boilerplate.time_analysis,
          parameter_samples = hm2.sampling.latin_hypercube(param_info, 100),
          observations = observations,
          init_func=SIR,
          run_func=run_model_without_time,
          replicates=1
        )

    def test_time_analysis_returns_correct_columns(self):
        def run_model_without_time(model):
            results = model.sim()
            results['prevalence'] = results['per_infected']
            results['Stdev'] = 1 #Junk value TODO
            return results

        observations = pd.DataFrame({
            'time':           [ 3,  15],
            'observation_id': [ 0,   1],
            'prevalence':     [15,  40],
            'Stdev':          [ 4, 2.3],
        })

        param_info = pd.DataFrame({
            'name': ['beta', 'gamma'],
            'min':  [  1e-6,    1e-6],
            'max':  [  0.01,     0.5]
        })

        results = hm2.boilerplate.time_analysis(
          parameter_samples = hm2.sampling.latin_hypercube(param_info, 100),
          observations = observations,
          init_func=SIR,
          run_func=run_model_without_time,
          replicates=1
        )

        self.assertEqual(results.columns.tolist(), ['sample_id', 'replicate', 'observation_id', 'time', 'prevalence', 'Stdev'])

    def test_time_analysis_automatically_adds_sample_id(self):
        def run_model_without_time(model):
            results = model.sim()
            results['prevalence'] = results['per_infected']
            results['Stdev'] = 1 #Junk value TODO
            return results

        observations = pd.DataFrame({
            'time':           [ 3,  15],
            'observation_id': [ 0,   1],
            'prevalence':     [15,  40],
            'Stdev':          [ 4, 2.3],
        })

        param_info = pd.DataFrame({
            'name': ['beta', 'gamma'],
            'min':  [  1e-6,    1e-6],
            'max':  [  0.01,     0.5]
        })

        parameter_samples = hm2.sampling.latin_hypercube(param_info, 100)
        del parameter_samples['sample_id']

        results = hm2.boilerplate.time_analysis(
          parameter_samples = parameter_samples,
          observations = observations,
          init_func=SIR,
          run_func=run_model_without_time,
          replicates=1
        )
