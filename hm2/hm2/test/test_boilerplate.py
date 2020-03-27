import unittest
import numpy as np
import pandas as pd

from hm2.examples.sir import SIR
import hm2.sampling
import hm2.boilerplate
from hm2.error import HistoryMatchingError



class ModelWrapperTest(unittest.TestCase):
  def test_missing_init(self):
    class SIRWrapper(hm2.boilerplate.ModelWrapper):
        @classmethod
        def init2(cls, **kwargs): #Init misnamed
            return SIR(**kwargs)
        @staticmethod
        def run(model):
          results = model.sim()
          results['prevalence'] = results['per_infected']
          results['Stdev'] = 1 #Junk value TODO
          return results
    self.assertRaises(TypeError, SIRWrapper)

  def test_missing_run(self):
    class SIRWrapper(hm2.boilerplate.ModelWrapper):
        @classmethod
        def init(cls, **kargs):
            return SIR(**kwargs)
        @staticmethod
        def run2(model): #Run misnamed
          results = model.sim()
          results['prevalence'] = results['per_infected']
          results['Stdev'] = 1 #Junk value TODO
          return results
    self.assertRaises(TypeError, SIRWrapper)

    

class BoilerplateTest(unittest.TestCase):
    def setUp(self):
        self.observations = pd.DataFrame({
            'time':           [ 3,  15],
            'observation_id': [ 0,   1],
            'prevalence':     [15,  40],
            'Stdev':          [ 4, 2.3],
        })

        self.param_info = pd.DataFrame({
            'name': ['beta', 'gamma'],
            'min':  [  1e-6,    1e-6],
            'max':  [  0.01,     0.5]
        })

    def test_observations_missing_time(self):

        class SIRWrapper(hm2.boilerplate.ModelWrapper):
            @classmethod
            def init(cls, **kwargs):
                return SIR(**kwargs)
            @staticmethod
            def run(model):
              results = model.sim()
              results['prevalence'] = results['per_infected']
              results['Stdev'] = 1 #Junk value TODO
              return results

        #Observations is missing `time`
        del self.observations['time']

        self.assertRaises(AssertionError,
          hm2.boilerplate.time_analysis,
          parameter_samples = hm2.sampling.latin_hypercube(self.param_info, 100),
          observations = self.observations,
          wrapped_model = SIRWrapper(),
          replicates=1
        )

    def test_observations_missing_observation_id(self):
        class SIRWrapper(hm2.boilerplate.ModelWrapper):
            @classmethod
            def init(cls, **kwargs):
                return SIR(**kwargs)
            @staticmethod
            def run(model):
              results = model.sim()
              results['prevalence'] = results['per_infected']
              results['Stdev'] = 1 #Junk value TODO
              return results

        #Observations is missing `observation_id`
        del self.observations['observation_id']

        self.assertRaises(AssertionError,
          hm2.boilerplate.time_analysis,
          parameter_samples = hm2.sampling.latin_hypercube(self.param_info, 100),
          observations = self.observations,
          wrapped_model=SIRWrapper(),
          replicates=1
        )

    def test_model_results_missing_time(self):
        class SIRWrapper(hm2.boilerplate.ModelWrapper):
            @classmethod
            def init(cls, **kwargs):
                return SIR(**kwargs)
            @staticmethod
            def run(model): #Runs model but doesn't return time
              results = model.sim()
              results['prevalence'] = results['per_infected']
              results['Stdev'] = 1 #Junk value TODO
              del results['time']
              return results

        self.assertRaises(AssertionError,
          hm2.boilerplate.time_analysis,
          parameter_samples = hm2.sampling.latin_hypercube(self.param_info, 100),
          observations = self.observations,
          wrapped_model=SIRWrapper(),
          replicates=1
        )

    def test_model_results_missing_observation_name(self):
        class SIRWrapper(hm2.boilerplate.ModelWrapper):
            @classmethod
            def init(cls, **kwargs):
                return SIR(**kwargs)
            @staticmethod
            def run(model):
              results = model.sim()
              # Oh no, we forgot to include prevalence in the results!
              # results['prevalence'] = results['per_infected']
              results['Stdev'] = 1 #Junk value TODO
              return results

        self.assertRaises(Exception,
          hm2.boilerplate.time_analysis,
          parameter_samples = hm2.sampling.latin_hypercube(self.param_info, 100),
          observations = self.observations,
          wrapped_model = SIRWrapper(),
          replicates=1
        )

    def test_time_analysis_returns_correct_columns(self):
        class SIRWrapper(hm2.boilerplate.ModelWrapper):
            @classmethod
            def init(cls, **kwargs):
                return SIR(**kwargs)
            @staticmethod
            def run(model):
              results = model.sim()
              results['prevalence'] = results['per_infected']
              results['Stdev'] = 1 #Junk value TODO
              return results

        results = hm2.boilerplate.time_analysis(
          parameter_samples = hm2.sampling.latin_hypercube(self.param_info, 100),
          observations = self.observations,
          wrapped_model = SIRWrapper(),
          replicates=1
        )

        self.assertEqual(results.columns.tolist(), ['sample_id', 'replicate', 'observation_id', 'time', 'prevalence', 'Stdev'])

    def test_time_analysis_automatically_adds_sample_id(self):
        class SIRWrapper(hm2.boilerplate.ModelWrapper):
            @classmethod
            def init(cls, **kwargs):
                return SIR(**kwargs)
            @staticmethod
            def run(model):
              results = model.sim()
              results['prevalence'] = results['per_infected']
              results['Stdev'] = 1 #Junk value TODO
              return results


        parameter_samples = hm2.sampling.latin_hypercube(self.param_info, 100)
        del parameter_samples['sample_id']

        results = hm2.boilerplate.time_analysis(
          parameter_samples = parameter_samples,
          observations = self.observations,
          wrapped_model = SIRWrapper(),
          replicates=1
        )

    def test_bad_wrap(self):
        class SIRWrapper: #Note that this is unwrapped
            @classmethod
            def init(cls, **kwargs):
                return SIR(**kwargs)
            @staticmethod
            def run(model):
              results = model.sim()
              results['prevalence'] = results['per_infected']
              results['Stdev'] = 1 #Junk value TODO
              return results

        parameter_samples = hm2.sampling.latin_hypercube(self.param_info, 100)

        self.assertRaises(HistoryMatchingError, hm2.boilerplate.time_analysis,
          parameter_samples = parameter_samples,
          observations = self.observations,
          wrapped_model = SIRWrapper(),
          replicates=1
        )