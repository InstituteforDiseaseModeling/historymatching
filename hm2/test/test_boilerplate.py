import numpy as np
import pandas as pd
import unittest

from hm2.error import HistoryMatchingError
from hm2.models.sir import SIR
import hm2.boilerplate
import hm2.sampling



class BoilerplateTest(unittest.TestCase):
    def setUp(self):
        self.observations = pd.DataFrame({
            'time':           [ 3,  15],
            'observation_id': [ 0,   1],
            'prevalence':     [15,  40],
            'stdev':          [ 4, 2.3],
        })

        self.param_info = pd.DataFrame({
            'name': ['beta', 'gamma'],
            'min':  [  1e-6,    1e-6],
            'max':  [  0.01,     0.5]
        })

    def test_observations_missing_time(self):

        def SIRWrapper(**kwargs):
            model = SIR(**kwargs)
            results = model.sim()
            results['prevalence'] = results['per_infected']
            results['stdev'] = 1 #Junk value TODO
            return results, None

        #Observations is missing `time`
        del self.observations['time']

        self.assertRaises(HistoryMatchingError,
          hm2.boilerplate.run_replicates,
          param_sets = hm2.sampling.latin_hypercube(self.param_info, 100),
          wrapped_model = SIRWrapper,
          replicates=1,
          processes=1
        )

    def test_observations_missing_observation_id(self):
        def SIRWrapper(**kwargs):
              model = SIR(**kwargs)
              results = model.sim()
              results['prevalence'] = results['per_infected']
              results['stdev'] = 1 #Junk value TODO
              return results, None

        #Observations is missing `observation_id`
        del self.observations['observation_id']

        self.assertRaises(HistoryMatchingError,
          hm2.boilerplate.run_replicates,
          param_sets = hm2.sampling.latin_hypercube(self.param_info, 100),
          wrapped_model=SIRWrapper,
          replicates=1,
          processes=1
        )

    def test_model_results_missing_time(self):
        def SIRWrapper(**kwargs):
              model = SIR(**kwargs) #Runs model but doesn't return time
              results = model.sim()
              results['prevalence'] = results['per_infected']
              results['stdev'] = 1 #Junk value TODO
              del results['time']
              return results, None

        self.assertRaises(HistoryMatchingError,
          hm2.boilerplate.run_replicates,
          param_sets = hm2.sampling.latin_hypercube(self.param_info, 100),
          wrapped_model=SIRWrapper,
          replicates=1,
          processes=1
        )

    def test_model_results_missing_observation_name(self):
        def SIRWrapper(**kwargs):
              model = SIR(**kwargs)
              results = model.sim()
              # Oh no, we forgot to include prevalence in the results!
              # results['prevalence'] = results['per_infected']
              results['stdev'] = 1 #Junk value TODO
              return results, None

        self.assertRaises(HistoryMatchingError,
          hm2.boilerplate.run_replicates,
          param_sets = hm2.sampling.latin_hypercube(self.param_info, 100),
          wrapped_model = SIRWrapper,
          replicates = 1,
          processes = 1
        )

    def test_time_analysis_returns_correct_columns(self):
        def SIRWrapper(**kwargs):
              model = SIR(**kwargs)
              results = model.sim()
              results['prevalence'] = results['per_infected']
              results = pd.melt(results, id_vars='time', var_name='observation')
              results['stdev'] = 1 #Junk value TODO
              results['observation_id'] = list(range(len(results)))
              return results, None

        # TODO
        # tp_results, su_results = hm2.boilerplate.run_replicates(
        #   param_sets = hm2.sampling.latin_hypercube(self.param_info, 100),
        #   wrapped_model = SIRWrapper,
        #   replicates=1,
        #   processes=1
        # )

        # self.assertEqual(tp_results.columns.tolist(), ['sample_id', 'replicate', 'observation_id', 'time', 'prevalence', 'stdev'])

    def test_bad_wrap(self):
        def SIRWrapper(**kwargs):
              model = SIR(**kwargs)
              results = model.sim()
              results['prevalence'] = results['per_infected']
              results['stdev'] = 1 #Junk value TODO
              return results, None

        parameter_samples = hm2.sampling.latin_hypercube(self.param_info, 100)

        self.assertRaises(HistoryMatchingError, hm2.boilerplate.run_replicates,
          wrapped_model = SIRWrapper,
          param_sets = parameter_samples,
          replicates = 1,
          processes = 1
        )

    def test_check_time_and_summary_frames(self):
        def SIRWrapper(**kwargs):
              model = SIR(**kwargs)
              results = model.sim()
              results['prevalence'] = results['per_infected']
              results['stdev'] = 1 #Junk value TODO
              return results  #Note that this is returning only one DataFrame

        parameter_samples = hm2.sampling.latin_hypercube(self.param_info, 100)

        self.assertRaises(HistoryMatchingError, hm2.boilerplate.run_replicates,
          wrapped_model = SIRWrapper,
          param_sets = parameter_samples,
          replicates = 1,
          processes = 1
        )

    def test_returns_tuple(self):
        def SIRWrapper(**kwargs):
              model = SIR(**kwargs)
              results = model.sim()
              results['prevalence'] = results['per_infected']
              results['stdev'] = 1 #Junk value TODO
              return [results, None]  #Note that this is returning a list

        parameter_samples = hm2.sampling.latin_hypercube(self.param_info, 100)

        self.assertRaises(HistoryMatchingError, hm2.boilerplate.run_replicates,
          param_sets = parameter_samples,
          wrapped_model = SIRWrapper,
          replicates=1,
          processes=1
        )