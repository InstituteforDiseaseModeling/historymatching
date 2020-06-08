import numpy as np
import pandas as pd
import unittest

from hm2.data_validation import ValidateSimFrame
from hm2.error import HistoryMatchingError
from hm2.models.sir import SIR
import hm2.boilerplate as bp
import hm2.sampling



def SIRWrapperForParallel(**kwargs):
    """Used for testing parallelism of run_replicates"""
    model = SIR(**kwargs)
    results = model.run()
    results['prevalence'] = results['per_infected']
    results = pd.melt(results, id_vars='time', var_name='observation')
    results['stdev'] = 1 #Junk value TODO
    results['observation_id'] = list(range(len(results)))
    return results, None



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

    def test_run_replicates_bad_processes(self):
        self.assertRaises(TypeError,
          hm2.boilerplate.run_replicates,
          param_sets = hm2.sampling.latin_hypercube(self.param_info, 100),
          wrapped_model = SIRWrapperForParallel,
          replicates=1,
          processes="hi"
        )
        self.assertRaises(ValueError,
          hm2.boilerplate.run_replicates,
          param_sets = hm2.sampling.latin_hypercube(self.param_info, 100),
          wrapped_model = SIRWrapperForParallel,
          replicates=1,
          processes=-1
        )
        self.assertRaises(ValueError,
          hm2.boilerplate.run_replicates,
          param_sets = hm2.sampling.latin_hypercube(self.param_info, 100),
          wrapped_model = SIRWrapperForParallel,
          replicates=1,
          processes=0
        )

    def test_single_threaded(self):
        replicates = 2
        parameter_samples = hm2.sampling.latin_hypercube(self.param_info, 20)

        results = hm2.boilerplate.run_replicates(
          param_sets = parameter_samples,
          wrapped_model = SIRWrapperForParallel,
          replicates=replicates,
          processes=1
        )

        self.assertTrue(len(results)==replicates*len(parameter_samples))
        for x in results:
          ValidateSimFrame(x)
        #TODO: Check results

    def test_multiprocessing(self):
        replicates = 2
        parameter_samples = hm2.sampling.latin_hypercube(self.param_info, 20)

        results = hm2.boilerplate.run_replicates(
          param_sets = parameter_samples,
          wrapped_model = SIRWrapperForParallel,
          replicates=replicates,
          processes=None
        )

        self.assertTrue(len(results)==replicates*len(parameter_samples))
        for x in results:
          ValidateSimFrame(x)
        #TODO: Check results

    def test_observations_missing_time(self):

        def SIRWrapper(**kwargs):
            model = SIR(**kwargs)
            results = model.run()
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
              results = model.run()
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
              results = model.run()
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
              results = model.run()
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
              results = model.run()
              results['prevalence'] = results['per_infected']
              results = pd.melt(results, id_vars='time', var_name='observation')
              results['stdev'] = 1 #Junk value TODO
              results['observation_id'] = list(range(len(results)))
              return results, None

    def test_bad_wrap(self):
        def SIRWrapper(**kwargs):
              model = SIR(**kwargs)
              results = model.run()
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
              results = model.run()
              results['prevalence'] = results['per_infected']
              results['stdev'] = 1 #Junk value TODO
              return results  #Note that this is returning only one DataFrame

        parameter_samples = hm2.sampling.latin_hypercube(self.param_info, 100)

        self.assertRaises(TypeError, hm2.boilerplate.run_replicates,
          wrapped_model = SIRWrapper,
          param_sets = parameter_samples,
          replicates = 1,
          processes = 1
        )

    def test_returns_tuple(self):
        def SIRWrapper(**kwargs):
              model = SIR(**kwargs)
              results = model.run()
              results['prevalence'] = results['per_infected']
              results['stdev'] = 1 #Junk value TODO
              return [results, None]  #Note that this is returning a list

        parameter_samples = hm2.sampling.latin_hypercube(self.param_info, 100)

        self.assertRaises(TypeError, hm2.boilerplate.run_replicates,
          param_sets = parameter_samples,
          wrapped_model = SIRWrapper,
          replicates=1,
          processes=1
        )

class TestMatches(unittest.TestCase):
    def setUp(self):
        param_info = pd.DataFrame({
            'name': ['beta', 'gamma'],
            'min':  [  1e-6,    1e-6],
            'max':  [  0.01,     0.5]
        })

        parameter_samples = hm2.sampling.latin_hypercube(param_info, 10)

        self.sim_outputs = hm2.boilerplate.run_replicates(
          param_sets = parameter_samples,
          wrapped_model = SIRWrapperForParallel,
          replicates=2,
          processes=None
        )

        self.time_observations = pd.DataFrame({
            'observation_id': [           0,            1],
            'time':           [         3.0,         15.0],
            'observation':    ['prevalence', 'prevalence'],
            'value':          [          15,           40],
            'stdev':          [           4,          2.3]
        })

        self.summary_observations = None


    def test_inputs(self):
        self.assertRaises(TypeError,
          hm2.boilerplate.match_sim_outputs_to_observations,
          "not a list",
          self.time_observations,
          self.summary_observations,
          processes=1
        )
        self.assertRaises(TypeError,
          hm2.boilerplate.match_sim_outputs_to_observations,
          ["not a list of tuples"],
          self.time_observations,
          self.summary_observations,
          processes=1
        )
        self.assertRaises(TypeError,
          hm2.boilerplate.match_sim_outputs_to_observations,
          self.sim_outputs,
          self.time_observations,
          self.summary_observations,
          processes="hi"
        )
        self.assertRaises(ValueError,
          hm2.boilerplate.match_sim_outputs_to_observations,
          self.sim_outputs,
          self.time_observations,
          self.summary_observations,
          processes=-1
        )
        self.assertRaises(ValueError,
          hm2.boilerplate.match_sim_outputs_to_observations,
          self.sim_outputs,
          self.time_observations,
          self.summary_observations,
          processes=0
        )

    def test_matching_single_threaded(self):
      breakpoint()
      matched = bp.match_sim_outputs_to_observations(
        self.sim_outputs,
        self.time_observations,
        self.summary_observations,
        processes=1
      )
      #TODO: Check results

    # def test_matching_multi_threaded(self):
    #   matched = bp.match_sim_outputs_to_observations(
    #     self.sim_outputs,
    #     self.time_observations,
    #     self.summary_observations,
    #     processes=None
    #   )
      #TODO: Check results


class TestImplausibility(unittest.TestCase):
  def test_implausibility_equation_zero(self):
    #Note that reality matches prediction
    reality = 3
    reality_stdev = 1
    prediction=3
    prediction_stdev=1
    model_stdev=1
    self.assertTrue(bp._implausibility_equ(reality,reality_stdev,prediction,prediction_stdev,model_stdev)==0)

  def test_implausibility_equation_nonzero(self):
    reality = 1
    reality_stdev = 2
    prediction=3
    prediction_stdev=2
    model_stdev=2
    self.assertAlmostEqual(bp._implausibility_equ(reality,reality_stdev,prediction,prediction_stdev,model_stdev),0.57735026918962576451)