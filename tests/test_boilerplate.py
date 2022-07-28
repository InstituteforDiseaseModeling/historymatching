import numpy as np
import pandas as pd
import unittest

from hm2.data_validation import ValidateSimFrame
from hm2.error import *
from hm2.models.sir import SIR
import hm2.boilerplate as bp
import hm2.sampling

TEST_MULTIPROCESS = False


def SIRWrapperForParallel(**kwargs):
    """Used for testing parallelism of run_replicates. multiprocessing cannot
    use class methods or lambdas."""
    model = SIR(**kwargs)
    return model.run()


class BoilerplateTest(unittest.TestCase):
    def setUp(self):
        self.observations = pd.DataFrame(
            {
                "observation_id": [0, 1],
                "time": [3.0, 15.0],
                "observation": ["prevalence", "prevalence"],
                "value": [15, 40],
                "stdev": [4, 2.3],
            }
        )

        self.param_info = pd.DataFrame(
            {"name": ["beta", "gamma"], "min": [1e-6, 1e-6], "max": [0.01, 0.5]}
        )

    def test_run_replicates_bad_processes(self):
        self.assertRaises(
            TypeError,
            hm2.boilerplate.run_replicates,
            param_sets=hm2.sampling.latin_hypercube(self.param_info, 100),
            wrapped_model=SIRWrapperForParallel,
            replicates=1,
            processes="hi",
        )
        self.assertRaises(
            ValueError,
            hm2.boilerplate.run_replicates,
            param_sets=hm2.sampling.latin_hypercube(self.param_info, 100),
            wrapped_model=SIRWrapperForParallel,
            replicates=1,
            processes=-1,
        )
        self.assertRaises(
            ValueError,
            hm2.boilerplate.run_replicates,
            param_sets=hm2.sampling.latin_hypercube(self.param_info, 100),
            wrapped_model=SIRWrapperForParallel,
            replicates=1,
            processes=0,
        )

    def test_single_threaded(self):
        replicates = 2
        parameter_samples = hm2.sampling.latin_hypercube(self.param_info, 20)

        results = hm2.boilerplate.run_replicates(
            param_sets=parameter_samples,
            wrapped_model=SIRWrapperForParallel,
            replicates=replicates,
            processes=1,
        )

        self.assertTrue(len(results) == replicates * len(parameter_samples))
        for x in results:
            ValidateSimFrame(x)
        # TODO: Check results

    def test_multiprocessing(self):
        if not TEST_MULTIPROCESS:
            return

        replicates = 2
        parameter_samples = hm2.sampling.latin_hypercube(self.param_info, 20)

        results = hm2.boilerplate.run_replicates(
            param_sets=parameter_samples,
            wrapped_model=SIRWrapperForParallel,
            replicates=replicates,
            processes=None,
        )

        self.assertTrue(len(results) == replicates * len(parameter_samples))
        for x in results:
            ValidateSimFrame(x)
        # TODO: Check results

    def test_sim_results_missing_observation_id(self):
        def SIRWrapper(**kwargs):
            model = SIR(**kwargs)
            results = model.run()
            del results["observation_id"]
            return results

        with self.assertRaises(HMMissingColumn) as cm:
            hm2.boilerplate.run_replicates(
                param_sets=hm2.sampling.latin_hypercube(self.param_info, 100),
                wrapped_model=SIRWrapper,
                replicates=1,
                processes=1,
            )
            self.assertTrue(cm.exception.missing_column == "observation_id")
            self.assertTrue(cm.exception.df_name == "ObservationsFrame")

    def test_model_results_missing_time(self):
        def SIRWrapper(**kwargs):
            model = SIR(**kwargs)  # Runs model but doesn't return time
            results = model.run()
            del results["time"]
            return results

        with self.assertRaises(HMMissingColumn) as cm:
            hm2.boilerplate.run_replicates(
                param_sets=hm2.sampling.latin_hypercube(self.param_info, 100),
                wrapped_model=SIRWrapper,
                replicates=1,
                processes=1,
            )
        self.assertTrue(cm.exception.missing_column == "time")
        self.assertTrue(cm.exception.df_name == "SimObservationsFrame")

    def test_time_analysis_returns_correct_columns(self):
        def SIRWrapper(**kwargs):
            model = SIR(**kwargs)
            results = model.run()
            results["prevalence"] = results["per_infected"]
            results = pd.melt(results, id_vars="time", var_name="observation")
            results["stdev"] = 1  # Junk value TODO
            results["observation_id"] = list(range(len(results)))
            return results


class TestMatches(unittest.TestCase):
    def setUp(self):
        param_info = pd.DataFrame(
            {"name": ["beta", "gamma"], "min": [1e-6, 1e-6], "max": [0.01, 0.5]}
        )

        parameter_samples = hm2.sampling.latin_hypercube(param_info, 10)

        self.sim_outputs = hm2.boilerplate.run_replicates(
            param_sets=parameter_samples,
            wrapped_model=SIRWrapperForParallel,
            replicates=2,
            processes=1,
        )

        self.real_observations = pd.DataFrame(
            {
                "observation_id": [0, 1],
                "time": [3.0, 15.0],
                "observation": ["prevalence", "prevalence"],
                "value": [15, 40],
                "stdev": [4, 2.3],
            }
        )

    def test_inputs(self):
        self.assertRaises(
            TypeError,
            hm2.boilerplate.match_sim_outputs_to_observations,
            "not a list",
            self.real_observations,
            processes=1,
        )
        self.assertRaises(
            HMNotADataFrame,
            hm2.boilerplate.match_sim_outputs_to_observations,
            ["not a list of SimFrame"],
            self.real_observations,
            processes=1,
        )
        self.assertRaises(
            TypeError,
            hm2.boilerplate.match_sim_outputs_to_observations,
            self.sim_outputs,
            self.real_observations,
            processes="hi",
        )
        self.assertRaises(
            ValueError,
            hm2.boilerplate.match_sim_outputs_to_observations,
            self.sim_outputs,
            self.real_observations,
            processes=-1,
        )
        self.assertRaises(
            ValueError,
            hm2.boilerplate.match_sim_outputs_to_observations,
            self.sim_outputs,
            self.real_observations,
            processes=0,
        )

    def test_matching_single_threaded(self):
        matched = bp.match_sim_outputs_to_observations(
            self.sim_outputs, self.real_observations, processes=1
        )
        # TODO: Check results

    def test_matching_multi_threaded(self):
        if not TEST_MULTIPROCESS:
            return
        matched = bp.match_sim_outputs_to_observations(
            self.sim_outputs, self.real_observations, processes=None
        )
        # TODO: Check results


class TestImplausibility(unittest.TestCase):
    def test_implausibility_equation_zero(self):
        # Note that reality matches prediction
        reality = 3
        reality_stdev = 1
        prediction = 3
        prediction_stdev = 1
        model_stdev = 1
        self.assertTrue(
            bp._implausibility_equ(
                reality, reality_stdev, prediction, prediction_stdev, model_stdev
            )
            == 0
        )

    def test_implausibility_equation_nonzero(self):
        reality = 1
        reality_stdev = 2
        prediction = 3
        prediction_stdev = 2
        model_stdev = 2
        self.assertAlmostEqual(
            bp._implausibility_equ(
                reality, reality_stdev, prediction, prediction_stdev, model_stdev
            ),
            0.57735026918962576451,
        )

    def test_filter_implausibility(self):
        implausibilities = pd.DataFrame(
            {"implausibility": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        )
        filtered = bp.filter_implausibilities(implausibilities, threshold=5)
        self.assertTrue(filtered["implausibility"].max() == 5)


class TestGetImplausibility(unittest.TestCase):
    def setUp(self):
        self.observations = pd.DataFrame(
            {
                "observation_id": [0, 1],
                "time": [3.0, 15.0],
                "observation": ["prevalence", "prevalence"],
                "value": [15, 40],
                "stdev": [4, 2.3],
            }
        )

        param_info = pd.DataFrame(
            {"name": ["beta", "gamma"], "min": [1e-6, 1e-6], "max": [0.01, 0.5]}
        )

        self.parameter_samples = hm2.sampling.latin_hypercube(param_info, 10)

    def test_not_an_emulator(self):
        self.assertRaises(
            HMNotAnEmulator,
            bp.get_implausibility,
            {1: "not an emulator"},
            self.parameter_samples,
            self.observations,
        )


if __name__ == "__main__":
    unittest.main()
