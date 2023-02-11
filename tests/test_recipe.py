#! /usr/bin/env python3

from typing import Dict, List, Tuple
import unittest

import pandas as pd

from history_matching import \
    BaseEmulator, \
    Config, \
    Recipe, \
    Situation, \
    do_step, \
    features_from_observations

class RecipeTests(unittest.TestCase):

    def test_recipe_order(self):

        parameter_space = pd.DataFrame(data=[["k", 0.0, 1.0]], columns=["parameter", "minimum", "maximum"])
        observations = pd.DataFrame(data=[["feature", 13.0, 1.0]], columns=["features", "means", "variances"])
        initial_sample_points = pd.DataFrame(data=[[0, 0.0], [0, 0.25], [0, 0.5], [0, 0.75], [0, 1.0]], columns=["iteration", "k"])

        messages = []

        def start_step(situation: Situation) -> None:

            messages.append("Start Step")
            self.assertTrue(isinstance(situation, Situation))

            return

        def run_simulators(iteration: int, test_points: pd.DataFrame, config: Config) -> pd.DataFrame:

            messages.append("Run Simulators")
            self.assertTrue(isinstance(iteration, int))
            self.assertTrue(isinstance(test_points, pd.DataFrame))
            self.assertTrue(isinstance(config, Config))

            columns = list(test_points.columns)
            columns.extend(features_from_observations(observations))

            results = []
            for row in test_points.itertuples():
                result = list(row[1:])    # iteration and parameter(s)
                while len(result) < len(columns):
                    result.append(0.0)
                results.append(result)

            results = pd.DataFrame(data=results, columns=columns)

            return results

        def select_features(iteration: int, observations: pd.DataFrame, simulator_results: pd.DataFrame, config: Config) -> List[str]:

            messages.append("Select Features")
            self.assertTrue(isinstance(iteration, int))
            self.assertTrue(isinstance(observations, pd.DataFrame))
            self.assertTrue(isinstance(simulator_results, pd.DataFrame))
            self.assertTrue(isinstance(config, Config))

            return features_from_observations(observations)

        def generate_emulators(iteration: int, selected_features: List[str], observations: pd.DataFrame, simulator_results: pd.DataFrame, generate_emulator_for_feature, config: Config) -> Dict[str, BaseEmulator]:

            messages.append("Generate Emulators")
            self.assertTrue(isinstance(iteration, int))
            self.assertTrue(isinstance(selected_features, list))
            self.assertTrue(all(map(lambda e: isinstance(e, str), selected_features)))
            self.assertTrue(isinstance(observations, pd.DataFrame))
            self.assertTrue(isinstance(simulator_results, pd.DataFrame))
            # TODO self.assertTrue(isinstance(generate_emulator_for_feature, TBD))
            self.assertTrue(isinstance(config, Config))

            return {}

        def next_point_generation(iteration: int, parameter_space: pd.DataFrame, observations: pd.DataFrame, emulator_bank: Dict[int, Dict[str, BaseEmulator]], config: Config) -> Tuple[pd.DataFrame, float]:

            messages.append("Next Point Generation")
            self.assertTrue(isinstance(iteration, int))
            self.assertTrue(isinstance(parameter_space, pd.DataFrame))
            self.assertTrue(isinstance(observations, pd.DataFrame))
            self.assertTrue(isinstance(emulator_bank, Dict))
            self.assertTrue(all(map(lambda k: isinstance(k, int), emulator_bank.keys())))
            self.assertTrue(all(map(lambda v: isinstance(v, dict), emulator_bank.values())))
            self.assertTrue(isinstance(config, Config))

            columns = ["iteration"]
            columns.extend(parameter_space.parameter)
            next_points = pd.DataFrame(columns=columns)

            return next_points, 1.0

        def end_step(situation: Situation) -> None:

            messages.append("End Step")
            self.assertTrue(isinstance(situation, Situation))

            return

        def predicate(iteration: int, non_implausible_fraction: float, config: Config) -> bool:

            messages.append("Exit Predicate")
            self.assertTrue(isinstance(iteration, int))
            self.assertTrue(isinstance(non_implausible_fraction, float))
            self.assertTrue(isinstance(config, Config))

            return True

        recipe = Recipe()

        recipe.start_step_callback         = start_step
        recipe.run_simulators              = run_simulators
        recipe.select_features             = select_features
        recipe.generate_emulators          = generate_emulators
        recipe.generate_next_sample_points = next_point_generation
        recipe.end_step_callback           = end_step
        recipe.exit_predicate              = predicate

        situation = Situation(parameter_space, observations, initial_sample_points)
        config = Config(max_iterations=42, implausibility_threshold=0.125, non_implausible_target=0.95)

        do_step(situation, recipe, config)

        self.assertEqual(messages[0], "Start Step")
        self.assertEqual(messages[1], "Run Simulators")
        self.assertEqual(messages[2], "Select Features")
        self.assertEqual(messages[3], "Generate Emulators")
        self.assertEqual(messages[4], "Next Point Generation")
        self.assertEqual(messages[5], "End Step")
        self.assertEqual(messages[6], "Exit Predicate")

        return

    def test_writeonly_properties(self):

        recipe = Recipe()

        with self.assertRaises(RuntimeError):
            recipe.default_feature_selection = lambda _: None

        with self.assertRaises(RuntimeError):
            recipe.default_emulator_generator = lambda _: None

        with self.assertRaises(RuntimeError):
            recipe.default_next_point_generator = lambda _: None

        with self.assertRaises(RuntimeError):
            recipe.default_exit_predicate = lambda _: None

        return


if __name__ == "__main__":
    unittest.main()
