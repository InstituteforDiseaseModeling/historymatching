# one step of history matching (see architecture diagram)
"""Version 2 of History Matching 2022

Description of this module.

Example:
    Or ``Examples``.

Attributes:

Todo:

"""

import logging
from typing import Dict

import pandas as pd

from history_matching.emulators import BaseEmulator

from .situation import Situation
from .recipe import Recipe
from .config import Config

logger = logging.getLogger()


def do_step(situation: Situation, recipe: Recipe, config: Config):

    logger.info(f"Starting step {situation.iteration}...")

    situation.validate()

    recipe.start_step_callback(situation)

    test_points = get_test_points_for_iteration(situation.iteration, situation.sample_points)

    test_results = recipe.run_simulators(situation.iteration, test_points, config)

    merge_results(situation.iteration, test_results, situation, config)

    selected_features = recipe.select_features(
        situation.iteration, situation.observations, situation.simulator_results, config
    )

    new_emulators = recipe.generate_emulators(
        situation.iteration,
        selected_features,
        situation.observations,
        situation.simulator_results,
        recipe.generate_emulator_for_feature,
        config
    )

    deposit_emulators(situation.iteration, new_emulators, situation, config)

    (next_sample_points, non_implausible_fraction) = recipe.generate_next_sample_points(
        situation.iteration,
        situation.parameter_space,
        situation.observations,
        situation.emulator_bank,
        config
    )
    logger.info(f"Remaining non-implausible space: {non_implausible_fraction*100:0.04}%")

    update_test_points(situation.iteration, next_sample_points, situation)

    recipe.end_step_callback(situation)

    logger.info(f"Finished step {situation.iteration}...")

    return recipe.exit_predicate(situation.iteration, non_implausible_fraction, config)


def get_test_points_for_iteration(
    iteration: int, sample_points: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        f"getting test points for iteration {iteration} from sample points dataframe"
    )
    test_points = sample_points[sample_points.iteration == iteration].copy()

    return test_points


def merge_results(
    iteration: int, test_results: pd.DataFrame, situation: Situation, config: Config
) -> None:

    logger.info(
        f"Merging {len(test_results)} new simulator results with {len(situation.simulator_results)} existing results..."
    )
    assert all(
        test_results.iteration == iteration
    ), "Test results include results from a different iteration."
    situation.simulator_results = pd.concat([situation.simulator_results, test_results])
    situation.simulator_results.reset_index(drop=True)

    return


def deposit_emulators(
    iteration: int, new_emulators: Dict[str, BaseEmulator], situation: Situation, config: Config
) -> None:

    logger.info(
        f"Adding {len(new_emulators.keys())} emulator(s) to emulator_bank on step {iteration}..."
    )
    situation.emulator_bank.update({iteration: new_emulators})

    return


def update_test_points(
    iteration: int, next_sample_points: pd.DataFrame, situation: Situation
) -> None:

    logger.info(
        f"Adding {len(next_sample_points)} new sample points on step {iteration}..."
    )
    next_sample_points["iteration"] = iteration + 1
    situation.sample_points = pd.concat([situation.sample_points, next_sample_points]).reset_index(drop=True)

    return


def do_staircase(situation: Situation, recipe: Recipe, config: Config) -> None:

    # do_step() returns results of exit_predicate()
    # exit_predicate return True when it's time to quit
    while not do_step(situation, recipe, config):
        situation.iteration += 1

    return
