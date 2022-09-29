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

from .state import State
from .recipe import Recipe
from .config import Config

logger = logging.getLogger()


def do_step(state: State, recipe: Recipe, config: Config):

    logger.info(f"Starting step{state.iteration}...")

    state.validate()

    recipe.start_step_callback(state)

    test_points = get_test_points_for_iteration(state.iteration, state.sample_points)

    test_results = recipe.run_simulators(state.iteration, test_points, config)

    merge_results(state.iteration, test_results, state, config)

    selected_features = recipe.select_features(
        state.iteration, state.observations, state.simulator_results, config
    )

    new_emulators = recipe.generate_emulators(
        state.iteration,
        selected_features,
        state.observations,
        state.simulator_results,
        recipe.generate_emulator_for_feature,
        config,
    )

    deposit_emulators(state.iteration, new_emulators, state, config)

    (next_sample_points, non_implausible_fraction) = recipe.generate_next_sample_points(
        state.iteration,
        state.parameter_space,
        state.observations,
        state.emulator_bank,
        config,
    )
    logger.info(f"Remaining non-implausible space: {non_implausible_fraction*100}%")

    update_test_points(state.iteration, next_sample_points, state)

    recipe.end_step_callback(state)

    logger.info(f"Finished step {state.iteration}...")

    return recipe.exit_predicate(state.iteration, non_implausible_fraction, config.non_implausible_target, config)


def get_test_points_for_iteration(
    iteration: int, sample_points: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        f"getting test points for iteration {iteration} from sample points dataframe"
    )
    test_points = sample_points[sample_points.iteration == iteration].copy()

    return test_points


def merge_results(
    iteration: int, test_results: pd.DataFrame, state: State, config: Config
) -> None:

    logger.info(
        f"Merging {len(test_results)} new simulator results with {len(state.simulator_results)} existing results..."
    )
    assert all(
        test_results.iteration == iteration
    ), "Test results include results from a different iteration."
    state.simulator_results = pd.concat([state.simulator_results, test_results])
    state.simulator_results.reset_index(drop=True)

    return


def deposit_emulators(
    iteration: int, new_emulators: Dict[str, BaseEmulator], state: State, config: Config
) -> None:

    logger.info(
        f"Adding {len(new_emulators.keys())} to emulator_bank on step {iteration}..."
    )
    state.emulator_bank.update({iteration: new_emulators})

    return


def update_test_points(
    iteration: int, next_sample_points: pd.DataFrame, state: State
) -> None:

    logger.info(
        f"Adding {len(next_sample_points)} new sample points on step {iteration}..."
    )
    next_sample_points["iteration"] = iteration+1
    state.sample_points = pd.concat([state.sample_points, next_sample_points])
    state.sample_points.reset_index(drop=True)

    return
