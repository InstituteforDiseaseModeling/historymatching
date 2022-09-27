# one step of history matching (see architecture diagram)
"""Version 2 of History Matching 2022

Description of this module.

Example:
    Or ``Examples``.

Attributes:

Todo:

"""

import logging
from math import nan
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class Config:
    def __init__(self):
        logger.info("Creating Config object")
        return


class State:
    def __init__(self, parameter_space: pd.DataFrame, observations: pd.DataFrame, initial_sample_points: pd.DataFrame, iteration:int=0) -> None:
        logger.info("Creating State object")
        self.iteration = iteration
        self.parameter_space = parameter_space
        self.sample_points = initial_sample_points
        columns = ["iteration", "replicate"]    # "iteration" isn't strictly necessary, but might assist in debugging
        columns.extend(parameter_space.parameter)
        columns.extend(observations.columns)
        self.simulator_results = pd.DataFrame(columns=columns)
        self.observations = observations
        self.emulator_bank = {}

        return


class Emulator:
    def evaluate(parameters: Tuple) -> float:

        return nan


class Recipe:
    def __init__(self):
        self.start_step_callback = Recipe.pirates
        self.run_simulators = Recipe.null_simulator
        self.select_features = Recipe.all_features
        self.generate_emulators = Recipe._generate_emulators
        self.generate_emulator_for_feature = Recipe._generate_emulator_for_feature
        self.generate_next_sample_points = Recipe.next_point_generation
        self.end_step_callback = Recipe.pirates
        self.exit_predicate = lambda : True
        return

    @staticmethod
    def pirates(*args):  # https://www.youtube.com/watch?v=XaWU1CmrJNc
        logger.info(f"Recipe.pirates() called with {args}")
        return

    @staticmethod
    def null_simulator(iteration: int, test_points: pd.DataFrame, config: Config) -> pd.DataFrame:
        """Method description

        Args:
            iteration: current iteration index (0 based)
            test_points: dataframe of parameter names in columns, each row represents a test point in parameter space
            config: history matching configuration

        Returns:
            pd.DataFrame: simulator results for the given test points in parameter space

        |iteration|<param0>|<param1>|...|<paramN>|replicate#|<feature1>|<feature2>|...|<featureM>|
        |---------|--------|--------|---|--------|----------|----------|----------|---|----------|
        |   int   | float  | float  |...| float  |   int    | float    | float    |...| float    |

        """

        logger.info(f"Running simulator for {len(test_points)} test points...")

        return pd.DataFrame()

    @staticmethod
    def all_features(iteration: int, observations: pd.DataFrame, simulator_results: pd.DataFrame, config: Config) -> List[str]:
        """Returns _all_ features found in the observations and simulator results.
        
        Args:
            iteration: current iteration index (0 based)
            observations: dataframe with feature names in columns, and one row of target values

        |<feature1>|<feature2>|...|<featureM>|
        |----------|----------|---|----------|
        | float    | float    |...| float    |

            simulator_results: dataframe with simulator results for various test points in parameter space

        |iteration|<param0>|<param1>|...|<paramN>|replicate#|<feature1>|<feature2>|...|<featureM>|
        |---------|--------|--------|---|--------|----------|----------|----------|---|----------|
        |   int   | float  | float  |...| float  |   int    | float    | float    |...| float    |

            config: history matching configuration

        """

        logger.info(f"Selecting features for iteration {iteration}...")
        selected_features = list(observations.columns)

        return selected_features

    @staticmethod
    def _generate_emulators(iteration: int, selected_features: List[str], observations: pd.DataFrame, simulator_results: pd.DataFrame, emulator_for_feature_fn, config: Config) -> Dict[str, object]:

        logger.info(f"Generating emulators for {len(selected_features)} features ({selected_features})...")
        emulators = {}

        for feature in selected_features:

            emulators[feature] = emulator_for_feature_fn(feature, observations, simulator_results, config)

        return emulators

    @staticmethod
    def _generate_emulator_for_feature(feature: str, observations: pd.DataFrame, simulator_results: pd.DataFrame, config: Config) -> Emulator:

        logger.info(f"Generating emulator for feature '{feature}'...")
        mean = simulator_results[feature].mean()

        def emulator(*args):
            print(f"emulator{args} => {mean}")
            return mean

        return emulator

    @staticmethod
    def next_point_generation(iteration: int, parameter_space: pd.DataFrame, emulator_bank: Dict[int, Dict[str, Any]], config:Config) -> Tuple[pd.DataFrame, float]:

        logger.info("Generating next set of test points in parameter space...")

        return pd.DataFrame(), 1.0


def do_step(state: State, setup: Recipe, config: Config):

    logger.info(f"Starting step{state.iteration}...")

    validate_state(state)

    setup.start_step_callback(state)

    test_points = get_test_points_for_iteration(state.iteration, state.sample_points)

    test_results = setup.run_simulators(state.iteration, test_points, config)

    merge_results(state.iteration, test_results, state, config)

    selected_features = setup.select_features(state.iteration, state.observations, state.simulator_results, config)

    new_emulators = setup.generate_emulators(state.iteration, selected_features, state.observations, state.simulator_results, setup.generate_emulator_for_feature, config)

    deposit_emulators(state.iteration, new_emulators, state, config)

    (next_sample_points, non_implausible_fraction) = setup.generate_next_sample_points(state.iteration, state.parameter_space, state.emulator_bank, config)
    logger.info(f"Remaining non-implausible space: {non_implausible_fraction*100}")

    update_test_points(state.iteration, next_sample_points, state)

    setup.end_step_callback(state)

    logger.info(f"Finished step {state.iteration}...")

    return setup.exit_predicate(state.iteration, non_implausible_fraction)


def validate_state(state: State) -> None:

    validate_iteration(state.iteration)
    validate_parameter_space(state.parameter_space)
    validate_sample_points(state.sample_points, state.parameter_space)
    validate_observations(state.observations)
    validate_simulator_results(state.simulator_results, state.parameter_space, state.observations)
    validate_emulator_bank(state.emulator_bank, state.observations)

    return


def validate_iteration(iteration: int) -> None:

    if not isinstance(iteration, (int, float, np.number)):
        raise TypeError(f"State iteration, {iteration}, should be numeric, not '{type(iteration)}'")
    if int(iteration) != iteration:
        raise ValueError(f"State iteration should be an integer value, not {iteration}")
    if iteration < 0:
        raise ValueError(f"State iteration should be >= 0, not {iteration}")

    return


def validate_parameter_space(parameter_space: pd.DataFrame) -> None:

    if not isinstance(parameter_space, pd.DataFrame):
        raise TypeError(f"State parameter space should be Pandas DataFrame, not '{type(parameter_space)}'")
    if not all([column in parameter_space.columns for column in ["parameter", "min", "max"]]):
        raise RuntimeError(f"State parameter space must contain the columns 'parameter', 'min', 'max'. Found {parameter_space.columns}.")
    if len(parameter_space) == 0:
        raise RuntimeError("State parameter space must specify at least one parameter. Found none.")
    ordered = True
    msg = ""
    for row in parameter_space.itertuples():
        if row.min > row.max:
            msg += f"Parameter '{row.parameter}' minimum ({row.min}) > maximum ({row.max}).\n"
            ordered = False
    if not ordered:
        raise RuntimeError(msg)

    return


def validate_sample_points(sample_points: pd.DataFrame, parameter_space: pd.DataFrame) -> None:

    if not isinstance(sample_points, pd.DataFrame):
        raise TypeError(f"State sample points should be Pandas DataFrame, not '{type(sample_points)}'")
    required_columns = ["iteration"]
    required_columns.extend(parameter_space.parameter)
    if not all([column in sample_points.columns for column in required_columns]):
        raise RuntimeError(f"State sample points must contain the columns {required_columns}. Found {sample_points.columns}.")
    if len(sample_points) == 0:
        raise RuntimeError("State sample points must specify at least one point in parameter space. Found none.")
    valid = True
    msg = ""
    for irow in range(len(sample_points)):
        row = sample_points.iloc[irow]
        for parameter_spec in parameter_space.itertuples():
            if (row[parameter_spec.parameter] < parameter_spec.min) or (row[parameter_spec.parameter] > parameter_spec.max):
                valid = False
                msg += f"Sample parameter, {row}, is outside parameter space."
    if not valid:
        raise RuntimeError(msg)

    return


def validate_observations(observations: pd.DataFrame) -> None:

    if not isinstance(observations, pd.DataFrame):
        raise TypeError(f"State observations should be Pandas DataFrame, not '{type(observations)}'")
    if len(observations.columns) == 0:
        raise RuntimeError("State observations must have at least one feature (column).")
    if len(observations) != 1:
        raise RuntimeError(f"State observations must have one row of observed features. Found {len(observations)} rows.")

    return


def validate_simulator_results(simulator_results: pd.DataFrame, parameter_space: pd.DataFrame, observations: pd.DataFrame) -> None:

    if not isinstance(simulator_results, pd.DataFrame):
        raise TypeError(f"State simulator results should be Pandas DataFrame, not '{type(simulator_results)}'")
    required_columns = ["replicate"]
    required_columns.extend(parameter_space.parameter)
    required_columns.extend(observations.columns)
    if not all([column in simulator_results.columns for column in required_columns]):
        raise RuntimeError(f"Simulator results must contain the columns {required_columns}. Found {simulator_results.columns}")

    return


def validate_emulator_bank(emulator_bank: Dict[int, Dict[str, Any]], observations: pd.DataFrame) -> None:

    if not isinstance(emulator_bank, dict):
        raise TypeError(f"State enumlator bank should be dictionary, not '{type(emulator_bank)}'")
    if len(emulator_bank) > 0:
        if not all([isinstance(key, int) for key in emulator_bank.keys()]):
            raise TypeError(f"State emulator bank should map integer iterations to dictionary of features:emulators. Found non-integral iteration/key.")
        if not all([isinstance(value, dict) for value in emulator_bank.values()]):
            raise TypeError(f"State emulator bank should map integer iterations to dictionary of features:emulators.")
    for iteration, emulators in emulator_bank.items():
        if not all([key in observations.columns for key in emulators.keys()]):
            raise ValueError(f"Found 'feature' in emulators dictionary ({emulators.keys()}, iteration {iteration}) which does not map to observation features ({observations.columns}).")

    return


def get_test_points_for_iteration(iteration: int, sample_points: pd.DataFrame) -> pd.DataFrame:

    logger.info(f"getting test points for iteration {iteration} from sample points dataframe")
    test_points = sample_points[sample_points.iteration == iteration].copy()

    return test_points


def merge_results(iteration: int, test_results: pd.DataFrame, state: State, config: Config) -> None:

    logger.info(f"Merging {len(test_results)} new simulator results with {len(state.simulator_results)} existing results...")
    assert all(test_results.iteration == iteration), "Test results include results from a different iteration."
    state.simulator_results = pd.concat([state.simulator_results, test_results])
    state.simulator_results.reset_index(drop=True)

    return


def deposit_emulators(iteration: int, new_emulators: Dict[str, Emulator], state: State, config: Config) -> None:

    logger.info(f"Adding {len(new_emulators.keys())} to emulator_bank on step {iteration}...")
    state.emulator_bank.update({iteration: new_emulators})

    return

def update_test_points(iteration: int, next_sample_points: pd.DataFrame, state: State) -> None:

    logger.info(f"Adding {len(next_sample_points)} new sample points on step {iteration}...")
    state.sample_points = pd.concat([state.sample_points, next_sample_points])
    state.sample_points.reset_index(drop=True)

    return
