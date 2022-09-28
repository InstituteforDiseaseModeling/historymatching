import logging
from typing import List, Dict, Tuple

import pandas as pd

from .config import Config

from history_matching.emulators import BaseEmulator

logger = logging.getLogger()


class Recipe:
    def __init__(self):
        self.start_step_callback = Recipe.pirates
        self.run_simulators = Recipe.null_simulator
        self.select_features = Recipe.all_features
        self.generate_emulators = Recipe._generate_emulators
        self.generate_emulator_for_feature = Recipe._generate_emulator_for_feature
        self.generate_next_sample_points = Recipe.next_point_generation
        self.end_step_callback = Recipe.pirates
        self.exit_predicate = lambda iter, frac, cfg: True
        return

    @staticmethod
    def pirates(*args):  # https://www.youtube.com/watch?v=XaWU1CmrJNc
        logger.info(f"Recipe.pirates() called with {args}")
        return

    @staticmethod
    def null_simulator(
        iteration: int, test_points: pd.DataFrame, config: Config
    ) -> pd.DataFrame:
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
    def all_features(
        iteration: int,
        observations: pd.DataFrame,
        simulator_results: pd.DataFrame,
        config: Config,
    ) -> List[str]:
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
    def _generate_emulators(
        iteration: int,
        selected_features: List[str],
        observations: pd.DataFrame,
        simulator_results: pd.DataFrame,
        emulator_for_feature_fn,
        config: Config,
    ) -> Dict[str, object]:

        logger.info(
            f"Generating emulators for {len(selected_features)} features ({selected_features})..."
        )
        emulators = {}

        for feature in selected_features:

            emulators[feature] = emulator_for_feature_fn(
                feature, observations, simulator_results, config
            )

        return emulators

    @staticmethod
    def _generate_emulator_for_feature(
        feature: str,
        observations: pd.DataFrame,
        simulator_results: pd.DataFrame,
        config: Config,
    ) -> BaseEmulator:

        logger.info(f"Generating emulator for feature '{feature}'...")
        mean = simulator_results[feature].mean()

        # TODO - input argument should be a pd.DataFrame of points in parameter space
        def emulator(*args):
            print(f"emulator{args} => {mean}")
            return mean

        return emulator

    @staticmethod
    def next_point_generation(
        iteration: int,
        parameter_space: pd.DataFrame,
        observations: pd.DataFrame,
        emulator_bank: Dict[int, Dict[str, BaseEmulator]],
        config: Config,
    ) -> Tuple[pd.DataFrame, float]:

        logger.info("Generating next set of test points in parameter space...")

        return pd.DataFrame(), 1.0
