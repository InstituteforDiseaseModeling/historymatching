import logging
from typing import Dict

import numpy as np
import pandas as pd

from .utils import features_from_observations

from history_matching.emulators import BaseEmulator

logger = logging.getLogger()


# https://en.wikipedia.org/wiki/Michael_Sorrentino
class Situation:
    def __init__(
        self,
        parameter_space: pd.DataFrame,
        observations: pd.DataFrame,
        initial_sample_points: pd.DataFrame,
        iteration: int = 0,
    ) -> None:

        logger.info("Creating Situation object")
        self.iteration = iteration
        self.parameter_space = parameter_space
        self.sample_points = initial_sample_points
        columns = [
            "iteration",
            "replicate",
        ]  # "iteration" isn't strictly necessary, but might assist in debugging
        columns.extend(parameter_space.parameter)
        features = features_from_observations(observations)
        columns.extend(features)
        self.simulator_results = pd.DataFrame(columns=columns)
        self.observations = observations
        self.emulator_bank = {}

        return

    def validate(self) -> None:

        Situation.validate_iteration(self.iteration)
        Situation.validate_parameter_space(self.parameter_space)
        Situation.validate_sample_points(self.sample_points, self.parameter_space)
        Situation.validate_observations(self.observations)
        Situation.validate_simulator_results(
            self.simulator_results, self.parameter_space, self.observations
        )
        Situation.validate_emulator_bank(self.emulator_bank, self.observations)

        return

    @staticmethod
    def validate_iteration(iteration: int) -> None:

        if not isinstance(iteration, (int, float, np.number)):
            raise TypeError(
                f"Situation iteration, {iteration}, should be numeric, not '{type(iteration)}'"
            )
        if int(iteration) != iteration:
            raise ValueError(
                f"Situation iteration should be an integer value, not {iteration}"
            )
        if iteration < 0:
            raise ValueError(f"Situation iteration should be >= 0, not {iteration}")

        return

    @staticmethod
    def validate_parameter_space(parameter_space: pd.DataFrame) -> None:

        if not isinstance(parameter_space, pd.DataFrame):
            raise TypeError(
                f"Situation parameter space should be Pandas DataFrame, not '{type(parameter_space)}'"
            )
        if not all(
            [
                column in parameter_space.columns
                for column in ["parameter", "minimum", "maximum"]
            ]
        ):
            raise RuntimeError(
                f"Situation parameter space must contain the columns 'parameter', 'minimum', 'maximum'. Found {parameter_space.columns}."
            )
        if len(parameter_space) == 0:
            raise RuntimeError(
                "Situation parameter space must specify at least one parameter. Found none."
            )
        ordered = True
        msg = ""
        for row in parameter_space.itertuples():
            if row.minimum > row.maximum:
                msg += f"Parameter '{row.parameter}' minimum ({row.minimum}) > maximum ({row.maximum}).\n"
                ordered = False
        if not ordered:
            raise ValueError(msg)

        return

    @staticmethod
    def validate_sample_points(
        sample_points: pd.DataFrame, parameter_space: pd.DataFrame
    ) -> None:

        if not isinstance(sample_points, pd.DataFrame):
            raise TypeError(
                f"Situation sample points should be Pandas DataFrame, not '{type(sample_points)}'"
            )
        required_columns = ["iteration"]
        required_columns.extend(parameter_space.parameter)
        if not all([column in sample_points.columns for column in required_columns]):
            raise RuntimeError(
                f"Situation sample points must contain the columns {required_columns}. Found {sample_points.columns}."
            )
        if len(sample_points) == 0:
            raise RuntimeError(
                "Situation sample points must specify at least one point in parameter space. Found none."
            )
        valid = True
        msg = ""
        for irow in range(len(sample_points)):
            row = sample_points.iloc[irow]
            for parameter_spec in parameter_space.itertuples():
                if (row[parameter_spec.parameter] < parameter_spec.minimum) or (
                    row[parameter_spec.parameter] > parameter_spec.maximum
                ):
                    valid = False
                    msg += f"Sample parameter, {row}, is outside parameter space."
        if not valid:
            raise ValueError(msg)

        return

    @staticmethod
    def validate_observations(observations: pd.DataFrame) -> None:

        if not isinstance(observations, pd.DataFrame):
            raise TypeError(
                f"Situation observations should be Pandas DataFrame, not '{type(observations)}'"
            )

        if not "statistic" in observations.columns:
            raise RuntimeError(f"Situation observations must have 'statistic' column. Found {observations.columns}")

        if len(observations.columns) <= 1:
            raise RuntimeError(
                f"Situation observations must have at least one feature column. Found {observations.columns}"
            )

        if len(observations[observations.statistic == "mean"]) != 1:
            raise RuntimeError(f"Situation observations must have one row for means. Found {observations.statistic}")

        if len(observations[observations.statistic == "variance"]) != 1:
            raise RuntimeError(f"Situation observations must have one row for variances. Found {observations.statistic}")

        return

    @staticmethod
    def validate_simulator_results(
        simulator_results: pd.DataFrame,
        parameter_space: pd.DataFrame,
        observations: pd.DataFrame,
    ) -> None:

        if not isinstance(simulator_results, pd.DataFrame):
            raise TypeError(
                f"Situation simulator results should be Pandas DataFrame, not '{type(simulator_results)}'"
            )
        required_columns = ["replicate"]
        required_columns.extend(parameter_space.parameter)
        features = features_from_observations(observations)
        required_columns.extend(features)
        if not all(
            [column in simulator_results.columns for column in required_columns]
        ):
            raise RuntimeError(
                f"Simulator results must contain the columns {required_columns}. Found {simulator_results.columns}"
            )

        return

    @staticmethod
    def validate_emulator_bank(
        emulator_bank: Dict[int, Dict[str, BaseEmulator]], observations: pd.DataFrame
    ) -> None:

        # emulator_bank must be a **dictionary** mapping int:dict
        if not isinstance(emulator_bank, dict):
            raise TypeError(
                f"Situation enumlator bank should be dictionary, not '{type(emulator_bank)}'"
            )

        # emulator_bank must be a dictionary mapping **int**:dict
        if not all(map(lambda k: isinstance(k, (int, np.number)), emulator_bank.keys())):
            raise TypeError(f"Situation emulator bank keys should be integer values. Found {list(emulator_bank.keys())}")

        # emulator_bank must be a dictionary mapping int:**dict**
        if not all(map(lambda v: isinstance(v, dict), emulator_bank.values())):
            raise TypeError(f"Situation emulator bank values must be dictionaries. Found {list(map(lambda v: type(v), emulator_bank.values()))}")

        # keys must be >= 0
        if not all(map(lambda k: k >= 0, emulator_bank.keys())):
            raise ValueError(f"Situation emulator bank keys must be >= 0. Found {list(emulator_bank.keys())}")

        # Values must be dictionaries mapping str:BaseEmulator
        for iteration, emulators in emulator_bank.items():

            # Values must be dictionaries mapping **str**:BaseEmulator
            if not all(map(lambda k: isinstance(k, str), emulators.keys())):
                raise TypeError(f"Situation emulator bank values should be dictionaries with string keys. Found {list(map(lambda k: type(k), emulators.keys()))}")

            # Values must be dictionaries mapping str:**BaseEmulator**
            if not all(map(lambda v: isinstance(v, BaseEmulator), emulators.values())):
                raise TypeError(f"Situation emulator bank values should be dictionaries mapping features to emulators. Found {list(map(lambda v: type(v), emulators.values()))}")

            # Keys must be features from observations
            if not all([key in observations.columns for key in emulators.keys()]):
                raise ValueError(
                    f"Found 'feature' in emulators dictionary ({emulators.keys()}, iteration {iteration}) which does not map to observation features ({observations.columns})."
                )

        return
