import logging
from typing import Dict

import numpy as np
import pandas as pd

from history_matching.emulators import BaseEmulator

logger = logging.getLogger()


class State:
    def __init__(
        self,
        parameter_space: pd.DataFrame,
        observations: pd.DataFrame,
        initial_sample_points: pd.DataFrame,
        iteration: int = 0,
    ) -> None:

        logger.info("Creating State object")
        self.iteration = iteration
        self.parameter_space = parameter_space
        self.sample_points = initial_sample_points
        columns = [
            "iteration",
            "replicate",
        ]  # "iteration" isn't strictly necessary, but might assist in debugging
        columns.extend(parameter_space.parameter)
        columns.extend(observations.columns)
        self.simulator_results = pd.DataFrame(columns=columns)
        self.observations = observations
        self.emulator_bank = {}

        return

    def validate(self) -> None:

        State.validate_iteration(self.iteration)
        State.validate_parameter_space(self.parameter_space)
        State.validate_sample_points(self.sample_points, self.parameter_space)
        State.validate_observations(self.observations)
        State.validate_simulator_results(
            self.simulator_results, self.parameter_space, self.observations
        )
        State.validate_emulator_bank(self.emulator_bank, self.observations)

        return

    @staticmethod
    def validate_iteration(iteration: int) -> None:

        if not isinstance(iteration, (int, float, np.number)):
            raise TypeError(
                f"State iteration, {iteration}, should be numeric, not '{type(iteration)}'"
            )
        if int(iteration) != iteration:
            raise ValueError(
                f"State iteration should be an integer value, not {iteration}"
            )
        if iteration < 0:
            raise ValueError(f"State iteration should be >= 0, not {iteration}")

        return

    @staticmethod
    def validate_parameter_space(parameter_space: pd.DataFrame) -> None:

        if not isinstance(parameter_space, pd.DataFrame):
            raise TypeError(
                f"State parameter space should be Pandas DataFrame, not '{type(parameter_space)}'"
            )
        if not all(
            [
                column in parameter_space.columns
                for column in ["parameter", "min", "max"]
            ]
        ):
            raise RuntimeError(
                f"State parameter space must contain the columns 'parameter', 'min', 'max'. Found {parameter_space.columns}."
            )
        if len(parameter_space) == 0:
            raise RuntimeError(
                "State parameter space must specify at least one parameter. Found none."
            )
        ordered = True
        msg = ""
        for row in parameter_space.itertuples():
            if row.min > row.max:
                msg += f"Parameter '{row.parameter}' minimum ({row.min}) > maximum ({row.max}).\n"
                ordered = False
        if not ordered:
            raise RuntimeError(msg)

        return

    @staticmethod
    def validate_sample_points(
        sample_points: pd.DataFrame, parameter_space: pd.DataFrame
    ) -> None:

        if not isinstance(sample_points, pd.DataFrame):
            raise TypeError(
                f"State sample points should be Pandas DataFrame, not '{type(sample_points)}'"
            )
        required_columns = ["iteration"]
        required_columns.extend(parameter_space.parameter)
        if not all([column in sample_points.columns for column in required_columns]):
            raise RuntimeError(
                f"State sample points must contain the columns {required_columns}. Found {sample_points.columns}."
            )
        if len(sample_points) == 0:
            raise RuntimeError(
                "State sample points must specify at least one point in parameter space. Found none."
            )
        valid = True
        msg = ""
        for irow in range(len(sample_points)):
            row = sample_points.iloc[irow]
            for parameter_spec in parameter_space.itertuples():
                if (row[parameter_spec.parameter] < parameter_spec.min) or (
                    row[parameter_spec.parameter] > parameter_spec.max
                ):
                    valid = False
                    msg += f"Sample parameter, {row}, is outside parameter space."
        if not valid:
            raise RuntimeError(msg)

        return

    @staticmethod
    def validate_observations(observations: pd.DataFrame) -> None:

        if not isinstance(observations, pd.DataFrame):
            raise TypeError(
                f"State observations should be Pandas DataFrame, not '{type(observations)}'"
            )
        if len(observations.columns) == 0:
            raise RuntimeError(
                "State observations must have at least one feature (column)."
            )
        if len(observations) != 1:
            raise RuntimeError(
                f"State observations must have one row of observed features. Found {len(observations)} rows."
            )

        return

    @staticmethod
    def validate_simulator_results(
        simulator_results: pd.DataFrame,
        parameter_space: pd.DataFrame,
        observations: pd.DataFrame,
    ) -> None:

        if not isinstance(simulator_results, pd.DataFrame):
            raise TypeError(
                f"State simulator results should be Pandas DataFrame, not '{type(simulator_results)}'"
            )
        required_columns = ["replicate"]
        required_columns.extend(parameter_space.parameter)
        required_columns.extend(observations.columns)
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

        if not isinstance(emulator_bank, dict):
            raise TypeError(
                f"State enumlator bank should be dictionary, not '{type(emulator_bank)}'"
            )
        if len(emulator_bank) > 0:
            if not all([isinstance(key, int) for key in emulator_bank.keys()]):
                raise TypeError(
                    f"State emulator bank should map integer iterations to dictionary of features:emulators. Found non-integral iteration/key."
                )
            if not all([isinstance(value, dict) for value in emulator_bank.values()]):
                raise TypeError(
                    f"State emulator bank should map integer iterations to dictionary of features:emulators."
                )
        for iteration, emulators in emulator_bank.items():
            if not all([key in observations.columns for key in emulators.keys()]):
                raise ValueError(
                    f"Found 'feature' in emulators dictionary ({emulators.keys()}, iteration {iteration}) which does not map to observation features ({observations.columns})."
                )

        return
