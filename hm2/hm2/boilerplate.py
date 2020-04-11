import os

import numpy as np
import pandas as pd
import pickle

from .error import HistoryMatchingError
from .data_validation import *
from .utility import drop_key
from .wrapped_model import *


def _column_comparison(model_results, observations, obs_name):
    """Determines if `model_results` and `observations` have the same values in
    their `measurement` columns. Otherwise throws an error identifying what's
    missing.

    Returns: 
      None
    """
    # Get the names of the properties that were observed
    observation_names = set()
    if observations is not None:
        observation_names = set(observations["observation"].unique().tolist())

    model_names = set()
    if model_results is not None:
        model_names = set(model_results["observation"].unique().tolist())

    diff_tm = observation_names - model_names
    diff_mt = model_names - observation_names
    if diff_tm or diff_mt:
        raise HistoryMatchingError(
            f"Model output is missing columns: {diff_tm}. Model {obs_name} is missing {diff_mt}."
        )


def match_sim_to_observations(sim_output, observations):
    if isinstance(sim_output, tuple):
        if isinstance(observations, tuple):
            tr = match_sim_to_observations(sim_output[0], observations[0])
            sr = match_sim_to_observations(sim_output[1], observations[1])
            return tr, sr
        else:
            raise HistoryMatchingError(
                "match_sim_to_observations expects either two single inputs or two tuples!"
            )

    if sim_output is None:
        return None

    sim_output = ValidateSimFrame(sim_output, copy=False)
    observations = ValidateObservationsFrame(observations, copy=False)

    if ("time" in sim_output.columns) ^ ("time" in observations.columns):
        raise HistoryMatchingError(
            "Attempting to match a TimeObservationsFrame to a SummaryObservationsFrame!"
        )

    time_frame = "time" in sim_output.columns

    if time_frame:
        # Left merge, matching each modeled and actual observation to its nearest
        # analogue by time
        temp = pd.merge_asof(
            observations,
            sim_output,
            on="time",
            by="observation",
            direction="nearest",
            suffixes=("_a", "_s"),
        )
        temp = temp.drop(columns="time")  # No longer need the time
    else:
        temp = pd.merge(
            observations, sim_output, on="observation", suffixes=("_a", "_s")
        )

    temp = temp.drop(
        columns=[
            "observation_id_s",  # Don't care about simulation observation ids
            "value_a",  # Drop actual value
            "stdev_a",  # Drop actual stdev
            "observation",  # Drop observation name
        ]
    )

    # Rename to drop suffixes
    temp = temp.rename(columns={"value_s": "value", "stdev_s": "stdev"})

    return temp


def _validated_run(wrapped_model, param_set, replicate):
    def add_to_frame(df, key, value):
        if df is not None:  # If there is a data frame
            if value is not None:  # and we have something to add to it
                df[key] = value  # then add the thing

    if not isinstance(param_set, dict):
        raise HistoryMatchingError("param_set must be a dictionary!")

    # Remove param_id, if present so that it isn't interpreted as a model
    # parameter
    param_id = param_set.get("param_id", None)
    param_set = drop_key(param_set, "param_id", ignore_missing=True)

    # Run the model
    results = wrapped_model(**param_set)
    # Ensure model returned the sorts of results we expected
    time_result, summary_result = ValidateObservationFrames(results)

    add_to_frame(time_result, "replicate", replicate)
    add_to_frame(time_result, "param_id", param_id)
    add_to_frame(summary_result, "replicate", replicate)
    add_to_frame(summary_result, "param_id", param_id)

    return (time_result, summary_result)


def run_replicates(wrapped_model, replicates, param_sets=None, processes=None):
    if param_sets is None:  # Means we run with default parameters
        param_sets = [dict()]
    elif isinstance(param_sets, dict):
        param_sets = [param_sets]
    elif isinstance(param_sets, pd.DataFrame):
        param_sets = ValidateParameterSamplesFrame(param_sets)
        param_sets = [x.to_dict() for _, x in param_sets.iterrows()]
    elif isinstance(param_sets, list) and all(
        [isinstance(x, dict) for x in param_sets]
    ):
        pass
    else:
        raise HistoryMatchingError(
            "`param_sets` should be a ParameterSamplesFrame, dictionary, a list of dictionaries, or None!"
        )

    if processes is None or processes > 1:
        mapper = multiprocessing.Pool(processes=processes).starmap
    elif processes == 1:
        mapper = itertools.starmap
    else:
        raise HistoryMatchingError("Unrecognized processes value!")

    return list(
        mapper(
            _validated_run,
            itertools.product([wrapped_model], param_sets, list(range(replicates))),
        )
    )


def match_sim_outputs_to_observations(
    sim_outputs, time_observations, summary_observations, processes=None
):
    if not isinstance(sim_outputs, list):
        raise TypeError("`sim_outputs` must be a list")
    if not all([isinstance(x, tuple) for x in sim_outputs]):
        raise TypeError("`sim_outputs` must be a list of tuples!")

    if processes is None or processes > 1:
        mapper = multiprocessing.Pool(processes=processes).starmap
    elif processes == 1:
        mapper = itertools.starmap
    else:
        raise HistoryMatchingError("Unrecognized processes value!")

    observations = [(time_observations, summary_observations)]

    matched = mapper(
        match_sim_to_observations, itertools.product(sim_outputs, observations)
    )

    aggregator = (
        lambda x: None if all(y is None for y in x) else pd.concat(x, ignore_index=True)
    )
    aggregate_time_results = aggregator([x[0] for x in matched])
    aggregate_summary_results = aggregator([x[1] for x in matched])

    return aggregate_time_results, aggregate_summary_results


def prep_emulator_data(param_samples, matched, observation_id):
    """Fit the Emulator

    Args:
        emulator: Emulator to fit
        param_samples: :ref:`ParameterSamplesFrame`
        model_output: A :ref:`TimeSimFrame` or 
                       :ref:`SummarySimFrame` built using parameters
                       from `param_samples`
        observation_key: Filter model_output by `observation_key` 
        maxiter (int): Number of training iterations

    Returns:
        None
    """
    param_samples = ValidateParameterSamplesFrame(param_samples, copy=False)
    matched = ValidateMatchedFrame(matched, copy=False)

    # Filter matched down to just the observation we are interested in. Doing
    # this early on makes subsequent operations faster.
    matched = matched[matched["observation_id_a"] == observation_id]
    # Drop observation_id_a column since we no longer need it
    matched = matched.drop(columns="observation_id_a")

    # Get all parameter samples used for observation
    params = matched[["param_id"]]
    # Pair them with their actual values
    params = pd.merge(params, param_samples, how="left", on="param_id")
    # Drop param_id column leaving only parameter values
    params = params.drop(columns="param_id")

    train_x = params
    train_y = matched["value"].to_numpy()
    stdev_y = matched["stdev"].to_numpy()

    return train_x, train_y, stdev_y


def get_data_for_emulators(param_samples, matched):
    """Fit the Emulator

    Args:
        param_samples (:ref:`ParameterSamplesFrame`)
        matched: A :ref:`TimeSimFrame` or 
                       :ref:`SummarySimFrame` built using parameters
                       from `param_samples`

    Returns:
        None
    """
    param_samples = ValidateParameterSamplesFrame(param_samples, copy=False)
    matched = ValidateMatchedFrame(matched, copy=False)
    merged = pd.merge(matched, param_samples, how="left", on="param_id")
    merged = merged.drop(columns="param_id")
    for observation_id_a, grouped in merged.groupby("observation_id_a"):
        train_y = grouped["value"].to_numpy()
        stdev_y = grouped["stdev"].to_numpy()
        grouped = grouped.drop(
            columns=["observation_id_a", "replicate", "value", "stdev"]
        )
        yield observation_id_a, grouped, train_y, stdev_y


def _implausibility_equ(
    reality, reality_stdev, prediction, prediction_stdev, model_stdev
):
    """Implements Equation 3 from Gardner2019"""
    return np.abs(reality - prediction) / np.sqrt(
        reality_stdev ** 2 + model_stdev ** 2 + prediction_stdev ** 2
    )


def get_implausibility(emulators, parameter_samples, observations, model_stdev=0):
    implausibilities = []
    for _, row in observations.iterrows():
        row = row.to_dict()
        if row["observation_id"] not in emulators:
            continue
        prediction, p_stdev = emulators[row["observation_id"]].predict(
            parameter_samples
        )
        implausibility = _implausibility_equ(
            reality=row["value"],
            reality_stdev=row["stdev"],
            prediction=prediction,
            prediction_stdev=p_stdev,
            model_stdev=model_stdev,
        )
        implausibility = pd.DataFrame(
            {
                "observation_id": row["observation_id"],
                "param_id": parameter_samples["param_id"],
                "implausibility": implausibility,
            }
        )
        implausibilities.append(implausibility)
    return pd.concat(implausibilities, ignore_index=True)


def max_implausibility_per_param(implausibilities):
    return implausibilities.groupby("param_id")["implausibility"].max().reset_index()


def filter_implausibilities(implausibilities, threshold):
    return implausibilities[implausibilities["implausibility"] <= threshold]


def get_plausible_parameters(implausibilities, parameter_samples):
    params = pd.merge(implausibilities, parameter_samples, how="left", on="param_id")
    params = params.drop(columns="implausibility")
    return params
