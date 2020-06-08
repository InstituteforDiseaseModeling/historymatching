import multiprocessing
import itertools
import os

import numpy as np
import pandas as pd
import pickle

from .error import HistoryMatchingError
from .data_validation import *
from .utility import drop_key



def _assert_processes_none_or_positive(processes):
    if processes is None:
        pass
    elif not isinstance(processes,int):
        raise TypeError("processes must be an integer!")
    elif processes<=0:
        raise ValueError("processes must be >0!")



def match_sim_to_observations(sim_output, observations):
    sim_output = ValidateSimFrame(sim_output, copy=False)
    observations = ValidateObservationsFrame(observations, copy=False)

    #TODO: Handle summary values

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
    temp = temp.drop(columns=[
        "time",              # No longer need the time
        "observation_id_s",  # Don't care about simulation observation ids
        "value_a",           # Drop actual value
        "stdev_a",           # Drop actual stdev
        "observation",       # Drop observation name
    ])

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

    # Run the model and ensure it returns valid results
    sim_results = ValidateObservationsFrame(wrapped_model(**param_set))

    add_to_frame(sim_results, "replicate", replicate)
    add_to_frame(sim_results, "param_id", param_id)

    return sim_results


def run_replicates(wrapped_model, replicates, param_sets=None, processes=None):
    """Runs a wrapped model `replicates` number of times for each row in param_sets

    Args:
        wrapped_model: A wrapped model (see :ref:`Wrapping A Model`)
        replicates: How many times to row the model per parameter set
        param_sets: A :ref:`ParameterSamplesFrame`.
        processes: Parallelize across this many processes. `None` implies using
                   as many processes as cores. `1` implies using a single core.

    Returns: A list of (:ref:`TimeSimFrame`, :ref:`SummarySimFrame`).
             Has length `replicates*len(param_sets)`.
    """
    param_sets = ValidateParameterSamplesFrame(param_sets)
    param_sets = [x.to_dict() for _, x in param_sets.iterrows()]

    _assert_processes_none_or_positive(processes)

    mapper_args = (
        _validated_run,
        itertools.product([wrapped_model], param_sets, list(range(replicates)))
    )

    if processes == 1:
        results = itertools.starmap(*mapper_args)
    else:
        pool = multiprocessing.Pool(processes=processes)
        results = pool.starmap(*mapper_args)
        pool.close()
        pool.join()

    return list(results)


def match_sim_outputs_to_observations(
    sim_outputs, real_observations, processes=None
):
    """Matches simulation outputs to actual observations.

    Args:
        sim_outputs(list): A list of :ref:`SimFrame`s.
        real_observations: An :ref:`ObservationsFrame`
        processes: Parallelize across this many processes. `None` implies using
                   as many processes as cores. `1` implies using a single core.

    Returns: A :ref:`MatchedFrame` which matches the simulation results to the
             observed time and summary results.
    """
    if not isinstance(sim_outputs, list):
        raise TypeError("`sim_outputs` must be a list")
    try:
        sim_outputs = [ValidateSimFrame(x) for x in sim_outputs]
    except HistoryMatchingError as hme:
        raise HistoryMatchingError(f"One of the simulation outputs did not validate! {hme}")

    _assert_processes_none_or_positive(processes)

    mapper_args = (
        match_sim_to_observations,
        itertools.product(sim_outputs, [real_observations])
    )

    if processes == 1:
        matched = itertools.starmap(*mapper_args)
    else:
        pool = multiprocessing.Pool(processes=processes)
        matched = pool.starmap(*mapper_args)
        pool.close()
        pool.join()

    if all(y is None for y in matched):
        raise HistoryMatchingError("No matches were possible for any simulation output!")

    return pd.concat(matched, ignore_index=True)

def prep_emulator_data(param_samples, matched, observation_id):
    """Fit the Emulator

    Args:
        emulator: Emulator to fit
        param_samples: A :ref:`ParameterSamplesFrame`
        model_output:  A :ref:`SimFrame` built using parameters
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
        matched: A :ref:`SimFrame` built using parameters from `param_samples`.

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
    #TODO: Check input type
    parameter_samples = ValidateParameterSamplesFrame(parameter_samples)

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
    #TODO: Check input type
    return implausibilities.groupby("param_id")["implausibility"].max().reset_index()


def filter_implausibilities(implausibilities, threshold):
    #TODO: Check input type
    return implausibilities[implausibilities["implausibility"] <= threshold]


def get_plausible_parameters(implausibilities, parameter_samples):
    #TODO: Check input type
    parameter_samples = ValidateParameterSamplesFrame(parameter_samples)
    params = pd.merge(implausibilities, parameter_samples, how="left", on="param_id")
    params = params.drop(columns="implausibility")
    return params
