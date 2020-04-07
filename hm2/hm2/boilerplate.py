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

  Returns: None
  """
  # Get the names of the properties that were observed
  observation_names = set()
  if observations is not None:
    observation_names = set(observations['observation'].unique().tolist())

  model_names = set()
  if model_results is not None:
    model_names = set(model_results['observation'].unique().tolist())

  diff_tm = observation_names-model_names
  diff_mt = model_names-observation_names
  if diff_tm or diff_mt:
    raise HistoryMatchingError(f'Model output is missing columns: {diff_tm}. Model {obs_name} is missing {diff_mt}.')



def _generate_time_standard_analysis_frame(
  param_id,
  replicate,
  results,
  observations
):
  if results is None:
    return None

  # We don't care about the observation_id produced by the model
  del results['observation_id']

  # We don't care about the paticular observation values here, only when they
  # took place and what quantity they were
  observations = observations[['observation_id', 'time', 'observation']]

  # Rename to match expected output.
  observations = observations.rename(columns={"observation_id": "aobservation_id"})

  # Left merge, matching each modeled and actual observation to its nearest
  # analogue by time
  temp = pd.merge_asof(observations, results, on='time', by='observation', direction='nearest')

  # No longer need the time
  del temp['time']

  # Add contextual information
  temp['param_id'] = param_id
  temp['replicate'] = replicate

  return temp



def _generate_summary_standard_analysis_frame(
  param_id,
  replicate,
  results,
  observations
):
  if results is None:
    return None

  # Everything needed to match to the observations data is already present: only
  # the values of the `observation` column are necessary.

  # Add contextual information
  results['param_id'] = param_id
  results['replicate'] = replicate

  return results



#TODO(r-barnes): parallelize
def standard_analysis(
  parameter_samples,
  time_observations,
  summary_observations,
  wrapped_model,
  replicates=1,
  cache_name=""
):
  """Perform a time analysis TODO

  This function is not parallelized!

  Args:
    parameter_samples - A ParameterSamplesFrame
    time_observations - A TimeObservationsFrame (may be None)
    summary_observations - A SummaryObservationsFrame (may be None)
    wrapped_model - A model instantiating the ModelWrapper class.
    replicates - Number of times to run the model for each parameter setting
    cache_name - If specified, results are pickled to a file of this name

  Returns: (TimeStandardAnalysisWithReplicatesFrame,SummaryStandardAnalysisWithReplicates)
  """
  if os.path.isfile(cache_name):
    return pickle.load(open(cache_name, "rb" ))

  if time_observations is None and summary_observations is None:
    raise HistoryMatchingError("time_analysis was passed None for both `time_observations` and `summary_observations`! At least one must be provided!")

  parameter_samples    = ValidateParameterSamplesFrame(parameter_samples)
  time_observations    = ValidateTimeObservationsFrame(time_observations)
  summary_observations = ValidateTimeObservationsFrame(summary_observations)

  reducer = StandardAnalysisReducer(time_observations, summary_observations)

  ret = RunReplicates(
    wrapped_model = wrapped_model,
    param_sets    = [x.to_dict() for _, x in parameter_samples.iterrows()],
    replicates    = replicates,
    show_hidden   = False,
    processes     = processes,
    reducer       = reducer
  )

  if cache_name:
    pickle.dump(ret, open(cache_name, "wb" ))

  return ret



#TODO: Is this good for anything?
def replicate_reducer(df, agg):
  """Reduce a TimeStandardAnalysisFrame or SummaryStandardAnalysisFrame to one
     without a replicate column by aggregating replicates according to `agg`.

  Args:
    df - TimeStandardAnalysisFrame or SummaryStandardAnalysisFrame
    agg - A dictionary describing the reduction a short name.
          Short names are: mean
          Dictionaries have the form:
              {"OBSERVATION NAME": (value_reducer_func, stdev_reducer_func)}
          If "OBSERVATION NAME" is "default" then the specified reducers are
          applied to any observations not otherwise specified.

  Return: The data frame with replicates aggregated.
  """
  if df is None:
    return None

  if agg=="mean":
    agg = {"default": (np.mean, np.mean)}
  elif not isinstance(agg,dict):
    raise HistoryMatchingError("`agg` argument to replicate_reducer must be a dict or a recognized reducer name!")

  if not isinstance(df,pd.DataFrame):
    raise HistoryMatchingError("`df` argument to replicate_reducer must be a DataFrame!")

  if 'aobservation_id' in df.columns:
    df = ValidateTimeStandardAnalysisWithReplicatesFrame(df)
  else:
    df = ValidateSummaryStandardAnalysisWithReplicatesFrame(df)

  # Function applied to each group
  def helper(group):
    observation_name = group['observation'].iloc[0]
    if observation_name in agg:
      group["value"] = agg[observation][0](group["value"])
      group["stdev"] = agg[observation][1](group["stdev"])
    else:
      group["value"] = agg["default"][0](group["value"])
      group["stdev"] = agg["default"][1](group["stdev"])
    group['replicate'] = 0 #Replicates are now binned to a single value
    return group

  grouping_keys = ['param_id', 'observation']
  if 'aobservation_id' in df.columns:
    grouping_keys.append('aobservation_id')
  ret = df.groupby(grouping_keys).apply(helper)

  return ret






def FitEmulatorWithParamsAndModel(emulator, param_samples, model_output):
    """Fit the Emulator

    Args:
        param_samples - ParameterSamplesFrame
        model_output - A TimeStandardAnalysisFrame or 
                       SummaryStandardAnalysisFrame built using parameters
                       from param_samples
        maxiter - Number of training iterations

    Returns: None
    """
    if not isinstance(emulator,EmulatorBase):
        raise HistoryMatchingError("`emulator` must inherit from EmulatorBase!")

    param_samples = ValidateParameterSamplesFrame(param_samples)
    model_output  = ValidateEmulatorInput(model_output)

    train_x = param_samples.iloc[model_output['param_id']]
    train_x = train_x.drop(columns=['param_id'])

    train_y = model_output['value']
    stdev_y = model_output['stdev']

    return emulator.fit(train_x, train_y, stdev_y, *args, **kwargs)





