import abc
import os

import pandas as pd
import pickle

from .error import HistoryMatchingError
from .data_validation import *
from .utility import drop_key


class ModelWrapper(abc.ABC):
  @classmethod
  @abc.abstractmethod
  def init(cls, **kwargs):
    """Function that returns an initialized model.

    Args:
      kwargs - Named arguments corresponding to the parameter settings to be 
       used in a set of runs of the model.

    Returns: A model initialized with `kwargs`. The model is treated as
             an opaque object which is passed to other methods.
    """
    pass

  @staticmethod
  @abc.abstractmethod
  def run(model):
    """Function that runs the initialized model.

    Args:
      model - A model returned by `init`.

    Returns:
      A tuple of DataFrames `(time_points, summary_points)`

      Either value of the tuple may also be None

      The `time_points` DataFrame has the form:
          <time> <observation name 1> [observation name 2]

      The `summary_points` DataFrame has the form:
          <summary name> <summary value>
    """
    pass



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
  if not isinstance(wrapped_model, ModelWrapper):
    raise HistoryMatchingError("wrapped_model was not an instance of ModelWrapper!")

  parameter_samples    = ValidateParameterSamplesFrame(parameter_samples)
  time_observations    = ValidateTimeObservationsFrame(time_observations)
  summary_observations = ValidateTimeObservationsFrame(summary_observations)

  aggregate_time_results = []
  aggregate_summary_results = []
  for _, parameter_sample in parameter_samples.iterrows():
    # Convert parameter_sample to dictionary
    parameter_sample = parameter_sample.to_dict() 
    # Initialize the model for these parameter settings
    model = wrapped_model.init(**drop_key(parameter_sample, 'param_id'))

    for replicate in range(replicates):
      #Run the model
      results = wrapped_model.run(model)
      #Ensure model returned the sorts of results we expected
      time_results, summary_results = ValidateWrappedModelResults(results)

      # Ensure that the modeled and actual observations agree on what quantities
      # were observed
      _column_comparison(time_results, time_observations, "TimeObservationsFrame")
      _column_comparison(summary_results, summary_observations, "SummaryObservationsFrame")

      # Ensure that time values are the same, so we can match them between
      # modeled and actual observations
      if time_results['time'].dtype!=time_observations['time'].dtype:
        raise HistoryMatchingError(f"Data type of `time` differs between modeled and actual observations: {time_results['time'].dtype} vs {time_observations['time'].dtype}!")

      # Match modeled and actual time observations
      time_results = _generate_time_standard_analysis_frame(
        parameter_sample['param_id'],
        replicate,
        time_results,
        time_observations
      )

      # Match modeled and actual summary observations
      summary_results = _generate_summary_standard_analysis_frame(
        parameter_sample['param_id'],
        replicate,
        summary_results,
        summary_observations
      )

      aggregate_time_results.append(time_results)
      aggregate_summary_results.append(summary_results)

  # Now we condense the lists of paired observations into single dataframes
  aggregator = lambda x: None if all(y is None for y in x) else pd.concat(x, ignore_index=True)
  aggregate_time_results    = aggregator(aggregate_time_results)
  aggregate_summary_results = aggregator(aggregate_summary_results)

  ret = aggregate_time_results, aggregate_summary_results

  if cache_name:
    pickle.dump(ret, open(cache_name, "wb" ))

  return ret



def replicate_reducer(df, agg, has_time):
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
    has_time - True if the this is a TimeStandardAnalysisFrame; otherwise, false.

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

  if has_time:
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
    del group['replicate']
    return group

  grouping_keys = ['param_id', 'observation']
  if has_time:
    grouping_keys.append('aobservation_id')
  ret = df.groupby(grouping_keys).apply(helper)

  return ret
