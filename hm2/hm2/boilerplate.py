import abc

import pandas as pd

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
  temp = pd.merge_asof(results, observations, on='time', by='observation', direction='nearest')

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
  replicates=1
):
  """Perform a time analysis TODO

  This function is not parallelized!

  Args:
    parameter_samples - A ParameterSamplesFrame
    time_observations - A TimeObservationsFrame (may be None)
    summary_observations - A SummaryObservationsFrame (may be None)
    wrapped_model - A model instantiating the ModelWrapper class.
    replicates - Number of times to run the model for each parameter setting
  """
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

  if time_observations is None:
    aggregate_time_results = None
  else:
    aggregate_time_results = pd.concat(aggregate_time_results, ignore_index=True)

  if summary_observations is None:
    aggregate_summary_results = None
  else:
    aggregate_summary_results = pd.concat(aggregate_summary_results, ignore_index=True)

  return aggregate_time_results, aggregate_summary_results
