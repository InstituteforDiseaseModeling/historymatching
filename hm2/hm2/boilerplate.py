import abc

import pandas as pd

from .error import HistoryMatchingError
from .data_validation import ValidateObservationsFrame, ValidateParameterSamplesFrame, ValidateWrappedModelResults



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



#TODO: Add summary observations argument
def time_analysis(parameter_samples, observations, wrapped_model, replicates=1):
  """Perform a time analysis TODO

  This function is not parallelized!

  Args:
    parameter_samples - A DataFrame of the form:
                   <sample_id> <Parameter1> [Parameter2] [Parameter3] [...]
    observations - A DataFrame of the form:
                   <observation_id> <time> <value1> [value2] [value3] [...]
    wrapped_model - A model instantiating the ModelWrapper class.
    replicates - Number of times to call `run_func` for each parameter setting
  """
  if not isinstance(wrapped_model, ModelWrapper):
    raise HistoryMatchingError("wrapped_model was not an instance of ModelWrapper!")

  parameter_samples = ValidateParameterSamplesFrame(parameter_samples)
  observations      = ValidateObservationsFrame(observations)

  # Get the names of the properties that were observed
  observation_names = observations.columns.tolist()
  observation_names.remove('observation_id')
  observation_names.remove('time')

  sim_tp_results = []
  sim_su_results = []
  for _, parameter_sample in parameter_samples.iterrows():
    # Convert parameter_sample to dictionary
    parameter_sample_for_init = parameter_sample.to_dict() 
    # Drop `sample_id` value from dictionary leaving only parameters behind
    del parameter_sample_for_init['sample_id']
    # Initialize the model for these parameter settings
    model = wrapped_model.init(**parameter_sample_for_init)

    for replicate in range(1):
      #Run the model
      results = wrapped_model.run(model)
      #Ensure model returned the sorts of results we expected
      #TODO(r-barnes): Handle summary results
      timepoint_results, _ = ValidateWrappedModelResults(results)

      # Ensure that the model run returns modeled observations for all our
      # matched observations
      name_check = set(observation_names)-set(timepoint_results.columns)
      if name_check:
        raise HistoryMatchingError(f'Model output is missing columns: {list(name_check)}. Found columns: {timepoint_results.columns.tolist()}')

      #Set the index so we can quickly find nearest times
      timepoint_results.set_index('time')

      # Because time in the model may proceed stochastically, the time vector
      # may not contain the exact observation time we require. Instead let's
      # find the closest ones to each observation.
      for _, obs in observations.iterrows():
        closest_time_index = timepoint_results.index.get_loc(obs['time'], method='nearest')
        model_observation  = timepoint_results.iloc[closest_time_index]
        model_time         = timepoint_results.index[closest_time_index]

        formatted_model_observation = [
          parameter_sample['sample_id'],
          replicate,
          obs['observation_id'],
          model_time
        ] + [model_observation[oname] for oname in observation_names]

        sim_tp_results.append(formatted_model_observation)

  sim_tp_results = pd.DataFrame(sim_tp_results, columns=['sample_id','replicate','observation_id','time'] + observation_names)

  #TODO: Return sim_su_results as well
  return sim_tp_results, None
