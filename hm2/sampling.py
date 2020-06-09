from pyDOE import lhs
import pandas as pd
import numpy as np

from .data_validation import ValidateParameterInfoFrame, ValidateParameterSamplesFrame
from .error import *



def parameter_info_frame_from_samples(
  parameter_samples:pd.DataFrame
) -> pd.DataFrame:
  """Generate a :ref:`ParameterInfoFrame` from a :ref:`ParameterSamplesFrame`

  Args:
    parameter_samples: A :ref:`ParamterSamplesFrame` to generate the
                       :ref:`ParameterInfoFrame` from.

  Returns: A :ref:`ParameterInfoFrame`
  """
  parameter_samples = ValidateParameterSamplesFrame(parameter_samples)
  # Build a ParameterInfoFrame
  params_and_ranges=[]
  for x in parameter_samples.drop(columns='param_id'):
    params_and_ranges.append({
      "name":x,
      "min":parameter_samples[x].min(),
      "max":parameter_samples[x].max()
    })
  return pd.DataFrame(params_and_ranges)



def latin_hypercube(
  param_info: pd.DataFrame,
  samples: int,
  random_state: int=None
):
  """
  Generate parameter hypercube given min and max values for parameters.

  Args:
    param_info (:ref:`ParameterInfoFrame`): Bounds of the parameters.
    samples (int): Number of samples to generate
    random_state (int): Used to generate samples reproducibly without affecting
                        random numbers in the rest of the program.

  Returns:
    A :ref:`ParameterSamplesFrame`.
  """
  assert isinstance(samples, int) and samples>=0
  assert isinstance(random_state, (int, type(None)))

  # Calculate ranges
  param_info = ValidateParameterInfoFrame(param_info)
  param_info["range"] = param_info['max']-param_info['min']
  param_info.set_index('name')

  #Swap in a new random state if the user requested one
  if random_state is not None:
    old_random_state = np.random.get_state()
    np.random.seed(random_state)

  # Generate a matrix with `n_params` columns and `n_samples_this_iter` rows
  # whose entries comprise the hypercube
  hypercube = lhs(len(param_info), samples)
  # Convert to a DataFrame where each column of the matrix is associated with a
  # parameter name from `param_info`
  samples = pd.DataFrame(hypercube, columns=param_info['name'].tolist())
  # Rescale hypercube to parameter range
  for index, row in param_info.iterrows():
      samples[row['name']][:] = row['min'] + samples[row['name']]*row['range']

  samples['param_id'] = list(range(len(samples)))

  #If the user requested a random state swap back to the old one here so
  #we don't intefere with the user's work
  if random_state is not None:
    np.random.set_state(old_random_state)

  return samples



def latin_hypercube_within(
  parameter_samples: pd.DataFrame,
  samples: int,
  random_state: int=None
):
  """
  Generate parameter hypercube bounded by another ParameterSamplesFrame.

  Args:
    parameter_samples (:ref:`ParameterSamplesFrame`):
        Parameter samples which bound the new frame.
    samples (int): Number of samples to generate for each parameter
    random_state (int): Used to generate samples reproducibly without affecting
                        random numbers in the rest of the program.

  Returns:
    A :ref:`ParameterSamplesFrame`.
  """
  assert isinstance(samples, int) and samples>=0
  assert isinstance(random_state, (int, type(None)))

  if len(parameter_samples)==0:
    raise HMParameterSamplesEmpty()

  # Strip down to parameters
  parameter_samples = ValidateParameterSamplesFrame(parameter_samples)
  parameter_samples = parameter_samples.drop(columns='param_id')

  # Get ranges of parameters
  params_and_ranges = parameter_info_frame_from_samples(parameter_samples)

  # Use parameter information to generate a new hypercube
  return latin_hypercube(params_and_ranges, samples, random_state=random_state)



def get_size_of_parameter_space(parameter_samples:pd.DataFrame) -> float:
  """Get the volume of the space defined by the parameter samples

  Args:
    parameter_samples: A :ref:`ParameterSamplesFrame` to get the volume for

  Returns: The volume of the space
  """
  #TODO: Should this throw on len(parameter_samples)==0 ?
  if len(parameter_samples)<=1:
    return 0.0

  parameter_samples = ValidateParameterSamplesFrame(parameter_samples)
  params_and_ranges = parameter_info_frame_from_samples(parameter_samples)
  ranges = params_and_ranges['max']-params_and_ranges['min']
  #Volume is product of the ranges
  return np.prod(ranges)

def percent_change_vol(vol_old:float, vol_new:float) -> float:
  return (vol_new-vol_old)/vol_old*100.0