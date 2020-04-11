from pyDOE import lhs
import pandas as pd

from .data_validation import *

def latin_hypercube(param_info, samples):
  """
  Generate parameter hypercube given min and max values for parameters.

  Args:
    param_info (ParameterInfoFrame): Bounds of the parameters.
    samples (int): Number of samples to generate

  Returns: A :ref:`ParameterSamplesFrame`.
  """
  # Calculate ranges
  param_info = ValidateParameterInfoFrame(param_info)
  param_info["range"] = param_info['max']-param_info['min']
  param_info.set_index('name')

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

  return samples



def latin_hypercube_within(parameter_samples, samples):
  """
  Generate parameter hypercube bounded by another ParameterSamplesFrame.

  Args:
    parameter_samples (ParameterSamplesFrame): Parameter samples which bound
        the new frame.
    samples (int): Number of samples to generate for each parameter

  Returns: A :ref:`ParameterSamplesFrame`.
  """
  # Strip down to parameters
  parameter_samples = ValidateParameterSamplesFrame(parameter_samples)
  parameter_samples = parameter_samples.drop(columns='param_id')

  # Build a ParameterInfoFrame
  params_and_ranges=[]
  for x in parameter_samples:
    params_and_ranges.append({
      "name":x, 
      "min":parameter_samples[x].min(), 
      "max":parameter_samples[x].max()
    })

  params_and_ranges = pd.DataFrame(params_and_ranges)

  print(params_and_ranges)

  # Use parameter information to generate a new hypercube
  return latin_hypercube(params_and_ranges, samples)
