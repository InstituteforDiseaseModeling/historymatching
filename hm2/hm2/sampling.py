from pyDOE import lhs
import pandas as pd

from .data_validation import ValidateParameterInfoFrame

def latin_hypercube(param_info, samples):
  """
  Generate parameter hypercube given min and max values for parameters.

  Args:
    param_info - A ParameterInfoFrame, e.g.:

                      param_info = pd.DataFrame({
                      'name': ['Beta', 'Gamma'],
                      'min':  [  1e-6,    1e-6],
                      'max':  [  0.01,     0.5]
                      })

    samples - Number of samples to generate for each parameter

  Returns: A ParameterSamplesFrame
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