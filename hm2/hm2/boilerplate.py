import pandas as pd

def time_analysis(parameter_samples, observations, init_func, run_func, replicates=1):
  """Perform a time analysis TODO

  This function is not parallelized!

  Args:
    parameter_samples - A DataFrame of the form:
                   [sample_id] <Parameter1> [Parameter2] [Parameter3] [...]
    observations - A DataFrame of the form:
                   <observation_id> <time> <value1> [value2] [value3] [...]
    init_func  - Function that returns an initialized model. This function is 
                 named arguments corresponding to the parameter settings to be 
                 used in a set of runs of the model.
                 We treat the return value of this function as an opaque object 
                 which is passed as an argument to `run_func`.
    run_func   - Function that runs the initialized model.
                 Takes the result of `init_func`.
                 Returns a DataFrame of the form:
                   <time> <observation name 1> [observation name 2]
    replicates - Number of times to call `run_func` for each parameter setting
  """
  parameter_samples = parameter_samples.copy()

  if 'sample_id' not in parameter_samples.columns:
    parameter_samples['sample_id'] = list(range(len(parameter_samples)))

  assert 'observation_id' in observations.columns
  assert 'time' in observations.columns

  # Get the names of the properties that were observed
  observation_names = observations.columns.tolist()
  observation_names.remove('observation_id')
  observation_names.remove('time')

  sim_results = []
  for _, parameter_sample in parameter_samples.iterrows():
    # Convert parameter_sample to dictionary
    parameter_sample_for_init = parameter_sample.to_dict() 
    # Drop `sample_id` value from dictionary leaving only parameters behind
    del parameter_sample_for_init['sample_id']
    # Initialize the model for these parameter settings
    model = init_func(**parameter_sample_for_init)

    for replicate in range(1):
      model_run_results = run_func(model)
      assert 'time' in model_run_results

      # Ensure that the model run returns modeled observations for all our
      # matched observations
      name_check = set(observation_names)-set(model_run_results.columns)
      if name_check:
        raise Exception(f'Model output is missing columns: {list(name_check)}. Found columns: {model_run_results.columns.tolist()}')

      #Set the index so we can quickly find nearest times
      model_run_results.set_index('time')

      # Because time in the model may proceed stochastically, the time vector
      # may not contain the exact observation time we require. Instead let's
      # find the closest ones to each observation.
      for _, obs in observations.iterrows():
        closest_time_index = model_run_results.index.get_loc(obs['time'], method='nearest')
        model_observation  = model_run_results.iloc[closest_time_index]
        model_time         = model_run_results.index[closest_time_index]

        formatted_model_observation = [
          parameter_sample['sample_id'],
          replicate,
          obs['observation_id'],
          model_time
        ] + [model_observation[oname] for oname in observation_names]

        sim_results.append(formatted_model_observation)

  return pd.DataFrame(sim_results, columns=['sample_id','replicate','observation_id','time'] + observation_names)
