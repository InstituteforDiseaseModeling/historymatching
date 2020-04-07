import itertools
import multiprocessing

import plotnine as pn

from .data_validation import *
from .utility import *



def validated_run(wrapped_model, param_set, replicate=None, show_hidden=False, reducer=None):
  def add_to_frame(df, key, value):
    if df is not None:       # If there is a data frame
      if value is not None:  # and we have something to add to it
        df[key] = value      # then add the thing

  if not isinstance(param_set,dict):
    raise HistoryMatchingError("param_set must be a dictionary!")

  #Remove param_id, if present so that it isn't interpreted as a model
  #parameter
  param_id  = param_set.get('param_id', None)
  param_set = drop_key(param_set, 'param_id', ignore_missing=True)

  #Run the model
  results = wrapped_model(show_hidden, **param_set)
  #Ensure model returned the sorts of results we expected
  time_results, summary_results = ValidateWrappedModelResults(results)

  add_to_frame(time_results,    'replicate', replicate)
  add_to_frame(time_results,    'param_id',  param_id )
  add_to_frame(summary_results, 'replicate', replicate)
  add_to_frame(summary_results, 'param_id',  param_id )

  if reducer is not None:
    time_results, summary_results = reducer(time_results, summary_results)
  return time_results, summary_results



def run_replicates(wrapped_model, replicates, param_sets=None, show_hidden=False, processes=None, reducer=None):
  if param_sets is None:
    #Means we run with default parameters
    param_sets = [dict()] 

  if not isinstance(param_sets,list) or not all([isinstance(x,dict) for x in param_sets]):
    raise HistoryMatchingError("param_sets should be a list of parameter dictionaries!")

  pool = multiprocessing.Pool(processes=processes)
  results = pool.starmap(
    validated_run, 
    itertools.product([wrapped_model], param_sets, list(range(replicates)), [show_hidden], [reducer])
  )

  aggregator = lambda x: None if all(y is None for y in x) else pd.concat(x, ignore_index=True)
  aggregate_time_results    = aggregator([x[0] for x in results])
  aggregate_summary_results = aggregator([x[1] for x in results])

  return {"time": aggregate_time_results, "summary": aggregate_summary_results}



def plot_runs(wrapped_model, params, replicates, processes=None):
  if params is None:
    params = dict()
  elif not isinstance(params, dict):
    raise HistoryMatchingError("`params` must be a dict!")

  results = run_replicates(
    wrapped_model = wrapped_model,
    param_sets    = [params],
    replicates    = replicates,
    show_hidden   = True,
    processes     = processes
  )

  return (pn.ggplot(results['time'], pn.aes('time', 'value', group='replicate')) + pn.geom_line() + pn.facet_wrap('~observation', scales='free_y'))

#TODO: Include observations above
# for i,obs in observations.iterrows():
#     ax.plot(obs['Times'], obs['Prevalence'], 'ko')
#     ax.plot(
#         [obs['Times'],obs['Times']], 
#         [obs['Prevalence']-2*obs['Stdev'],obs['Prevalence']+2*obs['Stdev']],
#         'k-')