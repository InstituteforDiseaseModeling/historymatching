import pandas as pd

from .error import HistoryMatchingError



def ValidateObservationsFrame(df):
  """Validates an observations DataFrame and returns a copy"""
  if not isinstance(df, pd.DataFrame):
    raise TypeError("observations-DataFrame must be a pandas DataFrame!")
  if 'observation_id' not in df.columns:
    raise HistoryMatchingError("observations-DataFrame must include `observation_id` column!")
  if 'time' not in df.columns:
    raise HistoryMatchingError("observations-DataFrame must include `time` column!")
  if not df['observation_id'].is_unique:
    raise HistoryMatchingError("observations-DataFrame's `observation_id` column has non-unique values!")
  if not df['time'].is_unique:
    raise HistoryMatchingError("observations-DataFrame's `time` column has non-unique values!")
  return df.copy()



def ValidateParameterSamplesFrame(df):
  """Validates a parameter sampling DataFrame and returns a copy"""
  if not isinstance(df, pd.DataFrame):
    raise TypeError("parameter_samples-DataFrame must be a pandas DataFrame!")
  if 'sample_id' not in df.columns:
    raise HistoryMatchingError("parameter_samples-DataFrame must include `sample_id` column!")
  return df.copy()



def ValidateRunsFrame(df):
  """Validates a runs-DataFrame and returns a copy"""
  if not isinstance(df, pd.DataFrame):
    raise TypeError("runs-DataFrame must be a pandas DataFrame!")
  if 'sample_id' not in df.columns:
    raise HistoryMatchingError("runs-DataFrame must include `sample_id` column!")
  if 'replicate' not in df.columns:
    raise HistoryMatchingError("runs-DataFrame must include `replicate` column!")
  if 'observation_id' not in df.columns:
    raise HistoryMatchingError("runs-DataFrame must include `observation_id` column!")
  if 'observation_id' not in df.columns:
    raise HistoryMatchingError("runs-DataFrame must include `time` column!")
  return df.copy()



def ValidateWrappedModelResults(results):
  if not isinstance(results, tuple) or len(results)!=2:
    raise HistoryMatchingError("Wrapped model must return a tuple with both a `time_points` and a `summary_points` DataFrame!")

  timepoint_results, summary_results = results

  if timepoint_results is not None and not isinstance(timepoint_results, pd.DataFrame):
    raise HistoryMatchingError("Wrapped model's `time_points` DataFrame be a DataFrame or None")
  if summary_results is not None and not isinstance(summary_results, pd.DataFrame):
    raise HistoryMatchingError("Wrapped model's `summary_points` DataFrame be a DataFrame or None")
  if not 'time' in timepoint_results:
    raise HistoryMatchingError("Wrapped model's `time_points` DataFrame did not include `time` column.")

  return timepoint_results, summary_results