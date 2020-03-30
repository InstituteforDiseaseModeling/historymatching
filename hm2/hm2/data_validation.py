import pandas as pd

from .error import HistoryMatchingError



def ValidateParameterInfoFrame(df):
  if not isinstance(df, pd.DataFrame):
    raise TypeError("ParameterInfoFrame must be a pandas DataFrame!")
  if 'name' not in df.columns:
    raise HistoryMatchingError("ParameterInfoFrame must include `name` column!")
  if 'min' not in df.columns:
    raise HistoryMatchingError("ParameterInfoFrame must include `min` column!")
  if 'max' not in df.columns:
    raise HistoryMatchingError("ParameterInfoFrame must include `max` column!")
  if not (df['min']<=df['max']).all():
    raise HistoryMatchingError("All entries of ParameterInfoFrame `min` column must be less than their corresponding `max` values!")
  return df.copy()



def ValidateParameterSamplesFrame(df):
  """Validates a parameter sampling DataFrame and returns a copy"""
  if not isinstance(df, pd.DataFrame):
    raise TypeError("parameter_samples-DataFrame must be a pandas DataFrame!")
  if 'sample_id' not in df.columns:
    raise HistoryMatchingError("parameter_samples-DataFrame must include `sample_id` column!")
  return df.copy()



def ValidateTimeObservationsFrame(df):
  """Validates an TimeObservationsFrame and returns a copy"""
  if not isinstance(df, pd.DataFrame):
    raise TypeError("TimeObservationsFrame must be a pandas DataFrame!")

  if 'observation_id' not in df.columns:
    raise HistoryMatchingError("TimeObservationsFrame must include `observation_id` column!")
  if 'time' not in df.columns:
    raise HistoryMatchingError("TimeObservationsFrame must include `time` column!")
  if 'observation' not in df.columns:
    raise HistoryMatchingError("TimeObservationsFrame must include `observation` column!")
  if 'value' not in df.columns:
    raise HistoryMatchingError("TimeObservationsFrame must include `value` column!")
  if 'stdev' not in df.columns:
    raise HistoryMatchingError("TimeObservationsFrame must include `stdev` column!")

  if len(df.columns)!=5:
    raise HistoryMatchingError("TimeObservationsFrame contains unexpected columns!")

  if not df['observation_id'].is_unique:
    raise HistoryMatchingError("TimeObservationsFrame's `observation_id` column has non-unique values!")

  if len(df[['time','observation']])!=len(df[['time','observation']].drop_duplicates()):
    raise HistoryMatchingError("TimeObservationsFrame contained the same observation made twice at the same time!")

  return df.copy()



def ValidateSummaryObservationsFrame(df):
  """Validates a SummaryObservationsFrame and returns a copy"""
  if not isinstance(df, pd.DataFrame):
    raise TypeError("SummaryObservationsFrame must be a pandas DataFrame!")

  if 'observation_id' not in df.columns:
    raise HistoryMatchingError("SummaryObservationsFrame must include `observation_id` column!")
  if 'observation' not in df.columns:
    raise HistoryMatchingError("SummaryObservationsFrame must include `observation` column!")
  if 'value' not in df.columns:
    raise HistoryMatchingError("SummaryObservationsFrame must include `value` column!")
  if 'stdev' not in df.columns:
    raise HistoryMatchingError("SummaryObservationsFrame must include `stdev` column!")

  if len(df.columns)!=4:
    raise HistoryMatchingError("SummaryObservationsFrame contains unexpected columns!")

  if not df['observation_id'].is_unique:
    raise HistoryMatchingError("SummaryObservationsFrame's `observation_id` column has non-unique values!")
  if not df['observation'].is_unique
    raise HistoryMatchingError("SummaryObservationsFrame contained non-unique measurements!")

  return df.copy()



def ValidateWrappedModelResults(results):
  if not isinstance(results, tuple) or len(results)!=2:
    raise HistoryMatchingError("Wrapped model must return a tuple with both a `time_points` and a `summary_points` DataFrame!")

  timepoint_results, summary_results = results

  if timepoint_results is not None:
    timepoint_results = ValidateTimeObservationsFrame(timepoint_results)
  if summary_results is not None:
    summary_results = ValidateSummaryObservationsFrame(summary_results)

  return timepoint_results, summary_results