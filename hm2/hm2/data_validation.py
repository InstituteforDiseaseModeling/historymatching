import pandas as pd

from .error import HistoryMatchingError



def _CheckColumns(df, cols, df_name, no_extra=True):
  for col in cols:
    if col not in df.columns:
      raise HistoryMatchingError(f"{df_name} must include `{col}` column!")
  if no_extra and len(df.columns)!=len(cols):
    raise HistoryMatchingError(f"{df_name} had extra columns! Found {len(df.columns)}, expected {len(cols)}!")



def ValidateParameterInfoFrame(df):
  if not isinstance(df, pd.DataFrame):
    raise TypeError("ParameterInfoFrame must be a pandas DataFrame!")
  _CheckColumns(df, ['name','min','max'], 'ParameterInfoFrame')
  if not (df['min']<=df['max']).all():
    raise HistoryMatchingError("All entries of ParameterInfoFrame `min` column must be less than their corresponding `max` values!")
  return df.copy()



def ValidateParameterSamplesFrame(df):
  """Validates a parameter sampling DataFrame and returns a copy"""
  if not isinstance(df, pd.DataFrame):
    raise TypeError("ParameterSamplesFrame must be a pandas DataFrame!")
  if 'param_id' not in df.columns:
    raise HistoryMatchingError("ParameterSamplesFrame must include `param_id` column!")
  return df.copy()



def ValidateTimeObservationsFrame(df):
  """Validates an TimeObservationsFrame and returns a copy"""
  if df is None:
    return None

  if not isinstance(df, pd.DataFrame):
    raise TypeError("TimeObservationsFrame must be a pandas DataFrame!")

  _CheckColumns(df, ['observation_id','time','observation','value','stdev'], 'TimeObservationsFrame')

  if not df['time'].is_monotonic_increasing:
    raise HistoryMatchingError("TimeObservationsFrame's `time` column is not monotonically increasing!")

  if not df['observation_id'].is_unique:
    raise HistoryMatchingError("TimeObservationsFrame's `observation_id` column has non-unique values!")

  unique_key = ['time','observation']
  if len(df[unique_key])!=len(df[unique_key].drop_duplicates()):
    raise HistoryMatchingError("TimeObservationsFrame contained the same observation made twice at the same time!")

  return df.copy()



def ValidateSummaryObservationsFrame(df):
  """Validates a SummaryObservationsFrame and returns a copy"""
  if df is None:
    return None

  if not isinstance(df, pd.DataFrame):
    raise TypeError("SummaryObservationsFrame must be a pandas DataFrame!")

  _CheckColumns(df, ['observation','value','stdev'], 'SummaryObservationsFrame')

  if not df['observation'].is_unique:
    raise HistoryMatchingError("SummaryObservationsFrame contained non-unique measurements!")

  return df.copy()



def ValidateWrappedModelResults(results):
  if not isinstance(results, tuple) or len(results)!=2:
    raise HistoryMatchingError("Wrapped model must return a tuple with both a TimeObservationsFrame and a SummaryObservationsFrame, or None for one of them!")

  time_results, summary_results = results
  time_results = ValidateTimeObservationsFrame(time_results)
  summary_results = ValidateSummaryObservationsFrame(summary_results)

  return time_results, summary_results



def ValidateTimeStandardAnalysisWithReplicatesFrame(df):
  if df is None:
    return None

  if not isinstance(df, pd.DataFrame):
    raise TypeError("TimeStandardAnalysisWithReplicatesFrame must be a pandas DataFrame!")

  _CheckColumns(df, ['param_id','replicate','observation','value','stdev','aobservation_id'], 'TimeStandardAnalysisWithReplicatesFrame')

  unique_key = ['param_id','replicate','aobservation_id','observation']
  if len(df[unique_key])!=len(df[unique_key].drop_duplicates()):
    raise HistoryMatchingError("TimeStandardAnalysisWithReplicatesFrame contained the same observation made twice at the same time within a given parameter+replicate!")

  return df.copy()


def ValidateSummaryStandardAnalysisWithReplicatesFrame(df):
  if df is None:
    return None

  if not isinstance(df, pd.DataFrame):
    raise TypeError("SummaryStandardAnalysisWithReplicatesFrame must be a pandas DataFrame!")

  _CheckColumns(df,['param_id','replicate','observation','value','stdev'], "SummaryStandardAnalysisWithReplicatesFrame")

  unique_key = ['param_id','replicate','observation']
  if len(df[unique_key])!=len(df[unique_key].drop_duplicates()):
    raise HistoryMatchingError("SummaryStandardAnalysisWithReplicatesFrame contained the same observation made twice within a given parameter+replicate!")

  return df.copy()