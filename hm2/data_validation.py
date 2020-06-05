import pandas as pd

from .error import HistoryMatchingError



def _CheckColumns(df, required_cols, df_name, optional_cols=None, no_extra=True):
  for col in required_cols:
    if col not in df.columns:
      raise HistoryMatchingError(f"{df_name} must include `{col}` column!")

  #What columns remaing after we have accounted for the required ones?
  remaining_columns = set(df.columns.tolist()) - set(required_cols)
  #Subtract off any optional columns
  if optional_cols is not None:
    remaining_columns -= set(optional_cols)
  #Now, check to see if there are any extra
  if no_extra and len(remaining_columns)>0:
    raise HistoryMatchingError("{df_name} had extra columns! Found {cf}. Required {cr} with {opt} optionally permitted! Extra columns were {extra}.".format(
      df_name=df_name,
      cf=sorted(df.columns.tolist()),
      cr=sorted(required_cols),
      opt=sorted(optional_cols) if optional_cols else [],
      extra=remaining_columns
    ))



def ValidateParameterInfoFrame(df, copy=True):
  if not isinstance(df, pd.DataFrame):
    raise TypeError("ParameterInfoFrame must be a pandas DataFrame!")
  _CheckColumns(df, ['name','min','max'], 'ParameterInfoFrame')
  if not (df['min']<=df['max']).all():
    raise HistoryMatchingError("All entries of ParameterInfoFrame `min` column must be less than their corresponding `max` values!")
  return df.copy() if copy else df



def ValidateParameterSamplesFrame(df, copy=True):
  """Validates a parameter sampling DataFrame and returns a copy"""
  if not isinstance(df, pd.DataFrame):
    raise TypeError("ParameterSamplesFrame must be a pandas DataFrame!")
  if 'param_id' not in df.columns:
    raise HistoryMatchingError("ParameterSamplesFrame must include `param_id` column!")
  return df.copy() if copy else df



def ValidateTimeObservationsFrame(df, copy=True):
  """Validates an TimeObservationsFrame and returns a copy"""
  if df is None:
    return None

  if not isinstance(df, pd.DataFrame):
    raise TypeError("TimeObservationsFrame must be a pandas DataFrame!")

  _CheckColumns(df, ['observation_id','time','observation','value','stdev'], 'TimeObservationsFrame')

  if not df.groupby('observation')['time'].is_monotonic_increasing.all():
    raise HistoryMatchingError("TimeObservationsFrame's `time` column is not monotonically increasing!")

  if not df['observation_id'].is_unique:
    raise HistoryMatchingError("TimeObservationsFrame's `observation_id` column has non-unique values!")

  unique_key = ['time','observation']
  if len(df[unique_key])!=len(df[unique_key].drop_duplicates()):
    raise HistoryMatchingError("TimeObservationsFrame contained the same observation made twice at the same time!")

  return df.copy() if copy else df



def ValidateSummaryObservationsFrame(df, copy=True):
  """Validates a SummaryObservationsFrame and returns a copy"""
  if df is None:
    return None

  if not isinstance(df, pd.DataFrame):
    raise TypeError("SummaryObservationsFrame must be a pandas DataFrame!")

  _CheckColumns(df, ['observation_id','observation','value','stdev'], 'SummaryObservationsFrame')

  if not df['observation'].is_unique:
    raise HistoryMatchingError("SummaryObservationsFrame contained non-unique measurements!")
  if not df['observation_id'].is_unique:
    raise HistoryMatchingError("SummaryObservationsFrame contained non-unique `observation_id`!")

  return df.copy() if copy else df



def ValidateObservationsFrame(df, copy=True):
  if 'time' in df.columns:
    return ValidateTimeObservationsFrame(df=df, copy=copy)
  else:
    return ValidateSummaryObservationsFrame(df=df, copy=copy)



def ValidateObservationFrames(results, copy=True):
  if not isinstance(results, tuple) or len(results)!=2:
    raise TypeError("Wrapped model must return a tuple with both a TimeObservationsFrame and a SummaryObservationsFrame, or None for one of them!")

  time_results = ValidateTimeObservationsFrame(results[0], copy=copy)
  summary_results = ValidateSummaryObservationsFrame(results[1], copy=copy)

  return time_results, summary_results



def ValidateTimeSimFrame(df, copy=True):
  if df is None:
    return None

  if not isinstance(df, pd.DataFrame):
    raise TypeError("TimeSimFrame must be a pandas DataFrame!")

  _CheckColumns(df, ['param_id','replicate','time','observation','value','stdev','observation_id'], 'TimeSimFrame')

  unique_key = ['param_id','replicate','time','observation_id','observation']
  if len(df[unique_key])!=len(df[unique_key].drop_duplicates()):
    raise HistoryMatchingError("TimeSimFrame contained the same observation made twice at the same time within a given parameter+replicate!")

  return df.copy() if copy else df


def ValidateSummarySimFrame(df, copy=True):
  if df is None:
    return None

  if not isinstance(df, pd.DataFrame):
    raise TypeError("SummarySimFrame must be a pandas DataFrame!")

  _CheckColumns(df,['param_id','replicate','observation','value','stdev','observation_id'], "SummarySimFrame")

  unique_key = ['param_id','replicate','observation']
  if len(df[unique_key])!=len(df[unique_key].drop_duplicates()):
    raise HistoryMatchingError("SummarySimFrame contained the same observation made twice within a given parameter+replicate!")

  return df.copy() if copy else df



def ValidateSimFrame(df, copy=True):
  if df is None:
    return None

  if isinstance(df,tuple):
    if len(df)!=2:
      raise HistoryMatchingError("Simulation frame tuples must have length 2!")
    return ValidateTimeSimFrame(df=df[0], copy=copy), ValidateSummarySimFrame(df=df[1], copy=copy)
  elif not isinstance(df, pd.DataFrame):
    raise TypeError("Simulation Frame must be a pandas DataFrame!")
  elif 'time' in df.columns:
    return ValidateTimeSimFrame(df=df, copy=copy)
  else:
    return ValidateSummarySimFrame(df=df, copy=copy)



def ValidateMatchedFrame(df, copy=True):
  if not isinstance(df, pd.DataFrame):
    raise TypeError("MatchedFrame must be a pandas DataFrame!")

  _CheckColumns(df, ['observation_id_a','replicate','value','stdev','param_id'], 'MatchedFrame')

  unique_key = ['param_id','replicate','observation_id_a']
  if len(df[unique_key])!=len(df[unique_key].drop_duplicates()):
    raise HistoryMatchingError("MatchedFrame contained the same observation made twice within a given parameter+replicate!")

  return df.copy() if copy else df
















def ValidateEmulatorInput(df, copy=True):
  if not isinstance(df, pd.DataFrame):
    raise TypeError("SingleEmulatorInput must be a pandas DataFrame!")

  _CheckColumns(df,['param_id','replicate','observation','value','stdev'], "SingleEmulatorInput", optional_cols=['aobservation_id'])

  if len(df["observation"].unique())!=1:
    raise HistoryMatchingError("SingleEmulatorInput must have a single observation type!")

  if 'aobservation_id' in df.columns and len(df["aobservation_id"].unique())!=1:
    raise HistoryMatchingError("SingleEmulatorInput for a time series must refer to only a single time point!")

  return df.copy() if copy else df