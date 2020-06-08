import pandas as pd

from .error import *



def ValidateParameterInfoFrame(df, copy=True):
  if not isinstance(df, pd.DataFrame):
    raise HMNotADataFrame("ParameterInfoFrame")

  if 'name' not in df.columns: raise HMMissingColumn("ParameterInfoFrame", "name")
  if 'min'  not in df.columns: raise HMMissingColumn("ParameterInfoFrame", "min")
  if 'max'  not in df.columns: raise HMMissingColumn("ParameterInfoFrame", "max")
  if len(df.columns)>3: raise HMExtraColumns("ParameterInfoFrame")

  if not (df['min']<=df['max']).all():
    raise HMMaxLessThanMin("ParameterInfoFrame")

  return df.copy() if copy else df



def ValidateParameterSamplesFrame(df, copy=True):
  """Validates a parameter sampling DataFrame and returns a copy"""
  if not isinstance(df, pd.DataFrame):
    raise HMNotADataFrame("ParameterSamplesFrame")
  if 'param_id' not in df.columns: raise HMMissingColumn("ParameterSamplesFrame", "param_id")
  return df.copy() if copy else df



def ValidateObservationsFrame(df, copy=True, frame_name="ObservationsFrame"):
  """Validates an ObservationsFrame and returns a copy"""
  if not isinstance(df, pd.DataFrame):
    raise HMNotADataFrame(f"{frame_name}")

  if "observation_id" not in df.columns: raise HMMissingColumn(f"{frame_name}", "observation_id")
  if "time"           not in df.columns: raise HMMissingColumn(f"{frame_name}", "time")
  if "observation"    not in df.columns: raise HMMissingColumn(f"{frame_name}", "observation")
  if "value"          not in df.columns: raise HMMissingColumn(f"{frame_name}", "value")
  if "stdev"          not in df.columns: raise HMMissingColumn(f"{frame_name}", "stdev")
  if len(df.columns)>5: raise HMExtraColumns("ParameterInfoFrame")

  if any(df['time'].isna()):
    raise HistoryMatchingError("ObservationFrame's `time` column contained NaNs!")

  if not df.groupby('observation')['time'].is_monotonic_increasing.all():
    raise HMTimeIsNotMonotonic(frame_name)

  if not df['observation_id'].is_unique:
    raise HMObservationIDsNotUnique(frame_name)

  unique_key = ['time','observation']
  if len(df[unique_key])!=len(df[unique_key].drop_duplicates()):
    raise HMTwoObservationsAtOneTime(frame_name)

  return df.copy() if copy else df

def ValidateSimObservationsFrame(df, copy=True):
  return ValidateObservationsFrame(df, copy=copy, frame_name="SimObservationsFrame")

def ValidateSimFrame(df, copy=True):
  if not isinstance(df, pd.DataFrame):
    raise HMNotADataFrame("SimFrame")

  if "param_id"       not in df.columns: raise HMMissingColumn("SimFrame", "param_id")
  if "replicate"      not in df.columns: raise HMMissingColumn("SimFrame", "replicate")
  if "time"           not in df.columns: raise HMMissingColumn("SimFrame", "time")
  if "observation"    not in df.columns: raise HMMissingColumn("SimFrame", "observation")
  if "value"          not in df.columns: raise HMMissingColumn("SimFrame", "value")
  if "stdev"          not in df.columns: raise HMMissingColumn("SimFrame", "stdev")
  if "observation_id" not in df.columns: raise HMMissingColumn("SimFrame", "observation_id")
  if len(df.columns)>7: raise HMExtraColumns("SimFrame")

  unique_key = ['param_id','replicate','time','observation_id','observation']
  if len(df[unique_key])!=len(df[unique_key].drop_duplicates()):
    raise HistoryMatchingError("SimFrame contained the same observation made twice at the same time within a given parameter+replicate!")

  return df.copy() if copy else df



def ValidateMatchedFrame(df, copy=True):
  if not isinstance(df, pd.DataFrame):
    raise HMNotADataFrame("MatchedFrame")

  if "observation_id_a" not in df.columns: raise HMMissingColumn("MatchedFrame", "observation_id_a")
  if "replicate"        not in df.columns: raise HMMissingColumn("MatchedFrame", "replicate")
  if "value"            not in df.columns: raise HMMissingColumn("MatchedFrame", "value")
  if "stdev"            not in df.columns: raise HMMissingColumn("MatchedFrame", "stdev")
  if "param_id"         not in df.columns: raise HMMissingColumn("MatchedFrame", "param_id")
  if len(df.columns)>5: raise HMExtraColumns("SimFrame")

  unique_key = ['param_id','replicate','observation_id_a']
  if len(df[unique_key])!=len(df[unique_key].drop_duplicates()):
    raise HistoryMatchingError("MatchedFrame contained the same observation made twice within a given parameter+replicate!")

  return df.copy() if copy else df
















# def ValidateEmulatorInput(df, copy=True):
#   if not isinstance(df, pd.DataFrame):
#     raise HMNotADataFrame("SingleEmulatorInput")

#   _CheckColumns(df,['param_id','replicate','observation','value','stdev'], "SingleEmulatorInput", optional_cols=['aobservation_id'])

#   if len(df["observation"].unique())!=1:
#     raise HistoryMatchingError("SingleEmulatorInput must have a single observation type!")

#   if 'aobservation_id' in df.columns and len(df["aobservation_id"].unique())!=1:
#     raise HistoryMatchingError("SingleEmulatorInput for a time series must refer to only a single time point!")

#   return df.copy() if copy else df