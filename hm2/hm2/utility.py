import pandas as pd
import sklearn

from hm2.error import HistoryMatchingError
from sklearn.preprocessing import MinMaxScaler



class Scaler:
  def __init__(self, data):
    if not isinstance(data, pd.DataFrame):
      raise TypeError("`data` must be a DataFrame!")
    self._columns = data.columns.tolist().copy()
    self._scaler = sklearn.preprocessing.MinMaxScaler().fit(data)

  def transform(self, data):
    if self._columns!=data.columns.tolist():
      raise HistoryMatchingError("Columns of DataFrame to be transformed don't match columns Scaler was built on!")
    return pd.DataFrame(self._scaler.transform(data), columns=self._columns)

  def __repr__(self):
    return str(pd.DataFrame({
        "feature": self._columns,
        "min": self._scaler.data_min_,
        "max": self._scaler.data_max_,
        "range": self._scaler.data_range_
    }))



def drop_key(dic, key, ignore_missing=False):
  """Returns a copy of the dictionary `dic` with the key `key` removed

  Args:
    dic - Dictionary to manipulated
    key - Key to remove
    ignore_missing - Don't throw error if key is missing
  """
  assert isinstance(dic, dict)
  dic = dic.copy()
  if key not in dic:
    if ignore_missing:
      return dic
    else:
      raise HistoryMatchingError(f"Key {key} was not in dict {dic}!")
  else:
    dic.pop(key)
    return dic