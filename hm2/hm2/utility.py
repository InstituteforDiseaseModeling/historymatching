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



def drop_key(dic, key):
  """Returns a copy of the dictionary `dic` with the key `key` removed"""
  assert isinstance(dic, dict)
  dic = dic.copy()
  dic.pop(key)
  return dic