import pandas as pd
from StringIO import StringIO


class FileAndDataException(Exception): pass
class InvalidFormat(Exception): pass
class MissingField(Exception): pass

# ck4, test and write tests
class ReferenceData(object):

    FIELD_KEY = 'field'
    VALUE_KEY = 'value'
    STDDEV_KEY = 'stddev'
    REQUIRED_KEYS = [FIELD_KEY, VALUE_KEY, STDDEV_KEY]

    def __init__(self, filename=None, data=None):
        if (filename is None) ^ (data is None):
            if filename:
                obj = filename
            else:
                obj = StringIO(data.strip())
            self._data = pd.read_csv(obj)
        else:
            raise FileAndDataException('Must provide a reference data filename or csv data string, not both.')

        available_keys = self._data.keys()
        if sorted(self.REQUIRED_KEYS) != sorted(available_keys):
            raise InvalidFormat('ReferenceData csv columns must be: %s' % self.REQUIRED_KEYS)
        self.fields = sorted(list(self._data[self.FIELD_KEY]))

    def value(self, field):
        if field not in self.fields:
            raise MissingField('No reference data for field: %s' % field)
        val = self._data.loc[self._data[self.FIELD_KEY] == field][self.VALUE_KEY].values[0]
        return float(val)

    def stddev(self, field):
        if field not in self.fields:
            raise MissingField('No reference data for field: %s' % field)
        val = self._data.loc[self._data[self.FIELD_KEY] == field][self.STDDEV_KEY].values[0]
        return float(val)

    def __eq__(self, other):
        return (self.fields == other.fields) and (str(self._data) == str(other._data))
