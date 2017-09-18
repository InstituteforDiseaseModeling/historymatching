import unittest
from history_matching.newlib.ReferenceData import *
import os
import tempfile
from copy import copy

class TestReferenceData(unittest.TestCase):

    def setUp(self):
        self.data = '%s,%s,%s\nfieldName1,2,3\nfieldName2,5.5,6.5'
        self.wrong_column_count_data ='%s,%s,%s,%s\nfieldName1,2,3,4\nfieldName2,5.5,6.5,7.5'
        # dumb, but windows won't allow reopening of tempfiles just written...
        self.temp_file_handle, self.temp_file_name = tempfile.mkstemp()

    def tearDown(self):
        try:
            os.close(self.temp_file_handle)
        except OSError: # already closed by the test; was used in a context
            pass
        if os.path.exists(self.temp_file_name):
            os.remove(self.temp_file_name)

    def test_load_from_file_is_same_as_from_data(self):
        columns = tuple(ReferenceData.REQUIRED_KEYS)

        from_data = ReferenceData(data=(self.data % columns))

        with os.fdopen(self.temp_file_handle, 'w+') as temp_file:
            temp_file.write(self.data % columns)
        from_file = ReferenceData(filename=self.temp_file_name)

        self.assertEqual(from_data, from_file)

    def test_failure_if_filename_and_data_given(self):
        self.assertRaises(FileAndDataException, ReferenceData, **{'filename': 'StartTheGameAlready.csv', 'data': self.data})

    def test_incorrect_columns(self):
        # incorrect column(s)
        columns = copy(ReferenceData.REQUIRED_KEYS)
        columns[0] = 'InvalidColumnName'
        columns = tuple(columns)

        with os.fdopen(self.temp_file_handle, 'w+') as temp_file:
            temp_file.write(self.data % columns)
        self.assertRaises(InvalidFormat, ReferenceData, **{'filename': self.temp_file_name})

    def test_wrong_column_count(self):
        # wrong column count
        columns = ReferenceData.REQUIRED_KEYS
        columns.append(columns[0])
        columns = tuple(columns)

        with os.fdopen(self.temp_file_handle, 'w+') as temp_file:
            temp_file.write(self.wrong_column_count_data % columns)

        self.assertRaises(InvalidFormat, ReferenceData, **{'filename': self.temp_file_name})

    def test_detects_missing_field(self):
        # fail if a field (row) is requested that is not present in the file
        columns = tuple(ReferenceData.REQUIRED_KEYS)

        with os.fdopen(self.temp_file_handle, 'w+') as temp_file:
            temp_file.write(self.data % columns)
        from_file = ReferenceData(filename=self.temp_file_name)

        self.assertRaises(MissingField, from_file.value,  **{'field': 'Terraforming Mars'})
        self.assertRaises(MissingField, from_file.stddev, **{'field': '7 Wonders'})

    def test_gets_right_data(self):
        columns = tuple(ReferenceData.REQUIRED_KEYS)

        with os.fdopen(self.temp_file_handle, 'w+') as temp_file:
            temp_file.write(self.data % columns)
        from_file = ReferenceData(filename=self.temp_file_name)

        self.assertEqual(sorted(from_file.fields), sorted(['fieldName1', 'fieldName2']))

        field = 'fieldName1'
        self.assertEqual(from_file.value(field=field), 2)
        self.assertEqual(from_file.stddev(field=field), 3)

        field = 'fieldName2'
        self.assertEqual(from_file.value(field=field), 5.5)
        self.assertEqual(from_file.stddev(field=field), 6.5)

if __name__ == '__main__':
    unittest.main()