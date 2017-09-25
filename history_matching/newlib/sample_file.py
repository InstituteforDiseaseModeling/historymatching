import os
import pandas as pd
from history_matching.newlib.quick_read import quick_read

# ck4, Homogenized candidate and sample files. They should now both have:
# a sheet: Values with cols: id, <all params> . These are used as samples for both.

# used for reading candidate and sample files
class SampleFile(object):
    def __init__(self, filename):
        self.filename = os.path.abspath(filename)
        self.samples = self._read_samples()

    def _read_samples(self):
        samples = quick_read(self.filename, 'Values')
        samples.index.name = 'id'
        return samples

    @classmethod
    def write(cls, samples, filename):
        writer = pd.ExcelWriter(filename)
        samples.to_excel(writer, sheet_name='Values')
        # ck4, doesn't appear to be needed in samples file. params.to_excel(writer, sheet_name='Params')
        writer.save()
