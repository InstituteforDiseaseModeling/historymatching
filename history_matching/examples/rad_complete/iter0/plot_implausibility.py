from history_matching import ProgressPlotting
import pandas as pd
import os, re

# Example constraint function:
#def day_sum(row):
#    return row[['Env Ramp Up', 'Env Ramp Down', 'Env Cutoff']].sum() < 365


samples = pd.read_excel(os.path.join('..', 'iter0', 'Data_20171214_180228', 'Samples.xlsx')) \
    .set_index('Sample')

# History Matching!
hm = ProgressPlotting(
    cut_dir = 'Cuts',
    samples = samples,
    iteration = int(re.search(r'[+-]?\d+', os.getcwd()).group())
)

### Cut #######################################################################
print("="*80, "\nPlotting\n", "="*80)
###############################################################################
fig = hm.plot()

# TODO: Save to candidates or pass in filename


