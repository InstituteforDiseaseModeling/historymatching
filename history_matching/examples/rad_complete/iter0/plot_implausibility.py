#! /usr/bin/env python3

from pathlib import Path
import re

import pandas as pd

from history_matching import ProgressPlotting

# Example constraint function:
#def day_sum(row):
#    return row[['Env Ramp Up', 'Env Ramp Down', 'Env Cutoff']].sum() < 365

WORK_DIR = Path(__file__).parent.absolute()
DATA_DIR = list((WORK_DIR.parent / "iter0").glob("Data_*"))[0]

samples = pd.read_excel(DATA_DIR / "Samples.xlsx").set_index("Sample")

# History Matching!
hm = ProgressPlotting(
    cut_dir = 'Cuts',
    samples = samples,
    iteration = int(re.search(r'[+-]?\d+', Path.cwd().parts[-1]).group())
)

### Cut #######################################################################
print(f"{'='*80}\nPlotting\n{'='*80}")
###############################################################################
fig = hm.plot()

# TODO: Save to candidates or pass in filename
