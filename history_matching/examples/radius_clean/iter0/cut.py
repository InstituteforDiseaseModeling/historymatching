from history_matching.HistoryMatchingCut import HistoryMatchingCut
import os, re

# Example constraint function:
#def day_sum(row):
#    return row[['Env Ramp Up', 'Env Ramp Down', 'Env Cutoff']].sum() < 365


# History Matching!
hm = HistoryMatchingCut(
    cut_dir = 'Cuts',
    iteration = int(re.search(r'[+-]?\d+', os.getcwd()).group())
)


### Cut #######################################################################
print("="*80, "\nCut\n", "="*80)
###############################################################################
(_, rejected_percent) = hm.cut(num_desired_candidates=50, constraint = None)

# TODO: Save to candidates or pass in filename


