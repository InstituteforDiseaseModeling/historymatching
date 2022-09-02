#! /usr/bin/env python3

from pathlib import Path
import re

from history_matching.HistoryMatchingCut import HistoryMatchingCut

WORK_DIR = Path(__file__).parent.absolute()

# History Matching!
hm = HistoryMatchingCut(
    cut_folder="Cuts",
    iteration=int(re.search(r"[+-]?\d+", WORK_DIR.parts[-1]).group()),
    iterdir_parent=WORK_DIR.parent
)

### Cut #######################################################################
print("="*80, "\nCut\n", "="*80)
###############################################################################
(_, rejected_percent) = hm.cut(num_desired_candidates=250, constraint = None)

# TODO: Save to candidates or pass in filename
