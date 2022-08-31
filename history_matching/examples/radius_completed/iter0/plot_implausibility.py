#! /usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
import re

import pandas as pd

from history_matching import ProgressPlotting

WORK_DIR = Path(__file__).parent.absolute()
PLOT_DIR = WORK_DIR / "Plots"


def main(data_dir=None):

    if data_dir is None:
        data_dir = sorted(WORK_DIR.glob("Data*"), key = lambda d : str(d))[0]

    samples = pd.read_excel( data_dir / "Samples.xlsx" ).set_index("Sample")

    hm = ProgressPlotting(
        experiment_dir = WORK_DIR.parent,
        cut_dir = 'Cuts',
        samples = samples,
        iteration = int(re.search(r"[+-]?\d+", WORK_DIR.parts[-1]).group())
    )

    ###############################################################################
    print(f"{'*'*80}\nPlotting\n{'*'*80}")
    ###############################################################################
    fig = hm.plot(display = False)
    
    if not PLOT_DIR.exists(): PLOT_DIR.mkdir(parents=True)
    plot_file = PLOT_DIR / "implausibility.png"
    print(f"Saving figure to '{plot_file}'...")
    fig.savefig( plot_file )

    return


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-d", "--data-dir", type=Path, default=None)

    args = parser.parse_args()

    main(args.data_dir)
