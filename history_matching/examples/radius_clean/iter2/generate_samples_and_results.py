#! /usr/bin/env python3

from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pyDOE import lhs

from history_matching.quick_read import quick_read_xl

WORK_DIR = Path(__file__).parent.absolute()


def main():

    iteration = int(re.search(r"[+-]?\d+", WORK_DIR.parts[-1]).group())
    experiment_dir = WORK_DIR / f"Data_{datetime.now():%Y%m%d_%H%M%S}"
    N_samples = 25  # <-- Only applies to iteration 0

    params = quick_read_xl(WORK_DIR.parent / "Params.xlsx", "Params").set_index("Name")

    samples = choose_samples(N_samples, iteration, params)

    # 2-norm with added random noise for this example.  Just one "replicate" per sample here.
    mu = 0
    sigma = 1
    results = np.sqrt(np.square(samples).sum(axis=1)).to_frame(name="Sim_Result")
    results["Sim_Result"] += np.random.normal(mu, sigma, results.shape[0])

    # Every Sample must have a unique simulation id to differentiate it from other replicates, if replicates are used
    results["Sim_Id"] = results.index.values

    if not experiment_dir.exists(): experiment_dir.mkdir(parents=True)

    samples_file = experiment_dir / "Samples.xlsx"
    writer = pd.ExcelWriter(samples_file)
    samples.to_excel(writer, sheet_name="Samples")
    params.to_excel(writer, sheet_name="Params")
    print(f"Saving samples and parameters to '{samples_file}'...")
    writer.save()

    results_file = experiment_dir / "Results.xlsx"
    print(f"Saving results to '{results_file}'...")
    results.to_excel(results_file)

    return


def choose_samples(num_samples, iteration, params):

    if iteration == 0:

        N_dim = params.shape[0]
        samples = pd.DataFrame( lhs(N_dim, samples=num_samples), columns=params.index.tolist() )

        for param_name in samples.columns.values:
            pmin, pmax = (params.loc[param_name, "Min"], params.loc[param_name, "Max"])
            samples[param_name] = pmin + samples[param_name] * (pmax - pmin)

        samples.index.name = "Sample"

    else:

        samples_file = WORK_DIR.parent / f"iter{iteration-1}" / f"Candidates_for_iter{iteration}.csv"
        print(f"Reading samples from '{samples_file}'...")
        samples = pd.read_csv( samples_file, skipinitialspace=True, )
        samples.index.name = "Sample"

    return samples


if __name__ == "__main__":
    main()
