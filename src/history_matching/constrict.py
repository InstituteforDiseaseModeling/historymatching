# constrict.py

import time
from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd

from history_matching import Config
from history_matching import latin_hypercube_sampler as lhs
from history_matching.emulators import BaseEmulator

_tictimes = []


def tic() -> int:
    t = time.time_ns()
    _tictimes.append(t)
    return t


def toc(msg: str = "", dopop: bool = True) -> int:
    t = time.time_ns()
    s = _tictimes.pop() if dopop else _tictimes[-1]
    elapsed = t - s
    print(f"{msg if msg else 'Elapsed: '} {elapsed} ns")
    return elapsed


def npg(
    iteration: int,
    parameter_space: pd.DataFrame,
    observations: pd.DataFrame,
    emulator_bank: Dict[int, Dict[str, BaseEmulator]],
    config: Config,
) -> Tuple[pd.DataFrame, float]:
    """Next Point Generation based on existing emulators and observations."""

    max_nSamples = 1000  # TODO - add to configuration?

    num_desired_candidates = config.candidates_per_iteration
    non_implausible_candidates = pd.DataFrame()
    num_candidates_considered = 0

    while (num_non_implausible_candidates := len(non_implausible_candidates)) < num_desired_candidates:
        if num_candidates_considered == 0:
            nSamples = num_desired_candidates
        else:
            nSamples = int(1.25 * (num_desired_candidates - num_non_implausible_candidates) * num_candidates_considered / num_non_implausible_candidates)

        nSamples = min(max_nSamples, nSamples)

        print(f"Generating {nSamples} new samples...")
        tic()
        new_samples = lhs(parameter_space, nSamples)
        toc(f"lhs({nSamples}): ")
        # TODO - filter with "business rules" constraint, e.g. initial cases <= 10% of population
        # new_samples = new_samples[constraint(new_samples)]
        new_candidates = pd.DataFrame(new_samples)

        plausibility = test_plausibility(new_candidates, emulator_bank, observations, config)
        num_candidates_considered += nSamples

        plausible_candidates = new_candidates[plausibility]
        print(f"Found {len(plausible_candidates)} plausible candidates.")

        non_implausible_candidates = pd.concat([non_implausible_candidates, plausible_candidates])
        print(f"{len(non_implausible_candidates)} non-implausible candidates so far from {num_candidates_considered} candidates ({len(non_implausible_candidates)/num_candidates_considered}).")

    plausible_fraction = len(non_implausible_candidates) / num_candidates_considered

    return non_implausible_candidates, plausible_fraction


def test_plausibility(candidates: pd.DataFrame, emulator_bank: Dict[int, Dict[str, BaseEmulator]], observations: pd.DataFrame, config: Config) -> pd.Series:
    """Run non-implausible candidates through each emulator and compare to observations."""

    # *** "non-implausible" is too hard to track - particularly when negated.
    # *** Use "plausible" instead, even if technically inaccurate.

    # Initially, all candidates are plausible
    plausible = np.ones(len(candidates), dtype=bool)

    for iteration in sorted(emulator_bank.keys(), reverse=True):
        for feature in emulator_bank[iteration]:
            emulator = emulator_bank[iteration][feature]
            tic()
            # candidates[f"{feature}_estimate"] = emulator.predict(candidates)
            predictions = emulator.predict(candidates[plausible])
            toc(f"{feature}_estimate: ")

            mean = predictions["value"]
            variance = predictions["variance"]

            # implausibility = abs(mean - observation) / sqrt(variance + observation_variance + discrepancy_variance)
            observation = observations.means[feature]
            observation_variance = observations.variances[feature]
            discrepancy_variance = config.discrepancy_variance
            implausibility = abs(mean - observation) / np.sqrt(variance + observation_variance + discrepancy_variance)

            implausible = implausibility > config.implausibility_threshold

            # plausible candidates are _still_ plausible only if _not_ determined to be implausible
            plausible[plausible] &= np.logical_not(implausible)

    return plausible
