#! /usr/bin/env python3

from argparse import ArgumentParser
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hm2
import hm2.samplers as samplers
import hm2.utils as utils
from history_matching.emulators import BaseEmulator, LinearModel


WORK_DIR = Path(__file__).parent.absolute()

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
# logger.addHandler(logging.FileHandler(WORK_DIR / "log.txt"))

__prng: np.random.Generator = None

# https://www.desmos.com/calculator/to2fujjctw
# Note: parameter 'a' is in log2 space.
ACTUAL_A = -9.0
ACTUAL_B = 0.0
ACTUAL_C = -1.5
ACTUAL_D = 5.0

FEATURE_POINTS = np.arange(0, 33)

def main(num_observations=33, num_initial_samples=32):

    config = dict(
        max_iterations = 10,
        implausibility_threshold = 0.5,
        non_implausible_target = 0.01,
        username = "clorton"
    )

    config = hm2.Config(**config)

    parameter_space = pd.DataFrame(
        [("a", ACTUAL_A-2, ACTUAL_A+2), ("b", ACTUAL_B-5, ACTUAL_B+5), ("c", ACTUAL_C-1.5, ACTUAL_C+1.5), ("d", ACTUAL_D-5, ACTUAL_D+5)],
        columns=["parameter", "minimum", "maximum"],
    )
    observations = generate_observations(parameter_space, num_observations)
    initial_sample_points = samplers.lhs(parameter_space, num_initial_samples)
    initial_sample_points["iteration"] = 0

    situation = hm2.Situation(parameter_space, observations, initial_sample_points)

    recipe = hm2.Recipe()
    recipe.start_step_callback = start_callback
    recipe.run_simulators = run_model
    recipe.generate_emulator_for_feature = emulator_for_feature
    recipe.generate_next_sample_points = next_point_generation

    hm2.do_step(situation, recipe, config)

    situation.iteration = 1
    start_callback(situation)    

    return


def generate_observations(
    parameter_space: pd.DataFrame,
    n_observations: int = 8,
    percent_variance: float = 10.0,
) -> pd.DataFrame:

    values = model(ACTUAL_A, ACTUAL_B, ACTUAL_C, ACTUAL_D)
    # noise = __prng.normal(size=values.shape[0])
    # results = values + noise
    ts = FEATURE_POINTS
    columns = list(map(lambda t: f"t{t}", ts))
    # observations = pd.DataFrame(data={c: [v] for c, v in zip(columns, results)})

    samples = []
    for _ in range(n_observations):
        noise = __prng.normal(size=values.shape[0])
        results = values + noise
        samples.append(results)
    observations = pd.DataFrame(data=samples, columns=columns)
    statistics = utils.mean_and_variance_for_observations(observations=observations)

    figure = plt.figure(figsize=(16, 9), dpi=300)
    for sample in samples:
        plt.plot(sample)
    plt.plot(values, color="black", linewidth=2)
    figure.savefig(WORK_DIR / "observations.png")

    return statistics


def model(a, b, c, d):

    ts = FEATURE_POINTS
    output = system(ts, 2.0**a, b, c, d)  # a is in log2 space

    return output


def system(t, a=1.0 / 512, b=0, c=-1.5, d=5):

    results = a * (t**3) + b * (t**2) + c * t + d

    return results


def start_callback(situation: hm2.Situation) -> None:

    figure2 = plt.figure(figsize=(16, 9), dpi=300)
    for row in situation.sample_points.loc[
        situation.sample_points.iteration == situation.iteration
    ].itertuples():
        values = model(row.a, row.b, row.c, row.d)
        plt.plot(values)

    actual = model(ACTUAL_A, ACTUAL_B, ACTUAL_C, ACTUAL_D)
    plt.plot(actual, "ro")

    figure2.savefig(WORK_DIR / f"samples_{situation.iteration}.png")

    return


def run_model(
    iteration: int, test_points: pd.DataFrame, config: hm2.Config
) -> pd.DataFrame:

    results = []
    for row in test_points.itertuples():
        model_output = model(row.a, row.b, row.c, row.d)
        results.append([iteration, 0, row.a, row.b, row.c, row.d])
        results[-1].extend(model_output)

    columns = ["iteration", "replicate", "a", "b", "c", "d"]
    columns.extend(map(lambda t: f"t{t}", FEATURE_POINTS))
    results = pd.DataFrame(results, columns=columns)

    return results


def emulator_for_feature(
    feature: str,
    observations: pd.DataFrame,
    simulator_results: pd.DataFrame,
    config: hm2.Config,
) -> BaseEmulator:

    x = simulator_results[["a", "b", "c", "d"]]
    y = simulator_results[feature]
    emulator = LinearModel(x=x, y=y)
    emulator.train()

    return emulator


def next_point_generation(
    iteration: int,
    parameter_space: pd.DataFrame,
    observations: pd.DataFrame,
    emulator_bank: Dict[int, Dict[str, Any]],
    config: hm2.Config,
) -> Tuple[pd.DataFrame, float]:

    # pick new sample points with a grid
    proposed_sample_points = samplers.grid(
        parameter_space=parameter_space, samples_per_dimension=16
    )
    proposed_sample_points = samplers.random(
        parameter_space=parameter_space, n_samples=65536
    )

    total_proposed = len(proposed_sample_points)

    # predict features from each emulator, disqualify points outside range
    for feature, emulator in emulator_bank[iteration].items():

        prediction = emulator.predict(proposed_sample_points)
        target = observations[feature][
            0
        ]  # observations[feature] is a Series, we want a scalar
        plausible = ((prediction.value - target) / target) < 0.25
        if not any(plausible):
            logger.info(
                f"Last remaining sample points:\n{proposed_sample_points.head()}"
            )
            break
        proposed_sample_points = proposed_sample_points[plausible]
        logger.info(
            f"{len(proposed_sample_points)} remaining after '{feature}' = {100.0*len(proposed_sample_points)/total_proposed}%"
        )

    non_implausible_fraction = len(proposed_sample_points) / total_proposed

    return proposed_sample_points, non_implausible_fraction


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=20220923)
    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        help="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        default="INFO",
    )

    args = parser.parse_args()

    print(f"Current logging level is {logger.getEffectiveLevel()}.")
    logger.setLevel(args.verbosity)
    logger.info(f"New logging level is {logger.getEffectiveLevel()}.")

    __prng = np.random.default_rng(args.seed)

    main()
