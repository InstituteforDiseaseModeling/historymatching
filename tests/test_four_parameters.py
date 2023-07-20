#! /usr/bin/env python3

import logging
import unittest
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from history_matching import Config, Recipe, Situation, grid_sampler, latin_hypercube_sampler, random_sampler, do_staircase
from history_matching import Config
from history_matching import Recipe
from history_matching import Situation
from history_matching import do_staircase
from history_matching import features_from_observations
from history_matching import latin_hypercube_sampler
from history_matching import mean_and_variance_for_observations
from history_matching import random_sampler
from history_matching.emulators import BaseEmulator
from history_matching.emulators import LinearModel

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

NUM_OBSERVATIONS = 33
NUM_INITIAL_SAMPLES = 32
NUM_RANDOM_SAMPLES = 8192

SEED = 20220923
__prng = np.random.default_rng(SEED)


class FourParameterTests(unittest.TestCase):
    def test_four_parameter(self):
        config = {
            "max_iterations": 10,
            "implausibility_threshold": 0.5,
            # non_implausible_target = 0.01,
            "non_implausible_target": 0.001,
            "candidates_per_iteration": 100,
            "username": "clorton",
        }

        config = Config(**config)

        parameter_space = pd.DataFrame(
            [("a", ACTUAL_A - 2, ACTUAL_A + 2), ("b", ACTUAL_B - 5, ACTUAL_B + 5), ("c", ACTUAL_C - 1.5, ACTUAL_C + 1.5), ("d", ACTUAL_D - 5, ACTUAL_D + 5)],
            columns=["parameter", "minimum", "maximum"],
        )
        observations = generate_observations(parameter_space, NUM_OBSERVATIONS)
        initial_sample_points = latin_hypercube_sampler(parameter_space, NUM_INITIAL_SAMPLES)
        initial_sample_points["iteration"] = 0

        situation = Situation(parameter_space, observations, initial_sample_points)

        recipe = Recipe()

        recipe.start_step_callback = start_callback

        recipe.run_simulators = run_model

        recipe.select_features = select_features

        recipe.generate_emulator_for_feature = emulator_for_feature
        recipe.generate_next_sample_points = next_point_generation

        do_staircase(situation, recipe, config)

        # make sure it ran at least 1 iteration but less than max
        assert 1 <= situation.iteration <= config.max_iterations, f"Invalid situation.iteration value: {situation.iteration}"
        # make sure implausible fraction < config value
        df = situation.sample_points
        implausible_fraction = len(df[df["iteration"] == situation.iteration + 1]) / NUM_RANDOM_SAMPLES
        assert implausible_fraction < config.non_implausible_target, f"implausible Fraction ({implausible_fraction}) >= config.non_implausible_target ({config.non_implausible_target})"

        situation.iteration = 1
        start_callback(situation)

        next_points = situation.sample_points[situation.sample_points.iteration == max(situation.sample_points.iteration)]
        print(f"Actual parameters: {ACTUAL_A}*x^3 + {ACTUAL_B}*x^2 + {ACTUAL_C}*x + {ACTUAL_D}")
        print(f"Observations:\n{observations}")
        print(f"Last selected points:\n{next_points}")

        # situation.save("situation.asdf")
        # copy = Situation.read("situation.asdf")

        return


def start_callback(situation: Situation) -> None:
    # figure2 = plt.figure(figsize=(16, 9), dpi=300)
    _ = plt.figure(figsize=(16, 9), dpi=300)
    for row in situation.sample_points.loc[situation.sample_points.iteration == situation.iteration].itertuples():
        values = model(row.a, row.b, row.c, row.d)
        plt.plot(values)

    actual = model(ACTUAL_A, ACTUAL_B, ACTUAL_C, ACTUAL_D)
    plt.plot(actual, "ro")

    # Comment for actual test run, uncomment savefig for manually run and debug
    # figure2.savefig(WORK_DIR / f"samples_{situation.iteration}.png")

    return


def run_model(iteration: int, test_points: pd.DataFrame, config: Config) -> pd.DataFrame:
    results = []
    for row in test_points.itertuples():
        model_output = model(row.a, row.b, row.c, row.d)
        results.append([iteration, 0, row.a, row.b, row.c, row.d])
        results[-1].extend(model_output)

    columns = ["iteration", "replicate", "a", "b", "c", "d"]
    columns.extend(f"t{t:02}" for t in FEATURE_POINTS)
    results = pd.DataFrame(results, columns=columns)

    return results


def select_features(iteration, observations, simulator_results, config) -> List[str]:
    all_features = features_from_observations(observations)

    by_iteration = {0: ["t00"], 1: ["t00", "t10"], 2: ["t00", "t10", "t20"], 3: ["t00", "t10", "t20", "t30"]}

    if iteration in by_iteration:
        selected_features = by_iteration[iteration]
    elif iteration < len(all_features):
        selected_features = all_features[0 : iteration + 1]
    else:
        selected_features = all_features

    print(f"Iteration {iteration} using features {selected_features}.")

    return selected_features


def emulator_for_feature(
    feature: str,
    observations: pd.DataFrame,
    simulator_results: pd.DataFrame,
    config: Config,
) -> BaseEmulator:
    x = simulator_results[["a", "b", "c", "d"]]
    y = simulator_results[feature]
    emulator = LinearModel(x=x, y=y)
    emulator.train()

    return emulator


def generate_observations(
    parameter_space: pd.DataFrame,
    n_observations: int = 8,
    percent_variance: float = 10.0,
) -> pd.DataFrame:
    values = model(ACTUAL_A, ACTUAL_B, ACTUAL_C, ACTUAL_D)
    # noise = __prng.normal(size=values.shape[0])
    # results = values + noise
    ts = FEATURE_POINTS
    columns = [f"t{t:02}" for t in ts]
    # observations = pd.DataFrame(data={c: [v] for c, v in zip(columns, results)})

    samples = []
    for _ in range(n_observations):
        noise = __prng.normal(size=values.shape[0])
        results = values + noise
        samples.append(results)
    observations = pd.DataFrame(data=samples, columns=columns)
    statistics = mean_and_variance_for_observations(observations=observations)

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


def next_point_generation(
    iteration: int,
    parameter_space: pd.DataFrame,
    observations: pd.DataFrame,
    emulator_bank: Dict[int, Dict[str, BaseEmulator]],
    config: Config,
) -> Tuple[pd.DataFrame, float]:
    # pick new sample points with a grid
    # proposed_sample_points = grid_sampler(
    #     parameter_space=parameter_space, samples_per_dimension=16
    # )
    proposed_sample_points = random_sampler(parameter_space=parameter_space, n_samples=NUM_RANDOM_SAMPLES)  # 65536

    total_proposed = len(proposed_sample_points)

    # predict features from each emulator, disqualify points outside range
    for feature, emulator in emulator_bank[iteration].items():
        prediction = emulator.predict(proposed_sample_points)
        target = observations.means[feature]  # observations[feature] is a Series, we want a scalar
        plausible = ((prediction.value - target) / target) < 0.25
        if not any(plausible):
            logger.info(f"Last remaining sample points:\n{proposed_sample_points.head()}")
            break
        proposed_sample_points = proposed_sample_points[plausible]
        logger.info(f"{len(proposed_sample_points)} remaining after '{feature}' = {100.0*len(proposed_sample_points)/total_proposed}%")

    non_implausible_fraction = len(proposed_sample_points) / total_proposed

    return proposed_sample_points, non_implausible_fraction


if __name__ == "__main__":
    unittest.main()
