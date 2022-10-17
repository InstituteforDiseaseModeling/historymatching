#! /usr/bin/env python3

import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hm2
from hm2.situation import Situation
import hm2.utils as utils
import hm2.samplers as samplers

from history_matching.emulators import BaseEmulator, LinearModel

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())  # defaults to stderr

WORK_DIR = Path(__file__).parent.absolute()
FIGURE_DIR = WORK_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

__prng = np.random.default_rng()


SLOPE = 0.75
ACTUAL_B = 42.0

def system(x):

    y = (x*SLOPE) + ACTUAL_B

    return y


def main():

    params = dict(
        max_iterations = 10,
        implausibility_threshold = 1.0,
        non_implausible_target = 0.01
    )
    config = hm2.Config(**params)

    parameter_space = pd.DataFrame([["b", ACTUAL_B-5, ACTUAL_B+5]], columns=["parameter", "minimum", "maximum"])

    figure_1(np.arange(0, 35), system(np.arange(0, 35)))

    taps = np.array([5.0, 8.0, 13.0, 21.0])
    solution = system(taps)
    figure_2(np.arange(0, 35), system(np.arange(0, 35)), taps, solution)
    features = list(map(lambda x: f"x{int(x)}", taps))
    # raw_observations = pd.DataFrame(data=[__prng.normal(solution) for _ in range(82)], columns=features)
    raw_observations = pd.DataFrame(data=[__prng.normal(solution) for _ in range(5)], columns=features)
    observations = utils.mean_and_variance_for_observations(raw_observations)
    figure_3(np.arange(0, 35), system(np.arange(0, 35)), taps, observations)

    initial_sample_points = samplers.lhs(parameter_space, 11)
    initial_sample_points["iteration"] = 0

    situation = hm2.Situation(parameter_space, observations, initial_sample_points)

    recipe = hm2.Recipe()

    def start_callback(situation: Situation) -> None:

        print(f"Starting iteration {situation.iteration}")

        figure_4(situation)

        return

    recipe.start_step_callback = start_callback

    def run_model(iteration: int, points: pd.DataFrame, config: hm2.Config) -> pd.DataFrame:

        results = []
        for b in points.b:
            results.append([iteration, 0, b])
            results[-1].extend((SLOPE*taps)+b)

        results = pd.DataFrame(data=results, columns=["iteration", "replicate", "b"]+features)

        return results

    recipe.run_simulators = run_model

    def generate_emulator(feature: str, observations: pd.DataFrame, simulator_results: pd.DataFrame, config: hm2.Config) -> BaseEmulator:

        x = pd.DataFrame()
        x["b"] = simulator_results.b
        y = pd.DataFrame()
        y[feature] = simulator_results[feature]

        emulator = LinearModel(x, y)
        emulator.train()

        return emulator

    recipe.generate_emulator_for_feature = generate_emulator

    def next_point_selection(iteration: int, parameter_space: pd.DataFrame, observations: pd.DataFrame, emulator_bank: Dict[int, Dict[str, BaseEmulator]], config: hm2.Config) -> Tuple[pd.DataFrame, float]:

        points = pd.DataFrame(columns=["iteration", "b"])
        candidate_count = 0
        effective_threshold = config.implausibility_threshold * (0.95**iteration)
        while len(points) < 11:
            candidates = samplers.random(parameter_space, 100)
            candidate_count += 100
            # candidates["iteration"] = iteration
            candidates["plausible"] = True
            for it in reversed(range(iteration+1)):
                emulators = emulator_bank[it]
                for feature, emulator in emulators.items():
                    prediction = emulator.predict(candidates.b)
                    difference = np.abs(prediction.value - np.float64(observations.loc[observations.statistic == "mean"][feature]))
                    # candidates.plausible &= (difference < config.implausibility_threshold)
                    candidates.plausible &= (difference < effective_threshold)
            points = pd.concat([points, candidates[candidates.plausible == True]])

        points.drop(columns="plausible", inplace=True)
        plausible_fraction = len(points) / candidate_count

        return points, plausible_fraction

    recipe.generate_next_sample_points = next_point_selection

    recipe.end_step_callback = lambda s: print(f"Finished iteration {s.iteration}")

    hm2.do_staircase(situation, recipe, config)    # do all steps

    next_points = situation.sample_points[situation.sample_points.iteration == max(situation.sample_points.iteration)].b
    print(f"Actual intercept = {ACTUAL_B}")
    # TODO - fix calculation
    # print(f"Correct intercept, based on noisy sampling = {np.float64(observations[observations.statistic == 'mean'].x5) - (SLOPE*5)}")
    print(observations)
    print(f"Min/max of last selected test points: {next_points.min():0.04f}/{next_points.max():0.04f}")
    print(f"Selected the following points for next iteration:\n{next_points}")

    figure_5(next_points, observations)

    return


def figure_1(x: np.ndarray, y: np.ndarray) -> None:

    figure = plt.figure(figsize=(16, 9), dpi=300)
    plt.plot(x, y)

    figure.savefig(FIGURE_DIR / "figure1 - system.png")

    return


def figure_2(x: np.ndarray, y: np.ndarray, taps: np.ndarray, solution: np.ndarray) -> None:

    figure = plt.figure(figsize=(16, 9), dpi=300)
    plt.plot(x, y)
    for tap, value in zip(taps, solution):
        plt.plot([tap, tap], [y.min(), value])

    figure.savefig(FIGURE_DIR / "figure2 - taps.png")

    return


def figure_3(x: np.ndarray, y: np.ndarray, taps: np.ndarray, observations: pd.DataFrame) -> None:

    figure = plt.figure(figsize=(16, 9), dpi=300)
    plt.plot(x, y)
    for row in observations.itertuples():
        if row.statistic == "mean":
            mean:np.ndarray = row[2:]
        elif row.statistic == "variance":
            variance:np.ndarray = row[2:]
        else:
            logger.warning(f"Unknown statistic in observations: '{row.statistic}'")

    plt.errorbar(taps, mean, yerr=np.sqrt(variance), fmt="go", ecolor="red")

    figure.savefig(FIGURE_DIR / "figure3 - observations.png")

    return


def figure_4(situation: Situation) -> None:

    figure = plt.figure(figsize=(16, 9), dpi=300)

    taps = np.array([5, 8, 13, 21], dtype=np.float32)
    for b in situation.sample_points[situation.sample_points.iteration == situation.iteration].b:
        plt.plot(taps, SLOPE*taps+np.float32(b), "x-")

    figure.savefig(FIGURE_DIR / f"figure4 - iteration{situation.iteration:02}.png")

    return


def figure_5(next_points: pd.DataFrame, observations: pd.DataFrame) -> None:

    figure = plt.figure(figsize=(16, 9), dpi=300)

    taps = np.array([5, 8, 13, 21], dtype=np.float32)
    for b in next_points.to_numpy():
        plt.plot(taps, SLOPE*taps+b, "-")

    for row in observations.itertuples():
        if row.statistic == "mean":
            mean:np.ndarray = row[2:]
        elif row.statistic == "variance":
            variance:np.ndarray = row[2:]
        else:
            logger.warning(f"Unknown statistic in observations: '{row.statistic}'")

    plt.errorbar(taps, mean, yerr=np.sqrt(variance), fmt="go", ecolor="red")

    figure.savefig(FIGURE_DIR / f"figure5 - final.png")

    return


if __name__ == "__main__":
    main()
