#! /usr/bin/env python3

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import hm2
import hm2.utils as utils
import hm2.samplers as samplers

from history_matching.emulators import BaseEmulator, LinearModel

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())  # defaults to stderr

__prng = np.random.default_rng()


def main():

    params = dict(
        max_iterations = 10,
        implausibility_threshold = 0.25,
        non_implausible_target = 0.01
    )
    config = hm2.Config(**params)

    SLOPE = 0.75
    ACTUAL_B = 42.0

    parameter_space = pd.DataFrame([["b", ACTUAL_B-5, ACTUAL_B+5]], columns=["parameter", "minimum", "maximum"])
    
    taps = np.array([5.0, 8.0, 13.0, 21.0])
    solution = (taps * SLOPE) + ACTUAL_B
    features = list(map(lambda x: f"x{int(x)}", taps))
    raw_observations = pd.DataFrame(data=[__prng.normal(solution) for _ in range(82)], columns=features)
    observations = utils.mean_and_variance_for_observations(raw_observations)

    initial_sample_points = samplers.lhs(parameter_space, 11)
    initial_sample_points["iteration"] = 0

    state = hm2.State(parameter_space, observations, initial_sample_points)

    recipe = hm2.Recipe()
    recipe.start_step_callback = lambda s: print(f"Starting iteration {s.iteration}")

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
        while len(points) < 11:
            candidates = samplers.random(parameter_space, 100)
            candidate_count += 100
            # candidates["iteration"] = iteration
            candidates["plausible"] = True
            for iteration in reversed(range(iteration+1)):
                emulators = emulator_bank[iteration]
                for feature, emulator in emulators.items():
                    prediction = emulator.predict(candidates.b)
                    difference = np.abs(prediction.value - np.float64(observations.loc[observations.statistic == "mean"][feature]))
                    candidates.plausible &= (difference < config.implausibility_threshold)
            points = pd.concat([points, candidates[candidates.plausible == True]])

        points.drop(columns="plausible", inplace=True)
        plausible_fraction = len(points) / candidate_count

        return points, plausible_fraction

    recipe.generate_next_sample_points = next_point_selection

    recipe.end_step_callback = lambda s: print(f"Finished iteration {s.iteration}")

    while not hm2.do_step(state, recipe, config):
        state.iteration += 1
        print("Continuing...")

    next_points = state.sample_points[state.sample_points.iteration == max(state.sample_points.iteration)].b
    print(f"Actual intercept = {ACTUAL_B}")
    print(f"Correct intercept, based on noisy sampling = {np.float64(observations[observations.statistic == 'mean'].x5) - (SLOPE*5)}")
    print(observations)
    print(f"Min/max of last selected test points: {next_points.min():0.04f}/{next_points.max():0.04f}")
    print(f"Selected the following points for next iteration:\n{next_points}")

    return

if __name__ == "__main__":
    main()
