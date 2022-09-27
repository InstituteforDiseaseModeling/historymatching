#! /usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hm2.step as hm2
from history_matching import Basis
from history_matching.glm import GLM


WORK_DIR = Path(__file__).parent.absolute()
__prng: np.random.Generator = None


def main(num_observations=33, num_initial_samples=32):

    # Note: parameter 'a' is in log2 space.
    parameter_space = pd.DataFrame([('a', -11.0, -7.0), ('b', -10.0, 10.0), ('c', 0, 3), ('d', 0, 5)], columns=["parameter", "min", "max"])
    observations = generate_observations(parameter_space, num_observations)
    initial_sample_points = lhs(parameter_space, num_initial_samples)
    initial_sample_points["iteration"] = 0

    state = hm2.State(parameter_space, observations, initial_sample_points)
    recipe = hm2.Recipe()
    recipe.start_step_callback = start_callback
    recipe.run_simulators = run_model
    recipe.generate_emulator_for_feature = emulator_for_feature
    config = hm2.Config()

    hm2.do_step(state, recipe, config)

    return


def generate_observations(parameter_space: pd.DataFrame, n_observations: int=8, percent_variance: float=10.0) -> pd.DataFrame:

    values = model(-9.0, 0, -1.5, 5)
    noise = __prng.normal(size=values.shape[0])
    results = values + noise
    ts = np.arange(0, 33)
    columns = list(map(lambda t: f"t{t}", ts))
    observations = pd.DataFrame(data={c:[v] for c,v in zip(columns, results)})

    figure = plt.figure(figsize=(16,9), dpi=300)
    plt.plot(values)
    plt.plot(results)
    figure.savefig(WORK_DIR / "observations.png")

    return observations


def lhs(parameter_space: pd.DataFrame, n_samples: int=8) -> pd.DataFrame:

    samples = pd.DataFrame()
    for entry in parameter_space.itertuples():
        values = np.linspace(entry.min, entry.max, n_samples)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        samples[entry.parameter] = values[indices]

    return samples


def model(a, b, c, d):

    ts = np.arange(0, 33)
    output = system(ts, 2.0**a, b, c, d)    # a is in log2 space

    return output


def system(t, a=1.0/512, b=0, c=-1.5, d=5):

    results = a*(t**3) + b*(t**2) + c*t + d

    return results


def start_callback(state: hm2.State) -> None:

    figure2 = plt.figure(figsize=(16, 9), dpi=300)
    for row in state.sample_points.loc[state.sample_points.iteration == state.iteration].itertuples():
        values = model(row.a, row.b, row.c, row.d)
        plt.plot(values)

    figure2.savefig(WORK_DIR / f"samples_{state.iteration}.png")

    return


def run_model(iteration: int, test_points: pd.DataFrame, config: hm2.Config) -> pd.DataFrame:

    results = []
    for row in test_points.itertuples():
        model_output = model(row.a, row.b, row.c, row.d)
        results.append([iteration, 0, row.a, row.b, row.c, row.d])
        results[-1].extend(model_output)

    columns = ["iteration", "replicate", "a", "b", "c", "d"]
    columns.extend(map(lambda t: f"t{t}", np.arange(0, 33)))
    results = pd.DataFrame(results, columns=columns)

    return results


def emulator_for_feature(feature: str, observations: pd.DataFrame, simulator_results: pd.DataFrame, config: hm2.Config) -> hm2.Emulator:

    basis = Basis.polynomial_basis(
        params=["a", "b", "c", "d"],    # TODO - get these from ???
        intercept=True,
        first_order=True,
        second_order=True,
        third_order=True,
        fourth_order=False,
        param_info=pd.DataFrame([("a", -11.0, -7.0), ("b", -10.0, 10.0), ("c", 0, 3), ("d", 0, 5)], columns=["Name", "Min", "Max"])    # TODO - get this from ???
    )
    glm = GLM(
        basis,
        Ycol=feature,
        training_data=simulator_results,
        reference_value=0,
        family="Poisson",
        fig_type="png",
        fitted_model=None
    )
    glm.fit(maxiter=1000)

    return emulator


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=20220923)

    args = parser.parse_args()

    __prng = np.random.default_rng(args.seed)

    main()
