#! /usr/bin/env python3

"""Example with Gaussian process regression."""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import ode

from history_matching import OBSERVATIONS_COLUMNS
from history_matching import PARAMETER_SPACE_COLUMNS
from history_matching import Config
from history_matching import Recipe
from history_matching import Situation
from history_matching import do_staircase
from history_matching import latin_hypercube_sampler as lhs

POPULATION = 1000
DURATION = 90


def model(_: float, y: np.array, params: np.array) -> np.array:
    """Model."""

    dydt = np.zeros_like(y)

    dydt[0] = -y[0] * y[1] * params[0] * params[1] / POPULATION  # S, susceptible, params[0] = beta, params[1] = gamma
    dydt[1] = +y[0] * y[1] * params[0] * params[1] / POPULATION - y[1] * params[1]  # I, infected
    dydt[2] = +y[1] * params[1]  # R, recovered

    return dydt


def jacobian(_: float, y: np.array, params: np.array) -> np.array:
    """Jacobian."""

    J = np.zeros((3, 3))
    J[0, 0] = -params[0] * y[1] / POPULATION
    J[0, 1] = -params[0] * y[0] / POPULATION
    J[1, 0] = +params[0] * y[1] / POPULATION
    J[1, 1] = +params[0] * y[0] / POPULATION - params[1]
    J[2, 1] = +params[1]

    return J


def trajectory(params: np.array) -> Tuple[np.array, np.array, np.array]:
    """Return one trajectory."""

    y0 = np.array([POPULATION - params[2], params[2], 0.0])  # S, I, R
    t0 = 0.0

    r = ode(model, jacobian).set_integrator("dopri5")
    r.set_initial_value(y0, t0).set_f_params(params[0:2]).set_jac_params(params[0:2])

    tmax = float(DURATION)
    dt = 1.0
    sus = np.zeros(int(tmax) + 1)
    sus[0] = y0[0]
    inf = np.zeros(int(tmax) + 1)
    inf[0] = y0[1]
    rec = np.zeros(int(tmax) + 1)
    rec[0] = y0[2]
    while r.successful() and r.t < tmax:
        r.integrate(r.t + dt)
        sus[int(r.t)] = r.y[0]
        inf[int(r.t)] = r.y[1]
        rec[int(r.t)] = r.y[2]

    return sus, inf, rec


def run_model(iteration: int, points: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Run model."""

    # features are:
    # infected at day 14
    # incidence at day 21
    # susceptible at day 90

    inf = np.zeros(len(points))
    inc = np.zeros(len(points))
    sus = np.zeros(len(points))

    for index, parameters in enumerate(points.itertuples()):
        params = np.array([parameters.beta, parameters.gamma, parameters.initial])
        susceptible, infected, _ = trajectory(params)  # We don't use recovered.
        inf[index] = infected[14]
        inc[index] = infected[21] - infected[20]
        sus[index] = susceptible[90]

    results = pd.DataFrame(
        {
            "iteration": iteration,
            "replicate": np.zeros(len(points), np.uint32),
            "beta": points.beta,
            "gamma": points.gamma,
            "initial": points.initial,
            "infected": inf,
            "incidence": inc,
            "susceptible": sus,
        }
    )

    return results


TARGET_BETA = 2.5
TARGET_GAMMA = 0.2
TARGET_INITIAL = 10.0


def main():
    """Main function."""

    # params = [beta, gamma, I0]
    # R0 = 2.5 (beta should be 1/R0?)
    # gamma = 1/5   (recovery rate, ~5 days of infectiousness)
    # I0 = 10       (initial number of infected individuals)
    params = np.array([TARGET_BETA, TARGET_GAMMA, TARGET_INITIAL])
    susceptible, infected, _ = trajectory(params)  # we don't use recovered
    inf14 = infected[14]
    inc21 = infected[21] - infected[20]
    sus90 = susceptible[90]

    # points = pd.DataFrame({"beta": [2.5, 0.5, 1.0, 1.5, 2.0, 2.5], "gamma": [1/5, 1/10, 1/8, 1/6, 1/5, 1/4], "infected": [10, 1, 2, 5, 10, 20]})
    # results = run_model(points)

    # import matplotlib.pyplot as plt
    # plt.plot(susceptible, label="S")
    # plt.plot(infected, label="I")
    # plt.plot(recovered, label="R")
    # plt.legend()
    # plt.show()

    parameter_space = pd.DataFrame([["beta", 0.5, 5.0], ["gamma", 1 / 10, 1 / 2], ["initial", 1, 20]], columns=PARAMETER_SPACE_COLUMNS)
    observations = pd.DataFrame([["infected", inf14, 0.0], ["incidence", inc21, 0.0], ["susceptible", sus90, 0.0]], columns=OBSERVATIONS_COLUMNS)
    initial_points = lhs(parameter_space, 250)
    initial_points["iteration"] = 0
    situation = Situation(parameter_space, observations, initial_points)

    recipe = Recipe()
    recipe.start_step_callback = lambda situation: print(f"Starting step {situation.iteration}")
    recipe.run_simulators = run_model
    recipe.end_step_callback = lambda situation: print(f"Ending step {situation.iteration}")

    config = Config(max_iterations=10, candidates_per_iteration=250, implausibility_threshold=1.5, non_implausible_target=0.01, model_variance=0.0)

    do_staircase(situation, recipe, config)

    for iteration in sorted(set(situation.simulator_results.iteration)):
        # # get rows of situation.simulator_results for this iteration
        # rows = situation.simulator_results[situation.simulator_results.iteration == iteration]
        # get the rows of situation.points for this iteration
        points = situation.sample_points[situation.sample_points.iteration == iteration]
        # iterate over each row, getting beta, gamma, and initial
        # for row in rows.itertuples():
        for row in points.itertuples():
            s, i, r = trajectory(np.array([row.beta, row.gamma, row.initial]))
            plt.subplot(1, 3, 1)
            plt.plot(s, linewidth=1)
            plt.subplot(1, 3, 2)
            plt.plot(i, linewidth=1)
            c = np.zeros_like(i)
            c[1:] = i[1:] - i[:-1]
            plt.subplot(1, 3, 3)
            plt.plot(c, linewidth=1)
        s, i, r = trajectory(np.array([TARGET_BETA, TARGET_GAMMA, TARGET_INITIAL]))
        plt.subplot(1, 3, 1)
        plt.plot(s, color="green", linewidth=2)
        plt.subplot(1, 3, 2)
        plt.plot(i, color="red", linewidth=2)
        c = np.zeros_like(i)
        c[1:] = i[1:] - i[:-1]
        plt.subplot(1, 3, 3)
        plt.plot(c, color="blue", linewidth=2)

        plt.subplot(1, 3, 1)
        plt.plot(90, s[90], "o", color="green")
        plt.subplot(1, 3, 2)
        plt.plot(14, i[14], "o", color="red")
        plt.subplot(1, 3, 3)
        plt.plot(21, i[21] - i[20], "o", color="blue")

        plt.title(f"Iteration {iteration} ({list(situation.emulator_bank[iteration].keys())[0]})")
        plt.tight_layout()
        plt.show()

    return


if __name__ == "__main__":
    main()
