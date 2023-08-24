#! /usr/bin/env python3

"""Example with Gaussian process regression."""

from argparse import ArgumentParser
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
    # infected at day 14 (default)
    # incidence at day 21 (default)
    # susceptible at day 90 (default)

    inf = np.zeros(len(points))
    inc = np.zeros(len(points))
    sus = np.zeros(len(points))

    for index, parameters in enumerate(points.itertuples()):
        params = np.array([parameters.beta, parameters.gamma, parameters.initial])
        susceptible, infected, _ = trajectory(params)  # We don't use recovered.
        inf[index] = infected[config.user.t_inf]
        inc[index] = infected[config.user.t_inc] - infected[config.user.t_inc - 1]
        sus[index] = susceptible[config.user.t_sus]

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

T_INF = 14
T_INC = 21
T_SUS = 90


def main():
    """Main function."""

    # params = [beta, gamma, I0]
    # R0 = 2.5 (beta should be 1/R0?)
    # gamma = 1/5   (recovery rate, ~5 days of infectiousness)
    # I0 = 10       (initial number of infected individuals)
    params = np.array([TARGET_BETA, TARGET_GAMMA, TARGET_INITIAL])
    susceptible, infected, recovered = trajectory(params)
    inf14 = infected[T_INF]
    inc21 = infected[T_INC] - infected[T_INC - 1]
    sus90 = susceptible[T_SUS]

    # plot_epidemic(susceptible, infected, recovered)

    parameter_space = pd.DataFrame([["beta", 0.5, 5.0], ["gamma", 1 / 10, 1 / 2], ["initial", 1, 20]], columns=PARAMETER_SPACE_COLUMNS)
    observations = pd.DataFrame([["infected", inf14, 0.0], ["incidence", inc21, 0.0], ["susceptible", sus90, 0.0]], columns=OBSERVATIONS_COLUMNS)
    initial_points = lhs(parameter_space, 250)
    initial_points["iteration"] = 0
    situation = Situation(parameter_space, observations, initial_points)

    recipe = Recipe()
    recipe.start_step_callback = lambda situation: print(f"Starting step {situation.iteration}")
    recipe.run_simulators = run_model
    recipe.end_step_callback = lambda situation: print(f"Ending step {situation.iteration}")

    config = Config(max_iterations=10, candidates_per_iteration=250, implausibility_threshold=1.5, non_implausible_target=0.01, model_variance=0.0, t_inf=T_INF, t_inc=T_INC, t_sus=T_SUS)

    do_staircase(situation, recipe, config)

    plot_epidemic(susceptible, infected, recovered)

    for iteration in sorted(set(situation.sample_points.iteration)):
        plot_iteration(iteration, situation)

    return


def plot_epidemic(susceptible: np.array, infected: np.array, recovered: np.array) -> None:
    plt.figure("Epidemic Scenario")
    plt.plot(susceptible, label="susceptible", color="blue")
    plt.plot(infected, label="infected", color="red")
    plt.plot(recovered, label="recovered", color="green")
    incidence = np.zeros_like(infected)
    incidence[1:] = infected[1:] - infected[:-1]
    plt.plot(incidence, label="incidence", color="purple")
    plt.plot(T_INF, infected[T_INF], "o", color="red")
    plt.plot(T_INC, incidence[T_INC], "o", color="purple")
    plt.plot(T_SUS, susceptible[T_SUS], "o", color="blue")
    plt.legend()
    plt.show()

    return


def plot_iteration(iteration: int, situation: Situation) -> None:
    print(f"Plotting iteration {iteration}...")

    title = f"Iteration {iteration} ({list(situation.emulator_bank[iteration-1].keys())[0]})" if iteration > 0 else f"Iteration {iteration}"
    plt.figure(title)

    points = situation.sample_points[situation.sample_points.iteration == iteration]
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

    # Get "actual" trajectory for comparison.
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
    plt.plot(T_SUS, s[T_SUS], "o", color="green")
    plt.subplot(1, 3, 2)
    plt.plot(T_INF, i[T_INF], "o", color="red")
    plt.subplot(1, 3, 3)
    plt.plot(T_INC, i[T_INC] - i[T_INC - 1], "o", color="blue")

    plt.tight_layout()
    plt.show()

    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tinf", type=int, default=14, help="Time of measurement for # of infected [14].")
    parser.add_argument("--tinc", type=int, default=21, help="Time of measurement of incidence [21].")
    parser.add_argument("--tsus", type=int, default=90, help="Time of measurement of 'final' susceptible population [90].")

    args = parser.parse_args()
    T_INF = args.tinf
    T_INC = args.tinc
    T_SUS = args.tsus

    main()
