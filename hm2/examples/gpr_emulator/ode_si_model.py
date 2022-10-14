#! /usr/bin/env python3

from collections import namedtuple
from pathlib import Path
from typing import Tuple

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import ode

import hm2
import hm2.samplers as samplers
from hm2.utils import mean_and_variance_for_observations
from history_matching.emulators import GprEmulator

WORK_DIR = Path(__file__).parent.absolute()


Visualization = namedtuple("Visualization", ["f1", "ax1", "f2", "ax2", "figure_directory", "format"])


def main(min_beta=0.025, max_beta=0.3) -> None:

    visualization = init_visualization()

    parameter_space = pd.DataFrame([["beta", min_beta, max_beta]], columns=["parameter", "minimum", "maximum"])

    sigma_y = 0.111
    X, y, observed_mean, target = get_observations(parameter_space, sigma_y, visualization=visualization)
    emulator = GprEmulator(X, y, test_fraction=0.0)
    emulator.train()
    beta = parameter_space.loc[parameter_space.parameter == "beta"]
    test = pd.DataFrame({"beta": np.linspace(float(beta.minimum), float(beta.maximum), 1000)})
    prediction, stddev = emulator.predict(test)
    prediction += observed_mean

    save_figure_6(test, prediction, stddev, visualization)

    implausibility = np.sqrt(
        (prediction - target) ** 2 / (stddev + sigma_y**2)
    )

    implausibility_thresh = 3
    # TODO - consolidate this with the same calculation in get_observations()
    beta_vec = np.linspace(float(beta.minimum), float(beta.maximum), 25)
    ax2_1, dBeta = save_figure_7(test, implausibility, implausibility_thresh, beta_vec, visualization)

    # TODO - reenable this when figure is fixed
    # save_figure_8(test, implausibility, implausibility_thresh, dBeta, visualization, ax2_1)

    return


def init_visualization() -> Visualization:

    # start = 0.0
    # stop = 1.0
    # number_of_lines = 25
    # cm_subsection = np.linspace(start, stop, number_of_lines)
    # colors = [cm.rainbow(x) for x in cm_subsection]
    f1, ax1 = plt.subplots(1, 1, figsize=(16, 9), dpi=300)
    f2, ax2 = plt.subplots(1, 1, figsize=(16, 9), dpi=300)

    figure_directory = WORK_DIR / "figures"

    if not figure_directory.exists():
        figure_directory.mkdir()

    visualization = Visualization(f1, ax1, f2, ax2, figure_directory, "png")

    return visualization


def get_observations(
    parameter_space: pd.DataFrame,
    sigma_y: float,
    num_samples=25,
    observed_indices=[0, 1, 8, 10, 13, 15],
    visualization: Visualization=None) -> Tuple[np.ndarray, np.ndarray, float, float]:

    beta = parameter_space.loc[parameter_space.parameter == "beta"]
    beta_vec = np.linspace(float(beta.minimum), float(beta.maximum), num_samples)
    t_star = 25
    y_star = []
    target_idx = 8  # Where meas[urement] is shown
    sigma_n = 0.025  # Measurement noise

    for beta_idx, beta in enumerate(beta_vec):

        y = [[0.95, 0.05]]
        t = [0]

        def f(t, y, beta):
            S = y[0]
            I = y[1]

            return [-beta * S * I, beta * S * I]

        r = ode(f).set_integrator("dopri5")
        r.set_initial_value(y[0], t[0]).set_f_params(beta)

        T = 50
        dt = 0.1  # <-- Doesn't affect accuracy, just plotting
        found = False
        while r.successful() and r.t < T:
            r.integrate(r.t + dt)
            t.append(r.t)
            y.append(r.y)

            if (not found) and (r.t >= t_star):
                y_star.append(r.y)
                found = True

        Y = np.vstack(y)

        if beta_idx % 5 == 0:
            visualization.ax1.plot(t, Y[:, 1], label="Beta=%.2f" % beta) if visualization else None
        else:
            visualization.ax1.plot(t, Y[:, 1], label=None) if visualization else None

    save_figure_1(t, visualization)

    # y_star is a _list_ of [S,I] values (one for each 0.025 <= beta <= 0.3)
    Y_star = np.stack(y_star)
    # Y_star is a Numpy array of shape (25,2), 25 different betas, S & I at time 25 (t_star)

    save_figure_2(t_star, Y_star, len(beta_vec), visualization)
    save_figure_3(beta_vec, Y_star, t_star, visualization)
    target = Y_star[target_idx, 1]
    save_figure_4(beta_vec, Y_star, target_idx, target, sigma_y, visualization)

    Y_noisy = Y_star[observed_indices, 1] + sigma_n * np.random.randn(len(observed_indices))
    Y_mean = np.mean(Y_star[observed_indices, 1])
    save_figure_5(beta_vec, observed_indices, Y_noisy, visualization)
    Y_noisy -= Y_mean

    X = pd.DataFrame(beta_vec[observed_indices], columns=["beta"])
    y = pd.DataFrame(Y_noisy, columns=["target"])

    return X, y, Y_mean, target


def save_figure_1(t:np.ndarray, visualization:Visualization) -> None:

    visualization.ax1.legend()
    visualization.ax1.set_xlim([t[0], t[-1]])
    visualization.ax1.set_ylim([0, 1])
    visualization.ax1.set_ylabel("Infected (%)")
    visualization.ax1.set_xlabel("Time")
    visualization.ax1.set_title("Infected Over Time for Various Values of β")

    visualization.f1.savefig( visualization.figure_directory / f"Infected_beta_sweep_1.{visualization.format}" )

    return


def save_figure_2(t_star: np.ndarray, Y_star: np.ndarray, D, visualization: Visualization) -> None:

    visualization.ax1.plot([t_star, t_star], [0, 1], "k", lw=2)
    visualization.ax1.plot(t_star * np.ones([1, D]), np.reshape(Y_star[:, 1], (1, D)), ".", ms=15)
    visualization.ax1.set_title("Infected Over Time for Various Values of β (Note Time = 25)")

    visualization.f1.savefig( visualization.figure_directory / f"Infected_beta_sweep_2.{visualization.format}" )

    return


def save_figure_3(beta_vec: np.ndarray, Y_star: np.ndarray, t_star: np.ndarray, visualization: Visualization) -> None:

    visualization.ax2.plot(beta_vec, Y_star[:, 1], "-", marker=".", ms=15)
    visualization.ax2.set_xlabel("Beta")
    visualization.ax2.set_ylabel("Infected at t=%.0f (%%)" % t_star)
    visualization.ax2.set_xlim(beta_vec[[0, -1]])
    visualization.ax2.set_ylim([0, 1])
    visualization.ax2.set_title("Infected @ t=25 for Different Values of β")

    visualization.f2.savefig( visualization.figure_directory / f"Infected_beta_sweep_3.{visualization.format}" )

    return


def save_figure_4(beta_vec: np.ndarray, Y_star: np.ndarray, idx: int, target: np.ndarray, sigma_y: float, visualization: Visualization) -> None:

    visualization.ax2.plot(beta_vec[[0, -1]], Y_star[[idx, idx], [1, 1]], "k", lw=2)
    visualization.ax2.plot(beta_vec[[idx, idx]], [0, target], "k:", lw=2)

    visualization.ax2.fill_between(
        [beta_vec[0], beta_vec[-1]],
        [Y_star[idx, 1] + 2 * sigma_y, Y_star[idx, 1] + 2 * sigma_y],
        [Y_star[idx, 1] - 2 * sigma_y, Y_star[idx, 1] - 2 * sigma_y],
        facecolor="k",
        alpha=0.25,
        zorder=0,
    )

    visualization.ax2.plot(beta_vec[idx], [0], "kx", ms=15)
    visualization.ax2.set_title("Infected @ t=25 for Different Values of β (Note 50% Infected with β ≈ 0.117")
    visualization.f2.savefig( visualization.figure_directory / f"Infected_beta_sweep_4.{visualization.format}" )

    return


def save_figure_5(beta_vec: np.ndarray, obs: np.ndarray, Y_noisy: np.ndarray, visualization: Visualization) -> None:

    visualization.ax2.plot(beta_vec[obs], Y_noisy, "c+", ms=15, mew=3)
    visualization.ax2.set_title("Infected @ t=25 for Different Values of β (with noisy observations)")

    visualization.f2.savefig( visualization.figure_directory / f"Infected_beta_sweep_5.{visualization.format}" )

    return


def save_figure_6(test: pd.DataFrame, ret: np.ndarray, stddev: np.ndarray, visualization: Visualization) -> None:

    visualization.ax2.plot(test["beta"], ret, "b-", lw=2)
    visualization.ax2.fill_between(
        test["beta"],
        ret + 2 * stddev,
        ret - 2 * stddev,
        facecolor="b",
        color="b",
        edgecolor="b",
        alpha=0.15,
    )
    visualization.ax2.set_title("Infected @ t=25 for Different Values of β (with noisy observations and emulator)")

    visualization.f2.savefig( visualization.figure_directory / f"Infected_beta_sweep_6.{visualization.format}" )

    return


def save_figure_7(test: pd.DataFrame, implausibility: np.ndarray, implausibility_thresh: float, beta_vec: np.ndarray, visualization: Visualization) -> None:

    ax2_1 = visualization.ax2.twinx()
    ax2_1.plot(test["beta"], implausibility, "g")
    ax2_1.set_ylim([0, 11])
    ax2_1.set_ylabel("Implausibility")
    ax2_1.spines["right"].set_color("g")
    ax2_1.yaxis.label.set_color("g")
    ax2_1.tick_params(axis="y", colors="g")

    cm = plt.get_cmap("seismic")

    dBeta = test.loc[1, "beta"] - test.loc[0, "beta"]
    for i, b in test["beta"].iteritems():
        visualization.ax2.fill_between(
            [b, b + dBeta],
            [0.05, 0.05],
            facecolor=cm(1 - min(implausibility[i] / implausibility_thresh, 1)),
        )

    visualization.ax2.set_xlim(beta_vec[[0, -1]])
    ax2_1.set_xlim(visualization.ax2.get_xlim())
    visualization.ax2.set_title("Infected @ t=25 for Different Values of β (with noisy observations, emulator, and implausibility)")

    visualization.f2.savefig( visualization.figure_directory / f"Infected_beta_sweep_7.{visualization.format}" )

    return ax2_1, dBeta


# TODO - fix this figure
def save_figure_8(test: np.ndarray, implausibility: np.ndarray, implausibility_thresh: float, dBeta, visualization: Visualization, ax2_1) -> None:

    ax2_1.plot(test["beta"], 3 * np.ones(test["beta"].shape[0]), "g:")
    for i, b in test["beta"].iteritems():
        """
        print(b, implausibility[i] >= implausibility_thresh)

        # From previous iter
        if (b >= 0.0558308308308 and b <= 0.096021021021) or (b >= 0.213288288288):
            ax2.fill_between([b, b+dBeta], [1,1], hatch = '///', facecolor='k', alpha=0.5)
        """

        if implausibility[i] >= implausibility_thresh:
            visualization.ax2.fill_between(
                [b, b + dBeta], [1, 1], hatch="\\\\\\", facecolor="k", alpha=0.5
            )

    visualization.f2.savefig( visualization.figure_directory / f"Infected_beta_sweep_8.{visualization.format}" )

    return


if __name__ == "__main__":
    main()
