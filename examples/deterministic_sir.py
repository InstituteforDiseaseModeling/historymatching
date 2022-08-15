#! /usr/bin/env python3

from pathlib import Path
import os

from scipy.integrate import ode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import matplotlib as mpl

mpl.rcParams.update({"font.size": 22})

import seaborn as sns

sns.set_palette(sns.color_palette("hls", 25))

from hm2.gpr import SkGPR
from hm2.basis import PolynomialBasis

np.random.seed(20220811)


def main():

    f1, ax1 = plt.subplots(1, 1, figsize=(15, 7))
    f2, ax2 = plt.subplots(1, 1, figsize=(15, 7))

    beta_vec = np.linspace(0.025, 0.3, 25)
    D = len(beta_vec)
    t_star = 25
    y_star = []

    implausibility_thresh = 3
    idx = 8  # Where meas is shown
    ##############################################
    sigma_y = 0.111
    ##############################################
    sigma_n = 0.025  # Meas noise
    obs = [0, 1, 8, 10, 13, 15]  # Observed inds

    fmt = "png"
    figdir = Path(f"fig_sigma_y={sigma_y}")
    figdir.mkdir() if not figdir.exists() else None

    for beta_idx, beta in enumerate(beta_vec):

        y = [np.array([0.95, 0.05])]
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

        while r.successful() and (r.t < T):

            r.integrate(r.t + dt)
            t.append(r.t)
            y.append(r.y)

            if (not found) and (r.t >= t_star):
                y_star.append(r.y)
                found = True

        Y = np.vstack(y)

        if beta_idx % 5 == 0:
            ax1.plot(t, Y[:, 1], label="Beta=%.2f" % beta)
        else:
            ax1.plot(t, Y[:, 1], label=None)

    save_infected_beta_sweep_1(ax1, t, f1, figdir, fmt)
    Y_star = np.stack(y_star)
    save_infected_beta_sweep_2(ax1, t_star, Y_star, D, f1, figdir, fmt)
    save_infected_beta_sweep_3(ax2, Y_star, t_star, beta_vec, f2, figdir, fmt)
    target = Y_star[idx, 1]
    save_infected_beta_sweep_4(ax2, beta_vec, Y_star, idx, target, sigma_y, f2, figdir, fmt)

    ### GPR
    params = ["Beta"]
    param_info = pd.DataFrame({"Name": ["Beta"], "Min": [0], "Max": [1]}).set_index(
        "Name"
    )
    basis = PolynomialBasis(degree=1, intercept=False, scale=True)

    Y_noisy = Y_star[obs, 1] + sigma_n * np.random.randn(len(obs))
    Y_mean = np.mean(Y_star[obs, 1])
    save_infected_beta_sweep_5(ax2, beta_vec, obs, Y_noisy, f2, figdir, fmt)
    Y_noisy -= Y_mean

    training_data = pd.DataFrame({"Beta": beta_vec[obs], "Sim_Result": Y_noisy})

    training_data.index.name = "Sample_Id"

    Ycol = "Sim_Result"
    g = SkGPR()
    x_train = np.array(training_data["Beta"]).reshape(-1, 1)
    y_train = np.array(training_data["Sim_Result"]).reshape(-1, 1)
    g.fit(
        x_train,
        y_train
    )
    sigma2_f_guess = 1
    sigma2_n_guess = 1
    lengthscale_guess = [0.2]   # basis.D * [0.2] # basis.D = # of params, which is 1
    x0 = np.array([sigma2_f_guess, sigma2_n_guess] + lengthscale_guess)

    sigma2_f = 5 * np.var(Y_noisy)
    sigma2_f_bounds = (0.99 * sigma2_f, 1.01 * sigma2_f)  # (0.5, 100)
    sigma2_n_bounds = ((0.99 * sigma_n) ** 2, (1.01**sigma_n) ** 2)  # (0.001, 10)
    lengthscale_bounds = (0.0001, 0.015)  # 0.0025

    # bounds = (sigma2_f_bounds,) + (sigma2_n_bounds,) + basis.D * (lengthscale_bounds,)
    bounds = (sigma2_f_bounds,) + (sigma2_n_bounds,) + (lengthscale_bounds,)

    test = pd.DataFrame(
        {
            "Beta": np.linspace(beta_vec[0], beta_vec[-1], 1000),
        }
    )

    ret = g.predict(test)
    predicted_mean, std_predictive = ret
    var_predictive = std_predictive**2
    predicted_mean += Y_mean

    save_infected_beta_sweep_6(ax2, test, predicted_mean, var_predictive, f2, figdir, fmt)

    implausibility = np.sqrt(
        (predicted_mean - target) ** 2 / (var_predictive + sigma_y**2)
    )

    dBeta = test.loc[1, "Beta"] - test.loc[0, "Beta"]
    ax2_1 = save_infected_beta_sweep_7(ax2, test, dBeta, implausibility, implausibility_thresh, beta_vec, f2, figdir, fmt)
    save_infected_beta_sweep_8(ax2_1, test, implausibility, implausibility_thresh, ax2, dBeta, f2, figdir, fmt)

    return


def save_infected_beta_sweep_1(ax1, t, f1, figdir, fmt):

    ax1.legend()
    ax1.set_xlim([t[0], t[-1]])
    ax1.set_ylim([0, 1])
    ax1.set_ylabel("Infected (%)")
    ax1.set_xlabel("Time")

    f1.savefig(figdir / f"Infected_beta_sweep_1.{fmt}")

    return


def save_infected_beta_sweep_2(ax1, t_star, Y_star, D, f1, figdir, fmt):

    ax1.plot([t_star, t_star], [0, 1], "k", lw=2)
    ax1.plot(t_star * np.ones([1, D]), np.reshape(Y_star[:, 1], (1, D)), ".", ms=15)
    f1.savefig(figdir / f"Infected_beta_sweep_2.{fmt}")

    return


def save_infected_beta_sweep_3(ax2, Y_star, t_star, beta_vec, f2, figdir, fmt):

    ax2.plot(beta_vec, Y_star[:, 1], "-", marker=".", ms=15)
    ax2.set_xlabel("Beta")
    ax2.set_ylabel("Infected at t=%.0f (%%)" % t_star)
    ax2.set_xlim(beta_vec[[0, -1]])
    ax2.set_ylim([0, 1])
    f2.savefig(figdir / f"Infected_beta_sweep_3.{fmt}")

    return


def save_infected_beta_sweep_4(ax2, beta_vec, Y_star, idx, target, sigma_y, f2, figdir, fmt):

    ax2.plot(beta_vec[[0, -1]], Y_star[[idx, idx], [1, 1]], "k", lw=2)
    ax2.plot(beta_vec[[idx, idx]], [0, target], "k:", lw=2)

    ax2.fill_between(
        [beta_vec[0], beta_vec[-1]],
        [Y_star[idx, 1] + 2 * sigma_y, Y_star[idx, 1] + 2 * sigma_y],
        [Y_star[idx, 1] - 2 * sigma_y, Y_star[idx, 1] - 2 * sigma_y],
        facecolor="k",
        alpha=0.25,
        zorder=0,
    )

    ax2.plot(beta_vec[idx], [0], "kx", ms=15)
    f2.savefig(figdir / f"Infected_beta_sweep_4.{fmt}")

    return


def save_infected_beta_sweep_5(ax2, beta_vec, obs, Y_noisy, f2, figdir, fmt):

    ax2.plot(beta_vec[obs], Y_noisy, "c+", ms=15, mew=3)
    f2.savefig(figdir / f"Infected_beta_sweep_5.{fmt}")

    return


def save_infected_beta_sweep_6(ax2, test, predicted_mean, var_predictive, f2, figdir, fmt):

    ax2.plot(test["Beta"], predicted_mean, "b-", lw=2)
    ax2.fill_between(
        test["Beta"],
        predicted_mean + 2 * np.sqrt(var_predictive),
        predicted_mean - 2 * np.sqrt(var_predictive),
        facecolor="b",
        color="b",
        edgecolor="b",
        alpha=0.15,
    )
    f2.savefig(figdir / f"Infected_beta_sweep_6.{fmt}")

    return


def save_infected_beta_sweep_7(ax2, test, dBeta, implausibility, implausibility_thresh, beta_vec, f2, figdir, fmt):

    ax2_1 = ax2.twinx()
    ax2_1.plot(test["Beta"], implausibility, "g")
    ax2_1.set_ylim([0, 11])
    ax2_1.set_ylabel("Implausibility")
    ax2_1.spines["right"].set_color("g")
    ax2_1.yaxis.label.set_color("g")
    ax2_1.tick_params(axis="y", colors="g")

    cm = plt.get_cmap("seismic")

    for i, b in test["Beta"].iteritems():
        ax2.fill_between(
            [b, b + dBeta],
            [0.05, 0.05],
            facecolor=cm(1 - min(implausibility[i] / implausibility_thresh, 1)),
        )

    ax2.set_xlim(beta_vec[[0, -1]])
    ax2_1.set_xlim(ax2.get_xlim())
    f2.savefig(figdir / f"Infected_beta_sweep_7.{fmt}")

    return ax2_1


def save_infected_beta_sweep_8(ax2_1, test, implausibility, implausibility_thresh, ax2, dBeta, f2, figdir, fmt):

    ax2_1.plot(test["Beta"], 3 * np.ones(test["Beta"].shape[0]), "g:")
    for i, b in test["Beta"].iteritems():

        if implausibility[i] >= implausibility_thresh:
            ax2.fill_between(
                [b, b + dBeta], [1, 1], hatch="\\\\\\", facecolor="k", alpha=0.5
            )

    f2.savefig(figdir / f"Infected_beta_sweep_8.{fmt}")

    return


if __name__ == "__main__":
    main()
