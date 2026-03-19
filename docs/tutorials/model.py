"""
Shared SIR model and helper functions for history matching examples.

This module defines the stochastic SIR model used in the tutorial notebooks,
as well as a helper to generate synthetic observed data for calibration.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SIR:
    """A stochastic SIR model using the tau-leap algorithm."""

    def __init__(
        self,
        beta=1.2,
        gamma=0.3,
        s0=990,
        i0=10,
        r0=0,
        n_days=20,
        step=0.1,
        seed=None,
    ):
        """
        Create and initialize the SIR model.

        Args:
            beta:   Transmission rate (> 0).
            gamma:  Recovery rate (> 0).
            s0:     Initial number of susceptible individuals (non-negative int).
            i0:     Initial number of infected individuals (non-negative int).
            r0:     Initial number of recovered individuals (non-negative int).
            n_days: Number of days to simulate (positive integer).
            step:   Step size in days (between 0 and 1).
            seed:   Seed for the random number generator.
        """
        assert beta > 0, "Transmission rate must be positive"
        assert gamma > 0, "Recovery rate must be positive"
        assert s0 >= 0 and i0 >= 0 and r0 >= 0, "Initial conditions must be non-negative"
        assert n_days > 0, "Simulation days must be positive"
        assert 0 < step <= 1, "Step size must be between 0 and 1"

        self.beta = beta
        self.gamma = gamma
        self.s0 = int(s0)
        self.i0 = int(i0)
        self.r0 = int(r0)
        self.n_days = int(n_days)
        self.step = step
        self.seed = seed
        self.simulation_complete = False

    def run(self):
        """Run the simulation and return S, I, and R arrays."""
        n_steps = math.ceil((self.n_days - 1) / self.step)
        steps_per_day = int(1 / self.step)
        N = self.s0 + self.i0 + self.r0

        s = np.zeros(n_steps + 1, dtype=int)
        i = np.zeros(n_steps + 1, dtype=int)
        r = np.zeros(n_steps + 1, dtype=int)
        s[0] = self.s0
        i[0] = self.i0
        r[0] = self.r0

        s_daily = np.zeros(self.n_days, dtype=int)
        i_daily = np.zeros(self.n_days, dtype=int)
        r_daily = np.zeros(self.n_days, dtype=int)
        s_daily[0] = s[0]
        i_daily[0] = i[0]
        r_daily[0] = r[0]

        if self.seed is not None:
            np.random.seed(self.seed)

        for j in range(1, n_steps + 1):
            lambda_transmission = (self.beta * s[j - 1] * i[j - 1] / N) * self.step
            lambda_recovery = self.gamma * i[j - 1] * self.step

            delta_Mt = np.random.poisson(lambda_transmission)
            delta_Mr = np.random.poisson(lambda_recovery)

            s[j] = max(s[j - 1] - delta_Mt, 0)
            i[j] = max(i[j - 1] + delta_Mt - delta_Mr, 0)
            r[j] = r[j - 1] + delta_Mr

        for j in range(1, self.n_days):
            k = j * steps_per_day
            s_daily[j] = s[k]
            i_daily[j] = i[k]
            r_daily[j] = r[k]

        self.s = s_daily
        self.i = i_daily
        self.r = r_daily
        self.simulation_complete = True
        return s_daily, i_daily, r_daily

    def get_incidence(self):
        """Return array with daily incidence (new infections per day)."""
        if not self.simulation_complete:
            self.run()
        n = len(self.i)
        incidence = np.zeros(n)
        for j in range(1, n):
            incidence[j] = self.s[j - 1] - self.s[j]
        return incidence

    def plot(self, title=None):
        """Display figure with S, I, and R curves."""
        if not self.simulation_complete:
            self.run()
        x = range(self.n_days)
        if title is None:
            title = f"beta = {self.beta};  gamma = {self.gamma}"
        plt.figure()
        plt.plot(x, self.s, "g")
        plt.plot(x, self.i, "r")
        plt.plot(x, self.r, "b")
        plt.legend(["S", "I", "R"])
        plt.ylabel("cases")
        plt.xlabel("days")
        plt.title(title)
        plt.grid(linestyle=":")

    def plot_incidence(self):
        """Display figure with the incidence curve."""
        if not self.simulation_complete:
            self.run()
        incidence = self.get_incidence()
        x = range(self.n_days)
        title = f"beta = {self.beta};  gamma = {self.gamma}"
        plt.figure()
        plt.plot(x, incidence, "b")
        plt.ylabel("new cases")
        plt.xlabel("days")
        plt.title(title)
        plt.grid(linestyle=":")


def generate_observed_data(
    beta_true=1.3,
    gamma_true=0.5,
    population_size=10_000,
    n_seed_infections=100,
    seed=0,
):
    """
    Generate synthetic observed incidence data from a SIR model.

    Returns a tuple (incidence_data, model) where incidence_data is a pandas
    Series indexed by day and model is the SIR instance used to generate it.
    """
    model = SIR(
        beta=beta_true,
        gamma=gamma_true,
        s0=population_size - n_seed_infections,
        i0=n_seed_infections,
        seed=seed,
    )
    model.run()
    incidence = model.get_incidence()
    incidence_series = pd.Series(incidence, name="incidence")
    incidence_series.index.name = "day"
    return incidence_series, model
