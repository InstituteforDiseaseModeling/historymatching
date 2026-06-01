"""Plotting helpers for history matching results."""

from typing import Optional, Sequence

import numpy as np


def plot_ensemble_fan(
    trajectories: Sequence[Sequence[float]],
    observed: Optional[Sequence[float]] = None,
    x: Optional[Sequence[float]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    ax=None,
    show: bool = False,
):
    """Fan plot of an ensemble of trajectories, optionally vs observed data.

    Draws the ensemble median, a shaded 5-95th and 25-75th percentile band, the
    faint individual trajectories, and (if given) the observed series on top.
    Handy for eyeballing how well a set of plausible (NROY) parameter sets
    reproduces the data.

    Args:
        trajectories: 2-D array-like, shape ``(n_runs, n_timepoints)`` — one row
            per simulated trajectory.
        observed: Optional observed series of length ``n_timepoints``.
        x: Optional x-axis values (defaults to ``0..n_timepoints-1``).
        xlabel, ylabel, title: Optional axis labels / title.
        ax: Optional matplotlib Axes to draw into (a new figure is made if None).
        show: If True, call ``plt.show()`` before returning.

    Returns:
        ``(fig, ax)`` from matplotlib.
    """
    import matplotlib.pyplot as plt

    arr = np.asarray(trajectories, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"trajectories must be 2-D (n_runs, n_timepoints); got shape {arr.shape}")

    n_runs, n_t = arr.shape
    if x is None:
        x = np.arange(n_t)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    # Faint individual trajectories.
    for i in range(n_runs):
        ax.plot(x, arr[i], color="gray", alpha=0.15, linewidth=0.8,
                label="Plausible simulations" if i == 0 else None)

    # Percentile bands + median.
    q05, q25, med, q75, q95 = np.percentile(arr, [5, 25, 50, 75, 95], axis=0)
    ax.fill_between(x, q05, q95, color="#3575b5", alpha=0.15, label="5-95th pct")
    ax.fill_between(x, q25, q75, color="#3575b5", alpha=0.30, label="25-75th pct")
    ax.plot(x, med, color="#1f4e8c", linewidth=2, label="Ensemble median")

    if observed is not None:
        ax.plot(x, np.asarray(observed, dtype=float), "ro-", markersize=4,
                linewidth=1.5, label="Observed")

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if show:
        plt.show()
    return fig, ax
