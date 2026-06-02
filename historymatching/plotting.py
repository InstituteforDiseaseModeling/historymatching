"""
Reusable plotting helpers for history matching.

This module collects the figures that recur in every history-matching
analysis — NROY parameter clouds, convergence curves, posterior marginals,
emulator-quality summaries, z-scores against targets, and ensemble fan plots
— into a single set of composable functions.

Design conventions (so the figures behave predictably in notebooks, scripts,
and the engine's on-disk output alike):

* Every function takes **primitive data** (DataFrames, dicts, arrays) rather
  than history-matching objects, so there are no import cycles and the
  functions can be reused anywhere.
* Every function accepts an ``ax`` (or ``axes``) argument.  When omitted a new
  figure is created; when supplied the function draws into the caller's axes so
  plots can be composed into larger grids.
* Every function **returns** the Matplotlib ``Axes`` (or array of axes) it drew
  into.  Nothing calls ``plt.show()`` or ``savefig`` — the caller decides
  whether to display, save, or further customise the result.

The thin ``plot_*`` methods on :class:`~historymatching.HistoryMatchingEngine`,
:class:`~historymatching.IterationResult`, and the domain objects all delegate
here.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Shared palette ──────────────────────────────────────────────────────────
NROY_COLOR = "#3575b5"     # steel blue — the surviving / non-implausible cloud
PRIOR_COLOR = "#bcbcbc"    # grey — the prior / earlier-wave cloud
TRUTH_COLOR = "#d44d4d"    # red — known true values (synthetic-recovery demos)
MEDIAN_COLOR = "#2a7f3f"   # green — estimated median / central value
TARGET_COLOR = "#d44d4d"   # red — observation targets


__all__ = [
    "plot_convergence",
    "plot_marginals",
    "plot_pairplot",
    "plot_ensemble_fan",
    "plot_zscores_vs_targets",
    "plot_constrained_dims",
    "plot_targets",
    "plot_parameter_bounds",
    "plot_emulator_quality",
    "plot_predicted_vs_actual",
    "variance_reduction",
    "marginal_variance_reduction",
]


# ── Small helpers ─────────────────────────────────────────────────────────────
def _get_ax(ax: Optional[plt.Axes], figsize: Tuple[float, float]) -> Tuple[plt.Figure, plt.Axes]:
    """Return ``(fig, ax)``, creating a new figure if ``ax`` is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _despine(ax: plt.Axes) -> None:
    """Hide the top and right spines for a cleaner look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _resolve_params(samples: pd.DataFrame, params: Optional[Sequence[str]]) -> List[str]:
    """Pick the parameter columns to plot, defaulting to all numeric columns."""
    if params is not None:
        missing = [p for p in params if p not in samples.columns]
        if missing:
            raise KeyError(f"Parameters not found in samples: {missing}")
        resolved = list(params)
    else:
        resolved = [c for c in samples.columns
                    if pd.api.types.is_numeric_dtype(samples[c])]
    if not resolved:
        raise ValueError("No parameters to plot — `samples` has no numeric "
                         "columns (pass `params=` to choose columns explicitly).")
    return resolved


# ── Convergence ───────────────────────────────────────────────────────────────
def plot_convergence(
    iterations: Sequence[int],
    fractions: Sequence[float],
    *,
    ax: Optional[plt.Axes] = None,
    log: bool = True,
    title: str = "NROY convergence",
) -> plt.Axes:
    """Plot the non-implausible (NROY) fraction at each wave.

    The NROY fraction is the share of fresh prior samples that survive *all*
    emulator constraints accumulated so far.  It should fall monotonically as
    waves add constraints; a plateau signals convergence (or an over-constrained
    model if it collapses toward zero).

    Args:
        iterations: Wave numbers (x-axis).
        fractions: NROY fraction for each wave, in ``[0, 1]``.
        ax: Existing axes to draw into; a new figure is created if omitted.
        log: Use a logarithmic y-axis (recommended — fractions span orders of
            magnitude as constraints tighten).
        title: Axes title.

    Returns:
        The Matplotlib ``Axes`` containing the plot.
    """
    fig, ax = _get_ax(ax, (7, 4))
    iterations = list(iterations)
    fractions = list(fractions)

    ax.bar(iterations, fractions, color=NROY_COLOR, alpha=0.85, edgecolor="white")
    for it, frac in zip(iterations, fractions):
        label = f"{frac:.2%}" if frac < 0.01 else f"{frac:.1%}"
        ax.annotate(label, (it, frac), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=8)

    ax.set_xlabel("Wave")
    ax.set_ylabel("Non-implausible (NROY) fraction")
    ax.set_title(title)
    ax.set_xticks(iterations)
    if log and any(f > 0 for f in fractions):
        ax.set_yscale("log")
        lo = min(f for f in fractions if f > 0)
        ax.set_ylim(lo * 0.5, 1.0)
    else:
        top = max(fractions) if fractions else 0.0
        ax.set_ylim(0, min(1.0, top * 1.3) if top > 0 else 1.0)
    ax.grid(axis="y", alpha=0.3)
    _despine(ax)
    return ax


# ── Posterior marginals ─────────────────────────────────────────────────────
def plot_marginals(
    samples: pd.DataFrame,
    params: Optional[Sequence[str]] = None,
    *,
    truth: Optional[Dict[str, float]] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    prior: Optional[pd.DataFrame] = None,
    show_median: bool = True,
    bins: int = 25,
    ncols: Optional[int] = None,
    axes: Optional[np.ndarray] = None,
    color: str = NROY_COLOR,
) -> np.ndarray:
    """Plot a marginal histogram for each parameter.

    Args:
        samples: DataFrame of parameter samples (e.g. an NROY cloud).
        params: Which columns to plot; defaults to all numeric columns.
        truth: Optional ``{name: value}`` of known true values, drawn as a
            dashed vertical line (useful for synthetic-recovery demos).
        bounds: Optional ``{name: (lo, hi)}`` to fix each x-axis to the prior
            range, so shrinkage is visible.
        prior: Optional second sample set drawn faintly behind (e.g. the
            prior or first wave) to show how the marginal tightened.
        show_median: Draw a solid line at each parameter's sample median.
        bins: Histogram bin count.
        ncols: Columns in the subplot grid; defaults to ``len(params)`` capped
            at 4.
        axes: Existing axes array to draw into; a new figure is created if
            omitted.
        color: Histogram colour.

    Returns:
        A flat NumPy array of the ``Axes`` used.
    """
    params = _resolve_params(samples, params)
    n = len(params)
    if ncols is None:
        ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))

    if axes is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows),
                                 squeeze=False)
        axes = axes.flatten()
    else:
        axes = np.atleast_1d(axes).flatten()

    for i, p in enumerate(params):
        ax = axes[i]
        if prior is not None and p in prior.columns:
            ax.hist(prior[p], bins=bins, density=True, alpha=0.35,
                    color=PRIOR_COLOR, edgecolor="none", label="prior")
        ax.hist(samples[p], bins=bins, density=True, alpha=0.75,
                color=color, edgecolor="none", label="NROY")
        if show_median:
            ax.axvline(samples[p].median(), color=MEDIAN_COLOR, lw=2,
                       label=f"median {samples[p].median():.3g}")
        if truth is not None and p in truth:
            ax.axvline(truth[p], color=TRUTH_COLOR, ls="--", lw=2,
                       label=f"true {truth[p]:.3g}")
        if bounds is not None and p in bounds:
            ax.set_xlim(*bounds[p])
        ax.set_xlabel(p)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        _despine(ax)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    return axes


# ── NROY parameter cloud / corner plot ──────────────────────────────────────
def plot_pairplot(
    samples: pd.DataFrame,
    params: Optional[Sequence[str]] = None,
    *,
    truth: Optional[Dict[str, float]] = None,
    prior: Optional[pd.DataFrame] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    max_params: int = 8,
    bins: int = 25,
    s: float = 8,
    color: str = NROY_COLOR,
    axes: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> np.ndarray:
    """Corner plot of a parameter cloud: marginals on the diagonal, pairwise
    scatter below it.

    This is the canonical history-matching output — the shape of the
    non-implausible (NROY) region.  Pass ``truth`` to overlay known values and
    ``prior`` to show the cloud you started from.

    Args:
        samples: DataFrame of parameter samples (foreground cloud).
        params: Columns to show; defaults to all numeric columns, capped at
            ``max_params``.
        truth: ``{name: value}`` drawn as crosshairs / vertical lines.
        prior: Optional background cloud (e.g. prior or first-wave samples).
        bounds: ``{name: (lo, hi)}`` to fix axis ranges to the prior, making
            shrinkage visible.
        max_params: Cap on the number of parameters shown (keeps the grid
            readable for high-dimensional problems).
        bins: Diagonal histogram bins.
        s: Scatter marker size.
        color: Foreground cloud colour.
        axes: Existing ``(p, p)`` axes array to draw into.
        title: Optional figure suptitle.

    Returns:
        The 2-D NumPy array of ``Axes`` (shape ``(p, p)``).
    """
    params = _resolve_params(samples, params)
    if len(params) > max_params:
        params = params[:max_params]
    p = len(params)

    if axes is None:
        fig, axes = plt.subplots(p, p, figsize=(2.4 * p, 2.4 * p), squeeze=False)
    else:
        axes = np.atleast_2d(axes)
        fig = axes[0, 0].figure

    for i, pi in enumerate(params):
        for j, pj in enumerate(params):
            ax = axes[i][j]
            if i == j:  # ── diagonal: marginal histogram ──
                if prior is not None and pi in prior.columns:
                    ax.hist(prior[pi], bins=bins, density=True, alpha=0.35,
                            color=PRIOR_COLOR, edgecolor="none")
                ax.hist(samples[pi], bins=bins, density=True, alpha=0.75,
                        color=color, edgecolor="none")
                if truth is not None and pi in truth:
                    ax.axvline(truth[pi], color=TRUTH_COLOR, ls="--", lw=1.5)
                if bounds is not None and pi in bounds:
                    ax.set_xlim(*bounds[pi])
            elif i > j:  # ── lower triangle: pairwise scatter ──
                if prior is not None and pi in prior.columns and pj in prior.columns:
                    ax.scatter(prior[pj], prior[pi], s=s, alpha=0.25,
                               color=PRIOR_COLOR, edgecolors="none")
                ax.scatter(samples[pj], samples[pi], s=s, alpha=0.5,
                           color=color, edgecolors="none")
                if truth is not None:
                    if pj in truth:
                        ax.axvline(truth[pj], color=TRUTH_COLOR, ls="--", lw=1.2)
                    if pi in truth:
                        ax.axhline(truth[pi], color=TRUTH_COLOR, ls="--", lw=1.2)
                if bounds is not None:
                    if pj in bounds:
                        ax.set_xlim(*bounds[pj])
                    if pi in bounds:
                        ax.set_ylim(*bounds[pi])
            else:  # ── upper triangle: hidden ──
                ax.set_visible(False)

            # Edge labels only
            if j == 0 and i > 0:
                ax.set_ylabel(pi, fontsize=8)
            if i == p - 1:
                ax.set_xlabel(pj, fontsize=8)
            ax.tick_params(labelsize=6)
            _despine(ax)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return axes


def plot_ensemble_fan(
    trajectories: Union[np.ndarray, Sequence[Sequence[float]], pd.DataFrame],
    *,
    observed: Optional[Sequence[float]] = None,
    x: Optional[Sequence[float]] = None,
    ax: Optional[plt.Axes] = None,
    ci: float = 0.95,
    member_color: str = "#888888",
    mean_color: str = NROY_COLOR,
    obs_color: str = TRUTH_COLOR,
    show_members: bool = True,
    show_mean: bool = True,
    show_band: bool = True,
    xlabel: str = "Index",
    ylabel: str = "Value",
    title: str = "Ensemble vs observed",
) -> plt.Axes:
    """Fan / spaghetti plot of an ensemble of trajectories against observed data.

    A posterior-predictive check: re-run the simulator at NROY parameter sets,
    pass the resulting trajectories here, and compare their spread to the
    observed series.  Model-agnostic — works for any ensemble of equal-length
    vectors.

    Args:
        trajectories: 2-D array/DataFrame/list of shape ``(n_members, n_points)``.
        observed: Optional observed series (length ``n_points``) drawn on top.
        x: Optional x-axis values (defaults to ``0..n_points-1``).
        ax: Existing axes to draw into.
        ci: Central probability mass for the shaded band (e.g. ``0.95``).
        member_color: Colour of individual trajectory lines.
        mean_color: Colour of the ensemble mean and band.
        obs_color: Colour of the observed series.
        show_members: Draw each member as a faint line.
        show_mean: Draw the ensemble mean.
        show_band: Shade the central ``ci`` band.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Axes title.

    Returns:
        The Matplotlib ``Axes`` containing the plot.
    """
    arr = np.asarray(trajectories.values if isinstance(trajectories, pd.DataFrame)
                     else trajectories, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"trajectories must be 2-D (n_members, n_points), got shape {arr.shape}")
    n_members, n_points = arr.shape
    if n_members == 0 or n_points == 0:
        raise ValueError(f"trajectories must be non-empty, got shape {arr.shape}")
    if x is None:
        x = np.arange(n_points)

    fig, ax = _get_ax(ax, (10, 5))

    if show_members:
        for k in range(n_members):
            ax.plot(x, arr[k], color=member_color, alpha=0.2, lw=0.8,
                    label="Ensemble members" if k == 0 else None)

    if show_band:
        lo_q = 100 * (1 - ci) / 2
        hi_q = 100 - lo_q
        lo = np.percentile(arr, lo_q, axis=0)
        hi = np.percentile(arr, hi_q, axis=0)
        ax.fill_between(x, lo, hi, color=mean_color, alpha=0.2,
                        label=f"{int(ci * 100)}% band")

    if show_mean:
        ax.plot(x, arr.mean(axis=0), color=mean_color, lw=2, label="Ensemble mean")

    if observed is not None:
        ax.plot(x, np.asarray(observed, dtype=float), "o-", color=obs_color,
                lw=2, markersize=4, label="Observed")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _despine(ax)
    return ax


# ── Z-scores vs targets ──────────────────────────────────────────────────────
def plot_zscores_vs_targets(
    waves: List[dict],
    targets: Dict[str, Tuple[float, float]],
    *,
    ax: Optional[plt.Axes] = None,
    threshold: float = 3.5,
) -> plt.Axes:
    """Plot standardised simulation outputs against every observation target.

    For each target the band shows ``(simulated - target_mean) / target_std``
    across the wave's samples: a thick bar for the inter-quartile range, a thin
    line for the 5th–95th percentile, and a dot at the median.  Outputs inside
    the green ``±threshold`` band are consistent with the target; bands drifting
    toward zero across waves show the calibration converging.

    Args:
        waves: List of ``{'iteration': int, 'sim_results': DataFrame,
            'selected_features': list[str]}`` dicts, one per wave.
        targets: ``{feature: (mean, std)}`` observation targets.
        ax: Existing axes to draw into.
        threshold: Half-width of the shaded acceptance band (in sigma).

    Returns:
        The Matplotlib ``Axes`` containing the plot.
    """
    target_names = [k for k in targets
                    if any(k in w["sim_results"].columns for w in waves)]
    n_targets = len(target_names)
    n_waves = len(waves)
    if n_targets == 0 or n_waves == 0:
        raise ValueError("No overlapping targets/waves to plot.")

    fig, ax = _get_ax(ax, (max(14, n_targets * 0.7), 7))
    cmap = plt.get_cmap("plasma")
    bar_width = 0.8 / n_waves
    ymin_data = ymax_data = 0.0

    # Which features were emulated in which waves (for the green star markers)
    emulated: Dict[str, List[int]] = {}
    for w in waves:
        for feat in w.get("selected_features", []):
            emulated.setdefault(feat, []).append(w["iteration"])

    for wi, w in enumerate(waves):
        sims = w["sim_results"]
        color = cmap(wi / max(n_waves, 1))
        for ti, key in enumerate(target_names):
            if key not in sims.columns:
                continue
            obs_mean, obs_std = targets[key]
            z = (sims[key].dropna() - obs_mean) / obs_std
            if len(z) == 0:
                continue
            x_pos = ti + (wi - n_waves / 2 + 0.5) * bar_width
            q05, q25, med, q75, q95 = np.percentile(z, [5, 25, 50, 75, 95])
            ymin_data, ymax_data = min(ymin_data, q05), max(ymax_data, q95)
            ax.plot([x_pos, x_pos], [q05, q95], color=color, lw=1.2, alpha=0.5,
                    solid_capstyle="round")
            ax.plot([x_pos, x_pos], [q25, q75], color=color, lw=3.5, alpha=0.7,
                    solid_capstyle="round")
            ax.plot(x_pos, med, "o", color=color, markersize=4, zorder=5)

    for wi, w in enumerate(waves):
        ax.plot([], [], color=cmap(wi / max(n_waves, 1)), lw=3.5,
                label=f"Wave {w['iteration']}")

    ax.axhline(0, color=TARGET_COLOR, lw=1.5, ls="--", alpha=0.7, label="Target")
    ax.axhline(threshold, color="green", lw=0.8, ls=":", alpha=0.4)
    ax.axhline(-threshold, color="green", lw=0.8, ls=":", alpha=0.4)
    ax.axhspan(-threshold, threshold, color="green", alpha=0.03)

    margin = max(abs(ymin_data), abs(ymax_data)) * 1.15
    if margin > 0:
        ax.set_ylim(-margin, margin)
        for ti, key in enumerate(target_names):
            if key in emulated:
                wlist = ",".join(str(w) for w in emulated[key])
                ax.annotate(f"★w{wlist}", (ti, -margin * 0.93), ha="center",
                            fontsize=7, color=MEDIAN_COLOR, fontweight="bold")

    ax.set_xticks(range(n_targets))
    ax.set_xticklabels([k.replace("_", "\n") for k in target_names],
                       fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("(Sim − Target) / Target σ", fontsize=12)
    ax.set_title("NROY z-scores across waves — thick=IQR, thin=5th–95th pctl, dot=median\n"
                 "Green ★ = target was emulated in that wave", fontsize=12)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=min(n_waves + 1, 8), framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    _despine(ax)
    fig.tight_layout()
    return ax


# ── Constrained directions (PCA) ─────────────────────────────────────────────
def variance_reduction(
    samples: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """PCA-based variance reduction of an NROY cloud vs a uniform prior.

    Samples are normalised to ``[0, 1]^d`` using the prior bounds and PCA is
    fit.  Per principal component, ``reduction = 1 - NROY_var / prior_var``
    where the prior (uniform) variance is ``1/12``: 0 means as wide as the
    prior, 1 means fully collapsed.

    Args:
        samples: NROY parameter samples.
        bounds: ``{name: (lo, hi)}`` prior bounds for each parameter.

    Returns:
        ``(reduction, components, param_names)`` sorted most-constrained first.
    """
    from sklearn.decomposition import PCA

    param_names = list(bounds.keys())
    X = np.empty((len(samples), len(param_names)))
    for j, name in enumerate(param_names):
        lo, hi = bounds[name]
        X[:, j] = (samples[name].to_numpy() - lo) / (hi - lo)

    prior_var = 1.0 / 12.0
    pca = PCA(n_components=len(param_names))
    pca.fit(X)
    reduction = np.clip(1.0 - pca.explained_variance_ / prior_var, 0.0, 1.0)
    order = np.argsort(reduction)[::-1]
    return reduction[order], pca.components_[order], param_names


def marginal_variance_reduction(
    samples: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    """Per-parameter marginal variance reduction vs a uniform prior.

    Simpler than the PCA version: compares each parameter's marginal variance
    in the NROY cloud to the prior variance.  Useful for ranking which
    parameters to show in a pairplot.

    Args:
        samples: NROY parameter samples.
        bounds: ``{name: (lo, hi)}`` prior bounds.

    Returns:
        ``{name: reduction}`` with each reduction in ``[0, 1]``.
    """
    prior_var = 1.0 / 12.0
    result = {}
    for name, (lo, hi) in bounds.items():
        x = (samples[name].to_numpy() - lo) / (hi - lo)
        result[name] = float(np.clip(1.0 - float(np.var(x)) / prior_var, 0.0, 1.0))
    return result


def plot_constrained_dims(
    samples: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]],
    *,
    n_top: int = 5,
    title: str = "Constrained directions",
    axes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Plot the directions in parameter space that history matching constrained.

    The top panel shows the variance-reduction spectrum (most-constrained
    principal components first).  Each following panel shows the loadings of one
    top component — which parameters combine to form that constrained direction
    (bar height = ``|loading|``, red = positive, blue = negative).

    Args:
        samples: NROY parameter samples.
        bounds: ``{name: (lo, hi)}`` prior bounds.
        n_top: Number of most-constrained components to detail.
        title: Figure suptitle.
        axes: Existing axes array (length ``1 + n_top``) to draw into.

    Returns:
        The array of ``Axes`` used.
    """
    reduction, components, param_names = variance_reduction(samples, bounds)
    n_params = len(param_names)
    n_top = min(n_top, n_params)

    if axes is None:
        fig, axes = plt.subplots(
            1 + n_top, 1,
            figsize=(max(10, n_params * 0.55), 4 + 2.5 * n_top),
            gridspec_kw={"height_ratios": [2.5] + [1.5] * n_top},
        )
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_1d(axes)
        fig = axes[0].figure

    fig.suptitle(
        f"{title}\nVariance reduction = 1 − NROY_var / prior_var  "
        "(bar height = |loading|, red = positive, blue = negative)",
        fontsize=10,
    )

    # Spectrum
    ax = axes[0]
    colors = ["firebrick" if r > 0.5 else "steelblue" for r in reduction]
    ax.bar(np.arange(n_params), reduction * 100, color=colors, edgecolor="none")
    ax.axhline(50, color="k", lw=0.8, ls="--", label="50% reduction")
    ax.set_ylabel("Variance reduction (%)")
    ax.set_xlabel("PC index (most-constrained first)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.set_xticks(np.arange(n_params))
    ax.set_xticklabels([f"PC{i + 1}" for i in range(n_params)], fontsize=7, rotation=45)

    # Loadings for top components
    for k in range(n_top):
        ax = axes[k + 1]
        loadings = components[k]
        bar_colors = ["firebrick" if v > 0 else "steelblue" for v in loadings]
        ax.bar(np.arange(n_params), np.abs(loadings), color=bar_colors, edgecolor="none")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("|Loading|")
        ax.set_title(f"PC{k + 1} — {reduction[k] * 100:.1f}% reduction", fontsize=9)
        ax.set_xticks(np.arange(n_params))
        ax.set_xticklabels(param_names, fontsize=6.5, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    return axes


# ── Domain-object views ──────────────────────────────────────────────────────
def plot_targets(
    targets: Dict[str, Tuple[float, float]],
    *,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot observation targets as means with ±1σ error bars.

    Args:
        targets: ``{feature: (mean, std)}`` observation targets.
        ax: Existing axes to draw into.

    Returns:
        The Matplotlib ``Axes`` containing the plot.
    """
    names = list(targets.keys())
    means = np.array([targets[k][0] for k in names])
    stds = np.array([targets[k][1] for k in names])

    fig, ax = _get_ax(ax, (max(6, len(names) * 0.8), 4))
    ax.errorbar(range(len(names)), means, yerr=stds, fmt="o", color=TARGET_COLOR,
                capsize=4, markersize=6, elinewidth=1.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8,
                       rotation=45, ha="right")
    ax.set_ylabel("Target value (±1σ)")
    ax.set_title("Observation targets")
    ax.grid(axis="y", alpha=0.3)
    _despine(ax)
    return ax


def plot_parameter_bounds(
    bounds: Dict[str, Tuple[float, float]],
    *,
    reference: Optional[Dict[str, Tuple[float, float]]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot parameter bounds as horizontal ranges, optionally vs a reference.

    Bounds are normalised to each ``reference`` range so shrinkage is visible on
    one axis.  Without a reference, raw widths are shown (each on its own scale
    label).

    Args:
        bounds: ``{name: (lo, hi)}`` current bounds.
        reference: Optional ``{name: (lo, hi)}`` original/prior bounds to
            normalise against (e.g. to show how much each parameter shrank).
        ax: Existing axes to draw into.

    Returns:
        The Matplotlib ``Axes`` containing the plot.
    """
    names = list(bounds.keys())
    fig, ax = _get_ax(ax, (7, max(3, len(names) * 0.5)))

    for i, name in enumerate(names):
        lo, hi = bounds[name]
        if reference is not None and name in reference:
            rlo, rhi = reference[name]
            span = (rhi - rlo) or 1.0
            ax.barh(i, (rhi - rlo) / span, left=0, color=PRIOR_COLOR, alpha=0.5,
                    height=0.6, label="prior" if i == 0 else None)
            ax.barh(i, (hi - lo) / span, left=(lo - rlo) / span, color=NROY_COLOR,
                    height=0.6, label="current" if i == 0 else None)
        else:
            ax.barh(i, hi - lo, left=lo, color=NROY_COLOR, height=0.6)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Range (normalised to prior)" if reference else "Value")
    ax.set_title("Parameter bounds")
    if reference is not None:
        ax.legend(fontsize=8)
    _despine(ax)
    return ax


# ── Emulator quality ─────────────────────────────────────────────────────────
def plot_emulator_quality(
    quality: Dict[str, Dict[str, float]],
    *,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Bar chart of per-feature emulator R² (a quick fit-quality overview).

    Args:
        quality: ``{feature: {'r2_score': ..., 'mse': ..., ...}}`` as returned
            by :meth:`IterationResult.get_emulator_quality_metrics`.
        ax: Existing axes to draw into.

    Returns:
        The Matplotlib ``Axes`` containing the plot.
    """
    feats = list(quality.keys())
    r2 = [quality[f].get("r2_score") for f in feats]
    r2 = [float("nan") if v is None else v for v in r2]

    fig, ax = _get_ax(ax, (max(6, len(feats) * 0.9), 4))
    colors = ["#d44d4d" if v < 0.7 else NROY_COLOR for v in r2]
    ax.bar(range(len(feats)), r2, color=colors, edgecolor="white")
    ax.axhline(0.7, color="grey", ls="--", lw=0.8, label="R²=0.7")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels([f.replace("_", "\n") for f in feats], fontsize=8,
                       rotation=45, ha="right")
    ax.set_ylabel("R²")
    finite = [v for v in r2 if np.isfinite(v)]
    ax.set_ylim(min(0.0, min(finite, default=0.0)), 1.05)
    ax.set_title("Emulator quality (R²)")
    ax.legend(fontsize=8)
    _despine(ax)
    return ax


def plot_predicted_vs_actual(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    ax: Optional[plt.Axes] = None,
    r2: Optional[float] = None,
    mse: Optional[float] = None,
    n_train: Optional[int] = None,
    title: str = "Predicted vs actual",
) -> plt.Axes:
    """Scatter of emulator predictions against true simulator outputs.

    Points hugging the dashed 1:1 line indicate a good fit.

    Args:
        y_true: True (held-out) simulator outputs.
        y_pred: Emulator predictions for the same points.
        ax: Existing axes to draw into.
        r2: Optional R² to annotate in the title.
        mse: Optional MSE to annotate in the title.
        n_train: Optional training-set size to annotate.
        title: Base title.

    Returns:
        The Matplotlib ``Axes`` containing the plot.
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("plot_predicted_vs_actual needs non-empty y_true/y_pred.")

    fig, ax = _get_ax(ax, (5.5, 5))
    ax.scatter(y_true, y_pred, s=12, alpha=0.6, edgecolors="none", color=NROY_COLOR)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    margin = (hi - lo) * 0.05 or 1.0
    ref = [lo - margin, hi + margin]
    ax.plot(ref, ref, "--", color="grey", lw=0.8, alpha=0.6)
    ax.set_xlim(ref)
    ax.set_ylim(ref)
    ax.set_xlabel("Simulation (true)")
    ax.set_ylabel("Emulator (predicted)")
    bits = []
    if r2 is not None:
        bits.append(f"R²={r2:.3f}")
    if mse is not None:
        bits.append(f"MSE={mse:.3g}")
    if n_train is not None:
        bits.append(f"n={n_train}")
    ax.set_title(title + ("\n" + "  ".join(bits) if bits else ""))
    ax.set_aspect("equal", adjustable="box")
    _despine(ax)
    return ax
