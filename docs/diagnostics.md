# Diagnostics & plotting

History matching produces a lot of intermediate information — emulator fits, the
shrinking parameter region, how close simulated outputs sit to their targets.
This page covers the built-in plots and summaries that let you inspect all of it,
and how to read each figure.

Every `plot_*` method returns the Matplotlib axes (or array of axes) it draws
into and accepts an `ax=` (or `axes=`) argument, so the plots render inline in
notebooks and compose into your own figures. Nothing is shown or saved
automatically — you decide.

The same functions are available three ways:

- **as engine / result / domain-object methods** — `engine.plot_nroy()`, the
  convenient entry points used below;
- **as module-level functions** — `historymatching.plot_pairplot(df, ...)`, for
  plotting arbitrary data (also re-exported at the top level, e.g.
  `hm.plot_ensemble_fan`);
- **as the figures the engine writes to `output_dir`** after each wave — these
  are produced by the very same methods, so what you see on disk matches what you
  get in a notebook.

## Quick reference

| Goal | Method |
|------|--------|
| Is the run converging? | `engine.plot_convergence()` |
| What does the plausible region look like? | `engine.plot_nroy()` |
| What are the posterior marginals per parameter? | `engine.plot_marginals()` |
| Are simulated outputs close to their targets? | `engine.plot_zscores()` |
| Which parameter directions got constrained? | `engine.plot_constrained_dims()` |
| Did the emulators fit well? | `result.plot_emulator_quality()`, `result.plot_predicted_vs_actual(feature)` |
| Compare an ensemble to observed data | `hm.plot_ensemble_fan(trajectories, observed=...)` |
| Text summaries | `engine.summary()`, `engine.nroy_summary()`, `result.quality_table()` |

## Text summaries

Before plotting, the fastest way to see what happened is the engine summary:

```python
print(engine.summary())
```

```
=== History Matching Summary ===
Waves completed:   4/4
Emulators trained: 5
Samples generated: 12,431
Samples accepted:  2,000
Acceptance rate:   16.090%

NROY fraction per wave:
  Wave 1:   56.2%   features: peak_incidence
  Wave 2:   12.3%   features: total_cases
  ...

Plausible (NROY) parameter ranges  [500 samples]:
  beta                 [1.05, 1.62]   median 1.31   (4.4× narrower)
  gamma                [0.38, 0.71]   median 0.51   (2.7× narrower)
```

For programmatic use, `engine.nroy_summary()` returns the same per-parameter
information as a DataFrame (`min`, `max`, `median`, `q05`, `q95`, `reduction`),
and `engine.nroy_bounds()` returns just the `(min, max)` ranges as a dict.

## Convergence

```python
engine.plot_convergence()
```

A bar per wave showing the **NROY fraction** — the share of fresh prior samples
that pass *all* emulator constraints accumulated so far. The y-axis is
logarithmic because the fraction typically falls by orders of magnitude.

**How to read it:** a steadily falling fraction means each wave is successfully
ruling out more parameter space. A fraction that plateaus means further waves are
adding little — you have likely converged. A fraction that collapses toward zero
(e.g. <0.1%) means the model may be *over-constrained*: no parameters can match
all targets within the threshold. If that happens, consider relaxing the
implausibility threshold or adding a model-discrepancy term.

## NROY parameter cloud

```python
engine.plot_nroy(truth={'beta': 1.3, 'gamma': 0.5})   # truth is optional
```

A corner plot of the non-implausible region: a marginal histogram for each
parameter on the diagonal and a pairwise scatter below it. This is the headline
result — the shape of the parameter region consistent with your observations.

- Pass `truth={name: value}` to overlay known values as dashed crosshairs (handy
  for synthetic-recovery checks).
- Pass `prior=<DataFrame>` to draw an earlier (e.g. first-wave) cloud faintly
  behind the current one, so the shrinkage is visible. The on-disk `pairplot.png`
  does exactly this, overlaying the current NROY cloud on the wave-1 cloud.
- For high-dimensional problems only the most-constrained parameters are shown
  (controlled by `max_params`); pass `params=[...]` to choose explicitly.

`engine.plot_marginals()` shows just the per-parameter marginal histograms in a
row, with the sample median and (optionally) the true value marked — a compact
alternative when you only care about one-dimensional posteriors.

## Z-scores vs targets

```python
engine.plot_zscores()
```

For every observed target, this shows the distribution of
`(simulated − target_mean) / target_std` across the wave's NROY samples: a thick
bar for the inter-quartile range, a thin line for the 5th–95th percentile, and a
dot at the median, coloured by wave. The green band marks the acceptance region
(±threshold σ); a green ★ under a target marks the waves in which it was emulated.

**How to read it:** outputs whose bands sit inside the green band and centre on
zero are consistent with that target. Bands drifting toward zero across
successive waves show the calibration tightening. A band stuck far from zero
flags a target the model struggles to match — a candidate for a model-discrepancy
term, or a sign of structural model error.

## Constrained directions

```python
engine.plot_constrained_dims()
```

Principal-component analysis of the NROY cloud relative to the (uniform) prior.
The top panel ranks directions by **variance reduction**
(`1 − NROY_var / prior_var`: 0 = as wide as the prior, 1 = fully collapsed); the
panels below show which parameters load onto each most-constrained direction (bar
height = |loading|, red = positive, blue = negative).

**How to read it:** this reveals *combinations* of parameters the data constrains,
which marginal plots miss. A highly constrained PC dominated by two parameters
with opposite-sign loadings, for example, means the data pins down their
difference (or ratio) even if each is individually free — a classic sign of
parameter non-identifiability.

## Emulator quality

The emulators are only useful if they faithfully reproduce the simulator. Each
wave's fit can be inspected per feature:

```python
result = engine.get_all_results()[-1]
result.quality_table()              # DataFrame: r2, mse, n_train per feature
result.plot_emulator_quality()      # bar chart of R² (red if < 0.7)
result.plot_predicted_vs_actual('peak_incidence')
result.plot_all_emulator_diagnostics()   # full per-emulator diagnostic figures
```

**How to read predicted-vs-actual:** points should hug the dashed 1:1 line.
Systematic curvature means the emulator is biased (try a more flexible emulator
such as `'gpr'`); large scatter means it is imprecise (try more samples per
wave). An R² below ~0.7 (highlighted in red) is a sign the emulator should not be
trusted to rule out parameter space — in the interactive workflow you can
`engine.drop_emulator_from_pending(feature)` before committing.

A single emulator object exposes the same `plot_predicted_vs_actual(ax=...)`
plus the richer `plot_diagnostics()` (residuals, train/test prediction accuracy,
error distributions), `plot_zscore()`, and `plot_implausibility()`.

## Ensemble fan plots

History matching constrains *parameters*; to check the *outputs*, re-run your
simulator at NROY parameter sets and compare the resulting trajectories to the
observed data:

```python
import numpy as np
nroy = engine.get_nroy_samples(50, method='lhs')   # unbiased sample
trajectories = np.array([my_model_trajectory(row) for _, row in nroy.iterrows()])

hm.plot_ensemble_fan(trajectories, observed=observed_series)
```

This draws each member faintly, the ensemble mean, and a shaded central band
(default 95%), with the observed series on top. The observed curve falling within
the ensemble spread is a posterior-predictive check that the calibrated region
actually reproduces the data. `plot_ensemble_fan` is model-agnostic — it works
for any 2-D array of equal-length trajectories.

## Plotting arbitrary data

All of the above are thin wrappers over functions in `historymatching.plotting`
that take plain DataFrames/arrays, so you can use them on any data — for example
to plot a custom NROY set:

```python
import historymatching as hm

nroy = engine.get_nroy_samples(5000, method='lhs')
hm.plot_pairplot(nroy, params=['beta', 'gamma'], truth={'beta': 1.3})
hm.plot_marginals(nroy, truth={'beta': 1.3, 'gamma': 0.5})
```

See the [API reference](api.md) for the full list of plotting functions and their
arguments.
