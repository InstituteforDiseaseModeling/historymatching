# What's new

## Version 2.1.0

### New features

- **Public NROY membership test.** `HistoryMatching.in_nroy(samples, batch_size=None)` returns a boolean mask: True where a point is inside the bounds and passes every committed emulator. `HistoryMatching.implausibility(samples)` returns each point's implausibility under every committed emulator, with (iteration, feature) columns. Both use the same test as `generate_nroy_design`. Use them for points you supply: a known truth in a re-identification run, uniform candidates for trajectory selection, MCMC proposals, or samples from another method.
- `historymatching.nroy_sampling.nroy_mask` is that test as a function of an emulator bank, for filtering without the engine.
- **Faster NROY tests with Bayes-linear emulators, same mask.** `nroy_mask` (behind `generate_nroy_design`, `in_nroy` and `get_nroy_samples`) first asks each emulator for `predict_bounds(x)`: bounds on the mean and variance that `predict` would return. A point that is implausible at every value within the bounds is ruled out, and one that is plausible at every value is ruled in. Only the points in between go through `predict`. `BayesLinear` bounds the variance below by `sigma2 * nugget` and above by the variance given only its first 500 training points, which costs about 2% of the full variance at 3,750 training points. Other emulators keep calling `predict` on every point, and `implausibility()` is unchanged. In a synthetic benchmark with 21 parameters, 3,750 training points, two outputs and 100,000 uniform candidates (`tests/benchmarks/benchmark_nroy_screening.py`), the bounds decided 80–94% of the points. Screening made `nroy_mask` 1.6–3.0× faster on macOS (Accelerate). On Linux with OpenBLAS it was 3.5–7.3× faster with one thread, the Coiled batch setting, and 1.3–2.1× with eight. Masks were identical in every run. With the faster `predict` below, `nroy_mask` is 7.6–10.7× faster than before on macOS and 8.5–16.4× faster with one OpenBLAS thread. (#277)
- **Bayes-linear `predict` and `train` are faster and use far less memory.** The correlations with the training points come from one matrix product (`|a|^2 + |b|^2 - 2 a.b`) instead of an `(n_new, n_train, d)` array of differences. At 3,750 training points × 21 parameters, `predict` on 5,000 points went from 5.9 s and 6.0 GB peak to 0.37 s and 0.28 GB. A batch of 20,000 points, which used to need about 24 GB, takes 1.7 s and 1.1 GB. One training likelihood evaluation went from 1.3 s and 4.5 GB to 0.32 s and 0.31 GB, and `train()` (`'mle'` nugget) from 37 s to 9 s on 900 points and from 768 s and 3.7 GB to 374 s and 0.26 GB on 3,375. The variance now takes one triangular solve instead of two, which is 2–3× faster. Results move at the rounding level, about 1e-13 on well-conditioned emulators. With near-duplicate training points and nuggets of 1e-6 to 1e-8, means move by up to 4e-7 and variances by up to 5e-6 relative. Fits can follow a slightly different optimizer path. (#277)

### Fixes

- **The linear-model and GLM emulators report the std of a new simulator run, so their implausibility is right.** Implausibility divides by the target's data uncertainty plus the emulator's variance, so that variance has to include the simulator's run-to-run noise, as `BayesLinear` (`sigma_sq * nugget`) and `GPR` (`predict_y`) already do. `LinearModel` (`'linear'`) and `GLM` (`'glm'`) returned only the standard error of the fitted mean, which understates it and rules out too much; `LinearModelScipy` returned the spread of all training outputs, which overstates it. They now return the predictive std: residual variance plus the mean's variance for `LinearModel` and `LinearModelScipy`, and the family's variance at the mean (scale for Gaussian, the mean for Poisson) plus the mean's variance for `GLM`. In a test with known noise, the standard deviation of (new run − predicted mean) / predicted std was 10.9 (`LinearModel`) and 0.44 (`LinearModelScipy`) before, and 0.99 for all of them now. Their interval columns follow the same definitions. (#273)
- **`GLM.predict` works on statsmodels 0.15**, which removed the `linear=` argument it used.
- **`LinearModel` and `GLM` can predict a single point.** `statsmodels.add_constant` skips the intercept when every column is constant, which a one-row frame always is, so one-point predictions failed with a shape error; in the NROY filter that happened whenever one candidate was left. The intercept is now always added, in training and prediction. `LinearModel`'s interval columns also no longer come back as NaN for frames with non-contiguous index labels.
- **Every wave draws a fresh design, and every run records its seed.** Previously, each wave's candidate generation restarted from `random_seed`, so consecutive waves re-drew the same plausible points. In a 21-parameter, 10-wave run, 34% of all design points repeated earlier ones. Each wave, wave 1 included, now draws from its own stream, derived from `(random_seed, wave)`.
- **`random_seed=None` (the default) now draws a seed from OS entropy instead of running unseeded.** The seed used, and whether it was user-supplied or drawn, is logged, written to `run_config.json` and stored in checkpoints, and resuming restores it. Passing that seed back reproduces the run under the same package version. This release changes which run a given seed produces, so seeded results differ from earlier versions.
- **`FastGPRPredictor`'s numpy fallback (no numba) returns the predictive variance, not zero**, matching the numba kernel and GPflow. Zero variance made every implausibility too large. (#272)

### Removed

- **The engine's separate GPR filter** (`_filter_fast`, `_filter_samples_slow`, `_build_fast_predictors`) and `historymatching.emulators.fast_predict.filter_nroy`. The engine's wave 1 sampler, its only caller, now filters through `nroy_mask` like `generate_nroy_design` and `in_nroy`, so NROY membership has one implementation. Normal runs never reached the removed path: it ran only when wave 1 started with emulators already in the bank. `FastGPRPredictor` stays in `emulators.fast_predict`.

---

## v2.0.1 (2026-06-07)

### New features

- **Plotting module — `historymatching.plotting`.** Composable figure helpers, each taking plain data, accepting an `ax=`/`axes=`, and returning the Matplotlib axes: `plot_pairplot` (NROY corner plot), `plot_marginals`, `plot_convergence`, `plot_zscores_vs_targets`, `plot_ensemble_fan`, `plot_constrained_dims`, `plot_emulator_quality`, `plot_predicted_vs_actual`, `plot_targets`, `plot_parameter_bounds`, plus `variance_reduction` / `marginal_variance_reduction`. The common ones are re-exported at the top level (e.g. `hm.plot_pairplot`).
- **Object-level plotting & summaries** that delegate to the module: `HistoryMatching.plot_convergence` / `plot_marginals` / `plot_nroy` / `plot_zscores` / `plot_constrained_dims` / `nroy_summary`; `IterationResult.quality_table` / `plot_emulator_quality` / `plot_predicted_vs_actual`; `ParameterSpace.plot_bounds` / `summary`; `ObservationData.plot_targets` / `summary`; `EmulatorBank.summary`.
- **Visualization tutorial** — `docs/tutorials/06_visualization.ipynb`.

### Deprecations

- `HistoryMatching.plot_nroy_parameters()` → use **`plot_nroy()`** (pass known values via `truth=` instead of `true_parameters=`). The old name still works with a `DeprecationWarning` and returns `(fig, axes)`.

### Fixes

- **Emulator predictions are now consistently indexed, fixing a crash in the NROY filter.** `EmulationResults` built its mean from whatever the emulator passed — a numpy array (getting a fresh `RangeIndex`) or a pandas Series (carrying the caller's labels) — and its std independently. When the two disagreed, downstream arithmetic like `abs(mean - target) / np.sqrt(variance)` aligned them *by label* rather than elementwise, returning the union of both indexes. Since the NROY filter predicts on `candidates.loc[mask]`, whose labels are non-contiguous after the first wave, this surfaced as `operands could not be broadcast together with shapes (15,) (29,) (15,)` from `_filter_nroy`. `LinearModel` was the emulator that actually tripped it (statsmodels returns an ndarray mean but a labelled Series for `mean_se`). All emulators now label mean, std, and additional data with the index of the DataFrame they predicted on, and `EmulationResults` forces the three onto a single shared index so the failure mode cannot recur.
- **The engine's implausibility filter no longer silently drops constraints.** `HistoryMatching._filter_samples_slow` was a near-verbatim copy of `_filter_nroy` in `historymatching.nroy_sampling`, but with one consequential difference: it caught emulator failures, logged a warning, and continued. Skipping a feature drops its constraint entirely, so points that should have been ruled implausible survived and the NROY region came out too wide — the exact failure mode the canonical implementation documents and deliberately guards against. The engine now delegates to `_filter_nroy`, so the two paths cannot drift apart and a failing emulator raises instead. Per-feature debug logging moved into the shared implementation and is unchanged.
- `plot_emulator_quality` reads the `r2` metric key produced by `get_emulator_quality_metrics()` (previously `r2_score`, which silently produced empty charts).
- `BaseEmulator.get_implausibility` now adds `model_discrepancy` in quadrature (i.e. squared), consistent with the production path `ObservationData.calculate_implausibility`. This diagnostic helper previously added it un-squared; the main implausibility filtering was already correct and is unaffected.
- Corrected the BIC value reported by `BaseEmulator.test()`: the parameter penalty was `2·k·log(n)` and is now the standard `k·log(n)` (AIC was already correct).

### Internal

- Hardened the emulator diagnostic plots (`plot_predictions`, `plot_zscores`) against index mismatches: columns built from emulator output are now assigned by position rather than relying on pandas label alignment, which would fill them with NaN rather than raise if the indexes ever disagreed.
- Removed unused legacy rejection-sampling methods (`_get_nroy_samples_serial`, `_compute_next_samples_serial`) and the now-orphaned `_filter_samples_with_bank` helper. The adaptive rejection-sampling loop now lives in a single place (`_generate_plausible_samples`); the public NROY path is unchanged (`generate_nroy_design`).
- Replaced the hardcoded Bayes-linear correlation-length optimisation bounds with a named, documented `LOG_THETA_BOUNDS` constant.
- Added `benchmarks/benchmark_predict.py` (and `benchmarks/README.md`) comparing the numba fast-predict path against GPflow, with a mean-agreement correctness guard that doubles as a regression check.

---

## Version 2.0.0 (2026)

### Breaking changes

- **New flat constructor API.** The fluent `HistoryMatchingBuilder` (`.with_*()` + `.build()`) is **removed** in favour of a single `HistoryMatching(...)` constructor that takes all options as keyword arguments. See the **[migration guide](https://docs.idmod.org/historymatching/migration.html)** for the full v1→v2 mapping.
- **`engine.set_simulation_function(f)` removed** — pass `function=f` to the constructor, or set `engine.function = f`.
- **`engine.update_*()` methods removed** — reconfigure via attribute assignment (e.g. `engine.max_iterations = n`).
- **`IterationResult` methods renamed/removed**: `summary_statistics()` → `summary()`, `get_emulator_for_feature()` → `get_emulator()`, `export_*()` → `save()`; `get_implausibility_scores()` removed.

### Changed defaults

- **Default emulator is now `bayes_linear`** (pure NumPy/SciPy, no TensorFlow dependency) — previously `gpr`. Pass `emulator_type='gpr'` for Gaussian Process emulation.
- **Default NROY method is now `'auto'`** (LHS first, escalating to `ray_resample` only when LHS acceptance is too low) — in v1, `get_nroy_samples()` defaulted to `ray_resample`.

The last release with the builder API is tagged **`v1.0.0`**.

---

## Version 1.0.0 development — later additions (untagged)

### New features

- **Bayes Linear emulator**: New `BayesLinear` emulator type (`'bayes_linear'`) inspired by the hmer R package. Uses an OLS regression trend plus squared-exponential correlated residuals with ARD correlation lengths. Pure numpy/scipy — no TensorFlow dependency. Good uncertainty quantification comparable to GPR, with faster training for moderate datasets.
- **Ray-resample NROY sampling**: New 4-stage pipeline for finding NROY samples, inspired by hmer's `generate_new_design()`. Stages: (1) LHS rejection, (2) ray sampling along pairs of distant NROY points, (3) PCA-oriented importance sampling, (4) maximin thinning for space-filling coverage. Much faster than pure rejection at low acceptance rates (<1%). Was the default in v1 (changed to `'auto'` in v2.0.0); pass `nroy_method='lhs'` for pure rejection or `'ray_resample'` to always use this pipeline. Tune via `nroy_options=dict(n_lines=20, points_per_line=50, ...)`.
- **Detailed run logging**: Engine writes `log.txt` to the output directory with per-phase timing, per-emulator training progress, and NROY sampling progress at 10% intervals. Hyperparameters (ARD lengthscales, etc.) saved to `metrics.json`.
- **Improved diagnostics**: Residuals/predictions wrap to 5-column grid for high-dimensional problems. Pred-vs-true split into train/test panels with error bars. Error scatter and histogram also split train/test. Convergence plot uses log y-axis.

### Slightly less new features

- **Auto-checkpointing**: Engine saves emulators, diagnostics, and checkpoint after each wave by default. Configure with the `output_dir=` and `run_name=` arguments. Disable with `output_dir=None`
- **Resume from checkpoint**: `engine.run(resume=True)` loads the latest checkpoint and continues from where it left off
- **Parallel rejection sampling**: the `n_jobs=n` argument parallelizes NROY candidate filtering across CPU cores. Workers load emulators from disk — no GPU required. Also available per-call: `engine.get_nroy_samples(10000, n_jobs=4)`
- **`get_nroy_samples(n)`**: Draw arbitrary number of NROY samples filtered through ALL emulators. Cheap — uses emulator predictions only
- **`drop_emulator_from_pending(feature)`**: Selectively remove a poor emulator before committing a wave
- **Per-wave diagnostics**: Auto-saved figures (predicted vs actual, convergence, NROY samples) in `wave{N}/{feature}/` subdirectories

### Improvements

- **GPR emulator**: Removed spurious bias column, data-informed initial hyperparameters, plain Softplus on noise variance (no artificial floor), don't abort on optimizer `success=False`
- **GPR diagnostics**: `plot_predictions()` uses observation CIs (includes noise) instead of latent-function CIs
- **`nroy_fraction`**: Per-wave fresh-LHS acceptance rate (replaces cumulative `non_implausible_fraction`)
- **Random sampler**: Fixed seed handling (`default_rng(seed)` instead of ignored `np.random.seed`)
- **`info()` display**: `len()` instead of `np.size()` for sample counts
- **Emulator quality metrics**: Lazy `test()` call + correct metric key lookup (`emulator_metrics['R2']`)
- **Column filtering**: Engine filters to parameter-space columns before emulator training/prediction (metadata like `rand_seed` is ignored)

### API changes

- `IterationResult.non_implausible_points` removed — use `engine.get_nroy_samples()`
- `IterationResult.non_implausible_fraction` renamed to `nroy_fraction`

---

## Version 1.0.0 (2025) — tagged release (the complete v1)

### New features

- **Object-oriented API**: New `HistoryMatching` class for single-call workflow configuration and execution
- **Domain objects**: `ParameterSpace`, `ObservationData`, `EmulatorBank`, and `IterationResult` for clean data management
- **Strategy pattern**: Pluggable sampling strategies (LHS, grid, random), feature selection (auto, manual), and emulator types (linear, GLM, GPR)
- **Interactive workflows**: Step-by-step execution with `step()` / `commit_step()` / `revert_step()`
- **Automatic workflows**: Multi-iteration execution with `run()` and convergence detection
- **Checkpoint/resume**: Save and restore engine state for long-running workflows
- **GPR with ARD**: Gaussian Process Regression emulators with Automatic Relevance Determination lengthscales
- **Auto feature selection**: Fano factor-based automatic feature selection with correlation filtering

### Improvements

- Replaced pyDOE2 with `scipy.stats.qmc` for Python 3.12+ compatibility
- Reproducible LHS sampling via proper scipy seed propagation
- Added `setuptools<81` pin for GPflow compatibility
- Comprehensive test suite (188 tests)
- Six tutorial notebooks covering basic through advanced workflows

### Removed

- Legacy procedural API (`Config`, `do_step`, `reduce_space`)
- Old `hm2` package
- Docker configuration
- Old DTK/radius/SIR examples
