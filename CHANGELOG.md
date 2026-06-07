# What's new

> **A note on versioning.** Only `v1.0.0` and `v2.0.0` were ever tagged and released.
> The "later/earlier additions" groupings below were informal, *untagged* development
> milestones (previously labelled "1.1.0" and "1.2.0"). None shipped as a separate
> release — they are all part of the `v1.0.0` tag.

## v2.0.1 (2026-06-07)

### New features

- **Plotting module — `historymatching.plotting`.** Composable figure helpers, each taking plain data, accepting an `ax=`/`axes=`, and returning the Matplotlib axes: `plot_pairplot` (NROY corner plot), `plot_marginals`, `plot_convergence`, `plot_zscores_vs_targets`, `plot_ensemble_fan`, `plot_constrained_dims`, `plot_emulator_quality`, `plot_predicted_vs_actual`, `plot_targets`, `plot_parameter_bounds`, plus `variance_reduction` / `marginal_variance_reduction`. The common ones are re-exported at the top level (e.g. `hm.plot_pairplot`).
- **Object-level plotting & summaries** that delegate to the module: `HistoryMatching.plot_convergence` / `plot_marginals` / `plot_nroy` / `plot_zscores` / `plot_constrained_dims` / `nroy_summary`; `IterationResult.quality_table` / `plot_emulator_quality` / `plot_predicted_vs_actual`; `ParameterSpace.plot_bounds` / `summary`; `ObservationData.plot_targets` / `summary`; `EmulatorBank.summary`.
- **Visualization tutorial** — `docs/tutorials/07_visualization.ipynb`.

### Deprecations

- `HistoryMatching.plot_nroy_parameters()` → use **`plot_nroy()`** (pass known values via `truth=` instead of `true_parameters=`). The old name still works with a `DeprecationWarning` and returns `(fig, axes)`.

### Fixes

- `plot_emulator_quality` reads the `r2` metric key produced by `get_emulator_quality_metrics()` (previously `r2_score`, which silently produced empty charts).
- `BaseEmulator.get_implausibility` now adds `model_discrepancy` in quadrature (i.e. squared), consistent with the production path `ObservationData.calculate_implausibility`. This diagnostic helper previously added it un-squared; the main implausibility filtering was already correct and is unaffected.
- Corrected the BIC value reported by `BaseEmulator.test()`: the parameter penalty was `2·k·log(n)` and is now the standard `k·log(n)` (AIC was already correct).

### Internal

- Removed unused legacy rejection-sampling methods (`_get_nroy_samples_serial`, `_compute_next_samples_serial`) and the now-orphaned `_filter_samples_with_bank` helper. The adaptive rejection-sampling loop now lives in a single place (`_generate_plausible_samples`); the public NROY path is unchanged (`generate_nroy_design`).
- Replaced the hardcoded Bayes-linear correlation-length optimisation bounds with a named, documented `LOG_THETA_BOUNDS` constant.
- Added `benchmarks/benchmark_predict.py` (and `benchmarks/README.md`) comparing the numba fast-predict path against GPflow, with a mean-agreement correctness guard that doubles as a regression check.

---

## v2.0.0 (2026)

### Breaking changes

- **New flat constructor API.** The fluent `HistoryMatchingBuilder` (`.with_*()` + `.build()`) is **removed** in favour of a single `HistoryMatching(...)` constructor that takes all options as keyword arguments. See the **[migration guide](docs/migration.md)** for the full v1→v2 mapping.
- **`engine.set_simulation_function(f)` removed** — pass `function=f` to the constructor, or set `engine.function = f`.
- **`engine.update_*()` methods removed** — reconfigure via attribute assignment (e.g. `engine.max_iterations = n`).
- **`IterationResult` methods renamed/removed**: `summary_statistics()` → `summary()`, `get_emulator_for_feature()` → `get_emulator()`, `export_*()` → `save()`; `get_implausibility_scores()` removed.

### Changed defaults

- **Default emulator is now `bayes_linear`** (pure NumPy/SciPy, no TensorFlow dependency) — previously `gpr`. Pass `emulator_type='gpr'` for Gaussian Process emulation.
- **Default NROY method is now `'auto'`** (LHS first, escalating to `ray_resample` only when LHS acceptance is too low) — in v1, `get_nroy_samples()` defaulted to `ray_resample`.

The last release with the builder API is tagged **`v1.0.0`**.

---

## v1.0.0 development — later additions (untagged)

### New features

- **Bayes Linear emulator**: New `BayesLinear` emulator type (`'bayes_linear'`) inspired by the hmer R package. Uses an OLS regression trend plus squared-exponential correlated residuals with ARD correlation lengths. Pure numpy/scipy — no TensorFlow dependency. Good uncertainty quantification comparable to GPR, with faster training for moderate datasets.
- **Ray-resample NROY sampling**: New 4-stage pipeline for finding NROY samples, inspired by hmer's `generate_new_design()`. Stages: (1) LHS rejection, (2) ray sampling along pairs of distant NROY points, (3) PCA-oriented importance sampling, (4) maximin thinning for space-filling coverage. Much faster than pure rejection at low acceptance rates (<1%). Was the default in v1 (changed to `'auto'` in v2.0.0); pass `nroy_method='lhs'` for pure rejection or `'ray_resample'` to always use this pipeline. Tune via `nroy_options=dict(n_lines=20, points_per_line=50, ...)`.
- **Detailed run logging**: Engine writes `log.txt` to the output directory with per-phase timing, per-emulator training progress, and NROY sampling progress at 10% intervals. Hyperparameters (ARD lengthscales, etc.) saved to `metrics.json`.
- **Improved diagnostics**: Residuals/predictions wrap to 5-column grid for high-dimensional problems. Pred-vs-true split into train/test panels with error bars. Error scatter and histogram also split train/test. Convergence plot uses log y-axis.

---

## v1.0.0 development — earlier additions (untagged)

### New features

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

## v1.0.0 (2025) — tagged release (the complete v1)

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
