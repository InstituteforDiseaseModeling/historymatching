# What's new

## v1.2.0 (2025)

### New features

- **Bayes Linear emulator**: New `BayesLinear` emulator type (`'bayes_linear'`) inspired by the hmer R package. Uses an OLS regression trend plus squared-exponential correlated residuals with ARD correlation lengths. Pure numpy/scipy — no TensorFlow dependency. Good uncertainty quantification comparable to GPR, with faster training for moderate datasets.
- **Ray-resample NROY sampling**: New 4-stage pipeline for finding NROY samples, inspired by hmer's `generate_new_design()`. Stages: (1) LHS rejection, (2) ray sampling along pairs of distant NROY points, (3) PCA-oriented importance sampling, (4) maximin thinning for space-filling coverage. Much faster than pure rejection at low acceptance rates (<1%). Now the default; pass `nroy_method='lhs'` for the old behavior. Tune via `nroy_options=dict(n_lines=20, points_per_line=50, ...)`.
- **Detailed run logging**: Engine writes `log.txt` to the output directory with per-phase timing, per-emulator training progress, and NROY sampling progress at 10% intervals. Hyperparameters (ARD lengthscales, etc.) saved to `metrics.json`.
- **Improved diagnostics**: Residuals/predictions wrap to 5-column grid for high-dimensional problems. Pred-vs-true split into train/test panels with error bars. Error scatter and histogram also split train/test. Convergence plot uses log y-axis.

---

## v1.1.0 (2025)

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

## v1.0.0 (2025)

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
