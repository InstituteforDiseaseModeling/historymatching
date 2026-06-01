# Usage guide

This guide covers the core concepts and common patterns for using the history matching library.

## Core concepts

### Parameter space

Define the parameters you want to calibrate with their bounds:

```python
import historymatching as hm

# From a dictionary
space = hm.ParameterSpace({
    'beta': (0.1, 0.5),
    'gamma': (0.01, 0.1),
    'R0': (1.0, 5.0),
})

# Query bounds
lo, hi = space.get_bounds('beta')
names = space.get_parameter_names()
```

### Observations

Define what you observed, with uncertainty:

```python
# Each entry is (mean, standard_deviation)
obs = hm.ObservationData({
    'peak_infected': (150.0, 20.0),
    'total_cases': (500.0, 50.0),
    'attack_rate': (0.3, 0.05),
})

# Query observations
mean, std = obs.get_target_for_feature('peak_infected')
features = obs.get_feature_names()
```

### Simulation function

Your simulation function takes a DataFrame of parameter samples and returns a DataFrame of outputs:

```python
def my_model(samples: pd.DataFrame) -> pd.DataFrame:
    results = []
    for _, row in samples.iterrows():
        # Run your model with these parameters
        output = run_simulation(beta=row['beta'], gamma=row['gamma'])
        results.append({
            'peak_infected': output.peak,
            'total_cases': output.total,
        })
    return pd.DataFrame(results)
```

## Building an engine

All history matching workflows start by constructing a `HistoryMatchingEngine` through the builder. You configure the builder by assigning to its public attributes; each one controls one aspect of the workflow. When configuration is complete, `.build()` validates the settings and returns a ready-to-use engine.

```python
builder = hm.HistoryMatchingBuilder.from_data(
    parameter_bounds={'beta': (0.1, 0.5), 'gamma': (0.01, 0.1)},
    observations={'peak_infected': (150.0, 20.0)},
)
builder.sampling_strategy = {'type': 'lhs', 'criterion': 'maximin'}  # How to sample
builder.feature_selection = {'method': 'mean_sq_z', 'max_features': 3}  # Which outputs to emulate
builder.emulator_type = 'gpr'              # Statistical surrogate type
builder.n_samples = 500                    # Samples per iteration
builder.max_iterations = 10                # Stopping criterion
builder.implausibility_threshold = 3.0     # Implausibility cutoff
builder.random_seed = 42                   # Reproducibility
engine = builder.build()                   # Validate and create engine
```

Every attribute is optional — the builder provides sensible defaults (LHS sampling, auto feature selection, GPR emulator, 1000 samples, 10 iterations, threshold of 3.0). The most common settings are:

| Attribute | Default | Meaning |
|-----------|---------|---------|
| `sampling_strategy` | `'lhs'` | How to draw samples: `'lhs'`, `'grid'`, `'random'`, or a config dict |
| `feature_selection` | auto (`mean_sq_z`) | Feature name list, or `{'method': ..., 'max_features': N}` |
| `emulator_type` | `'gpr'` | `'gpr'`, `'bayes_linear'`, `'glm'`, `'linear'` |
| `n_samples` | `1000` | Samples generated per wave |
| `max_iterations` | `10` | Maximum number of waves |
| `implausibility_threshold` | `3.0` | Implausibility cutoff (typically 2.5–4.0) |
| `random_seed` | `None` | Seed for reproducibility |
| `output_dir` | `'./hm_output'` | Directory for diagnostics/checkpoints (`None` disables disk output) |
| `run_name` | auto timestamp | Subdirectory under `output_dir` |
| `nroy_method` | `'auto'` | NROY sampling: `'auto'`, `'ray'`, or `'lhs'` |
| `convergence_threshold` | `0.0` (off) | Stop early when the NROY acceptance rate falls below this |

You can also build from DataFrames:

```python
param_df = pd.DataFrame({
    'parameter': ['beta', 'gamma'],
    'minimum': [0.1, 0.01],
    'maximum': [0.5, 0.1],
})

obs_df = pd.DataFrame({
    'feature': ['peak_infected', 'total_cases'],
    'mean': [150.0, 500.0],
    'std': [20.0, 50.0],
})

builder = hm.HistoryMatchingBuilder.from_dataframes(param_df, obs_df)
```

## Running the workflow

### Automated execution

Run all iterations automatically:

```python
engine.set_simulation_function(my_model)
results = engine.run()

# Inspect results
for r in results:
    print(f"Iteration {r.iteration}: {len(r.samples)} samples, "
          f"features={r.selected_features}")
```

### Interactive step-by-step

For more control, execute one iteration at a time:

```python
# Run one iteration
result = engine.step()

# Inspect before committing
print(f"Features: {result.selected_features}")
print(f"NROY fraction: {result.nroy_fraction:.1%}")

# Accept or reject
engine.commit_step()   # Accept this iteration
# engine.revert_step() # Or reject and retry
```

### Changing strategies mid-workflow

You can switch strategies between iterations:

```python
# Start with automatic feature selection
result1 = engine.step()
engine.commit_step()

# Switch to manual features for iteration 2
engine.update_feature_selection(['peak_infected', 'attack_rate'])
result2 = engine.step()
engine.commit_step()

# Switch emulator type
engine.update_emulator_type('linear')
result3 = engine.step()
engine.commit_step()
```

## Emulator types

The library includes four emulator types:

| Emulator | Best for | Speed | Uncertainty |
|----------|----------|-------|-------------|
| `'linear'` | Linear relationships, fast prototyping | Fast | Limited |
| `'glm'` | Generalized linear relationships | Fast | Limited |
| `'gpr'` | Nonlinear relationships, small–medium data (default) | Slower | Excellent |
| `'bayes_linear'` | Nonlinear, no TensorFlow dependency | Medium | Good |

`'bayes_linear'` (inspired by the [hmer](https://cran.r-project.org/package=hmer) R package) fits an OLS regression trend plus squared-exponential correlated residuals with ARD correlation lengths, in pure NumPy/SciPy — useful when you want GPR-like uncertainty without TensorFlow.

Select via the builder:

```python
builder.emulator_type = 'gpr'           # Gaussian Process Regression (default)
builder.emulator_type = 'bayes_linear'  # Bayes Linear (no TensorFlow)
builder.emulator_type = 'linear'        # Linear regression
builder.emulator_type = 'glm'           # Generalized linear model
```

## Sampling strategies

| Strategy | Description |
|----------|-------------|
| `'lhs'` | Latin Hypercube Sampling — good space-filling (default) |
| `'grid'` | Regular grid — uniform coverage |
| `'random'` | Uniform random — simple baseline |

## Feature selection

Pass a list of feature names to emulate exactly those, or a config dict to rank
features automatically each wave. Automatic selection ranks candidate features
by a statistic, drops those too correlated with already-selected ones, and keeps
the top `max_features`.

| `feature_selection` value | Description |
|---------------------------|-------------|
| `['peak_infected', 'total_cases']` | Manual list — emulate exactly these |
| `{'method': 'mean_sq_z'}` | Auto: rank by mean squared z-score vs target (default) |
| `{'method': 'fano'}` | Auto: rank by Fano factor (variance / mean) |
| `{'method': 'var'}` | Auto: rank by variance |
| `{'method': 'mean'}` / `{'method': 'std'}` | Auto: rank by mean / standard deviation |

```python
builder.feature_selection = {'method': 'mean_sq_z', 'max_features': 3}
```

## Choosing an NROY sampling method

After each wave the engine searches for non-implausible candidates for the next
wave. As the plausible region shrinks, naive rejection sampling becomes slow, so
several strategies are available via `builder.nroy_method`:

| Method | Description |
|--------|-------------|
| `'auto'` | LHS rejection, escalating to ray + importance sampling if acceptance is low (default) |
| `'ray'` | Skip straight to ray + importance sampling — best for very small NROY regions |
| `'lhs'` | Pure LHS rejection — simplest, and **unbiased**, but slow at low acceptance |

```python
builder.nroy_method = 'auto'
builder.nroy_options = {'n_lines': 30, 'points_per_line': 100}  # optional tuning
```

!!! note "Unbiased samples for posterior analysis"
    `'ray'` and `'auto'` bias the sample density toward the region they explore.
    For final posterior summaries or trajectory selection, draw an unbiased set
    with `engine.get_nroy_samples(n, method='lhs')`.

## Inspecting and visualising results

Once a run completes, the engine and result objects provide summaries and plots
(see the [Diagnostics & plotting](diagnostics.md) guide for the full catalogue):

```python
print(engine.summary())            # per-wave convergence + NROY parameter ranges
engine.nroy_summary()              # DataFrame: min/max/median/quantiles per parameter

engine.plot_convergence()          # NROY fraction per wave
engine.plot_nroy(truth=...)        # corner plot of the plausible region
engine.plot_marginals()            # per-parameter posterior marginals
engine.plot_zscores()              # standardised outputs vs every target
```

Every `plot_*` method returns Matplotlib axes, so you can compose or save them.

## Dropping a poor emulator

In the interactive workflow you can inspect a wave's emulators and drop any with
a poor fit before committing, so it does not constrain future waves:

```python
result = engine.step()
print(result.quality_table())            # R² / MSE / n_train per feature
engine.drop_emulator_from_pending('noisy_feature')
engine.commit_step()
```

## Output and checkpoints

By default the engine saves emulators, diagnostics, a log, and a checkpoint after
each wave:

```python
builder.output_dir = './hm_output'    # default; None disables all disk output
builder.run_name = 'my_calibration'   # default: auto-generated timestamp
engine = builder.build()

engine.set_simulation_function(my_model)
results = engine.run()

print(engine.run_dir)
# ./hm_output/my_calibration/
```

Output layout:

```
hm_output/my_calibration/
  wave1/
    peak_infected/              # one directory per emulated feature
      emulator.pkl              # pickled trained emulator
      diagnostics_0.png ...     # predicted-vs-actual, residuals, error plots
      metrics.json              # r2_score, mse, training_size, hyperparameters
    total_cases/
      ...
    convergence.png             # NROY fraction per wave (log scale)
    zscores_vs_targets.png      # standardised sim outputs vs ALL targets, by wave
    constrained_dims.png        # PCA directions constrained most (needs ≥10 NROY samples)
    pairplot.png                # NROY parameter cloud vs wave 1 (from wave 2 on)
    nroy_samples.csv            # plausible candidates for the next wave
  wave2/
    ...
  checkpoint.pkl                # latest engine state (overwritten each wave)
  run_config.json              # parameter bounds, observations, settings
  log.txt                       # per-phase timing and progress for the whole run
```

The figures saved here are exactly those produced by the engine's `plot_*`
methods; see [Diagnostics & plotting](diagnostics.md) for how to read each one.

### Resume from checkpoint

```python
builder.run_name = 'my_calibration'
engine = builder.build()
engine.set_simulation_function(my_model)
results = engine.run(resume=True)   # loads checkpoint.pkl, continues
```

### Manual checkpoints

For finer control, save and load checkpoints explicitly:

```python
engine.save_checkpoint('checkpoint.pkl')

loaded = hm.HistoryMatchingEngine.load_checkpoint(
    'checkpoint.pkl',
    sampling_strategy=hm.SamplingStrategyFactory.create('lhs'),
    feature_selection_strategy=hm.AutoFeatureSelection(method='mean_sq_z'),
    emulator_factory=hm.EmulatorFactory('gpr'),
)
loaded.set_simulation_function(my_model)
```

## NROY samples and trajectory selection

After `run()` completes, draw NROY samples filtered through ALL emulators:

```python
# Default: returns pre-computed samples (~n_samples)
nroy = engine.get_nroy_samples()

# Request more (cheap — emulator predictions only, no new sims)
nroy = engine.get_nroy_samples(10000)

# Unbiased draw for posterior analysis / trajectory selection
nroy = engine.get_nroy_samples(10000, method='lhs')
```

**Trajectory selection** is a common post-calibration step: re-run your simulator
(including its stochastic seed) at NROY parameter sets, weight each trajectory by
a pseudo-likelihood against the observed data, and resample to obtain a calibrated
ensemble. The library supplies the NROY scaffolding; you supply the weighting. See
the [Trajectory selection tutorial](tutorials/05_trajectory_selection.ipynb) for a
complete worked example.

