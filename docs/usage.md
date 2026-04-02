# Usage guide

This guide covers the core concepts and common patterns for using the history matching library.

## Core concepts

### Parameter space

Define the parameters you want to calibrate with their bounds:

```python
import history_matching as hm

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

All history matching workflows start by constructing a `HistoryMatchingEngine` through the builder. The builder uses a fluent interface — each `.with_*()` call configures one aspect of the workflow and returns the builder itself, so calls can be chained. When configuration is complete, `.build()` validates the settings and returns a ready-to-use engine.

```python
builder = hm.HistoryMatchingBuilder.from_data(
    parameter_bounds={'beta': (0.1, 0.5), 'gamma': (0.01, 0.1)},
    observations={'peak_infected': (150.0, 20.0)},
)
builder.with_sampling_strategy({'type': 'lhs', 'criterion': 'maximin'})  # How to sample
builder.with_feature_selection({'method': 'fano', 'max_features': 3})    # Which outputs to emulate
builder.with_emulator_type('gpr')                # Statistical surrogate type
builder.with_samples_per_iteration(500)          # Samples per iteration
builder.with_max_iterations(10)                  # Stopping criterion
builder.with_implausibility_threshold(3.0)       # Implausibility cutoff
builder.with_random_seed(42)                     # Reproducibility
engine = builder.build()                         # Validate and create engine
```

Every `.with_*()` call is optional — the builder provides sensible defaults (LHS sampling, auto feature selection, GPR emulator, 1000 samples, 10 iterations, threshold of 3.0).

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

The library includes three emulator types:

| Emulator | Best for | Speed | Uncertainty |
|----------|----------|-------|-------------|
| `'linear'` | Linear relationships, fast prototyping | Fast | Limited |
| `'glm'` | Generalized linear relationships | Fast | Limited |
| `'gpr'` | Nonlinear relationships, small-medium data | Slower | Excellent |

Select via the builder:

```python
builder.with_emulator_type('gpr')   # Gaussian Process Regression (default)
builder.with_emulator_type('linear') # Linear regression
builder.with_emulator_type('glm')    # Generalized linear model
```

## Sampling strategies

| Strategy | Description |
|----------|-------------|
| `'lhs'` | Latin Hypercube Sampling — good space-filling (default) |
| `'grid'` | Regular grid — uniform coverage |
| `'random'` | Uniform random — simple baseline |

## Feature selection

| Strategy | Description |
|----------|-------------|
| Manual list | Specify exact features: `['peak_infected', 'total_cases']` |
| `{'method': 'fano'}` | Automatic selection via Fano factor |
| `{'method': 'var'}` | Automatic selection via variance |

## Output and checkpoints

By default the engine saves emulators, diagnostics, and checkpoints after each wave:

```python
engine = builder \
    .with_output_dir('./hm_output')       # default; None disables all disk output
    .with_run_name('my_calibration')      # default: auto-generated timestamp
    .build()

engine.set_simulation_function(my_model)
results = engine.run()

print(engine.run_dir)
# ./hm_output/my_calibration/
```

Output layout:

```
hm_output/my_calibration/
  wave1/
    peak_infected/
      emulator.pkl          # pickled emulator
      diagnostics_*.png     # predicted vs actual, residuals
      metrics.json          # R², MSE, training size
    total_cases/
      ...
    convergence.png         # NROY fraction bar chart
    nroy_samples.csv        # candidates for next wave
  wave2/
    ...
  checkpoint.pkl            # latest engine state (overwritten each wave)
  run_config.json           # parameter bounds, observations, settings
```

### Resume from checkpoint

```python
engine = builder.with_run_name('my_calibration').build()
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
    feature_selection_strategy=hm.AutoFeatureSelection(method='fano'),
    emulator_factory=hm.EmulatorFactory('gpr'),
)
loaded.set_simulation_function(my_model)
```

## NROY samples and trajectory selection

After `run()` completes, draw NROY samples filtered through ALL emulators:

```python
# Default: returns pre-computed samples (~samples_per_iteration)
nroy = engine.get_nroy_samples()

# Request more (cheap — emulator predictions only, no new sims)
nroy = engine.get_nroy_samples(10000)
```

## Parallel rejection sampling

When the NROY fraction is small (high-dimensional problems), rejection sampling
can be slow. Parallelize across CPU cores:

```python
# Set default parallelism via builder
engine = builder.with_n_jobs(4).build()

# Or override per-call
nroy = engine.get_nroy_samples(10000, n_jobs=8)

# -1 = all available cores
engine = builder.with_n_jobs(-1).build()
```

Workers load emulators from disk (saved by auto-checkpointing) and filter
LHS candidates independently. No GPU required — prediction is CPU-bound
matrix multiplication.
