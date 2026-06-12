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

## Creating an engine

All history matching workflows start by constructing a `HistoryMatching` engine. Everything is configured in a single constructor call — pass your simulator `function`, the parameter bounds, and observations, then any options as keyword arguments. Every keyword argument is optional and has a sensible default.

```python
engine = hm.HistoryMatching(
    function=my_model,                                       # the simulation function
    bounds={'beta': (0.1, 0.5), 'gamma': (0.01, 0.1)},
    observations={'peak_infected': (150.0, 20.0)},
    sampling_strategy={'type': 'lhs', 'criterion': 'maximin'},  # How to sample
    feature_selection={'method': 'fano', 'max_features': 3},    # Which outputs to emulate
    emulator_type='bayes_linear',                            # Statistical surrogate type (default)
    n_samples=500,                                           # Parameter samples per wave (one wave = one iteration)
    max_iterations=10,                                       # Stopping criterion
    implausibility_threshold=3.0,                            # Drop points >3 std devs from target (~95% for any unimodal dist.; see Glossary)
    random_seed=42,                                          # Reproducibility
)
```

Every keyword argument is optional — the constructor provides sensible defaults (LHS sampling, automatic output selection by `mean_sq_z`, Bayes linear emulator, 1000 samples, 10 iterations, threshold of 3.0).

You can also build from DataFrames — just pass them in place of the dicts:

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

engine = hm.HistoryMatching(param_df, obs_df, function=my_model)
```

## Running the workflow

### Automated execution

Run all iterations automatically (the simulation function was passed as `function=my_model` above; you can also set it later with `engine.function = my_model`):

```python
results = engine.run()

# Inspect results
for r in results:
    print(f"Iteration {r.iteration}: {len(r.samples)} samples, "
          f"outputs={r.emulated_outputs}")
```

### Interactive step-by-step

For more control, execute one iteration at a time:

```python
# Run one iteration
result = engine.step()

# Inspect before committing
print(f"Outputs: {result.emulated_outputs}")
print(f"NROY fraction: {result.nroy_fraction:.1%}")

# Accept or reject
engine.commit_step()   # Accept this iteration
# engine.revert_step() # Or reject and retry
```

### Changing strategies mid-workflow

You can switch strategies between iterations by assigning to the attribute (e.g.
`engine.feature_selection = ['peak']`); the coercing property setters accept the same
values as the constructor:

```python
# Start with automatic feature selection
result1 = engine.step()
engine.commit_step()

# Switch to manual outputs for iteration 2
engine.feature_selection = ['peak_infected', 'attack_rate']
result2 = engine.step()
engine.commit_step()

# Switch emulator type
engine.emulator_type = 'linear'
result3 = engine.step()
engine.commit_step()
```

## Emulator types

The library includes four [emulator](glossary.md#emulator) types:

| Emulator | Best for | Speed | Uncertainty |
|----------|----------|-------|-------------|
| `'bayes_linear'` | Linear relationships with calibrated uncertainty (default) | Fast | Good |
| `'linear'` | Linear relationships, fast prototyping | Fast | Limited |
| `'glm'` | Generalized linear relationships | Fast | Limited |
| `'gpr'` | Nonlinear relationships, small-medium data | Slower | Excellent |

Select via the `emulator_type` argument:

```python
hm.HistoryMatching(..., emulator_type='bayes_linear') # Bayes linear (default)
hm.HistoryMatching(..., emulator_type='gpr')          # Gaussian Process Regression
hm.HistoryMatching(..., emulator_type='linear')       # Linear regression
hm.HistoryMatching(..., emulator_type='glm')          # Generalized linear model
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
| Manual list | Specify exact outputs: `['peak_infected', 'total_cases']` |
| `{'method': 'mean_sq_z'}` | Automatic selection by mean squared z-score — how far each output sits from its target in std units (default) |
| `{'method': 'fano'}` | Automatic selection via Fano factor — the variance-to-mean ratio of each output |
| `{'method': 'var'}` | Automatic selection via variance |

## Output and checkpoints

By default the engine saves emulators, diagnostics, and checkpoints after each wave:

```python
engine = hm.HistoryMatching(
    function=my_model,
    bounds=parameter_bounds,
    observations=observations,
    output_dir='./hm_output',       # default; None disables all disk output
    run_name='my_calibration',      # default: auto-generated timestamp
)
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
    zscores_vs_targets.png  # sim outputs vs ALL targets across waves
    pairplot.png            # parameter space shrinkage (top 8 most-constrained parameters)
    nroy_samples.csv        # candidates for next wave
  wave2/
    ...
  checkpoint.pkl            # latest engine state (overwritten each wave)
  run_config.json           # parameter bounds, observations, settings
```

### Resume from checkpoint

```python
engine = hm.HistoryMatching(
    function=my_model,
    bounds=parameter_bounds,
    observations=observations,
    run_name='my_calibration',
)
results = engine.run(resume=True)   # loads checkpoint.pkl, continues
```

### Manual checkpoints

For finer control, save and load checkpoints explicitly:

```python
engine.save_checkpoint('checkpoint.pkl')

loaded = hm.HistoryMatching.load_checkpoint(
    'checkpoint.pkl',
    sampling_strategy=hm.SamplingStrategyFactory.create('lhs'),
    feature_selection=hm.AutoFeatureSelection(method='fano'),
    emulator_factory=hm.EmulatorFactory('gpr'),
)
loaded.function = my_model
```

## NROY samples and trajectory selection

The **NROY** ("Not Ruled Out Yet") region is the part of parameter space that has
not been ruled out as [implausible](glossary.md#implausibility) — see the
[Glossary](glossary.md#nroy).

After `run()` completes, draw NROY samples filtered through ALL emulators:

```python
# Default: returns pre-computed samples (~n_samples)
nroy = engine.get_nroy_samples()

# Request more (cheap — emulator predictions only, no new sims)
nroy = engine.get_nroy_samples(10000)
```


