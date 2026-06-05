# Migrating from v1 to v2

`historymatching` v2.0.0 replaces the fluent **builder** API
(`HistoryMatchingBuilder` + `.with_*()` + `.build()`) with a single flat
**constructor**, `HistoryMatching(...)`. This is a breaking change for any code
written against v1.

**Staying on v1.** The last release with the builder API is tagged `v1.0.0`:

```bash
pip install "historymatching @ git+https://github.com/InstituteforDiseaseModeling/historymatching@v1.0.0"
```

## At a glance

**v1 (builder):**

```python
import historymatching as hm

builder = hm.HistoryMatchingBuilder.from_data(
    parameter_bounds={'beta': (0.1, 0.5), 'gamma': (0.01, 0.1)},
    observations={'peak_infected': (150.0, 20.0)},
)
engine = (builder
    .with_sampling_strategy('lhs')
    .with_emulator_type('bayes_linear')
    .with_samples_per_iteration(500)
    .with_max_iterations(5)
    .with_output_dir('./hm_output')
    .with_run_name('my_calibration')
    .build())
engine.set_simulation_function(my_model)
results = engine.run()
```

**v2 (constructor):**

```python
import historymatching as hm

engine = hm.HistoryMatching(
    function=my_model,
    bounds={'beta': (0.1, 0.5), 'gamma': (0.01, 0.1)},
    observations={'peak_infected': (150.0, 20.0)},
    sampling_strategy='lhs',
    emulator_type='bayes_linear',
    n_samples=500,
    max_iterations=5,
    output_dir='./hm_output',
    run_name='my_calibration',
)
results = engine.run()
```

## Construction

`HistoryMatchingBuilder` and `.build()` are gone — pass everything to the
`HistoryMatching(...)` constructor. `bounds` and `observations` accept the same
inputs the builder did (a dict, a DataFrame, or a `ParameterSpace` /
`ObservationData`), so the separate `from_data` and `from_dataframes` entry
points collapse into one argument each.

| v1 builder | v2 constructor argument |
|---|---|
| `HistoryMatchingBuilder.from_data(parameter_bounds=…, observations=…)` | `bounds=…`, `observations=…` |
| `HistoryMatchingBuilder.from_dataframes(…)` | `bounds=…`, `observations=…` (DataFrames accepted) |
| `.with_sampling_strategy(s)` | `sampling_strategy=s` |
| `.with_feature_selection(f)` | `feature_selection=f` |
| `.with_emulator_type(t)` | `emulator_type=t` |
| `.with_emulator_factory(f)` | `emulator_factory=f` |
| `.with_emulator_bank(b)` | `emulator_bank=b` |
| `.with_samples_per_iteration(n)` | `n_samples=n` |
| `.with_max_iterations(n)` | `max_iterations=n` |
| `.with_implausibility_threshold(t)` | `implausibility_threshold=t` |
| `.with_random_seed(s)` | `random_seed=s` |
| `.with_convergence_threshold(t)` | `convergence_threshold=t` |
| `.with_space_reduction(b)` | `auto_reduce_space=b` |
| `.with_oversample_factor(f)` | `oversample_factor=f` |
| `.with_max_batch_size(n)` | `max_batch_size=n` |
| `.with_nroy_method(m)` | `nroy_method=m` |
| `.with_nroy_options(**kw)` | `nroy_options=dict(**kw)` |
| `.with_output_dir(p)` | `output_dir=p` |
| `.with_run_name(n)` | `run_name=n` |
| `.with_setting(k, v)` | pass `k=v` directly |
| `.build()` | *(not needed — the constructor returns the engine)* |

## The simulator function

`set_simulation_function()` is removed. Pass the function to the constructor, or
assign it as an attribute:

```python
# v1
engine.set_simulation_function(my_model)

# v2 — in the constructor…
engine = hm.HistoryMatching(function=my_model, ...)
# …or as an attribute
engine.function = my_model
```

## Reconfiguring between waves

The `update_*` methods are replaced by plain attribute assignment, which also
works mid-run (between `step()` calls):

| v1 | v2 |
|---|---|
| `engine.update_emulator_type(t)` | `engine.emulator_type = t` |
| `engine.update_feature_selection(f)` | `engine.feature_selection = f` |
| `engine.update_max_iterations(n)` | `engine.max_iterations = n` |
| `engine.update_sampling_strategy(s)` | `engine.sampling_strategy = s` |

## `IterationResult` methods

| v1 | v2 |
|---|---|
| `result.summary_statistics()` | `result.summary()` |
| `result.get_emulator_for_feature(f)` | `result.get_emulator(f)` |
| `result.export_results()` / `.export_samples_and_results()` / `.export_emulators()` | `result.save(directory)` |
| `result.get_implausibility_scores()` | *removed — use `ObservationData.calculate_implausibilities(...)` or `BaseEmulator.get_implausibility(...)`* |

## Changed defaults

- **Default emulator is now `bayes_linear`** (pure NumPy/SciPy, no TensorFlow
  dependency) — previously `gpr`. Pass `emulator_type='gpr'` to keep Gaussian
  Process emulation.
- **Default NROY method is now `'auto'`** — Latin Hypercube sampling first,
  escalating to the `ray_resample` pipeline only when LHS acceptance is too low.
  In v1, `get_nroy_samples()` defaulted to `ray_resample`. Pass
  `nroy_method='ray_resample'` or `nroy_method='lhs'` to force a specific method.

## What hasn't changed

The wave mechanics are identical to v1: `step()` / `commit_step()` /
`revert_step()`, sampling at the end of each wave for the next one,
multi-output emulation against a single simulation set,
`drop_emulator_from_pending()`, `run(resume=True)`, and `get_nroy_samples()`
all behave exactly as before.
