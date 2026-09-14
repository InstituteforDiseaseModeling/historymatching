# API reference

Documentation is generated from the `historymatching` package using [mkdocstrings](https://mkdocstrings.github.io/).

## History matching

The single public entry point. Configure and run an entire workflow through one `HistoryMatching(...)` constructor call.

::: historymatching.HistoryMatching

## Domain objects

### Parameter space

::: historymatching.ParameterSpace

### Observation data

::: historymatching.ObservationData

### Emulator bank

::: historymatching.EmulatorBank

### Iteration result

::: historymatching.IterationResult

## Strategies

### Sampling

::: historymatching.sampling.SamplingStrategy

::: historymatching.sampling.LatinHypercubeSampling

::: historymatching.sampling.GridSampling

::: historymatching.sampling.RandomSampling

::: historymatching.sampling.SamplingStrategyFactory

### Feature selection

::: historymatching.feature_selection.FeatureSelectionStrategy

::: historymatching.feature_selection.AutoFeatureSelection

::: historymatching.feature_selection.ManualFeatureSelection

### Emulator factory

::: historymatching.emulators.factory.EmulatorFactory

## Emulators

::: historymatching.emulators.base.BaseEmulator

::: historymatching.emulators.linear.LinearModel

::: historymatching.emulators.glm.GLM

::: historymatching.emulators.gpr.GPR

### Bayes linear learned nugget

::: historymatching.emulators.bayes_linear.BayesLinear

`BayesLinear` accepts two scalar nugget modes:

| Nugget | Behavior |
|--------|----------|
| `1e-6` or another number | Fixed scalar diagonal noise term, preserving the default deterministic behavior |
| `'mle'` | Learn a single scalar nugget with the squared-exponential correlation lengths |

Use `nugget='mle'` for stochastic simulators when replicate noise is close to
constant across the input space:

```python
factory = hm.EmulatorFactory(
    default_type='bayes_linear',
    nugget='mle',
)
```

::: historymatching.emulators.results.EmulationResults

## NROY sampling

::: historymatching.nroy_sampling.generate_nroy_design

::: historymatching.nroy_sampling.NROYResult

## Plotting

The `historymatching.plotting` module provides composable plotting functions (each
returns Matplotlib axes and accepts an `ax=`/`axes=` argument). The most commonly
used are re-exported at the top level, e.g. `historymatching.plot_pairplot`.

::: historymatching.plotting
