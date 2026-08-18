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

### Bayes linear stochastic noise

::: historymatching.emulators.bayes_linear.BayesLinear

`BayesLinear` accepts three nugget modes:

| Nugget | Behavior |
|--------|----------|
| `1e-6` or another number | Fixed scalar diagonal noise term, preserving the default deterministic behavior |
| `'mle'` | Learn a single scalar nugget with the squared-exponential correlation lengths |
| `'adaptive'` | Learn a smoothly varying simulator-variance surface from replicated parameter sites |

Use `nugget='mle'` when replicate noise is close to constant across the input
space. Use `nugget='adaptive'` for heteroskedastic stochastic simulators where
replicated sites can estimate simulator variance across the region of interest:

```python
factory = hm.EmulatorFactory(
    default_type='bayes_linear',
    nugget='adaptive',
)
```

Adaptive nuggets follow hmer's variance-emulator pattern at a smaller scope:
training rows with identical parameter values are collapsed into per-site means,
sites with at least two replicates train an internal log-variance emulator, and
the mean emulator is trained with per-site variance divided by replicate count.
The adaptive mode uses `exp(E[log(sample_variance)])` as a positive plug-in
estimate of raw simulator variance, so hmer's truncated negative-variance
correction is intentionally out of scope.
You may include unreplicated sites for the mean surface as long as the replicated
subset spans the parameter region; otherwise the variance surface must
extrapolate.

::: historymatching.emulators.results.EmulationResults

## NROY sampling

::: historymatching.nroy_sampling.generate_nroy_design

::: historymatching.nroy_sampling.NROYResult

## Plotting

The `historymatching.plotting` module provides composable plotting functions (each
returns Matplotlib axes and accepts an `ax=`/`axes=` argument). The most commonly
used are re-exported at the top level, e.g. `historymatching.plot_pairplot`.

::: historymatching.plotting
