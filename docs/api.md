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

::: historymatching.emulators.results.EmulationResults
