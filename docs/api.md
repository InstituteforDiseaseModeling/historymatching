# API reference

Documentation is generated from the `history_matching` package using [mkdocstrings](https://mkdocstrings.github.io/).

## Builder

::: history_matching.HistoryMatchingBuilder

## Engine

::: history_matching.HistoryMatchingEngine

## Domain objects

### Parameter space

::: history_matching.ParameterSpace

### Observation data

::: history_matching.ObservationData

### Emulator bank

::: history_matching.EmulatorBank

### Iteration result

::: history_matching.IterationResult

## Strategies

### Sampling

::: history_matching.sampling.SamplingStrategy

::: history_matching.sampling.LatinHypercubeSampling

::: history_matching.sampling.GridSampling

::: history_matching.sampling.RandomSampling

::: history_matching.sampling.SamplingStrategyFactory

### Feature selection

::: history_matching.feature_selection.FeatureSelectionStrategy

::: history_matching.feature_selection.AutoFeatureSelection

::: history_matching.feature_selection.ManualFeatureSelection

### Emulator factory

::: history_matching.emulators.factory.EmulatorFactory

## Emulators

::: history_matching.emulators.base.BaseEmulator

::: history_matching.emulators.linear.LinearModel

::: history_matching.emulators.glm.GLM

::: history_matching.emulators.gpr.GPR

::: history_matching.emulators.results.EmulationResults
