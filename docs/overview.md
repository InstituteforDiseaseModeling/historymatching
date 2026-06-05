# Overview

History matching is an iterative calibration technique that progressively rules out regions of parameter space that are *implausible* — i.e., where simulated outputs are inconsistent with observations. Rather than finding a single "best fit," it identifies the set of all parameter combinations that could plausibly have produced the observed data.

## How it works

Each iteration of the algorithm:

1. **Sample** points from the current non-implausible parameter space
2. **Simulate** the model at each sample point
3. **Select features** — choose which model outputs to emulate
4. **Train emulators** — build fast statistical surrogates (e.g., Gaussian Process Regression) of the simulation
5. **Filter** — use emulator predictions and implausibility scores to discard implausible regions

The parameter space shrinks with each iteration until it converges on the region consistent with observations.

## Key features

- **Single-constructor API**: Configure an entire workflow in one `HistoryMatching(...)` call
- **Interactive engine**: Step-by-step control with `step()` / `commit_step()` / `revert_step()`, or fully automated execution with `run()`
- **Multiple emulators**: Linear, GLM, and Gaussian Process Regression (GPflow-based with ARD kernels)
- **Pluggable strategies**: Swap sampling (LHS, grid, random), feature selection (auto Fano-factor, manual), and emulator types at any point
- **Domain objects**: `ParameterSpace`, `ObservationData`, `EmulatorBank`, and `IterationResult` for clean data management
- **Checkpoint/resume**: Save and restore engine state for long-running workflows

## Architecture

```
HistoryMatching                 # Configure and execute the workflow
    ├── ParameterSpace          # Parameter bounds
    ├── ObservationData         # Target observations (mean, std)
    ├── SamplingStrategy        # How to generate samples (LHS, grid, random)
    ├── FeatureSelectionStrategy # Which outputs to emulate (auto, manual)
    ├── EmulatorFactory         # Which emulator to use (bayes_linear, linear, glm, gpr)
    ├── step()                  # Run one iteration
    ├── commit_step()           # Accept the iteration
    ├── revert_step()           # Reject and retry
    └── run()                   # Fully automated multi-iteration
            │
            ▼
IterationResult                 # Immutable results per iteration
    ├── samples                 # Parameter samples used
    ├── simulation_results      # Model outputs
    ├── emulators               # Trained emulators
    └── nroy_fraction           # Fresh-LHS acceptance rate (convergence diagnostic)
```

## When to use history matching

History matching is well-suited for:

- **Expensive simulations** where each run takes minutes to hours
- **Multiple uncertain parameters** (2-20+ dimensions)
- **Multiple output features** to match against observations
- **Uncertainty quantification** — you want the set of plausible parameters, not just a point estimate
- **Iterative refinement** — you want to progressively learn which regions of parameter space are viable

It complements other calibration approaches like MCMC (which finds posterior distributions) and optimization (which finds point estimates).
