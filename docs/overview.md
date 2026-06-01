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

## Implausibility and NROY

The key quantity is **implausibility** — how many standard deviations the emulator's prediction sits from the observed target, accounting for every source of uncertainty. For a single output feature \(f\):

$$
I_f(\theta) = \frac{\lvert \, \mathbb{E}[f(\theta)] - z_f \, \rvert}{\sqrt{\operatorname{Var}_\text{em}[f(\theta)] + \sigma_{\text{obs},f}^2 + \sigma_{\text{disc},f}^2}}
$$

where \(z_f\) is the observed target, \(\sigma_{\text{obs},f}\) its uncertainty, \(\operatorname{Var}_\text{em}\) the emulator's predictive variance at parameter set \(\theta\), and \(\sigma_{\text{disc},f}\) an optional model-discrepancy term. With several features, the overall implausibility is the **maximum** across them, so a parameter set must be consistent with *every* target at once.

A parameter set is ruled out when its implausibility exceeds a **threshold** (default 3.0 — roughly "more than 3σ from at least one target"). The set of parameters *not* ruled out is the **NROY region** ("Not Ruled Out Yet"). Each wave trains emulators on new features and removes more of parameter space; the **NROY fraction** — the share of fresh prior samples that still pass all constraints — is the headline convergence diagnostic, visualised by `engine.plot_convergence()`.

History matching deliberately does *not* return a single best fit or a posterior density. It returns the whole region that could plausibly have produced the data, which is more honest about what the data can and cannot constrain.

## Key features

- **Builder pattern**: Configure workflows by assigning attributes on `HistoryMatchingBuilder`, then call `build()`
- **Interactive engine**: Step-by-step control with `step()` / `commit_step()` / `revert_step()`, or fully automated execution with `run()`
- **Multiple emulators**: Linear, GLM, Gaussian Process Regression (GPflow-based with ARD kernels), and Bayes Linear (pure NumPy/SciPy, no TensorFlow)
- **Pluggable strategies**: Swap sampling (LHS, grid, random), feature selection (automatic by mean-squared z-score or Fano factor, or manual), and emulator types at any point
- **Domain objects**: `ParameterSpace`, `ObservationData`, `EmulatorBank`, and `IterationResult` for clean data management
- **Diagnostics & plotting**: Built-in `plot_*` and `summary()` methods for convergence, NROY parameter clouds, z-scores, and emulator quality — see [Diagnostics & plotting](diagnostics.md)
- **Checkpoint/resume**: Save and restore engine state for long-running workflows

## Architecture

```
HistoryMatchingBuilder          # Configure the workflow
    ├── ParameterSpace          # Parameter bounds
    ├── ObservationData         # Target observations (mean, std)
    ├── SamplingStrategy        # How to generate samples (LHS, grid, random)
    ├── FeatureSelectionStrategy # Which outputs to emulate (auto, manual)
    └── EmulatorFactory         # Which emulator to use (linear, glm, gpr)
            │
            ▼
HistoryMatchingEngine           # Execute the workflow
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
