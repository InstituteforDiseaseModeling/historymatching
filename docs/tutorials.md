# Tutorials

These tutorials walk through how to use the history matching library, from basic setup to advanced workflows. To run locally, start a Jupyter environment in the `docs/tutorials` folder.

## Getting started

- [Basic workflow](tutorials/01_basic_workflow.ipynb) — Complete walkthrough calibrating a model with the single-constructor `HistoryMatching` API
- [Manual workflow](tutorials/02_manual_workflow.ipynb) — Fine-grained control over each iteration with manual inspection and decision points

## During calibration

- [Advanced configuration](tutorials/03_advanced_configuration.ipynb) — Custom strategies, strategy switching mid-workflow, checkpoint/resume
- [Emulator showcase](tutorials/04_emulator_showcase.ipynb) — Comparison of Bayes linear, linear, GLM, and GPR emulators; a key takeaway is that **Bayes linear matches GPR quality at a fraction of the cost** (hence the default)
- [NROY sampling methods](tutorials/05_nroy_sampling_methods.ipynb) — Comparison of sampling strategies for exploring the non-implausible space
- [Visualization](tutorials/06_visualization.ipynb) — Composable plotting helpers for NROY corner plots, convergence, marginals, and emulator quality

## After calibration

- [Trajectory selection](tutorials/07_trajectory_selection.ipynb) — Draw plausible `(θ, seed)` pairs (here `θ = (β, γ)`) for stochastic simulation using importance resampling
