# Tutorials

These tutorials walk through how to use the history matching library, from basic setup to advanced workflows. To run locally, start a Jupyter environment in the `docs/tutorials` folder.

## Getting started

- [Basic workflow](tutorials/01_basic_workflow.ipynb) — Complete walkthrough calibrating a model using the builder pattern
- [Manual workflow](tutorials/02_manual_workflow.ipynb) — Fine-grained control over each iteration with manual inspection and decision points

## Advanced workflows

- [Advanced configuration](tutorials/03_advanced_configuration.ipynb) — Custom strategies, strategy switching mid-workflow, checkpoint/resume
- [Emulator showcase](tutorials/04_emulator_showcase.ipynb) — Comparison of Linear, GLM, and GPR emulators on different test functions

## Post-calibration

- [Trajectory selection](tutorials/05_trajectory_selection.ipynb) — Select plausible `(parameter, seed)` pairs for stochastic simulation using importance resampling
- [NROY sampling methods](tutorials/06_nroy_sampling_methods.ipynb) — Comparison of sampling strategies for exploring the NROY space
