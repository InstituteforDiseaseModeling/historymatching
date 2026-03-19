# Tutorials

These tutorials walk through how to use the history matching library, from basic setup to advanced workflows. To run locally, start a Jupyter environment in the `docs/tutorials` folder.

## Getting started

- [Quick start](tutorials/01_basic_sir_model.ipynb) — Complete walkthrough calibrating an SIR model using the builder pattern
- [Interactive workflow](tutorials/01_basic_sir_oop_demo.ipynb) — Step-by-step execution with manual inspection and decision points
- [Manual control](tutorials/01_manual_history_matching_oop.ipynb) — Fine-grained control over each iteration using the builder pattern

## Advanced workflows

- [Advanced configuration](tutorials/02_advanced_configuration.ipynb) — Custom strategies, strategy switching mid-workflow, checkpoint/resume
- [Automatic workflow](tutorials/03_automatic_workflow.ipynb) — Fully automated multi-iteration execution with convergence detection

## Emulators

- [Emulator showcase](tutorials/04_emulator_showcase.ipynb) — Comparison of Linear, GLM, and GPR emulators on different test functions

## Post-calibration

- [Trajectory selection](tutorials/05_trajectory_selection.ipynb) — Select plausible `(parameter, seed)` pairs for stochastic simulation using importance resampling
