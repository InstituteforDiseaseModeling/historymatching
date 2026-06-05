# History Matching — Developer Guide for Claude

## What this project is

A Python library implementing the **Bayesian History Matching** algorithm for calibrating complex disease models. It is developed at the **Institute for Disease Modeling (IDM)** at the Gates Foundation.

History matching iteratively constrains a model's parameter space by comparing simulation outputs against observed data. In each iteration it:
1. Samples points in parameter space (e.g., Latin Hypercube Sampling)
2. Runs the user's simulator at those points
3. Selects informative output features (via Fano factor or manually)
4. Trains statistical emulators (Gaussian Process Regression by default) on those outputs
5. Uses the emulators to identify and discard "implausible" regions of parameter space
6. Repeats with a smaller parameter space until convergence

The library provides the scaffolding. Users supply their own simulator function.

## Environment

This project uses `uv`. Always prefix Python/pytest commands with `uv run`:

```bash
uv run pytest tests/
uv run python -c "import historymatching"
```

Set up the dev environment (reproduces the locked versions CI uses):
```bash
uv sync --extra notebooks --extra test
```

On Apple Silicon, optionally add Metal GPU support:
```bash
uv sync --extra notebooks --extra test --extra mac
```

## Running tests

```bash
uv run pytest tests/           # all tests
uv run pytest tests/ -x -q    # fail fast, quiet
```

234 tests, runs in ~10 seconds. No network or external dependencies required.

## Code structure

```
historymatching/         # flat package — everything at the top level
    __init__.py           # re-exports all public API; users just: import historymatching as hm
    engine.py             # HistoryMatching — single public class; configures and runs the iterative loop (HistoryMatchingEngine is a back-compat alias)
    parameter_space.py    # ParameterSpace — wraps parameter bounds (DataFrame)
    observation_data.py   # ObservationData — wraps target observations (mean, std)
    emulator_bank.py      # EmulatorBank — stores trained emulators by iteration and feature
    iteration_result.py   # IterationResult — immutable result from one iteration
    feature_selection.py  # AutoFeatureSelection, ManualFeatureSelection
    sampling.py           # LatinHypercubeSampling, GridSampling, RandomSampling, SamplingStrategyFactory
    plotting.py           # Composable plot_* helpers (take primitive data, return Axes); re-exported at top level
    utils.py              # Column name constants and helper functions
    emulators/            # Emulator implementations (the one subdirectory)
        base.py           # BaseEmulator abstract class
        gpr.py            # Gaussian Process Regression (uses GPflow/TensorFlow)
        glm.py            # Generalized Linear Model
        linear.py         # Linear regression
        bayes_linear.py   # Bayes linear emulator (the default)
        factory.py        # EmulatorFactory — creates emulators by type name
        results.py        # EmulationResults dataclass

tests/
    fixtures.py           # Shared test data factories (TestDataFactory, TestAssertions)
    test_*.py             # One test file per source module
```

## Key concepts

**The typical user workflow:**
```python
import historymatching as hm

engine = hm.HistoryMatching(
    function=my_simulator,
    bounds={'beta': (0.1, 0.5), 'gamma': (0.01, 0.1)},
    observations={'peak_infected': (150.0, 20.0)},  # (mean, std)
    emulator_type='bayes_linear',
    n_samples=500,
)
results = engine.run()
```

**Emulator types:** `'bayes_linear'` (default), `'gpr'`, `'glm'`, `'linear'`

**Sampling strategies:** `'lhs'` (Latin Hypercube, default), `'grid'`, `'random'`

**Feature selection:** `AutoFeatureSelection` (Fano factor), `ManualFeatureSelection` (explicit list)

**Implausibility:** A point is "implausible" if the emulator prediction differs from the observed target by more than the threshold (default 3.0) in units of combined variance. Implausible points are excluded in subsequent iterations.

## Key dependencies

- **GPflow + TensorFlow** — for Gaussian Process emulators (the main emulator type)
- **tf-keras** — required by GPflow with TF 2.18+
- `setuptools<81` is pinned as a dependency because GPflow uses `pkg_resources` which was removed in setuptools 81

## Documentation plans

Docs will be migrated to **mkdocs** (see `pyproject.toml [project.optional-dependencies] docs`). The old Sphinx docs have been removed. Look at `~/GIT/tbsim`, `starsim`, or `fpsim` for mkdocs patterns used in sibling IDM projects.

## Things to know

- Do not add literal `\n` characters in notebook metadata sections — they cause JSON formatting errors
- The `examples/` directory contains Jupyter notebooks; `04_emulator_showcase.ipynb.bak` is intentionally kept (in use by another agent)
- The GPR emulator uses `compile=False` in the Scipy optimizer — removing it causes a hang on TF 2.20
- The `architecture.md` file contains the original design proposal with useful pseudocode and data structure diagrams; it predates the current implementation
