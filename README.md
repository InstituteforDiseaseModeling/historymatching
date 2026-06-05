# History Matching

A Python implementation of the Bayesian History Matching algorithm for model calibration and uncertainty quantification.

History matching iteratively constrains a model's parameter space by comparing simulation outputs against observed data through statistical emulators. It is particularly useful for calibrating expensive computational models where exhaustive parameter sweeps are impractical.

## Requirements

Python 3.11+, with TensorFlow 2.18+.

## Installation

> **Note:** historymatching will be published to PyPI soon. Until the first release lands, install from GitHub as shown below; afterward, `pip install historymatching` will work directly.

To use historymatching in your own project, install it from GitHub:

```bash
pip install "historymatching @ git+https://github.com/InstituteforDiseaseModeling/historymatching"
```

Add optional extras as needed — `notebooks` for the tutorial notebooks, `mac` for Metal GPU acceleration on Apple Silicon. Combine them in a single bracket to install both at once:

```bash
pip install "historymatching[notebooks] @ git+https://github.com/InstituteforDiseaseModeling/historymatching"
pip install "historymatching[notebooks,mac] @ git+https://github.com/InstituteforDiseaseModeling/historymatching"
```

If you use [uv](https://docs.astral.sh/uv/), `uv pip install` is a drop-in replacement for the `pip install` commands above: it mirrors pip's interface, resolving dependencies fresh from package metadata into the active environment without consulting a lockfile. uv's resolver is also noticeably faster and more reliable than plain pip on the TensorFlow/GPflow dependency tree (with its pinned `setuptools<81` and `tf-keras` requirements).

### Developing historymatching

To work on historymatching itself, clone the repository and set up the environment with [uv](https://docs.astral.sh/uv/) (recommended). Unlike `uv pip`, `uv sync` reads the committed `uv.lock`, creates a `.venv` automatically, and reproduces the *exact* dependency versions CI uses (pruning anything not in the lock) — so your environment matches CI:

```bash
git clone https://github.com/InstituteforDiseaseModeling/historymatching
cd historymatching
uv sync --extra notebooks --extra test    # add --extra mac on Apple Silicon
```

Run commands inside that environment with `uv run`, e.g. `uv run pytest tests/`.

Prefer plain pip? Install in editable mode instead (this resolves dependencies fresh rather than from the lockfile):

```bash
pip install -e ".[notebooks,test]"
```

## Quick start

```python
import historymatching as hm

# Configure the engine
engine = hm.HistoryMatching(
    function=my_model,                  # the simulation function
    bounds={
        'beta': (0.1, 0.5),
        'gamma': (0.01, 0.1),
    },
    observations={
        'peak_infected': (150.0, 20.0),  # (mean, std)
        'total_cases': (500.0, 50.0),
    },
    sampling_strategy='lhs',
    emulator_type='bayes_linear',       # default; or 'gpr', 'linear', 'glm'
    n_samples=500,
    max_iterations=5,
    output_dir='./hm_output',
    run_name='my_calibration',
)

# Run
results = engine.run()

# Emulators, diagnostics, and checkpoints are saved automatically to
# hm_output/my_calibration/wave1/, wave2/, etc.

# Get NROY samples (filtered through ALL emulators)
nroy = engine.get_nroy_samples(10000)
```

### Resume from checkpoint

```python
engine = hm.HistoryMatching(
    function=my_model,
    bounds=parameter_bounds,
    observations=observations,
    run_name='my_calibration',
)
results = engine.run(resume=True)  # continues from last committed wave
```

### NROY sampling methods

The default `auto` method draws with Latin Hypercube sampling first and only escalates to the more expensive `ray_resample` pipeline (LHS → ray sampling → importance sampling → maximin thinning) when LHS acceptance is too low to fill small NROY regions:

```python
engine = hm.HistoryMatching(
    function=my_model,
    bounds=parameter_bounds,
    observations=observations,
    nroy_method='auto',                   # default; LHS first, escalates to 'ray_resample'
    nroy_options=dict(n_lines=30, points_per_line=100),  # optional tuning for ray_resample
)
```

For simple problems, pure LHS rejection is fine:

```python
engine = hm.HistoryMatching(
    function=my_model,
    bounds=parameter_bounds,
    observations=observations,
    nroy_method='lhs',
)
```

## Documentation

Full documentation, tutorials, and API reference are available in the `docs/` folder. To build locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## License

MIT License. See [LICENSE](LICENSE) for details.
