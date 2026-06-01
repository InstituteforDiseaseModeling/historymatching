# History Matching

A Python implementation of the Bayesian History Matching algorithm for model calibration and uncertainty quantification.

History matching iteratively constrains a model's parameter space by comparing simulation outputs against observed data through statistical emulators. It is particularly useful for calibrating expensive computational models where exhaustive parameter sweeps are impractical.

## Requirements

Python 3.9-3.12, with TensorFlow 2.18+.

## Installation

Install from the repository:

```bash
git clone https://github.com/InstituteforDiseaseModeling/historymatching
cd historymatching
pip install -e .
```

For notebook support and development:

```bash
pip install -e ".[notebooks,dev]"
```

On Apple Silicon Macs, optionally install Metal GPU acceleration:

```bash
pip install -e ".[mac]"
```

### Installation via uv

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager that can serve as a drop-in replacement for `pip`. To install with uv:

```bash
git clone https://github.com/InstituteforDiseaseModeling/historymatching
cd historymatching
uv pip install -e .
```

The optional dependency groups work the same way:

```bash
uv pip install -e ".[notebooks,dev]"
uv pip install -e ".[mac]"  # Metal GPU acceleration on Apple Silicon
```

uv is especially helpful on macOS. The TensorFlow and GPflow dependency tree (including the pinned `setuptools<81` and `tf-keras` requirements) can be slow and error-prone to resolve with `pip`, and uv's resolver handles it quickly and reliably. uv can also manage the Python interpreter itself, which makes it easy to get a supported version (3.9-3.12) without touching the system Python that macOS ships with:

```bash
uv python install 3.12
uv venv --python 3.12
uv pip install -e ".[notebooks,dev,mac]"
```

## Quick start

```python
import historymatching as hm

# Configure the engine
engine = hm.HistoryMatching(
    parameter_bounds={
        'beta': (0.1, 0.5),
        'gamma': (0.01, 0.1),
    },
    observations={
        'peak_infected': (150.0, 20.0),  # (mean, std)
        'total_cases': (500.0, 50.0),
    },
    function=my_model,                  # the simulation function
    sampling_strategy='lhs',
    emulator_type='gpr',                # or 'linear', 'glm'
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
    parameter_bounds=parameter_bounds,
    observations=observations,
    function=my_model,
    run_name='my_calibration',
)
results = engine.run(resume=True)  # continues from last committed wave
```

### NROY sampling methods

The default `ray_resample` method uses a 4-stage pipeline (LHS → ray sampling → importance sampling → maximin thinning) that efficiently explores small NROY regions:

```python
engine = hm.HistoryMatching(
    parameter_bounds=parameter_bounds,
    observations=observations,
    function=my_model,
    nroy_method='ray_resample',           # default; or 'lhs' for pure rejection
    nroy_options=dict(n_lines=30, points_per_line=100),  # optional tuning
)
```

For simple problems, pure LHS rejection is fine:

```python
engine = hm.HistoryMatching(
    parameter_bounds=parameter_bounds,
    observations=observations,
    function=my_model,
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
