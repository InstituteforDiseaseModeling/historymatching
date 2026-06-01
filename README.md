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

# Configure the engine. The builder is configured by assigning to its
# attributes; every setting is optional and has a sensible default.
builder = hm.HistoryMatchingBuilder.from_data(
    parameter_bounds={
        'beta': (0.1, 0.5),
        'gamma': (0.01, 0.1),
    },
    observations={
        'peak_infected': (150.0, 20.0),  # (mean, std)
        'total_cases': (500.0, 50.0),
    },
)
builder.sampling_strategy = 'lhs'
builder.emulator_type = 'gpr'          # or 'bayes_linear', 'linear', 'glm'
builder.n_samples = 500
builder.max_iterations = 5
builder.output_dir = './hm_output'
builder.run_name = 'my_calibration'
engine = builder.build()

# Provide a simulation function and run
engine.set_simulation_function(my_model)
results = engine.run()

# Emulators, diagnostics, and checkpoints are saved automatically to
# hm_output/my_calibration/wave1/, wave2/, etc.

# Inspect and visualise the result
print(engine.summary())            # NROY ranges + per-wave convergence
engine.plot_convergence()          # NROY fraction per wave
engine.plot_nroy()                 # corner plot of the plausible region

# Get NROY samples (filtered through ALL emulators)
nroy = engine.get_nroy_samples(10000)
```

### Resume from checkpoint

```python
builder.run_name = 'my_calibration'
engine = builder.build()
engine.set_simulation_function(my_model)
results = engine.run(resume=True)  # continues from last committed wave
```

### NROY sampling methods

The default `auto` method uses a multi-stage pipeline (LHS rejection, escalating to ray sampling + PCA-oriented importance sampling + maximin thinning) that efficiently explores small NROY regions:

```python
builder.nroy_method = 'auto'   # default; or 'ray', or 'lhs' for pure rejection
builder.nroy_options = {'n_lines': 30, 'points_per_line': 100}  # optional tuning
engine = builder.build()
```

For simple problems, pure LHS rejection is fine:

```python
builder.nroy_method = 'lhs'
engine = builder.build()
```

## Documentation

Full documentation, tutorials, and API reference are available in the `docs/` folder. To build locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## License

MIT License. See [LICENSE](LICENSE) for details.
