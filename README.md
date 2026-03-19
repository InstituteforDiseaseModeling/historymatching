# History Matching

A Python implementation of the Bayesian History Matching algorithm for model calibration and uncertainty quantification.

History matching iteratively constrains a model's parameter space by comparing simulation outputs against observed data through statistical emulators. It is particularly useful for calibrating expensive computational models where exhaustive parameter sweeps are impractical.

## Requirements

Python 3.9-3.12, with TensorFlow 2.18+.

## Installation

Install from the repository:

```bash
git clone https://github.com/InstituteforDiseaseModeling/history_matching
cd history_matching
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

## Quick start

```python
import history_matching as hm

# Configure the engine
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
engine = builder \
    .with_sampling_strategy('lhs') \
    .with_emulator_type('gpr') \
    .with_samples_per_iteration(500) \
    .with_max_iterations(5) \
    .build()

# Provide a simulation function and run
engine.set_simulation_function(my_model)
results = engine.run()
```

## Documentation

Full documentation, tutorials, and API reference are available in the `docs/` folder. To build locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## License

MIT License. See [LICENSE](LICENSE) for details.
