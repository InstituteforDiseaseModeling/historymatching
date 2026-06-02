# History Matching

[![Tests](https://github.com/InstituteforDiseaseModeling/historymatching/actions/workflows/tests.yml/badge.svg)](https://github.com/InstituteforDiseaseModeling/historymatching/actions)

History Matching is a Python library for Bayesian History Matching — an iterative algorithm for calibrating computational models against observed data. It is designed for problems where simulations are expensive and exhaustive parameter sweeps are impractical.

The library provides a modern, object-oriented API built around the **Builder/Engine** pattern, with pluggable strategies for sampling, feature selection, and emulation.

## Requirements

Python 3.11+, with TensorFlow 2.18+.

## Installation

```bash
pip install -e .
```

See [Installation](installation.md) for full details including optional dependencies.

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
    emulator_type='gpr',
    n_samples=500,
    max_iterations=5,
)

# Run
results = engine.run()
```

See the [Usage guide](usage.md) for a complete walkthrough, or jump into the [Tutorials](tutorials.md).

## Contributing

Questions or comments can be directed to the project's [GitHub](https://github.com/InstituteforDiseaseModeling/historymatching) page.

## Disclaimer

The code in this repository was developed by IDM and other collaborators to support our joint research on model calibration. We've made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as permitted under the MIT License.
