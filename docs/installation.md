# Installation

## Prerequisites

- Python 3.9-3.12
- TensorFlow 2.18+ (installed automatically)

## Standard installation

Clone the repository and install:

```bash
git clone https://github.com/InstituteforDiseaseModeling/history_matching
cd history_matching
pip install -e .
```

## Optional dependencies

### Notebooks

For running the tutorial notebooks:

```bash
pip install -e ".[notebooks]"
```

### Apple Silicon (Metal GPU)

On M1/M2/M3 Macs, TensorFlow can use the Metal plugin for GPU acceleration:

```bash
pip install -e ".[mac]"
```

!!! note
    The Metal plugin provides GPU acceleration for many TensorFlow operations, but Cholesky decomposition (used internally by GPflow) is not yet implemented on Metal. GPR training will fall back to CPU for this operation.

### Development

For running tests and linting:

```bash
pip install -e ".[dev]"
```

### Documentation

For building the documentation locally:

```bash
pip install -e ".[docs]"
```

## Using uv

If you use [uv](https://docs.astral.sh/uv/) for environment management:

```bash
uv sync --extra notebooks --extra test
```

On Apple Silicon:

```bash
uv sync --extra notebooks --extra test --extra mac
```

## Verify installation

```python
import history_matching as hm

engine = (hm.HistoryMatchingBuilder
    .from_data(
        parameter_bounds={'x': (0, 1)},
        observations={'y': (0.5, 0.1)},
    )
    .build()
)
print(engine)  # Should print engine status
```
