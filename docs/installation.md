# Installation

## Prerequisites

- Python 3.9+ (3.11+ recommended)
- TensorFlow 2.18+ (installed automatically)

## Install from PyPI

historymatching is published on PyPI, so the usual one-liner works:

```bash
pip install historymatching
```

Add optional extras as needed — `notebooks` for the tutorial notebooks, `mac` for
Metal GPU acceleration on Apple Silicon:

```bash
pip install "historymatching[notebooks]"
pip install "historymatching[notebooks,mac]"
```

If you use [uv](https://docs.astral.sh/uv/), `uv pip install historymatching` is a
faster drop-in replacement for `pip install`.

!!! note "Apple Silicon (Metal GPU)"
    The `mac` extra installs the TensorFlow Metal plugin for GPU acceleration. Note
    that Cholesky decomposition (used internally by GPflow) is not yet implemented on
    Metal, so GPR training falls back to CPU for that operation.

## Development setup (recommended: uv)

To work on historymatching itself — or just to get an environment that exactly matches
CI — use [uv](https://docs.astral.sh/uv/). Unlike a bare `pip install`, `uv sync`
creates an isolated `.venv`, **manages the Python interpreter for you** (so the build
doesn't depend on whatever Python happens to be on your machine), and installs the
project with the extras you ask for:

```bash
git clone https://github.com/InstituteforDiseaseModeling/historymatching
cd historymatching
uv sync --extra notebooks --extra test    # add --extra mac on Apple Silicon
```

Run anything inside that environment with `uv run`, e.g. `uv run pytest tests/` or
`uv run jupyter lab`.

(Plain `pip install -e ".[test]"` also works for an editable checkout, but it installs
into whatever Python and environment are already active.)

## Verify installation

```python
import historymatching as hm

engine = hm.HistoryMatching(
    bounds={'x': (0, 1)},
    observations={'y': (0.5, 0.1)},
)
print(engine)  # Should print engine status
```
