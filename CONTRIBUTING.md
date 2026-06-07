# Contributing

Thanks for your interest in improving History Matching. This guide covers how to
set up a development environment, run the tests, and submit changes.

## Development setup

Set up the environment with [uv](https://docs.astral.sh/uv/) (recommended). This
reads the committed `uv.lock` and reproduces the exact dependency versions CI uses:

```bash
git clone https://github.com/InstituteforDiseaseModeling/historymatching
cd historymatching
uv sync --extra notebooks --extra test    # add --extra mac on Apple Silicon
```

Or with plain pip in editable mode:

```bash
pip install -e ".[notebooks,test]"
```

## Running tests

With uv:

```bash
uv run pytest tests/           # all tests (~10 s, no network required)
uv run pytest tests/ -x -q     # fail fast, quiet
```

The tutorial notebooks are executed end-to-end separately (pytest does not run
them):

```bash
bash docs/tutorials/run_tutorials.sh
```

## Pull requests

1. Create a feature branch from `main`.
2. Make your changes, adding or updating tests as appropriate.
3. Ensure `uv run pytest tests/` passes and, if you touched the tutorials, that
   `docs/tutorials/run_tutorials.sh` passes.
4. Submit a pull request describing the change and its motivation.

Questions or bug reports are welcome on the
[issue tracker](https://github.com/InstituteforDiseaseModeling/historymatching/issues).
