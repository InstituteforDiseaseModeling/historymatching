# Contributing

Thanks for your interest in improving History Matching. This guide covers how to
set up a development environment, run the tests, and submit changes.

## Development setup

Set up the environment with [uv](https://docs.astral.sh/uv/) (recommended). This
creates a `.venv` and installs the project with the chosen extras:

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
4. Bump the version and update the changelog in the same pull request, so
   `main` always says which version it is (see below).
5. Submit a pull request describing the change and its motivation.

## Versions and the changelog

Every pull request that changes the package, its behavior or its results bumps
`__version__` and sets `__versiondate__` in `historymatching/__init__.py`, and
adds its entries under a heading for that version at the top of
`CHANGELOG.md` (`## Version X.Y.Z (YYYY-MM-DD)`). There is no "Unreleased"
section. Bump the patch number for fixes that leave results unchanged, the
minor number for new features or changed results, and the major number for
breaking API changes. Pull requests that touch only tests, docs or CI don't
bump the version. If another open pull request already takes the next version
number, take the one after it, and rebase if that pull request changes.

Questions or bug reports are welcome on the
[issue tracker](https://github.com/InstituteforDiseaseModeling/historymatching/issues).
