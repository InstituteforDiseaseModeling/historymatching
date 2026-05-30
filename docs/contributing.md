# Contributing

## Development setup

```bash
git clone https://github.com/InstituteforDiseaseModeling/historymatching
cd historymatching
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/
```

## Code style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting. Run checks with:

```bash
ruff check .
```

## Pull requests

1. Create a feature branch from `main`
2. Make your changes
3. Ensure all tests pass
4. Submit a pull request
