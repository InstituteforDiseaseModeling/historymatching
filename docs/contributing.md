# Contributing

## Development setup

```bash
git clone https://github.com/InstituteforDiseaseModeling/historymatching
cd historymatching
uv sync --extra notebooks --extra test
```

Or with plain pip in editable mode:

```bash
pip install -e ".[test]"
```

## Running tests

```bash
pytest tests/
```

## Pull requests

1. Create a feature branch from `main`
2. Make your changes
3. Ensure all tests pass
4. Submit a pull request
