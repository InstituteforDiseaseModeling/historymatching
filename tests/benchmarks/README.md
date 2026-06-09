# Benchmarks

Performance benchmarks for History Matching. These are standalone scripts (not
part of the pytest suite) that you run manually to measure and sanity-check
hot paths.

## `benchmark_predict.py`

Compares the numba **fast-predict** path against the **GPflow** GPR predict
path — the dominant cost in the NROY rejection-sampling loop. It trains a GPR
emulator on synthetic data, extracts a `FastGPRPredictor`, verifies the two
paths agree on the predicted mean, then times both on a large batch.

```bash
# Default: 200 train points, 4 dims, 20k prediction batch
uv run --python 3.13 --extra test python benchmarks/benchmark_predict.py

# Larger batch, more repeats
uv run --python 3.13 --extra test python benchmarks/benchmark_predict.py --n-test 50000 --repeats 5
```

The script asserts that the fast path matches GPflow to a relative tolerance of
`1e-4`, so it also acts as a regression guard on the numba kernel. Reported
speedups depend on hardware, batch size, and whether numba is installed (it
falls back to vectorized numpy otherwise).
