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

## `benchmark_nroy_screening.py`

Times `nroy_mask` with and without the Bayes-linear screening
(`BayesLinear.predict_bounds`), on synthetic data shaped like a calibration
wave: 21 parameters, 3,750 training points, two outputs, and 100,000 uniform
candidates. Three scenarios place the targets so that about 0.3%, 1.5% and 77%
of the box is plausible. It checks that the screened and unscreened masks are
identical and reports the fraction of emulator evaluations decided without
`predict`.

```bash
uv run python tests/benchmarks/benchmark_nroy_screening.py
# Linux (OpenBLAS / MKL): limit BLAS threads through threadpoolctl
uv run python tests/benchmarks/benchmark_nroy_screening.py --threads 1
# macOS (Accelerate ignores threadpoolctl)
VECLIB_MAXIMUM_THREADS=1 uv run python tests/benchmarks/benchmark_nroy_screening.py
```

`--unscreened-only` runs only the baseline, which also works on package
versions without screening. Point `PYTHONPATH` at another checkout's package to
compare against it.
