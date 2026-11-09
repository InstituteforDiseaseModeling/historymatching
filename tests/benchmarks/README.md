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

## `benchmark_mle_nugget.py`

Checks the Bayes-linear hyperparameter fit, with fixed and learned nuggets. A fixed
nugget inside `nugget_bounds` is a feasible point of the learned-nugget search,
so a working learned fit never ends with a higher NLL than a fixed nugget on
the same training data. The script fits a grid of synthetic scenarios (a smooth
function of 5 active inputs plus Gaussian noise, the other inputs inactive)
with `nugget='mle'` and with fixed nuggets of 1e-6 (the default), 0.01 and
0.03, on the same train/test split, and writes NLL, nugget, held-out R²,
iterations, log correlation lengths on their lower bound and wall time to a
CSV. `--compare` counts, per CSV, the learned fits more than `--tol` NLL above
a fixed-nugget fit or above the lowest NLL any fit reached on that scenario,
and the learned nuggets on a bound. For each fixed nugget, it counts the fits
more than `--tol` above the lowest NLL any CSV reached at that nugget, and the
fits that stopped after at most one iteration with a log correlation length on
its lower bound.

```bash
uv run python tests/benchmarks/benchmark_mle_nugget.py --out new.csv
PYTHONPATH=/path/to/other/checkout uv run python tests/benchmarks/benchmark_mle_nugget.py --out old.csv
uv run python tests/benchmarks/benchmark_mle_nugget.py --compare old.csv new.csv
```

`--n`, `--d`, `--noise`, `--kind` and `--seeds` set the grid (by default 48
scenarios: 300 and 1,000 points, 10 and 21 dimensions, noise sd 0.1, 0.3 and
0.6, two functions, two seeds). `--fit-only mle` skips the fixed-nugget fits.
