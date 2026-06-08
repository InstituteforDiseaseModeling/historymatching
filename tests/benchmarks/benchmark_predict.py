"""
Benchmark: numba fast-predict path vs the GPflow GPR predict path.

History matching spends most of its wall-clock time in the NROY rejection
loop, calling ``emulator.predict`` on large batches of candidate points.  To
make this affordable the library extracts a :class:`FastGPRPredictor` from each
trained GPflow model and runs predictions in pure numpy/numba, bypassing
TensorFlow's Python->graph overhead.

This script trains a GPR emulator on synthetic data, builds the fast predictor
from it, and times both paths on a large batch.  It also checks that the two
paths agree on the predicted mean (a correctness guard), so the benchmark
doubles as a regression test for the fast path.

Run with:

    uv run --python 3.13 --extra test python benchmarks/benchmark_predict.py
    uv run --python 3.13 --extra test python benchmarks/benchmark_predict.py --n-test 50000 --repeats 5

Notes:
  * The first call to each path is excluded from timing (JIT / graph warm-up).
  * If numba is not installed, FastGPRPredictor transparently falls back to a
    vectorized numpy implementation; the speedup will be smaller but the script
    still runs.
"""

import argparse
import time

import numpy as np
import pandas as pd

from historymatching.emulators.fast_predict import _HAS_NUMBA, FastGPRPredictor
from historymatching.emulators.gpr import GPR


def _make_training_data(n_train, n_dims, seed):
    """Generate a smooth synthetic function to train the emulator on."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=(n_train, n_dims))
    # A smooth, anisotropic target so the GP has something real to fit.
    freqs = np.linspace(1.0, 2.0, n_dims)
    y = np.sin(x @ freqs) + 0.5 * (x ** 2).sum(axis=1)
    x_df = pd.DataFrame(x, columns=[f"p{i}" for i in range(n_dims)])
    y_df = pd.DataFrame({"feature": y})
    return x_df, y_df


def _time_call(fn, repeats):
    """Return the best (minimum) wall-clock time over ``repeats`` calls."""
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=200, help="training points")
    parser.add_argument("--n-dims", type=int, default=4, help="parameter dimensions")
    parser.add_argument("--n-test", type=int, default=20000, help="prediction batch size")
    parser.add_argument("--repeats", type=int, default=3, help="timed repeats (best is reported)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = parser.parse_args()

    print(f"numba available: {_HAS_NUMBA}")
    print(f"config: n_train={args.n_train}, n_dims={args.n_dims}, "
          f"n_test={args.n_test:,}, repeats={args.repeats}\n")

    # --- Train the GPflow GPR emulator ---
    x_df, y_df = _make_training_data(args.n_train, args.n_dims, args.seed)
    emulator = GPR(x_df, y_df, test_fraction=0.1)
    emulator.train()

    # --- Build the fast predictor from the trained model ---
    fast = FastGPRPredictor.from_emulator(emulator)

    # --- Prediction batch (raw parameter space) ---
    rng = np.random.default_rng(args.seed + 1)
    x_test = rng.uniform(0.0, 1.0, size=(args.n_test, args.n_dims))
    x_test_df = pd.DataFrame(x_test, columns=x_df.columns)

    # --- Warm up both paths (excluded from timing) ---
    gpflow_warm = emulator.predict(x_test_df)
    fast_mean, _ = fast.predict(x_test)

    # --- Correctness guard: the two paths must agree on the mean ---
    gpflow_mean = gpflow_warm.get_mean().to_numpy()
    max_abs_diff = float(np.max(np.abs(gpflow_mean - fast_mean)))
    scale = float(np.std(gpflow_mean)) or 1.0
    rel_diff = max_abs_diff / scale
    print(f"mean agreement: max|Δ|={max_abs_diff:.3e}  (relative={rel_diff:.3e})")
    assert rel_diff < 1e-4, (
        f"fast path disagrees with GPflow (relative diff {rel_diff:.3e}); "
        f"the numba kernel may be out of sync with the GPflow model"
    )

    # --- Time both paths ---
    t_gpflow = _time_call(lambda: emulator.predict(x_test_df), args.repeats)
    t_fast = _time_call(lambda: fast.predict(x_test), args.repeats)

    speedup = t_gpflow / t_fast if t_fast > 0 else float("inf")
    print()
    print(f"{'path':<22}{'best time (s)':>16}")
    print(f"{'-' * 38}")
    print(f"{'GPflow predict':<22}{t_gpflow:>16.4f}")
    print(f"{'fast (numba/numpy)':<22}{t_fast:>16.4f}")
    print(f"{'-' * 38}")
    print(f"speedup: {speedup:.1f}x")


if __name__ == "__main__":
    main()
