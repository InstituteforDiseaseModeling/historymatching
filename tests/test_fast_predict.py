"""
Regression test for the numba/numpy fast-predict path.

History matching spends most of its time calling `emulator.predict` on large
batches inside the NROY rejection loop, so the GPR emulator extracts a
`FastGPRPredictor` that bypasses TensorFlow. This test is the in-suite
guard (a fast counterpart to `tests/benchmarks/benchmark_predict.py`) that the
fast path stays numerically in sync with the GPflow model — if the numba kernel
drifts out of sync with the trained GPR, the fast path would silently return
wrong NROY decisions.
"""

import numpy as np
import pandas as pd
import pytest

gpflow = pytest.importorskip("gpflow")

from historymatching.emulators.fast_predict import FastGPRPredictor
from historymatching.emulators.gpr import GPR


def _make_training_data(n_train, n_dims, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=(n_train, n_dims))
    freqs = np.linspace(1.0, 2.0, n_dims)
    y = np.sin(x @ freqs) + 0.5 * (x ** 2).sum(axis=1)
    x_df = pd.DataFrame(x, columns=[f"p{i}" for i in range(n_dims)])
    y_df = pd.DataFrame({"feature": y})
    return x_df, y_df


def test_fast_predictor_matches_gpflow_mean():
    """The fast predictor's mean must agree with the GPflow predict path to
    within a tight relative tolerance."""
    n_dims = 3
    x_df, y_df = _make_training_data(n_train=120, n_dims=n_dims, seed=0)
    emulator = GPR(x_df, y_df, test_fraction=0.1)
    emulator.train()

    fast = FastGPRPredictor.from_emulator(emulator)

    rng = np.random.default_rng(1)
    x_test = rng.uniform(0.0, 1.0, size=(2000, n_dims))
    x_test_df = pd.DataFrame(x_test, columns=x_df.columns)

    gpflow_mean = emulator.predict(x_test_df).get_mean().to_numpy()
    fast_mean, _ = fast.predict(x_test)

    max_abs_diff = float(np.max(np.abs(gpflow_mean - fast_mean)))
    scale = float(np.std(gpflow_mean)) or 1.0
    rel_diff = max_abs_diff / scale
    assert rel_diff < 1e-4, (
        f"fast path disagrees with GPflow (relative diff {rel_diff:.3e}); "
        f"the numba kernel may be out of sync with the GPflow model"
    )
