"""
Benchmark: screening in `nroy_mask` for Bayes-linear emulators.

`nroy_mask` decides most candidates from `BayesLinear.predict_bounds()`, cheap
bounds on each point's mean and variance, and calls `predict` only for the
points those bounds can't decide. This script times the screened mask against
the same emulators with the bounds hidden (every point through `predict`),
checks that the two masks are identical, and reports the fraction of
emulator evaluations decided without `predict`.

The data are synthetic, shaped like a TB calibration wave: 21 parameters,
3,750 training points, a smooth function plus run-to-run noise, two outputs,
and uniform candidates in the unit box. Each scenario places the targets so
that a different fraction of the box is plausible.

Run with (from the repository root):

    uv run python tests/benchmarks/benchmark_nroy_screening.py
    uv run python tests/benchmarks/benchmark_nroy_screening.py --n-candidates 1000000

BLAS threads: on Linux (OpenBLAS, MKL) pass `--threads 1`, which uses
threadpoolctl. Apple's Accelerate ignores threadpoolctl; set
`VECLIB_MAXIMUM_THREADS=1` in the environment instead. `--unscreened-only`
runs just the baseline, which also works on package versions without
screening, for comparing against an earlier release.

Notes:
  * Hyperparameters (correlation lengths, nugget) are fitted on a subsample
    of `--n-fit` points, then the emulator is conditioned on all training
    points with them. Fitting on 3,750 points takes far longer than the
    benchmark and doesn't change what is being timed.
  * Candidates go through `nroy_mask` in batches of `--batch-size`, as
    `HistoryMatching.in_nroy(batch_size=...)` does.
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

import historymatching as hm
from historymatching.nroy_sampling import nroy_mask

SCENARIOS = {
    # name: (target quantile of the predicted mean, target sd as a fraction of the output sd)
    'hard': (0.01, 0.01),
    'medium': (0.05, 0.02),
    'easy': (0.50, 0.50),
}


class Unscreened(hm.BaseEmulator):
    """The wrapped emulator without variance bounds: every point goes through `predict`."""

    def __init__(self, emulator):
        super().__init__()
        self.emulator = emulator
        self.training_complete = True
        self.rows_predicted = 0

    def train(self):
        pass

    def predict(self, x):
        self.rows_predicted += len(x)
        return self.emulator.predict(x)

    def print_emulator_description(self):
        return 'unscreened wrapper'

    def get_hyperparameters(self):
        return {}


class Counted(Unscreened):
    """The wrapped emulator with its bounds, counting rows screened and rows predicted."""

    def __init__(self, emulator):
        super().__init__(emulator)
        self.rows_tested = 0

    def predict_bounds(self, x):
        self.rows_tested += len(x)
        return self.emulator.predict_bounds(x)


def make_function(n_dims, seed):
    """A smooth function of `n_dims` parameters with decaying sensitivities."""
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=n_dims) * np.geomspace(3.0, 0.1, n_dims)
    curvature = rng.uniform(0, 1, n_dims) * np.geomspace(2.0, 0.05, n_dims)
    return lambda x: np.sin(x @ weights) + (x ** 2) @ curvature


def make_emulator(f, x, noise_sd, n_fit, rng):
    """Bayes-linear emulator: hyperparameters fitted on `n_fit` points, conditioned on all."""
    columns = [f'p{i}' for i in range(x.shape[1])]
    y = f(x) + rng.normal(0, noise_sd, len(x))
    fit = rng.choice(len(x), n_fit, replace=False)
    em = hm.BayesLinear(pd.DataFrame(x[fit], columns=columns),
                        pd.DataFrame({'y': y[fit]}), test_fraction=0.1, nugget='mle')
    em.train()

    # Condition on every training point with the fitted theta and nugget, as
    # the end of `BayesLinear.train` does.
    X_norm = em._normalize_x(x)
    y_z = (y - em._y_mean) / em._y_std
    G = em._design_matrix(X_norm)
    em._beta_hat = np.linalg.solve(G.T @ G, G.T @ y_z)
    residuals = y_z - G @ em._beta_hat
    em._L, em._lower = cho_factor(em._correlation_with_noise(X_norm, em._theta, em.nugget))
    em._alpha = cho_solve((em._L, em._lower), residuals)
    em._sigma2 = float(residuals @ em._alpha) / len(x)
    em._X_train_norm = X_norm
    em._residuals = residuals
    return em, y


def run_mask(bank, candidates, observations, threshold, batch_size):
    """`nroy_mask` over the candidates in batches; returns (mask, seconds)."""
    columns = list(candidates.columns)
    t0 = time.perf_counter()
    mask = np.concatenate([
        nroy_mask(candidates.iloc[s:s + batch_size], bank, observations, threshold, columns)
        for s in range(0, len(candidates), batch_size)])
    return mask, time.perf_counter() - t0


def blas_description(threads):
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError:
        return 'threadpoolctl not installed', None
    info = threadpool_info()
    blas = [i for i in info if i.get('user_api') == 'blas']
    if threads is not None and not blas:
        print('WARNING: threadpoolctl found no BLAS it can limit (Accelerate?); '
              '--threads has no effect. Use VECLIB_MAXIMUM_THREADS / OMP_NUM_THREADS.')
    limiter = threadpool_limits(threads, user_api='blas') if threads is not None else None
    desc = ', '.join(f"{i['internal_api']} {i.get('version')} ({i['num_threads']} threads)"
                     for i in (threadpool_info() if limiter else info)
                     if i.get('user_api') == 'blas') or 'BLAS not visible to threadpoolctl'
    return desc, limiter


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-train', type=int, default=3750)
    parser.add_argument('--n-dims', type=int, default=21)
    parser.add_argument('--n-fit', type=int, default=400, help='points used to fit hyperparameters')
    parser.add_argument('--n-candidates', type=int, default=100_000)
    parser.add_argument('--batch-size', type=int, default=2000)
    parser.add_argument('--threshold', type=float, default=3.0)
    parser.add_argument('--threads', type=int, default=None, help='BLAS threads (threadpoolctl)')
    parser.add_argument('--scenarios', default=','.join(SCENARIOS))
    parser.add_argument('--unscreened-only', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    desc, _limiter = blas_description(args.threads)
    print(f'BLAS: {desc}')
    print(f'config: n_train={args.n_train}, n_dims={args.n_dims}, '
          f'n_candidates={args.n_candidates:,}, batch={args.batch_size}, threshold={args.threshold}')

    np.random.seed(args.seed)  # BaseEmulator's train/test split
    rng = np.random.default_rng(args.seed)
    x_train = rng.uniform(size=(args.n_train, args.n_dims))
    emulators, outputs = {}, {}
    t0 = time.perf_counter()
    for k in range(2):
        f = make_function(args.n_dims, args.seed + 100 + k)
        noise_sd = 0.05 * np.std(f(x_train))
        emulators[f'y{k}'], outputs[f'y{k}'] = make_emulator(f, x_train, noise_sd, args.n_fit, rng)
    print(f'emulators ready [{time.perf_counter() - t0:.0f}s]')
    for name, em in emulators.items():
        ys2 = em._y_std ** 2
        print(f'  {name}: nugget={em.nugget:.3g}, sd of output={np.std(outputs[name]):.3g}, '
              f'emulator sd between {np.sqrt(em._sigma2 * em.nugget * ys2):.3g} (floor) and '
              f'{np.sqrt(em._sigma2 * (1 + em.nugget) * ys2):.3g} (far from data)')

    columns = [f'p{i}' for i in range(args.n_dims)]
    candidates = pd.DataFrame(rng.uniform(size=(args.n_candidates, args.n_dims)), columns=columns)
    pilot = candidates.iloc[:5000]

    rows = []
    for scenario in args.scenarios.split(','):
        quantile, sd_fraction = SCENARIOS[scenario]
        targets = {name: (float(np.quantile(em.predict(pilot).get_mean(), quantile)),
                          float(sd_fraction * np.std(outputs[name])))
                   for name, em in emulators.items()}
        observations = hm.ObservationData(targets)

        plain = hm.EmulatorBank()
        for name, em in emulators.items():
            plain.add_emulator(1, name, Unscreened(em))
        mask_plain, t_plain = run_mask(plain, candidates, observations, args.threshold, args.batch_size)
        row = {'scenario': scenario, 'NROY fraction': mask_plain.mean(), 'unscreened (s)': t_plain}

        if not args.unscreened_only:
            screened = hm.EmulatorBank()
            for name, em in emulators.items():
                screened.add_emulator(1, name, Counted(em))
            mask, t_screen = run_mask(screened, candidates, observations, args.threshold, args.batch_size)
            tested = sum(e.rows_tested for e in screened.get_emulators_for_iteration(1).values())
            exact = sum(e.rows_predicted for e in screened.get_emulators_for_iteration(1).values())
            row.update({'screened (s)': t_screen, 'speed-up': t_plain / t_screen,
                        'decided by bounds': 1 - exact / tested,
                        'masks identical': bool(np.array_equal(mask, mask_plain))})
            assert row['masks identical'], f'{scenario}: screened mask differs from unscreened'
        rows.append(row)
        print(pd.DataFrame([row]).to_string(index=False, float_format=lambda v: f'{v:.4g}'))

    print()
    print(pd.DataFrame(rows).to_markdown(index=False, floatfmt='.3g')
          if _has_tabulate() else pd.DataFrame(rows).to_string(index=False))


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == '__main__':
    main()
