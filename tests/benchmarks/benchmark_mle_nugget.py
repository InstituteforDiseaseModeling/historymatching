"""
Benchmark: robustness and cost of the Bayes-linear hyperparameter fit, fixed and learned nuggets.

A learned nugget is a search over the correlation lengths and the nugget
together, and a fixed nugget inside `nugget_bounds` is one feasible point of
that search. So on the same training data, a working `nugget='mle'` fit never
ends with a higher NLL than a fixed-nugget fit. This script fits synthetic
scenarios (a smooth function of a few active inputs plus Gaussian run-to-run
noise, with the rest inactive) with `nugget='mle'` and with fixed nuggets of
1e-6 (the default), 0.01 and 0.03, on the same train/test split, and writes one
CSV row per fit: NLL, nugget, held-out R², iterations, log correlation lengths
on their lower bound and wall time.

Run it once per package version, then compare the CSVs:

    uv run python tests/benchmarks/benchmark_mle_nugget.py --out new.csv
    PYTHONPATH=/path/to/other/checkout uv run python \\
        tests/benchmarks/benchmark_mle_nugget.py --out old.csv
    uv run python tests/benchmarks/benchmark_mle_nugget.py --compare old.csv new.csv

The comparison counts, per CSV, the scenarios where the learned fit ends more
than `--tol` NLL above the fixed-nugget fits, or more than `--tol` above the
lowest NLL any fit in any of the CSVs reached on that scenario, and the
learned nuggets on a bound. For each fixed nugget it counts the fits more than
`--tol` above the lowest NLL any CSV reached at that nugget, and the collapsed
fits: at most one iteration with a log correlation length on its lower bound.
`--n`, `--d`, `--noise`, `--kind` and `--seeds` set the grid; `--fit-only mle`
skips the fixed-nugget fits.
"""

import argparse
import csv
import itertools
import time
from collections import defaultdict

import numpy as np
import pandas as pd

import historymatching as hm
from historymatching.emulators.bayes_linear import LOG_THETA_BOUNDS

FIXED_NUGGETS = [1e-6, 0.01, 0.03]


def make_data(n, d, noise_sd, seed, kind):
    """Synthetic scenario: `n` uniform points in `d` dimensions, 5 inputs active."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, d))
    if kind == 'sin':
        w = np.zeros(d)
        w[:5] = [3, 2, 1.5, 1, 0.5][:d]
        f = np.sin(X @ w) + 0.3 * X[:, 5:].sum(axis=1) / max(d - 5, 1)
    elif kind == 'friedman':  # Friedman #1, scaled by 1/10
        f = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
             + 10 * X[:, 3] + 5 * X[:, 4]) / 10
    else:
        raise ValueError(kind)
    y = f + rng.normal(0, noise_sd, n)
    return (pd.DataFrame(X, columns=[f'p{i}' for i in range(d)]),
            pd.DataFrame({'y': y}))


def fit(X, y, seed, nugget):
    np.random.seed(seed)  # the same train/test split for every fit of a scenario
    em = hm.BayesLinear(X, y, test_fraction=0.25, nugget=nugget)
    t0 = time.perf_counter()
    em.train()
    seconds = time.perf_counter() - t0
    em.test()
    hp = em.get_hyperparameters()
    lo, hi = em.nugget_bounds
    starts = hp.get('starts', hp.get('mle_starts', []))  # 'mle_starts' before 2.2.0
    return {
        'nll': hp['nll'],
        'nugget': hp['nugget'],
        'at_bound': nugget == 'mle' and (np.isclose(hp['nugget'], lo, rtol=1e-3)
                                         or np.isclose(hp['nugget'], hi, rtol=1e-3)),
        'r2': em.emulator_metrics['R2'],
        'nit': em._opt_result.nit,
        'n_theta_at_lower': int(np.sum(np.log(em._theta) <= LOG_THETA_BOUNDS[0] + 1e-6)),
        'seconds': seconds,
        'n_runs': len(starts),
        'selected': next((f"{r['start']} ({r['stage']})" for r in starts if r.get('selected')), ''),
    }


def run(args):
    rows = []
    grid = itertools.product(args.kind, args.n, args.d, args.noise, range(args.seeds))
    print(f"historymatching {hm.__version__} from {hm.__file__}", flush=True)
    fits = ['mle'] if args.fit_only == 'mle' else ['mle'] + FIXED_NUGGETS
    for kind, n, d, noise, seed in grid:
        X, y = make_data(n, d, noise, seed, kind)
        for nugget in fits:
            r = fit(X, y, seed, nugget)
            row = {'kind': kind, 'n': n, 'd': d, 'noise': noise, 'seed': seed,
                   'fit': str(nugget), **r}
            rows.append(row)
            print(f"{kind:8s} n={n:5d} d={d:2d} noise={noise:.2f} seed={seed} {str(nugget):5s} "
                  f"NLL={r['nll']:11.2f} nugget={r['nugget']:9.3g} R2={r['r2']:.3f} "
                  f"nit={r['nit']:3d} {r['seconds']:6.1f}s {r['selected']}", flush=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def compare(paths, tol):
    tables = {}
    for p in paths:
        with open(p) as fh:
            tables[p] = list(csv.DictReader(fh))
    key = lambda r: (r['kind'], r['n'], r['d'], r['noise'], r['seed'])
    # Fixed-nugget fits from any CSV count.
    best = defaultdict(lambda: np.inf)
    fixed_nll = defaultdict(lambda: np.inf)
    for rows in tables.values():
        for r in rows:
            best[key(r)] = min(best[key(r)], float(r['nll']))
            if r['fit'] != 'mle':
                fixed_nll[key(r)] = min(fixed_nll[key(r)], float(r['nll']))
    print(f"{'csv':40s} {'scen':>5s} {'>fixed':>7s} {'>best':>6s} {'bound':>6s} "
          f"{'worst gap':>10s} {'med R2':>7s} {'med s':>7s} {'total s':>8s}")
    for p, rows in tables.items():
        by = defaultdict(dict)
        for r in rows:
            by[key(r)][r['fit']] = r
        mle = [(k, v['mle']) for k, v in by.items() if 'mle' in v]
        above_fixed = sum(float(r['nll']) > fixed_nll[k] + tol for k, r in mle)
        gaps = [float(r['nll']) - best[k] for k, r in mle]
        above_best = sum(g > tol for g in gaps)
        at_bound = sum(r['at_bound'] == 'True' for _, r in mle)
        secs = [float(r['seconds']) for _, r in mle]
        r2 = [float(r['r2']) for _, r in mle]
        print(f"{p[-40:]:40s} {len(mle):5d} {above_fixed:7d} {above_best:6d} {at_bound:6d} "
              f"{max(gaps):10.2f} {np.median(r2):7.3f} {np.median(secs):7.2f} {sum(secs):8.1f}")

    # Fixed nuggets: each against the lowest NLL any CSV reached at the same nugget.
    best_at = defaultdict(lambda: np.inf)
    for rows in tables.values():
        for r in rows:
            if r['fit'] != 'mle':
                best_at[key(r), r['fit']] = min(best_at[key(r), r['fit']], float(r['nll']))
    print(f"\n{'csv':40s} {'nugget':>7s} {'scen':>5s} {'>best':>6s} {'collapsed':>9s} "
          f"{'worst gap':>10s} {'med R2':>7s} {'med s':>7s} {'total s':>8s}")
    for p, rows in tables.items():
        for fit in sorted({r['fit'] for r in rows if r['fit'] != 'mle'}, key=float):
            sel = [r for r in rows if r['fit'] == fit]
            gaps = [float(r['nll']) - best_at[key(r), fit] for r in sel]
            collapsed = sum(int(r['nit']) <= 1 and int(r.get('n_theta_at_lower', 0)) > 0 for r in sel)
            secs = [float(r['seconds']) for r in sel]
            print(f"{p[-40:]:40s} {fit:>7s} {len(sel):5d} {sum(g > tol for g in gaps):6d} "
                  f"{collapsed:9d} {max(gaps):10.2f} {np.median([float(r['r2']) for r in sel]):7.3f} "
                  f"{np.median(secs):7.2f} {sum(secs):8.1f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='mle_nugget.csv')
    ap.add_argument('--compare', nargs='+', metavar='CSV')
    ap.add_argument('--tol', type=float, default=1.0, help='NLL tolerance for counting a failure')
    ap.add_argument('--kind', nargs='+', default=['sin', 'friedman'])
    ap.add_argument('--n', nargs='+', type=int, default=[300, 1000])
    ap.add_argument('--d', nargs='+', type=int, default=[10, 21])
    ap.add_argument('--noise', nargs='+', type=float, default=[0.1, 0.3, 0.6])
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--fit-only', choices=['all', 'mle'], default='all')
    args = ap.parse_args()
    if args.compare:
        compare(args.compare, args.tol)
    else:
        run(args)
