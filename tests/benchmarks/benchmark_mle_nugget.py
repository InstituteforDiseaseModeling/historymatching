"""
Benchmark: robustness and cost of the Bayes-linear `nugget='mle'` fit.

A learned nugget is a search over the correlation lengths and the nugget
together, and a fixed nugget inside `nugget_bounds` is one feasible point of
that search. So on the same training data, a working `nugget='mle'` fit never
ends with a higher NLL than a fixed-nugget fit. This script fits synthetic
scenarios (a smooth function of a few active inputs plus Gaussian run-to-run
noise, with the rest inactive) with `nugget='mle'` and with fixed nuggets of
0.01 and 0.03, on the same train/test split, and writes one CSV row per fit:
NLL, nugget, held-out R², iterations and wall time.

Run it once per package version, then compare the CSVs:

    uv run python tests/benchmarks/benchmark_mle_nugget.py --out new.csv
    PYTHONPATH=/path/to/other/checkout uv run python \\
        tests/benchmarks/benchmark_mle_nugget.py --out old.csv
    uv run python tests/benchmarks/benchmark_mle_nugget.py --compare old.csv new.csv

The comparison counts, per CSV, the scenarios where the learned fit ends more
than `--tol` NLL above the fixed-nugget fits, or more than `--tol` above the
lowest NLL any fit in any of the CSVs reached on that scenario, and the
learned nuggets on a bound. `--n`, `--d`, `--noise`, `--kind` and `--seeds`
set the grid; `--fit-only mle` skips the fixed-nugget fits.
"""

import argparse
import csv
import itertools
import time
from collections import defaultdict

import numpy as np
import pandas as pd

import historymatching as hm


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
    starts = hp.get('mle_starts', [])
    return {
        'nll': hp['nll'],
        'nugget': hp['nugget'],
        'at_bound': nugget == 'mle' and (np.isclose(hp['nugget'], lo, rtol=1e-3)
                                         or np.isclose(hp['nugget'], hi, rtol=1e-3)),
        'r2': em.emulator_metrics['R2'],
        'nit': em._opt_result.nit,
        'seconds': seconds,
        'n_runs': len(starts),
        'selected': next((f"{r['start']} ({r['stage']})" for r in starts if r.get('selected')), ''),
    }


def run(args):
    rows = []
    grid = itertools.product(args.kind, args.n, args.d, args.noise, range(args.seeds))
    fits = ['mle'] if args.fit_only == 'mle' else ['mle', 0.01, 0.03]
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
    # Fixed-nugget fits from any CSV count (pass `--fit-only mle` for versions
    # whose fixed-nugget fits are unchanged).
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
