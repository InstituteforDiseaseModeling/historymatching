"""Screening in `nroy_mask` gives the same mask as calling `predict` on every point.

`BayesLinear` offers `predict_bounds`, so `nroy_mask` decides most points from
cheap bounds on the mean and variance and calls `predict` only for the rest.
These tests compare that against the same emulators wrapped so they offer no
bounds, which sends every point through `predict`.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

import historymatching as hm
from historymatching.emulators.results import EmulationResults
from historymatching.nroy_sampling import nroy_mask

PARAMS = ['p0', 'p1', 'p2', 'p3']
FEATURES = ['f0', 'f1']


def screen(emulator, x):
    """`(mean, mean_tol, var_lo, var_hi)` from `predict_bounds`."""
    return tuple(emulator.predict_bounds(x))


class Unscreened(hm.BaseEmulator):
    """Wraps an emulator and hides its variance bounds, so every point gets `predict`."""

    def __init__(self, emulator):
        super().__init__()
        self.emulator = emulator
        self.training_complete = True

    def train(self):
        pass

    def predict(self, x):
        return self.emulator.predict(x)

    def print_emulator_description(self):
        return 'unscreened wrapper'

    def get_hyperparameters(self):
        return {}


class ShiftedSum(hm.BaseEmulator):
    """Deterministic emulator without variance bounds: mean = sum of parameters."""

    def __init__(self, std=0.3):
        super().__init__()
        self.std = std
        self.training_complete = True

    def train(self):
        pass

    def predict(self, x):
        return EmulationResults(mean=x.to_numpy().sum(axis=1), std=np.full(len(x), self.std),
                                index=x.index)

    def print_emulator_description(self):
        return 'sum of parameters'

    def get_hyperparameters(self):
        return {}


def simulate(x, feature, rng, noise):
    """A smooth function of four parameters, one per feature, plus run noise."""
    w = {'f0': np.array([1.0, 2.0, 0.5, 1.5]), 'f1': np.array([2.5, 0.5, 1.0, 0.3])}[feature]
    return np.sin(x @ w) + (x ** 2).sum(axis=1) + rng.normal(0, noise, len(x))


def train_wave(nugget, seed, lo=0.0, hi=1.0, n=80):
    """Train one Bayes-linear emulator per feature on `n` points in `[lo, hi]^4`."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, size=(n, len(PARAMS)))
    X = pd.DataFrame(x, columns=PARAMS)
    emulators = {}
    for feature in FEATURES:
        y = pd.DataFrame({feature: simulate(x, feature, rng, noise=0.05)})
        # The bounds the fixture was built with; the screening tests depend on these emulators.
        emulator = hm.BayesLinear(X, y, test_fraction=0.1, nugget=nugget, nugget_bounds=(1e-8, 1.0))
        emulator.train()
        emulators[feature] = emulator
    return emulators


@pytest.fixture(scope='module', params=[1e-6, 'mle'], ids=['fixed-nugget', 'mle-nugget'])
def waves(request):
    """Three waves of Bayes-linear emulators, each trained on a smaller box."""
    np.random.seed(0)  # train_test_split inside BaseEmulator
    return [train_wave(request.param, seed=10 + w, lo=0.1 * w, hi=1.0 - 0.1 * w)
            for w in range(3)]


def make_banks(waves, extra=None):
    """The same emulators twice: as they are, and wrapped to hide their bounds."""
    screened, unscreened = hm.EmulatorBank(), hm.EmulatorBank()
    for iteration, emulators in enumerate(waves, start=1):
        for feature, emulator in emulators.items():
            screened.add_emulator(iteration, feature, emulator)
            unscreened.add_emulator(iteration, feature, Unscreened(emulator))
    for iteration, feature, emulator in (extra or []):
        screened.add_emulator(iteration, feature, emulator)
        unscreened.add_emulator(iteration, feature, emulator)
    return screened, unscreened


def uniform(n, seed, lo=-0.1, hi=1.1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.uniform(lo, hi, size=(n, len(PARAMS))), columns=PARAMS)


def random_observations(emulators, rng, candidates):
    """Targets near the emulators' predicted range at the candidates, with random sds."""
    obs = {}
    for feature, emulator in emulators.items():
        mean = emulator.predict(candidates).get_mean()
        target = rng.uniform(*np.quantile(mean, [0.05, 0.95]))
        obs[feature] = (float(target), float(rng.uniform(0.01, 0.15)))
    return hm.ObservationData(obs)


def boundary_points(emulator, feature, observations, threshold, rng, candidates, n_lines=8):
    """Candidates right at a decision boundary of the screen or of the exact test.

    Along lines from low-mean to high-mean candidates, find where the distance
    from the target equals `threshold * sqrt(hi + sd^2)` (the rule-out screen),
    `threshold * sqrt(lo + sd^2)` (the rule-in screen), and where the exact
    implausibility equals `threshold`, then add points a hair either side.
    """
    target, sd = observations.get_target_for_feature(feature)

    def at(t, a, b):
        return pd.DataFrame([a + t * (b - a)], columns=PARAMS)

    def rule_out_gap(t, a, b):
        mean, tol, _, hi = screen(emulator, at(t, a, b))
        return abs(mean[0] - target) - tol[0] - threshold * np.sqrt(hi[0] + sd ** 2)

    def rule_in_gap(t, a, b):
        mean, tol, lo, _ = screen(emulator, at(t, a, b))
        return abs(mean[0] - target) + tol[0] - threshold * np.sqrt(lo[0] + sd ** 2)

    def exact_gap(t, a, b):
        pred = emulator.predict(at(t, a, b))
        return (abs(pred.get_mean().iloc[0] - target)
                / np.sqrt(pred.get_variance().iloc[0] + sd ** 2) - threshold)

    order = np.argsort(screen(emulator, candidates)[0])
    k = max(1, len(order) // 20)
    starts = candidates.to_numpy()[rng.choice(order[:k], n_lines)]
    ends = candidates.to_numpy()[rng.choice(order[-k:], n_lines)]
    points, kinds = [], set()
    ts = np.linspace(0, 1, 41)
    for a, b in zip(starts, ends):
        for gap in (rule_out_gap, rule_in_gap, exact_gap):
            values = np.array([gap(t, a, b) for t in ts])
            for i in np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:])):
                t = brentq(gap, ts[i], ts[i + 1], args=(a, b),
                           xtol=1e-15, rtol=4 * np.finfo(float).eps)
                kinds.add(gap.__name__)
                for dt in (0.0, -1e-12, 1e-12, -1e-9, 1e-9):
                    points.append(a + (t + dt) * (b - a))
    assert kinds == {'rule_out_gap', 'rule_in_gap', 'exact_gap'}, kinds
    return pd.DataFrame(points, columns=PARAMS)


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('threshold', [2.0, 3.0, 3.5])
def test_screened_mask_equals_unscreened(waves, seed, threshold):
    rng = np.random.default_rng(seed)
    candidates = uniform(3000, seed)
    observations = random_observations(waves[-1], rng, candidates)
    newest_first = next(iter(waves[-1]))
    boundary = boundary_points(waves[-1][newest_first], newest_first, observations,
                               threshold, rng, candidates)
    candidates = pd.concat([candidates, boundary], ignore_index=True)
    screened, unscreened = make_banks(waves)

    mask = nroy_mask(candidates, screened, observations, threshold, PARAMS)
    expected = nroy_mask(candidates, unscreened, observations, threshold, PARAMS)

    np.testing.assert_array_equal(mask, expected)
    assert 0 < expected.sum() < len(candidates)


def test_starting_mask_and_non_contiguous_index(waves):
    rng = np.random.default_rng(5)
    candidates = uniform(2000, 5).set_axis(rng.permutation(10_000)[:2000])
    observations = random_observations(waves[-1], rng, candidates)
    start = rng.uniform(size=len(candidates)) < 0.7
    screened, unscreened = make_banks(waves)

    mask = nroy_mask(candidates, screened, observations, 3.0, PARAMS, mask=start)
    expected = nroy_mask(candidates, unscreened, observations, 3.0, PARAMS, mask=start)

    np.testing.assert_array_equal(mask, expected)
    assert not mask[~start].any()


def test_bank_mixing_screened_and_unscreened_emulators(waves):
    rng = np.random.default_rng(6)
    candidates = uniform(3000, 6, lo=0.0, hi=1.0)
    observations = random_observations(waves[-1], rng, candidates)
    observations = hm.ObservationData({**observations.get_all_targets(), 'total': (2.0, 0.2)})
    # One emulator without bounds in the newest wave, one in a wave of its own.
    extra = [(len(waves), 'total', ShiftedSum()), (len(waves) + 1, 'total', ShiftedSum(0.5))]
    screened, unscreened = make_banks(waves, extra)

    mask = nroy_mask(candidates, screened, observations, 3.0, PARAMS)
    expected = nroy_mask(candidates, unscreened, observations, 3.0, PARAMS)

    np.testing.assert_array_equal(mask, expected)
    assert 0 < expected.sum() < len(candidates)


def test_bounds_contain_predict(waves):
    """`predict`'s mean is within the tolerance and its variance within the bounds."""
    rng = np.random.default_rng(7)
    spread = uniform(2000, 7, lo=-0.5, hi=1.5)
    far = uniform(50, 8, lo=1e3, hi=2e3)
    for emulators in waves:
        for emulator in emulators.values():
            near = emulator.X_train[:50] + rng.normal(0, 1e-4, size=(50, len(PARAMS)))
            x = pd.concat([spread, pd.DataFrame(near, columns=PARAMS), far], ignore_index=True)
            mean, tol, lo, hi = screen(emulator, x)
            pred = emulator.predict(x)
            variance = pred.get_variance().to_numpy()
            s2, ys2 = emulator._sigma2, emulator._y_std ** 2

            assert np.all(np.abs(mean - pred.get_mean().to_numpy()) <= tol)
            # The tolerance is tiny next to the output scale.
            assert np.all(tol < 1e-6 * emulator._y_std)
            assert np.all(0 < lo) and np.all(lo <= hi)
            np.testing.assert_allclose(lo, s2 * emulator.nugget * ys2, rtol=1e-12)
            assert np.all(hi <= s2 * (1 + emulator.nugget) * ys2 * (1 + 1e-12))
            assert np.all(variance >= lo)   # exactly, for get_variance() itself
            assert np.all(variance <= hi)
            # Next to the first training points, which the bound conditions on,
            # it is much tighter than the variance far from the data...
            m = min(500, len(emulator._X_train_norm) // 4)
            assert np.all(hi[2000:2000 + m] - lo[2000:2000 + m] < 0.9 * s2 * ys2)
            # ...and far from every training point both reach that variance.
            np.testing.assert_allclose(variance[-50:], hi[-50:], rtol=1e-12)
            np.testing.assert_allclose(hi[-50:], s2 * (1 + emulator.nugget) * ys2, rtol=1e-12)


def test_screening_skips_most_predictions(waves, monkeypatch):
    """Count the rows sent to `predict`: only those between the two screens."""
    emulator = waves[-1]['f0']
    candidates = uniform(5000, 10)
    mean, tol, lo, hi = screen(emulator, candidates)
    target, sd, threshold = float(np.quantile(mean, 0.02)), 0.05, 3.0
    observations = hm.ObservationData({'f0': (target, sd)})
    bank = hm.EmulatorBank()
    bank.add_emulator(1, 'f0', emulator)

    rows = []
    predict = emulator.predict
    monkeypatch.setattr(emulator, 'predict', lambda x: rows.append(len(x)) or predict(x))
    mask = nroy_mask(candidates, bank, observations, threshold, PARAMS)
    monkeypatch.undo()

    distance = np.abs(mean - target)
    undecided = ((distance - tol <= threshold * np.sqrt(hi + sd ** 2))
                 & (distance + tol > threshold * np.sqrt(lo + sd ** 2)))
    assert sum(rows) == undecided.sum()
    assert 0 < sum(rows) < 0.2 * len(candidates)
    unscreened = hm.EmulatorBank()
    unscreened.add_emulator(1, 'f0', Unscreened(emulator))
    np.testing.assert_array_equal(
        mask, nroy_mask(candidates, unscreened, observations, threshold, PARAMS))


def test_no_screening_without_a_positive_variance_floor(waves, monkeypatch):
    """Without a positive floor on the total variance, every point gets `predict`."""
    emulator = waves[0]['f0']
    candidates = uniform(300, 11)
    original = emulator.predict_bounds
    monkeypatch.setattr(emulator, 'predict_bounds',
                        lambda x: original(x)._replace(var_lo=np.full(len(x), -1.0)))
    rows = []
    predict = emulator.predict
    monkeypatch.setattr(emulator, 'predict', lambda x: rows.append(len(x)) or predict(x))
    bank = hm.EmulatorBank()
    bank.add_emulator(1, 'f0', emulator)
    target = float(np.median(original(candidates).mean))

    nroy_mask(candidates, bank, hm.ObservationData({'f0': (target, 0.5)}), 3.0, PARAMS)

    assert rows == [len(candidates)]


def test_plain_emulators_offer_no_screening():
    assert ShiftedSum().predict_bounds(pd.DataFrame({'p0': [0.5]})) is None


def test_prediction_failure_still_raises(waves):
    candidates = uniform(100, 12)
    candidates.iloc[3, 0] = np.nan
    observations = random_observations(waves[-1], np.random.default_rng(12), uniform(100, 13))
    screened, _ = make_banks(waves)
    with pytest.raises(RuntimeError, match='NROY filtering failed'):
        nroy_mask(candidates, screened, observations, 3.0, PARAMS)



def test_variance_bounds_hold_exactly_after_the_std_round_trip():
    """Far from the data the variance equals the upper bound's cap, and get_variance() (std**2)
    can exceed the computed variance by about one ulp. The bounds must still contain it exactly.
    Checked on emulators with many output scales, since whether std**2 rounds up depends on it."""
    rng = np.random.default_rng(278)
    d, n = 5, 40
    cols = [f"p{i}" for i in range(d)]
    far = pd.DataFrame(rng.uniform(1e3, 2e3, (20, d)), columns=cols)
    for k in range(40):
        x = pd.DataFrame(rng.uniform(0, 1, (n, d)), columns=cols)
        y = pd.DataFrame({"y": rng.lognormal(0, 2) * np.sin(x.to_numpy() @ rng.normal(0, 1, d)) + rng.normal(0, 0.1, n)})
        np.random.seed(k)
        em = hm.BayesLinear(x=x, y=y, nugget=float(10.0 ** rng.uniform(-6, 0)))
        em.train()
        bounds = em.predict_bounds(far)
        variance = em.predict(far).get_variance().to_numpy()
        assert np.all(variance >= bounds.var_lo), k
        assert np.all(variance <= bounds.var_hi), k
