import numpy as np
import pandas as pd
import pytest

import historymatching as hm


@pytest.fixture
def smooth_training_data():
    """Create a small smooth training set for BayesLinear tests."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        'x1': rng.uniform(0, 1, 40),
        'x2': rng.uniform(0, 1, 40),
    })
    y = pd.DataFrame({
        'output': (
            np.sin(2 * np.pi * X['x1'])
            + 0.5 * X['x2']
            + rng.normal(0, 0.01, 40)
        )
    })

    return X, y


@pytest.fixture
def noisy_repeated_training_data():
    """Create repeated-design data with noise the nugget should learn."""
    rng = np.random.default_rng(123)
    x_unique = np.linspace(0, 1, 10)
    x = np.repeat(x_unique, 5)
    signal = np.sin(2 * np.pi * x)
    noise = rng.normal(0, 0.4, len(x))

    X = pd.DataFrame({'x': x})
    y = pd.DataFrame({'output': signal + noise})

    return X, y


def test_numeric_nugget_remains_fixed(smooth_training_data):
    """Numeric nuggets should not be optimized."""
    np.random.seed(42)
    X, y = smooth_training_data
    emulator = hm.BayesLinear(X, y, nugget=0.123, test_fraction=0.25)

    emulator.train()
    hyperparameters = emulator.get_hyperparameters()

    assert emulator.nugget == pytest.approx(0.123)
    assert hyperparameters['nugget'] == pytest.approx(0.123)
    assert hyperparameters['nugget_learned'] is False


def test_mle_nugget_moves_above_lower_bound_on_replicates(noisy_repeated_training_data):
    """Replicated noisy observations should push the learned nugget off the floor."""
    np.random.seed(42)
    X, y = noisy_repeated_training_data
    lower_bound = 1e-8
    emulator = hm.BayesLinear(
        X,
        y,
        nugget='mle',
        nugget_bounds=(lower_bound, 1.0),
        test_fraction=0.25,
    )

    emulator.train()
    hyperparameters = emulator.get_hyperparameters()

    assert emulator.nugget > lower_bound * 10
    assert hyperparameters['nugget'] == pytest.approx(emulator.nugget)
    assert hyperparameters['nugget_learned'] is True


def test_emulator_factory_passes_mle_nugget_to_bayes_linear(
    noisy_repeated_training_data,
):
    """Factory defaults should flow through to BayesLinear nugget learning."""
    np.random.seed(42)
    X, y = noisy_repeated_training_data
    factory = hm.EmulatorFactory(
        default_type='bayes_linear',
        nugget='mle',
        nugget_bounds=(1e-8, 1.0),
    )

    emulator = factory.create_and_train_emulator(X, y)
    hyperparameters = emulator.get_hyperparameters()

    assert isinstance(emulator, hm.BayesLinear)
    assert hyperparameters['nugget_learned'] is True
    assert 1e-8 < hyperparameters['nugget'] <= 1.0


def test_correlation_matches_exact_differences():
    """The matrix-product correlations agree with the difference form to ~1e-12."""
    rng = np.random.default_rng(7)
    X1, X2 = rng.uniform(size=(300, 21)), rng.uniform(size=(200, 21))
    X1[:50] = X2[:50] + 1e-6 * rng.normal(size=(50, 21))  # near-coincident pairs
    for theta in (np.exp(rng.uniform(-4, 4, 21)), np.full(21, 0.3)):
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        expected = np.exp(-np.sum((diff / theta) ** 2, axis=2))

        R = hm.BayesLinear._sq_exp_corr(X1, X2, theta)

        np.testing.assert_allclose(R, expected, rtol=1e-10, atol=1e-12)


def test_training_correlation_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(8)
    X = rng.uniform(size=(150, 6))
    R = hm.BayesLinear._sq_exp_corr(X, X, np.exp(rng.uniform(-2, 1, 6)))
    np.testing.assert_array_equal(R, R.T)
    np.testing.assert_array_equal(np.diag(R), 1.0)


@pytest.fixture
def trained_5d():
    """A Bayes-linear emulator trained on 120 points of a smooth 5-D function."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.uniform(size=(120, 5)), columns=[f'p{i}' for i in range(5)])
    y = pd.DataFrame({'out': np.sin(X.to_numpy() @ np.linspace(1, 3, 5)) + rng.normal(0, 0.05, 120)})
    emulator = hm.BayesLinear(X, y, test_fraction=0.1, nugget='mle')
    emulator.train()
    return emulator


def test_predictive_variance_matches_two_solve_formula(trained_5d):
    """One triangular solve gives the same variance as c C^{-1} c^T via cho_solve."""
    from scipy.linalg import cho_solve

    em = trained_5d
    rng = np.random.default_rng(4)
    x = pd.DataFrame(rng.uniform(-0.2, 1.2, size=(500, 5)), columns=[f'p{i}' for i in range(5)])
    c = em._sq_exp_corr(em._normalize_x(x), em._X_train_norm, em._theta)
    v = cho_solve((em._L, em._lower), c.T)
    var_z = np.maximum(em._sigma2 * (1 - np.sum(c.T * v, axis=0)), 0) + em._sigma2 * em.nugget

    np.testing.assert_allclose(em.predict(x).get_variance(), var_z * em._y_std ** 2,
                               rtol=1e-10, atol=0)


# ----------------------------------------------------------------------
# nugget='mle': multistart search
# ----------------------------------------------------------------------

def _friedman_data(n=300, d=10, noise_sd=0.1, seed=0):
    """Friedman #1 (scaled by 1/10) in `d` dimensions, 5 active, plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, d))
    f = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
         + 10 * X[:, 3] + 5 * X[:, 4]) / 10
    return (pd.DataFrame(X, columns=[f'p{i}' for i in range(d)]),
            pd.DataFrame({'y': f + rng.normal(0, noise_sd, n)}))


def _train(X, y, split_seed=0, **kwargs):
    np.random.seed(split_seed)  # the same train/test split for every fit
    em = hm.BayesLinear(X, y, test_fraction=0.25, **kwargs)
    em.train()
    return em


def _single_start_nll(em):
    """The NLL the pre-multistart `nugget='mle'` fit reached: one L-BFGS-B start from
    theta = 1 and the geometric middle of the nugget bounds, finite-difference gradients."""
    from scipy.optimize import minimize
    from historymatching.emulators.bayes_linear import LOG_THETA_BOUNDS

    X, r = em._X_train_norm, em._residuals
    d = X.shape[1]
    log_bounds = np.log(em.nugget_bounds)
    result = minimize(lambda z: em._nll(z[:-1], float(np.exp(z[-1])), X, r),
                      np.append(np.zeros(d), log_bounds.mean()), method='L-BFGS-B',
                      bounds=[LOG_THETA_BOUNDS] * d + [tuple(log_bounds)],
                      options={'maxiter': 200, 'ftol': em.ftol, 'gtol': em.gtol})
    return result.fun, float(np.exp(result.x[-1]))


@pytest.fixture(scope='module')
def friedman_fits():
    """The synthetic reproduction: the single start stops near its starting nugget."""
    X, y = _friedman_data()
    return {nugget: _train(X, y, nugget=nugget) for nugget in ('mle', 0.01, 0.03)}


def test_mle_fit_beats_single_start_and_fixed_nuggets(friedman_fits):
    """A fixed nugget is a feasible point of the learned-nugget search, so the learned
    fit's NLL is at most every fixed-nugget NLL, and at most the old single start's."""
    mle = friedman_fits['mle']
    nll = mle.get_hyperparameters()['nll']
    old_nll, old_nugget = _single_start_nll(mle)

    assert old_nll > friedman_fits[0.03].get_hyperparameters()['nll'] + 10  # the defect
    assert nll <= old_nll - 10
    for fixed in (0.01, 0.03):
        assert nll <= friedman_fits[fixed].get_hyperparameters()['nll']


def test_mle_nugget_is_interior_and_near_the_truth(friedman_fits):
    """The noise variance is 0.1^2; the learned nugget times sigma^2 should recover it."""
    hp = friedman_fits['mle'].get_hyperparameters()
    lo, hi = friedman_fits['mle'].nugget_bounds
    assert 1e3 * lo < hp['nugget'] < hi / 10
    noise_var = hp['nugget'] * hp['sigma_sq']
    assert 0.5 * 0.1 ** 2 < noise_var < 2 * 0.1 ** 2


@pytest.mark.parametrize('noise_sd, warm_nugget', [(0.1, 1e-2), (0.6, 1e-2), (0.3, 0.2), (0.1, 1e-9)])
def test_mle_nll_never_above_theta1_search_at_warm_nugget(noise_sd, warm_nugget):
    """Guaranteed by construction: the warm start's warm-up is a candidate, and it is the
    same search as a fixed-nugget fit's theta = 1 refit at the warm nugget."""
    X, y = _friedman_data(n=200, d=8, noise_sd=noise_sd, seed=1)
    mle = _train(X, y, split_seed=1, nugget='mle', warm_nugget=warm_nugget)
    w = float(np.clip(warm_nugget, *mle.nugget_bounds))
    fixed = _train(X, y, split_seed=1, nugget=w)
    theta1 = [r for r in fixed.start_results if r['start'] == 'theta=1']
    warm = [r for r in mle.start_results if r['stage'] == 'warm-up']
    assert len(theta1) == 1 and len(warm) == 1 and warm[0]['nugget'] == w
    assert warm[0]['nll'] == theta1[0]['nll']
    assert mle.get_hyperparameters()['nll'] <= warm[0]['nll']


def test_start_records(friedman_fits):
    hp = friedman_fits['mle'].get_hyperparameters()
    records = hp['starts']
    # 'warm' is two runs (warm-up, then the search); 'midpoint' and 1e-2 are one each.
    assert [(r['start'], r['stage']) for r in records] == [
        ('warm', 'warm-up'), ('warm', 'search'),
        ('midpoint', 'search'), ('nugget=0.01', 'search')]
    for r in records:
        assert set(r) >= {'start', 'stage', 'start_nugget', 'nll', 'nugget', 'nit', 'nfev',
                          'success', 'message', 'seconds', 'selected'}
    selected = [r for r in records if r['selected']]
    assert len(selected) == 1
    assert selected[0]['nll'] == hp['nll'] and selected[0]['nugget'] == hp['nugget']
    assert all(r['candidate'] for r in records)  # with 'mle', every run is a candidate
    assert all(r['nll'] >= hp['nll'] for r in records)
    assert records[2]['start_nugget'] == pytest.approx(1e-4)  # middle of (1e-8, 1)
    import json
    json.dumps(hp)


def test_single_midpoint_start():
    """One start is one free-nugget search from the old starting point."""
    X, y = _friedman_data()
    em = _train(X, y, nugget='mle', starts=('midpoint',))
    hp = em.get_hyperparameters()
    assert [(r['start'], r['stage']) for r in hp['starts']] == [('midpoint', 'search')]
    assert hp['starts'][0]['start_nugget'] == pytest.approx(1e-4)
    assert hp['starts'][0]['selected'] is True


@pytest.mark.parametrize('nugget', [1e-7, 1e-3, 0.05, 0.9])
def test_nll_gradient_matches_central_differences(nugget):
    X, y = _friedman_data(n=150, d=6, noise_sd=0.1, seed=3)
    em = _train(X, y, split_seed=3, nugget='mle', starts=('midpoint',))
    Xn, r = em._X_train_norm, em._residuals
    z = np.append(np.random.default_rng(5).uniform(-1.5, 1.5, 6), np.log(nugget))

    def f(z):
        return em._nll(z[:-1], float(np.exp(z[-1])), Xn, r)

    nll, grad = em._nll_and_grad(z[:-1], float(np.exp(z[-1])), Xn, r)
    h = 1e-5
    central = np.array([(f(z + h * e) - f(z - h * e)) / (2 * h) for e in np.eye(len(z))])
    assert nll == f(z)
    np.testing.assert_allclose(grad, central, rtol=1e-6, atol=1e-6 * np.abs(central).max())


def _independent_nll(Xn, r, log_theta, nugget):
    """The concentrated NLL, written out independently of the emulator."""
    from scipy.linalg import cho_factor, cho_solve

    n = len(r)
    R = hm.BayesLinear._sq_exp_corr(Xn, Xn, np.exp(log_theta)) + nugget * np.eye(n)
    try:
        L, lower = cho_factor(R)
    except np.linalg.LinAlgError:
        return 1e10
    s2 = float(r @ cho_solve((L, lower), r)) / n
    return 0.5 * (n * np.log(s2) + 2.0 * np.sum(np.log(np.diag(L))))


def test_fixed_nugget_runs_the_same_starts_then_refits():
    """A fixed nugget runs the learned-nugget searches, then refits theta at the fixed
    nugget from each of their solutions and from theta = 1, and keeps the nugget."""
    X, y = _friedman_data(n=200, d=6, noise_sd=0.1, seed=2)
    em = _train(X, y, split_seed=2, nugget=1e-6)
    mle = _train(X, y, split_seed=2, nugget='mle')
    hp = em.get_hyperparameters()
    assert em.nugget == 1e-6 and hp['nugget'] == 1e-6 and hp['nugget_learned'] is False
    records = hp['starts']
    phase1 = [('warm', 'warm-up'), ('warm', 'search'), ('midpoint', 'search'), ('nugget=0.01', 'search')]
    assert [(r['start'], r['stage']) for r in records] == phase1 + [
        ('theta=1', 'refit'), ('warm', 'refit of warm-up'), ('warm', 'refit of search'),
        ('midpoint', 'refit of search'), ('nugget=0.01', 'refit of search')]
    # The first phase is the learned-nugget fit, run for run.
    for r, m in zip(records[:4], mle.get_hyperparameters()['starts']):
        assert (r['nll'], r['nugget']) == (m['nll'], m['nugget'])
    assert not any(r['candidate'] for r in records[:4])
    assert all(r['candidate'] and r['nugget'] == 1e-6 for r in records[4:])
    selected = [r for r in records if r['selected']]
    assert len(selected) == 1 and selected[0]['nll'] == min(r['nll'] for r in records[4:])
    assert hp['nll'] <= records[4]['nll']  # never above the theta = 1 search alone
    assert hp['nll'] == pytest.approx(
        _independent_nll(em._X_train_norm, em._residuals, np.log(em._theta), 1e-6), rel=1e-10)


def test_fixed_nugget_fit_no_longer_collapses():
    """The old fixed-nugget search (theta = 1 start, raw NLL, finite differences) took
    one step that put log correlation lengths on their lower bound, where the NLL is
    flat, and stopped there. The shared search does not."""
    from scipy.optimize import minimize
    from historymatching.emulators.bayes_linear import LOG_THETA_BOUNDS

    X, y = _friedman_data(n=300, d=21, noise_sd=0.1, seed=0)
    em = _train(X, y, nugget=1e-6)
    Xn, r = em._X_train_norm, em._residuals
    old = minimize(lambda z: _independent_nll(Xn, r, z, 1e-6), np.zeros(21), method='L-BFGS-B',
                   bounds=[LOG_THETA_BOUNDS] * 21,
                   options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-4})

    assert old.nit <= 1 and np.sum(old.x <= LOG_THETA_BOUNDS[0] + 1e-9) >= 3  # the defect
    hp = em.get_hyperparameters()
    assert hp['nll'] <= old.fun - 50
    assert np.all(np.log(em._theta) > LOG_THETA_BOUNDS[0] + 1e-6)
    em.test()
    assert em.emulator_metrics['R2'] > 0.9


def test_starts_help_a_fixed_nugget():
    """From theta = 1 at nugget 1e-6, the search ends on the near-white-noise plateau;
    a refit from the learned-nugget searches' solutions does not."""
    X, y = _friedman_data(n=300, d=10, noise_sd=0.3, seed=0)
    hp = _train(X, y, nugget=1e-6).get_hyperparameters()
    theta1 = next(r for r in hp['starts'] if r['start'] == 'theta=1')
    assert hp['nll'] <= theta1['nll'] - 10
    assert not theta1['selected']


@pytest.mark.parametrize('bad', [(), ('cold',), (0.0,), (-1e-3,), (True,)])
@pytest.mark.parametrize('nugget', ['mle', 1e-6])
def test_starts_validation(bad, nugget):
    with pytest.raises(ValueError):
        hm.BayesLinear(nugget=nugget, starts=bad)


def test_starts_accepts_a_single_entry_and_numbers():
    assert hm.BayesLinear(starts='warm').starts == ('warm',)
    assert hm.BayesLinear(starts=[1e-3, 'midpoint']).starts == (1e-3, 'midpoint')
    with pytest.raises(ValueError):
        hm.BayesLinear(warm_nugget=0)
