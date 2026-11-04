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
