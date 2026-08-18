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


@pytest.fixture
def sparse_replicated_training_data():
    """Create replicated corners plus unreplicated interior mean sites."""
    rng = np.random.default_rng(456)
    replicated_sites = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    interior_sites = np.array([
        [0.25, 0.25],
        [0.25, 0.75],
        [0.50, 0.50],
        [0.75, 0.25],
        [0.75, 0.75],
    ])

    rows = []
    outputs = []
    for x1, x2 in replicated_sites:
        simulator_variance = 0.02 + 0.3 * x1 + 0.5 * x2
        for _ in range(8):
            rows.append((x1, x2))
            outputs.append(
                1.0
                + x1
                - 0.5 * x2
                + rng.normal(0, np.sqrt(simulator_variance))
            )

    for x1, x2 in interior_sites:
        rows.append((x1, x2))
        outputs.append(1.0 + x1 - 0.5 * x2)

    X = pd.DataFrame(rows, columns=['x1', 'x2'])
    y = pd.DataFrame({'output': outputs})

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


def test_adaptive_nugget_trains_from_sparse_replicated_subset(
    sparse_replicated_training_data,
):
    """Adaptive nuggets learn variance from replicated sites only."""
    X, y = sparse_replicated_training_data
    emulator = hm.BayesLinear(X, y, nugget='adaptive', test_fraction=0)

    emulator.train()
    hyperparameters = emulator.get_hyperparameters()
    results = emulator.predict(pd.DataFrame({
        'x1': [0.2, 0.8],
        'x2': [0.2, 0.8],
    }))
    additional = results.get_additional_data()

    assert hyperparameters['nugget'] == 'adaptive'
    assert hyperparameters['nugget_adaptive'] is True
    assert hyperparameters['n_train'] == len(X)
    assert hyperparameters['n_adjusted'] == len(X.drop_duplicates())
    assert emulator._variance_emulator is not None
    assert emulator._variance_emulator.get_hyperparameters()['n_train'] == 4
    assert emulator._noise_diag_train.shape == (len(X.drop_duplicates()),)
    assert np.all(additional['simulator_variance'] > 0)


def test_adaptive_nugget_adds_known_noise_outside_profiled_sigma2(
    sparse_replicated_training_data,
):
    """Adaptive simulator variance should be additive, not scaled by sigma^2."""
    X, y = sparse_replicated_training_data
    emulator = hm.BayesLinear(X, y, nugget='adaptive', test_fraction=0)

    emulator.train()

    expected_noise_diag = (
        emulator._simulator_variance_train
        / emulator._sample_counts_train
        / emulator._y_std ** 2
    )
    expected_data_cov = (
        emulator._sigma2
        * emulator._sq_exp_corr(
            emulator._X_train_norm,
            emulator._X_train_norm,
            emulator._theta,
        )
        + np.diag(expected_noise_diag)
    )

    np.testing.assert_allclose(emulator._noise_diag_train, expected_noise_diag)
    np.testing.assert_allclose(emulator._data_cov_train, expected_data_cov)


def test_holdout_split_groups_replicated_parameter_sites(noisy_repeated_training_data):
    """Repeated parameter sites must not be split between train and test."""
    np.random.seed(42)
    X, y = noisy_repeated_training_data
    emulator = hm.BayesLinear(X, y, nugget='mle', test_fraction=0.3)

    train_sites = set(map(tuple, emulator.X_train))
    test_sites = set(map(tuple, emulator.X_test))

    assert train_sites.isdisjoint(test_sites)
    assert len(train_sites) == 7
    assert len(test_sites) == 3


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
