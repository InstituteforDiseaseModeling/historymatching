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
