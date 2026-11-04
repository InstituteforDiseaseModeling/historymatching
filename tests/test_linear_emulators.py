"""Linear-family emulators report the std of a new simulator run (#273).

Implausibility divides by sqrt(target variance + emulator variance), and targets carry the data
uncertainty only, so an emulator's predictive std has to include the simulator's run-to-run
noise, as BayesLinear (sigma_sq * nugget) and GPR (predict_y) do. Then, at fresh points,
(new run - predicted mean) / predicted std has a standard deviation close to 1.
"""
import numpy as np
import pandas as pd
import pytest

from historymatching.emulators.glm import GLM
from historymatching.emulators.linear import LinearModel, LinearModelScipy

N_TRAIN, N_TEST, NOISE_SD = 400, 4000, 0.3


def gaussian_world(rng, n):
    x = pd.DataFrame(rng.uniform(0, 1, (n, 2)), columns=["a", "b"])
    mean = (1.0 + 2.0 * x["a"] - 1.0 * x["b"]).to_numpy()
    return x, mean + rng.normal(0, NOISE_SD, n)


def poisson_world(rng, n):
    x = pd.DataFrame(rng.uniform(0, 1, (n, 2)), columns=["a", "b"])
    mean = np.exp(1.0 + 1.5 * x["a"] - 0.5 * x["b"]).to_numpy()  # about 1.6 to 12
    return x, rng.poisson(mean).astype(float)


CASES = {
    "LinearModel": (LinearModel, {}, gaussian_world),
    "LinearModelScipy": (LinearModelScipy, {}, gaussian_world),
    "GLM gaussian": (GLM, {"link": "linear"}, gaussian_world),
    "GLM poisson": (GLM, {"link": "poisson"}, poisson_world),
}


def trained(name, seed=273):
    cls, kwargs, world = CASES[name]
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # the train/test split draws from NumPy's global RNG
    x, y = world(rng, N_TRAIN)
    em = cls(x, pd.DataFrame({"y": y}), **kwargs)
    em.train()
    return em, rng, world


@pytest.mark.parametrize("name", CASES)
def test_predictive_std_is_calibrated(name):
    em, rng, world = trained(name)
    x_new, y_new = world(rng, N_TEST)
    pred = em.predict(x_new)
    z = (y_new - pred.get_mean().to_numpy()) / pred.get_std().to_numpy()
    assert 0.9 < z.std() < 1.1, f"{name}: sd of standardized residuals {z.std():.3f}"


@pytest.mark.parametrize("name", ["LinearModel", "LinearModelScipy", "GLM gaussian"])
def test_predictive_std_includes_run_to_run_noise(name):
    em, rng, world = trained(name)
    x_new, _ = world(rng, 200)
    assert np.all(em.predict(x_new).get_std().to_numpy() >= NOISE_SD * 0.9)


@pytest.mark.parametrize("name", CASES)
def test_one_point_matches_batch(name):
    em, rng, world = trained(name)
    x_new, _ = world(rng, 10)
    batch = em.predict(x_new)
    one = em.predict(x_new.iloc[[7]])
    assert len(one.get_mean()) == 1
    assert one.get_mean().iloc[0] == pytest.approx(batch.get_mean().iloc[7])
    assert one.get_std().iloc[0] == pytest.approx(batch.get_std().iloc[7])


@pytest.mark.parametrize("name", CASES)
def test_intervals_follow_the_predicted_rows(name):
    """Rows predicted from a filtered frame (non-contiguous labels) keep their intervals."""
    em, rng, world = trained(name)
    x_new, _ = world(rng, 30)
    subset = x_new.iloc[::3]
    extra = em.predict(subset).get_additional_data()
    assert extra.index.equals(subset.index)
    assert not extra.isna().any().any()
