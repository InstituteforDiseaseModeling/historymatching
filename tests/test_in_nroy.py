"""Tests for the public NROY membership test: in_nroy() and implausibility()."""


import numpy as np
import pandas as pd
import pytest

import historymatching as hm
from historymatching.emulators.results import EmulationResults
from historymatching.nroy_sampling import _filter_nroy


class LinearInOneParam(hm.BaseEmulator):
    """Deterministic emulator: mean = scale * x[param], fixed std."""

    def __init__(self, param, scale, std=0.0):
        super().__init__()
        self.param, self.scale, self.std = param, scale, std
        self.training_complete = True
        self.calls = 0

    def train(self):
        self.training_complete = True

    def predict(self, x):
        self.calls += 1
        mean = self.scale * x[self.param].to_numpy(dtype=float)
        return EmulationResults(mean=mean, std=np.full(len(x), self.std))

    def print_emulator_description(self):
        return "deterministic test emulator"

    def get_hyperparameters(self):
        return {}


@pytest.fixture
def engine():
    """Engine with bounds param1, param2 in [0, 1] and two targets.

    output1 = 10 * param1, target 5 ± 1 → in NROY iff param1 in [0.2, 0.8].
    output2 = 10 * param2, target 5 ± 1 → in NROY iff param2 in [0.2, 0.8].
    """
    return hm.HistoryMatching(
        bounds={"param1": (0.0, 1.0), "param2": (0.0, 1.0)},
        observations={"output1": (5.0, 1.0), "output2": (5.0, 1.0)},
        sampling_strategy=hm.RandomSampling(),
        feature_selection=hm.ManualFeatureSelection(["output1"]),
        emulator_factory=hm.EmulatorFactory("linear"),
        n_samples=20,
        output_dir=None,
    )


def grid(n=11):
    values = np.linspace(0.0, 1.0, n)
    p1, p2 = np.meshgrid(values, values)
    return pd.DataFrame({"param1": p1.ravel(), "param2": p2.ravel()})


def expected(df):
    ok1 = (df["param1"] >= 0.2 - 1e-12) & (df["param1"] <= 0.8 + 1e-12)
    ok2 = (df["param2"] >= 0.2 - 1e-12) & (df["param2"] <= 0.8 + 1e-12)
    return (ok1 & ok2).to_numpy()


def add_two_waves(engine):
    engine.emulator_bank.add_emulator(1, "output1", LinearInOneParam("param1", 10.0))
    engine.emulator_bank.add_emulator(2, "output2", LinearInOneParam("param2", 10.0))


class TestInNroy:
    def test_no_emulators_means_bounds_only(self, engine):
        samples = pd.DataFrame({"param1": [0.5, 1.5, -0.1], "param2": [0.5, 0.5, 0.5]})
        assert engine.in_nroy(samples).tolist() == [True, False, False]

    def test_matches_analytic_region(self, engine):
        add_two_waves(engine)
        samples = grid()
        np.testing.assert_array_equal(engine.in_nroy(samples), expected(samples))

    def test_agrees_with_internal_filters(self, engine):
        add_two_waves(engine)
        samples = grid(21)
        mask = engine.in_nroy(samples)
        engine_filtered = engine._filter_samples_by_implausibility(samples)
        design_filtered = _filter_nroy(samples, engine.emulator_bank, engine.observations,
                                       engine.implausibility_threshold, ["param1", "param2"])
        assert samples.index[mask].tolist() == engine_filtered.index.tolist()
        assert samples.index[mask].tolist() == design_filtered.index.tolist()

    def test_out_of_bounds_point_is_ruled_out_even_if_emulators_pass(self, engine):
        # output1 = 10 * 0.5 passes; param2 = 0.5 passes; but param1 is checked
        # against the box, and 5.0 for a [0, 1] parameter is outside it.
        engine.emulator_bank.add_emulator(1, "output1", LinearInOneParam("param2", 10.0))
        samples = pd.DataFrame({"param1": [0.5, 5.0], "param2": [0.5, 0.5]})
        assert engine.in_nroy(samples).tolist() == [True, False]

    def test_keeps_row_order_with_any_index(self, engine):
        add_two_waves(engine)
        samples = pd.DataFrame({"param1": [0.9, 0.5, 0.1], "param2": [0.5, 0.5, 0.5]},
                               index=[10, 3, 7])
        assert engine.in_nroy(samples).tolist() == [False, True, False]

    def test_batch_size_gives_same_answer(self, engine):
        add_two_waves(engine)
        samples = grid(21)
        np.testing.assert_array_equal(engine.in_nroy(samples, batch_size=7),
                                      engine.in_nroy(samples))

    def test_extra_columns_are_ignored(self, engine):
        add_two_waves(engine)
        samples = pd.DataFrame({"param1": [0.5], "param2": [0.5], "seed": [123]})
        assert engine.in_nroy(samples).tolist() == [True]

    def test_missing_parameter_column_raises(self, engine):
        with pytest.raises(ValueError, match="param2"):
            engine.in_nroy(pd.DataFrame({"param1": [0.5]}))

    def test_empty_samples(self, engine):
        add_two_waves(engine)
        empty = pd.DataFrame({"param1": [], "param2": []})
        assert engine.in_nroy(empty).shape == (0,)

    def test_emulator_variance_widens_the_region(self, engine):
        # |10 * p1 - 5| / sqrt(std_em**2 + 1) <= 3; with std_em = 2, |10 p1 - 5| <= 6.7
        engine.emulator_bank.add_emulator(1, "output1",
                                          LinearInOneParam("param1", 10.0, std=2.0))
        samples = pd.DataFrame({"param1": [0.05, 0.95], "param2": [0.5, 0.5]})
        assert engine.in_nroy(samples).tolist() == [True, True]

    def test_uses_committed_emulators_only(self, engine):
        def simulate(samples):
            n = len(samples)
            return pd.DataFrame({"output1": 10 * samples["param1"].to_numpy(),
                                 "output2": np.full(n, 5.0)})

        engine.function = simulate
        engine.step()
        samples = grid()
        # Pending wave: its emulator isn't committed yet, so only bounds apply
        assert engine.in_nroy(samples).all()
        engine.commit_step()
        mask = engine.in_nroy(samples)
        assert not mask.all()
        # Every committed NROY sample is in NROY by the public test
        assert engine.in_nroy(engine.get_nroy_samples()).all()


class TestImplausibility:
    def test_one_column_per_committed_emulator(self, engine):
        add_two_waves(engine)
        samples = pd.DataFrame({"param1": [0.5, 0.0], "param2": [0.9, 0.5]}, index=["a", "b"])
        table = engine.implausibility(samples)
        assert list(table.columns) == [(1, "output1"), (2, "output2")]
        assert table.columns.names == ["iteration", "feature"]
        assert table.index.tolist() == ["a", "b"]
        np.testing.assert_allclose(table[(1, "output1")], [0.0, 5.0])
        np.testing.assert_allclose(table[(2, "output2")], [4.0, 0.0])

    def test_max_implausibility_agrees_with_in_nroy(self, engine):
        add_two_waves(engine)
        samples = grid(21)
        by_table = (engine.implausibility(samples).max(axis=1)
                    <= engine.implausibility_threshold).to_numpy()
        np.testing.assert_array_equal(by_table, engine.in_nroy(samples))

    def test_no_emulators_gives_empty_table(self, engine):
        table = engine.implausibility(pd.DataFrame({"param1": [0.5], "param2": [0.5]}))
        assert table.shape == (1, 0)


class TestEveryEmulatorApplied:
    def test_engine_filter_applies_every_emulator(self, engine):
        """Each emulator in the bank constrains the engine's filter, whatever its type."""
        first = LinearInOneParam("param1", 10.0)    # rules on param1
        second = LinearInOneParam("param2", 10.0)   # rules on param2
        engine.emulator_bank.add_emulator(1, "output1", first)
        engine.emulator_bank.add_emulator(2, "output2", second)

        samples = grid()
        filtered = engine._filter_samples_by_implausibility(samples)

        assert filtered.index.tolist() == samples.index[expected(samples)].tolist()
        assert first.calls >= 1 and second.calls >= 1
