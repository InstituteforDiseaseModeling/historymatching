"""
Unit tests for EmulationResults index handling.

Regression coverage for the index-alignment bug: emulators build their outputs
from a mix of numpy arrays (which get a fresh RangeIndex) and pandas Series
(which carry the caller's labels). When mean and std ended up on *different*
indexes, downstream arithmetic such as `abs(mean - target) / np.sqrt(var)`
aligned them by label instead of elementwise, silently returning the union of
both indexes -- a longer result than the number of points predicted on.

This matters because the NROY filter predicts on `candidates.loc[mask]`, whose
labels are non-contiguous after the first wave.
"""

import numpy as np
import pandas as pd
import pytest

import historymatching as hm
from historymatching.emulators.linear import LinearModel, LinearModelScipy
from historymatching.emulators.results import EmulationResults


@pytest.fixture
def training_data():
    """Well-conditioned linear training data."""
    rng = np.random.default_rng(0)
    x = pd.DataFrame({'beta': rng.random(40), 'gamma': rng.random(40)})
    # A little noise: a noise-free target makes GLM's IRLS fit report perfect
    # separation, which is unrelated to what these tests are checking.
    y = pd.DataFrame({
        'peak': 2.0 * x['beta'] - x['gamma'] + 1.0 + rng.normal(0, 0.01, 40)
    })
    return x, y


@pytest.fixture
def noncontiguous_inputs():
    """Prediction inputs with non-contiguous labels, as the NROY filter produces."""
    rng = np.random.default_rng(1)
    candidates = pd.DataFrame({'beta': rng.random(30), 'gamma': rng.random(30)})
    mask = np.zeros(30, dtype=bool)
    mask[15:] = True  # labels 15..29, disjoint from a fresh RangeIndex
    return candidates.loc[mask, ['beta', 'gamma']]


class TestEmulationResultsIndex:
    """Index consistency of the results container itself."""

    def test_mixed_array_and_series_share_one_index(self):
        """An ndarray mean and a label-indexed Series std must not misalign."""
        index = pd.Index([15, 16, 17])
        mean = np.array([1.0, 2.0, 3.0])            # would get RangeIndex(3)
        std = pd.Series([0.5, 0.5, 0.5], index=index)  # carries labels 15..17

        results = EmulationResults(mean=mean, std=std, index=index)

        assert results.get_mean().index.equals(index)
        assert results.get_variance().index.equals(index)
        # The actual failure mode: label-aligned arithmetic returning the union.
        implausibility = abs(results.get_mean() - 1.0) / np.sqrt(results.get_variance())
        assert len(implausibility) == 3

    def test_index_defaults_to_series_index(self):
        """With no explicit index, a Series operand's labels are adopted."""
        index = pd.Index([7, 8, 9])
        results = EmulationResults(mean=pd.Series([1.0, 2.0, 3.0], index=index),
                                   std=np.array([0.1, 0.1, 0.1]))
        assert results.get_mean().index.equals(index)
        assert results.get_std().index.equals(index)

    def test_index_defaults_to_range_for_plain_arrays(self):
        results = EmulationResults(mean=np.array([1.0, 2.0]), std=np.array([0.1, 0.1]))
        assert results.get_mean().index.equals(pd.RangeIndex(2))

    def test_additional_data_follows_the_shared_index(self):
        index = pd.Index([15, 16])
        additional = pd.DataFrame({'ci_obs_low': [0.0, 1.0]}, index=pd.RangeIndex(2))
        results = EmulationResults(mean=np.array([1.0, 2.0]), std=np.array([0.1, 0.1]),
                                   additional_data=additional, index=index)
        assert results.get_additional_data().index.equals(index)

    def test_length_mismatch_still_raises(self):
        with pytest.raises(ValueError, match="same length"):
            EmulationResults(mean=np.array([1.0, 2.0]), std=np.array([0.1]))

    def test_additional_data_length_mismatch_still_raises(self):
        with pytest.raises(ValueError, match="same length"):
            EmulationResults(mean=np.array([1.0, 2.0]), std=np.array([0.1, 0.1]),
                             additional_data=pd.DataFrame({'a': [1.0]}))


@pytest.mark.parametrize('emulator_class', [
    LinearModel,
    LinearModelScipy,
    hm.GLM,
    hm.BayesLinear,
    hm.GPR,
])
class TestEmulatorPredictIndex:
    """Every emulator must label predictions with the input's index."""

    def test_predict_preserves_noncontiguous_index(
        self, emulator_class, training_data, noncontiguous_inputs
    ):
        x, y = training_data
        emulator = emulator_class(x, y)
        emulator.train()

        results = emulator.predict(noncontiguous_inputs)
        expected = noncontiguous_inputs.index

        assert results.get_mean().index.equals(expected)
        assert results.get_variance().index.equals(expected)
        assert results.get_additional_data().index.equals(expected)

    def test_implausibility_stays_elementwise(
        self, emulator_class, training_data, noncontiguous_inputs
    ):
        """Regression: label-aligned arithmetic used to inflate the result length."""
        x, y = training_data
        emulator = emulator_class(x, y)
        emulator.train()

        results = emulator.predict(noncontiguous_inputs)
        implausibility = (abs(results.get_mean() - 1.0)
                          / np.sqrt(results.get_variance() + 1.0))

        assert len(implausibility) == len(noncontiguous_inputs)
        assert not implausibility.isna().any()


class TestFilterConsistency:
    """The engine and the public NROY path must use one filter implementation."""

    @staticmethod
    def _engine():
        return hm.HistoryMatching(
            function=lambda samples: pd.DataFrame({'peak': np.zeros(len(samples))}),
            bounds={'beta': (0.0, 1.0), 'gamma': (0.0, 1.0)},
            observations={'peak': (1.0, 0.5)},
        )

    @staticmethod
    def _bank_with(emulator):
        bank = hm.EmulatorBank()
        bank.add_emulator(1, 'peak', emulator)
        return bank

    def test_engine_delegates_to_canonical_filter(self, monkeypatch):
        """_filter_samples_slow must call _filter_nroy, not a private copy."""
        from historymatching import nroy_sampling

        calls = []
        original = nroy_sampling._filter_nroy

        def spy(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        monkeypatch.setattr(nroy_sampling, '_filter_nroy', spy)

        engine = self._engine()
        candidates = pd.DataFrame({'beta': [0.1, 0.2], 'gamma': [0.3, 0.4]})
        engine._filter_samples_slow(candidates, self._bank_with(_ConstantEmulator()))

        assert len(calls) == 1, "engine should delegate to nroy_sampling._filter_nroy"

    def test_broken_emulator_raises_rather_than_dropping_a_constraint(self):
        """A failing emulator must fail loud: silently skipping it widens NROY."""
        engine = self._engine()
        candidates = pd.DataFrame({'beta': [0.1, 0.2], 'gamma': [0.3, 0.4]})

        with pytest.raises(RuntimeError, match="NROY filtering failed"):
            engine._filter_samples_slow(candidates, self._bank_with(_BrokenEmulator()))


class _ConstantEmulator(hm.BaseEmulator):
    """Emulator that predicts a constant, correctly indexed by the input."""

    def __init__(self):
        super().__init__()
        self.training_complete = True

    def train(self):
        self.training_complete = True

    def predict(self, x):
        from historymatching.emulators.results import EmulationResults
        return EmulationResults(mean=np.ones(len(x)), std=np.ones(len(x)), index=x.index)


class _BrokenEmulator(hm.BaseEmulator):
    """Emulator whose predict() always fails."""

    def __init__(self):
        super().__init__()
        self.training_complete = True

    def train(self):
        self.training_complete = True

    def predict(self, x):
        raise ValueError("emulator is broken")
