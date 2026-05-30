"""
Tests for nroy_sampling module — NROY fraction reporting and sampling pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from historymatching.nroy_sampling import (
    NROYResult,
    generate_nroy_design,
    _lhs_reject_loop,
)
from historymatching.emulator_bank import EmulatorBank
from historymatching.observation_data import ObservationData
from historymatching.parameter_space import ParameterSpace
from historymatching.emulators.base import BaseEmulator
from historymatching.emulators.results import EmulationResults


class PassthroughEmulator(BaseEmulator):
    """Emulator that predicts the first input column as the output.

    With a tight observation target around 0.5, roughly half the [0,1]
    uniform prior will be implausible — giving a measurable, non-trivial
    acceptance rate.
    """

    def __init__(self, feature_name="output"):
        super().__init__(feature_name)

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        mean = X.iloc[:, 0].values if isinstance(X, pd.DataFrame) else X[:, 0]
        # Small constant std so implausibility is dominated by |pred - obs|
        std = np.full_like(mean, np.sqrt(0.001))
        return EmulationResults(mean=pd.Series(mean), std=pd.Series(std))

    def get_hyperparameters(self):
        return {"type": "passthrough"}


@pytest.fixture
def simple_setup():
    """2-parameter, 1-observation setup with a passthrough emulator."""
    param_bounds = {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}
    # Target x1 ≈ 0.5 ± 0.1 → with threshold=3.0, accept x1 in [0.2, 0.8]
    # Expected acceptance ≈ 60%
    observations = {"output": (0.5, 0.1)}

    param_space = ParameterSpace(param_bounds)
    obs_data = ObservationData(observations)

    emulator = PassthroughEmulator("output")
    # Minimal training data (not used by passthrough, but required by API)
    X_dummy = pd.DataFrame({"x1": [0.5], "x2": [0.5]})
    y_dummy = pd.Series([0.5])
    emulator.train(X_dummy, y_dummy)

    bank = EmulatorBank()
    bank.add_emulator(1, "output", emulator)

    return param_space, obs_data, bank


class TestNROYFractionReporting:
    """Verify that NROYResult reports the actual number of candidates tested,
    not a hard-coded max_candidates ceiling."""

    def test_lhs_success_reports_actual_tested(self, simple_setup):
        """When LHS finds enough points quickly, lhs_tested should reflect
        actual candidates generated, NOT max_candidates."""
        param_space, obs_data, bank = simple_setup

        result = generate_nroy_design(
            n_points=50,
            parameter_space=param_space,
            emulator_bank=bank,
            observations=obs_data,
            threshold=3.0,
            method='auto',
            seed=42,
            max_candidates=10_000_000,  # Deliberately large
        )

        assert len(result.samples) == 50
        # The passthrough emulator accepts ~60% of uniform draws.
        # To get 50 points we need ~85 candidates, certainly not 10M.
        assert result.lhs_tested < 10_000, (
            f"lhs_tested={result.lhs_tested} — should reflect actual candidates "
            f"generated, not max_candidates"
        )
        # Sanity: tested >= accepted
        assert result.lhs_tested >= result.lhs_accepted

    def test_nroy_fraction_is_reasonable(self, simple_setup):
        """The NROY fraction computed from NROYResult should match the
        actual acceptance rate, not be artificially deflated."""
        param_space, obs_data, bank = simple_setup

        result = generate_nroy_design(
            n_points=100,
            parameter_space=param_space,
            emulator_bank=bank,
            observations=obs_data,
            threshold=3.0,
            method='auto',
            seed=123,
            max_candidates=10_000_000,
        )

        fraction = result.lhs_accepted / result.lhs_tested
        # With target=0.5±0.1 and threshold=3.0, acceptance should be ~60%
        assert fraction > 0.3, (
            f"NROY fraction={fraction:.4f} is implausibly low — "
            f"likely a reporting bug (accepted={result.lhs_accepted}, "
            f"tested={result.lhs_tested})"
        )

    def test_lhs_reject_loop_tracks_total_generated(self):
        """_lhs_reject_loop should attach total_generated to the result."""
        param_space = ParameterSpace({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})

        from historymatching.sampling import SamplingStrategyFactory
        strategy = SamplingStrategyFactory.create('lhs')

        # filter_fn that accepts everything
        result = _lhs_reject_loop(
            n_points=50,
            parameter_space=param_space,
            sampling_strategy=strategy,
            filter_fn=lambda df: df,  # accept all
            seed=42,
        )

        assert len(result) == 50
        total = result.attrs.get('total_generated', None)
        assert total is not None, "total_generated not attached to result"
        # Accept-all means we only need one batch
        assert total < 200, f"total_generated={total}, expected ~55 (one batch)"

    def test_ray_mode_reports_actual_seed_count(self, simple_setup):
        """In ray mode, lhs_tested should be the seed LHS size, not max_candidates."""
        param_space, obs_data, bank = simple_setup

        result = generate_nroy_design(
            n_points=20,
            parameter_space=param_space,
            emulator_bank=bank,
            observations=obs_data,
            threshold=3.0,
            method='ray',
            seed=42,
            max_candidates=10_000_000,
        )

        assert len(result.samples) == 20
        # Ray mode generates max(n_points, lhs_factor * n_dims) = max(20, 10*2) = 20 seed candidates
        assert result.lhs_tested <= 1000, (
            f"lhs_tested={result.lhs_tested} — ray mode should report "
            f"actual seed LHS size, not max_candidates"
        )
