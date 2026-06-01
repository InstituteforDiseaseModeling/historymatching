"""
Unit tests for the HistoryMatching configuration API.

These cover the single-constructor configuration that replaced the old
HistoryMatchingBuilder: friendly values (dicts / strings / lists / objects) are
accepted for each option and coerced into the underlying domain objects.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import historymatching as hm
from historymatching import HistoryMatching
from historymatching.emulator_bank import EmulatorBank
from historymatching.observation_data import ObservationData
from historymatching.parameter_space import ParameterSpace
from historymatching.emulators.factory import EmulatorFactory
from historymatching.feature_selection import AutoFeatureSelection
from historymatching.feature_selection import ManualFeatureSelection
from historymatching.sampling import LatinHypercubeSampling
from historymatching.sampling import RandomSampling


@pytest.fixture
def parameter_bounds():
    return {
        "param1": (0.0, 1.0),
        "param2": (-5.0, 5.0),
        "param3": (10.0, 100.0),
    }


@pytest.fixture
def observations():
    return {
        "output1": (25.0, 5.0),  # (mean, std)
        "output2": (100.0, 10.0),
        "output3": (0.5, 0.1),
    }


@pytest.fixture
def parameter_df():
    return pd.DataFrame({
        "parameter": ["param1", "param2", "param3"],
        "minimum": [0.0, -5.0, 10.0],
        "maximum": [1.0, 5.0, 100.0],
    })


@pytest.fixture
def observations_df():
    return pd.DataFrame({
        "feature": ["output1", "output2", "output3"],
        "mean": [25.0, 100.0, 0.5],
        "std": [5.0, 10.0, 0.1],
    })


def make(parameter_bounds, observations, **kwargs):
    """Build a HistoryMatching with disk output disabled by default."""
    kwargs.setdefault("output_dir", None)
    return HistoryMatching(parameter_bounds=parameter_bounds, observations=observations, **kwargs)


class TestConstruction:
    """Constructing HistoryMatching from plain data."""

    def test_public_aliases(self):
        assert hm.HistoryMatch is hm.HistoryMatching
        assert hm.HistoryMatchingEngine is hm.HistoryMatching

    def test_defaults(self, parameter_bounds, observations):
        match = make(parameter_bounds, observations)
        assert match.n_samples == 1000
        assert match.implausibility_threshold == 3.0
        assert match.max_iterations == 10
        assert match.random_seed is None
        assert match.auto_reduce_space is False
        assert match.oversample_factor == 1.1

    def test_from_dicts(self, parameter_bounds, observations):
        match = make(parameter_bounds, observations)
        assert isinstance(match.parameter_space, ParameterSpace)
        assert isinstance(match.observations, ObservationData)
        assert set(match.get_parameter_names()) == set(parameter_bounds)
        assert set(match.get_feature_names()) == set(observations)

    def test_from_dataframes(self, parameter_df, observations_df):
        match = make(parameter_df, observations_df)
        assert set(match.get_parameter_names()) == set(parameter_df["parameter"])
        assert set(match.get_feature_names()) == set(observations_df["feature"])

    def test_from_existing_objects(self, parameter_bounds, observations):
        ps = ParameterSpace(parameter_bounds)
        obs = ObservationData(observations)
        match = make(ps, obs)
        assert match.parameter_space is ps
        assert match.observations is obs

    def test_constructor_kwargs(self, parameter_bounds, observations):
        match = make(
            parameter_bounds, observations,
            n_samples=500, max_iterations=5,
            sampling_strategy="grid", emulator_type="linear",
        )
        assert match.n_samples == 500
        assert match.max_iterations == 5
        assert "Grid" in match.sampling_strategy.get_strategy_name()
        assert match.emulator_factory.get_default_type() == "linear"

    def test_unknown_kwargs_go_to_settings(self, parameter_bounds, observations):
        match = make(parameter_bounds, observations, max_candidate_factor=2000)
        assert match.settings["max_candidate_factor"] == 2000


class TestSamplingStrategy:
    def test_string(self, parameter_bounds, observations):
        match = make(parameter_bounds, observations, sampling_strategy="grid")
        assert "Grid" in match.sampling_strategy.get_strategy_name()

    def test_object(self, parameter_bounds, observations):
        strategy = RandomSampling()
        match = make(parameter_bounds, observations, sampling_strategy=strategy)
        assert match.sampling_strategy is strategy

    def test_dict(self, parameter_bounds, observations):
        match = make(parameter_bounds, observations,
                     sampling_strategy={"type": "lhs", "criterion": "center"})
        assert isinstance(match.sampling_strategy, LatinHypercubeSampling)
        assert "center" in match.sampling_strategy.get_strategy_name()

    def test_default_is_lhs(self, parameter_bounds, observations):
        match = make(parameter_bounds, observations)
        assert isinstance(match.sampling_strategy, LatinHypercubeSampling)


class TestFeatureSelection:
    def test_list(self, parameter_bounds, observations):
        match = make(parameter_bounds, observations, feature_selection=["output1", "output2"])
        assert isinstance(match.feature_selection_strategy, ManualFeatureSelection)

    def test_string(self, parameter_bounds, observations):
        match = make(parameter_bounds, observations, feature_selection="output1")
        assert isinstance(match.feature_selection_strategy, ManualFeatureSelection)

    def test_object(self, parameter_bounds, observations):
        strategy = AutoFeatureSelection(method="var", max_features=2)
        match = make(parameter_bounds, observations, feature_selection=strategy)
        assert match.feature_selection_strategy is strategy

    def test_dict(self, parameter_bounds, observations):
        match = make(parameter_bounds, observations,
                     feature_selection={"method": "var", "max_features": 2, "threshold": 0.5})
        strategy = match.feature_selection_strategy
        assert isinstance(strategy, AutoFeatureSelection)
        assert strategy.method == "var"
        assert strategy.max_features == 2
        assert strategy.threshold == 0.5

    def test_default_is_auto(self, parameter_bounds, observations):
        match = make(parameter_bounds, observations)
        assert isinstance(match.feature_selection_strategy, AutoFeatureSelection)


class TestEmulatorConfig:
    def test_emulator_type(self, parameter_bounds, observations):
        match = make(parameter_bounds, observations, emulator_type="linear")
        assert isinstance(match.emulator_factory, EmulatorFactory)
        assert match.emulator_factory.get_default_type() == "linear"

    def test_emulator_factory_overrides_type(self, parameter_bounds, observations):
        factory = EmulatorFactory("gpr", kernel="rbf")
        match = make(parameter_bounds, observations,
                     emulator_type="linear", emulator_factory=factory)
        assert match.emulator_factory is factory

    def test_emulator_bank(self, parameter_bounds, observations):
        bank = EmulatorBank()
        match = make(parameter_bounds, observations, emulator_bank=bank)
        assert match.emulator_bank is bank

    def test_default_is_gpr(self, parameter_bounds, observations):
        match = make(parameter_bounds, observations)
        assert match.emulator_factory.get_default_type() == "gpr"


class TestWorkflowParameters:
    def test_workflow_parameters(self, parameter_bounds, observations):
        match = make(parameter_bounds, observations,
                     n_samples=2000, max_iterations=20,
                     implausibility_threshold=2.5, random_seed=42)
        assert match.n_samples == 2000
        assert match.max_iterations == 20
        assert match.implausibility_threshold == 2.5
        assert match.random_seed == 42

    def test_space_reduction_options(self, parameter_bounds, observations):
        match = make(parameter_bounds, observations,
                     auto_reduce_space=True, oversample_factor=5.0)
        assert match.auto_reduce_space is True
        assert match.oversample_factor == 5.0

    def test_full_configuration(self, parameter_bounds, observations):
        match = make(
            parameter_bounds, observations,
            sampling_strategy="lhs",
            feature_selection=["output1"],
            emulator_type="gpr",
            n_samples=1500,
            max_iterations=15,
            implausibility_threshold=2.8,
            random_seed=123,
            auto_reduce_space=True,
            oversample_factor=3.0,
        )
        assert match.n_samples == 1500
        assert match.max_iterations == 15
        assert match.implausibility_threshold == 2.8
        assert match.random_seed == 123
        assert match.auto_reduce_space is True
        assert match.oversample_factor == 3.0
        assert isinstance(match.feature_selection_strategy, ManualFeatureSelection)


class TestValidation:
    def test_missing_parameter_bounds(self, observations):
        with pytest.raises(ValueError, match="parameter_bounds is required"):
            HistoryMatching(observations=observations, output_dir=None)

    def test_missing_observations(self, parameter_bounds):
        with pytest.raises(ValueError, match="observations is required"):
            HistoryMatching(parameter_bounds=parameter_bounds, output_dir=None)

    def test_rejects_nonpositive_n_samples(self, parameter_bounds, observations):
        with pytest.raises(ValueError):
            make(parameter_bounds, observations, n_samples=0)

    def test_rejects_negative_max_iterations(self, parameter_bounds, observations):
        with pytest.raises(ValueError):
            make(parameter_bounds, observations, max_iterations=-1)

    def test_rejects_nonpositive_threshold(self, parameter_bounds, observations):
        with pytest.raises(ValueError):
            make(parameter_bounds, observations, implausibility_threshold=0)

    def test_rejects_small_oversample(self, parameter_bounds, observations):
        with pytest.raises(ValueError):
            make(parameter_bounds, observations, oversample_factor=0.5)

    def test_invalid_sampling_strategy_type(self, parameter_bounds, observations):
        with pytest.raises(ValueError):
            make(parameter_bounds, observations, sampling_strategy=123)

    def test_invalid_feature_selection_type(self, parameter_bounds, observations):
        with pytest.raises(ValueError):
            make(parameter_bounds, observations, feature_selection=123)

    def test_empty_parameter_bounds(self):
        with pytest.raises(ValueError):
            make({}, {"output": (1.0, 0.1)})

    def test_empty_observations(self):
        with pytest.raises(ValueError):
            make({"param": (0, 1)}, {})

    def test_inconsistent_bounds_tuple(self):
        with pytest.raises((ValueError, TypeError)):
            make({"param": (0,)}, {"output": (1.0, 0.1)})


class TestSimulationFunction:
    def test_function_via_constructor(self, parameter_bounds, observations):
        def sim(samples):
            return pd.DataFrame({"output1": np.ones(len(samples))})
        match = make(parameter_bounds, observations, function=sim)
        assert match.function is sim
        assert match._simulation_function is sim

    def test_function_via_setter(self, parameter_bounds, observations):
        def sim(samples):
            return pd.DataFrame({"output1": np.ones(len(samples))})
        match = make(parameter_bounds, observations)
        assert match.function is None
        match.set_simulation_function(sim)
        assert match.function is sim
        # The `function` property setter is equivalent.
        match2 = make(parameter_bounds, observations)
        match2.function = sim
        assert match2.function is sim

    def test_output_dataframe_passthrough(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        out = HistoryMatching._coerce_simulation_output(df, 3)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["a", "b"]
        assert len(out) == 3
        assert out["a"].tolist() == [1, 2, 3]

    def test_output_records_normalized(self):
        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        out = HistoryMatching._coerce_simulation_output(records, 2)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["a", "b"]
        assert len(out) == 2

    def test_output_dict_of_columns(self):
        out = HistoryMatching._coerce_simulation_output({"a": [1, 2], "b": [3, 4]}, 2)
        assert list(out.columns) == ["a", "b"]
        assert len(out) == 2

    def test_output_wrong_length_raises(self):
        with pytest.raises(ValueError, match="exactly one row"):
            HistoryMatching._coerce_simulation_output([{"a": 1}], 3)


class TestConveniences:
    """End-to-end checks of the friendly accessors after a short run."""

    @pytest.fixture
    def ran_match(self, parameter_bounds, observations):
        np.random.seed(0)

        def sim(samples):
            # list-of-dicts output (records), exercising auto-normalization
            rows = []
            for _, row in samples.iterrows():
                rows.append({
                    "output1": 25.0 + np.random.normal(0, 0.5),
                    "output2": 100.0 + np.random.normal(0, 1.0),
                    "output3": 0.5 + np.random.normal(0, 0.02),
                })
            return rows

        match = make(
            parameter_bounds, observations,
            function=sim,
            emulator_type="linear",
            feature_selection=["output1"],
            n_samples=30,
            max_iterations=2,
            random_seed=1,
        )
        match.run()
        return match

    def test_len_and_results(self, ran_match):
        assert len(ran_match) == len(ran_match.results)
        assert 1 <= len(ran_match) <= 2

    def test_enumerate(self, ran_match):
        rows = list(ran_match.enumerate())
        assert len(rows) == len(ran_match)
        for i, result, samples in rows:
            assert result.iteration == i
            assert len(samples) == len(result.samples)

    def test_progress_properties(self, ran_match):
        assert ran_match.samples_generated > 0
        assert ran_match.samples_accepted > 0
        assert ran_match.emulators_trained >= 1

    def test_print_emulator_quality_metrics(self, ran_match, capsys):
        metrics = ran_match.print_emulator_quality_metrics()
        out = capsys.readouterr().out
        assert "output1" in metrics
        assert "wave" in out.lower()

    def test_plot_nroy_parameters(self, ran_match):
        fig, axes = ran_match.plot_nroy_parameters(
            derived={"sum": lambda df: df["param1"] + df["param2"]},
            true_parameters={"param1": 0.5},
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestRepr:
    def test_repr(self, parameter_bounds, observations):
        match = make(parameter_bounds, observations)
        text = repr(match)
        assert "HistoryMatching(" in text
        assert "state=initialized" in text
