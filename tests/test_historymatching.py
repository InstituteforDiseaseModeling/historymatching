"""
Unit tests for the hm.HistoryMatching configuration API.

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
    return hm.HistoryMatching(bounds=parameter_bounds, observations=observations, **kwargs)


class TestConstruction:
    """Constructing HistoryMatching from plain data."""

    def test_public_namespace_is_curated(self):
        # One canonical class name; the old aliases are gone.
        assert "HistoryMatching" in hm.__all__
        assert not hasattr(hm, "HistoryMatch")
        assert not hasattr(hm, "HistoryMatchingEngine")
        # No leaked submodules or bare constants in the documented surface.
        assert "engine" not in hm.__all__
        assert "PARAMETER_SPACE_COLUMNS" not in hm.__all__

    def test_defaults(self, parameter_bounds, observations):
        engine = make(parameter_bounds, observations)
        assert engine.n_samples == 1000
        assert engine.implausibility_threshold == 3.0
        assert engine.max_iterations == 10
        assert engine.random_seed is None
        assert engine.auto_reduce_space is False
        assert engine.oversample_factor == 1.1

    def test_from_dicts(self, parameter_bounds, observations):
        engine = make(parameter_bounds, observations)
        assert isinstance(engine.parameter_space, hm.ParameterSpace)
        assert isinstance(engine.observations, hm.ObservationData)
        assert set(engine.parameters) == set(parameter_bounds)
        assert set(engine.outputs) == set(observations)

    def test_from_dataframes(self, parameter_df, observations_df):
        engine = make(parameter_df, observations_df)
        assert set(engine.parameters) == set(parameter_df["parameter"])
        assert set(engine.outputs) == set(observations_df["feature"])

    def test_from_existing_objects(self, parameter_bounds, observations):
        ps = hm.ParameterSpace(parameter_bounds)
        obs = hm.ObservationData(observations)
        engine = make(ps, obs)
        assert engine.parameter_space is ps
        assert engine.observations is obs

    def test_constructor_kwargs(self, parameter_bounds, observations):
        engine = make(
            parameter_bounds, observations,
            n_samples=500, max_iterations=5,
            sampling_strategy="grid", emulator_type="linear",
        )
        assert engine.n_samples == 500
        assert engine.max_iterations == 5
        assert "Grid" in engine.sampling_strategy.get_strategy_name()
        assert engine.emulator_factory.get_default_type() == "linear"

    def test_tuning_knobs_are_explicit(self, parameter_bounds, observations):
        # Formerly-hidden **settings knobs are now real, discoverable kwargs.
        engine = make(parameter_bounds, observations,
                     max_candidate_factor=2000, convergence_threshold=0.02, nroy_method="lhs")
        assert engine.max_candidate_factor == 2000
        assert engine.convergence_threshold == 0.02
        assert engine.nroy_method == "lhs"

    def test_unknown_kwarg_raises(self, parameter_bounds, observations):
        # A typo'd option is no longer silently swallowed.
        with pytest.raises(TypeError):
            make(parameter_bounds, observations, max_iteratons=5)


class TestSamplingStrategy:
    def test_string(self, parameter_bounds, observations):
        engine = make(parameter_bounds, observations, sampling_strategy="grid")
        assert "Grid" in engine.sampling_strategy.get_strategy_name()

    def test_object(self, parameter_bounds, observations):
        strategy = hm.RandomSampling()
        engine = make(parameter_bounds, observations, sampling_strategy=strategy)
        assert engine.sampling_strategy is strategy

    def test_dict(self, parameter_bounds, observations):
        engine = make(parameter_bounds, observations,
                     sampling_strategy={"type": "lhs", "criterion": "center"})
        assert isinstance(engine.sampling_strategy, hm.LatinHypercubeSampling)
        assert "center" in engine.sampling_strategy.get_strategy_name()

    def test_default_is_lhs(self, parameter_bounds, observations):
        engine = make(parameter_bounds, observations)
        assert isinstance(engine.sampling_strategy, hm.LatinHypercubeSampling)


class TestFeatureSelection:
    def test_list(self, parameter_bounds, observations):
        engine = make(parameter_bounds, observations, feature_selection=["output1", "output2"])
        assert isinstance(engine.feature_selection, hm.ManualFeatureSelection)

    def test_string(self, parameter_bounds, observations):
        engine = make(parameter_bounds, observations, feature_selection="output1")
        assert isinstance(engine.feature_selection, hm.ManualFeatureSelection)

    def test_object(self, parameter_bounds, observations):
        strategy = hm.AutoFeatureSelection(method="var", max_features=2)
        engine = make(parameter_bounds, observations, feature_selection=strategy)
        assert engine.feature_selection is strategy

    def test_dict(self, parameter_bounds, observations):
        engine = make(parameter_bounds, observations,
                     feature_selection={"method": "var", "max_features": 2, "threshold": 0.5})
        strategy = engine.feature_selection
        assert isinstance(strategy, hm.AutoFeatureSelection)
        assert strategy.method == "var"
        assert strategy.max_features == 2
        assert strategy.threshold == 0.5

    def test_default_is_auto(self, parameter_bounds, observations):
        engine = make(parameter_bounds, observations)
        assert isinstance(engine.feature_selection, hm.AutoFeatureSelection)


class TestEmulatorConfig:
    def test_emulator_type(self, parameter_bounds, observations):
        engine = make(parameter_bounds, observations, emulator_type="linear")
        assert isinstance(engine.emulator_factory, hm.EmulatorFactory)
        assert engine.emulator_factory.get_default_type() == "linear"

    def test_emulator_factory_overrides_type(self, parameter_bounds, observations):
        factory = hm.EmulatorFactory("gpr", kernel="rbf")
        engine = make(parameter_bounds, observations,
                     emulator_type="linear", emulator_factory=factory)
        assert engine.emulator_factory is factory

    def test_emulator_bank(self, parameter_bounds, observations):
        bank = hm.EmulatorBank()
        engine = make(parameter_bounds, observations, emulator_bank=bank)
        assert engine.emulator_bank is bank

    def test_default_is_bayes_linear(self, parameter_bounds, observations):
        engine = make(parameter_bounds, observations)
        assert engine.emulator_factory.get_default_type() == "bayes_linear"


class TestWorkflowParameters:
    def test_workflow_parameters(self, parameter_bounds, observations):
        engine = make(parameter_bounds, observations,
                     n_samples=2000, max_iterations=20,
                     implausibility_threshold=2.5, random_seed=42)
        assert engine.n_samples == 2000
        assert engine.max_iterations == 20
        assert engine.implausibility_threshold == 2.5
        assert engine.random_seed == 42

    def test_space_reduction_options(self, parameter_bounds, observations):
        engine = make(parameter_bounds, observations,
                     auto_reduce_space=True, oversample_factor=5.0)
        assert engine.auto_reduce_space is True
        assert engine.oversample_factor == 5.0

    def test_full_configuration(self, parameter_bounds, observations):
        engine = make(
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
        assert engine.n_samples == 1500
        assert engine.max_iterations == 15
        assert engine.implausibility_threshold == 2.8
        assert engine.random_seed == 123
        assert engine.auto_reduce_space is True
        assert engine.oversample_factor == 3.0
        assert isinstance(engine.feature_selection, hm.ManualFeatureSelection)


class TestValidation:
    def test_missing_bounds(self, observations):
        with pytest.raises(ValueError, match="bounds is required"):
            hm.HistoryMatching(observations=observations, output_dir=None)

    def test_missing_observations(self, parameter_bounds):
        with pytest.raises(ValueError, match="observations is required"):
            hm.HistoryMatching(bounds=parameter_bounds, output_dir=None)

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
        engine = make(parameter_bounds, observations, function=sim)
        assert engine.function is sim

    def test_function_via_setter(self, parameter_bounds, observations):
        def sim(samples):
            return pd.DataFrame({"output1": np.ones(len(samples))})
        engine = make(parameter_bounds, observations)
        assert engine.function is None
        engine.function = sim
        assert engine.function is sim
        # Assigning `function` after construction is equivalent.
        engine2 = make(parameter_bounds, observations)
        engine2.function = sim
        assert engine2.function is sim

    def test_output_dataframe_passthrough(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        out = hm.HistoryMatching._coerce_simulation_output(df, 3)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["a", "b"]
        assert len(out) == 3
        assert out["a"].tolist() == [1, 2, 3]

    def test_output_records_normalized(self):
        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        out = hm.HistoryMatching._coerce_simulation_output(records, 2)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["a", "b"]
        assert len(out) == 2

    def test_output_dict_of_columns(self):
        out = hm.HistoryMatching._coerce_simulation_output({"a": [1, 2], "b": [3, 4]}, 2)
        assert list(out.columns) == ["a", "b"]
        assert len(out) == 2

    def test_output_wrong_length_raises(self):
        with pytest.raises(ValueError, match="exactly one row"):
            hm.HistoryMatching._coerce_simulation_output([{"a": 1}], 3)


class TestConveniences:
    """End-to-end checks of the friendly accessors after a short run."""

    @pytest.fixture
    def ran_engine(self, parameter_bounds, observations):
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

        engine = make(
            parameter_bounds, observations,
            function=sim,
            emulator_type="linear",
            feature_selection=["output1"],
            n_samples=30,
            max_iterations=2,
            random_seed=1,
        )
        engine.run()
        return engine

    def test_len_and_results(self, ran_engine):
        assert len(ran_engine) == len(ran_engine.results)
        assert 1 <= len(ran_engine) <= 2

    def test_enumerate(self, ran_engine):
        rows = list(ran_engine.enumerate())
        assert len(rows) == len(ran_engine)
        for i, result, samples in rows:
            assert result.iteration == i
            assert len(samples) == len(result.samples)

    def test_progress_properties(self, ran_engine):
        assert ran_engine.samples_generated > 0
        assert ran_engine.samples_accepted > 0
        assert ran_engine.emulators_trained >= 1

    def test_print_emulator_quality_metrics(self, ran_engine, capsys):
        metrics = ran_engine.print_emulator_quality_metrics()
        out = capsys.readouterr().out
        assert "output1" in metrics
        assert "wave" in out.lower()

    def test_plot_nroy_parameters_deprecated(self, ran_engine):
        # Deprecated forwarder: warns, still works, and returns (fig, axes).
        import matplotlib.pyplot as plt
        with pytest.warns(DeprecationWarning):
            fig, axes = ran_engine.plot_nroy_parameters(
                derived={"sum": lambda df: df["param1"] + df["param2"]},
                true_parameters={"param1": 0.5},
            )
        assert fig is not None
        plt.close("all")

    # ── plotting / summary wrappers (delegate to historymatching.plotting) ──
    def test_plot_convergence(self, ran_engine):
        import matplotlib.pyplot as plt
        assert ran_engine.plot_convergence() is not None
        plt.close("all")

    def test_plot_marginals(self, ran_engine):
        import matplotlib.pyplot as plt
        assert ran_engine.plot_marginals(truth={"param1": 0.5}) is not None
        plt.close("all")

    def test_plot_nroy(self, ran_engine):
        import matplotlib.pyplot as plt
        axes = ran_engine.plot_nroy(
            truth={"param1": 0.5},
            derived={"sum": lambda df: df["param1"] + df["param2"]},
        )
        assert axes is not None
        plt.close("all")

    def test_plot_zscores(self, ran_engine):
        import matplotlib.pyplot as plt
        assert ran_engine.plot_zscores() is not None
        plt.close("all")

    def test_plot_constrained_dims(self, ran_engine):
        import matplotlib.pyplot as plt
        assert ran_engine.plot_constrained_dims(n_top=2) is not None
        plt.close("all")

    def test_nroy_summary(self, ran_engine, capsys):
        text = ran_engine.nroy_summary()
        out = capsys.readouterr().out
        assert "NROY" in text and "NROY" in out
        assert "param1" in text

    def test_iterationresult_plotting(self, ran_engine):
        import matplotlib.pyplot as plt
        result = ran_engine.results[-1]
        table = result.quality_table()
        assert "r2" in table.columns
        assert result.plot_emulator_quality() is not None
        assert result.plot_predicted_vs_actual("output1") is not None
        plt.close("all")

    def test_domain_object_summaries_and_plots(self, ran_engine):
        import matplotlib.pyplot as plt
        assert "ParameterSpace" in ran_engine.parameter_space.summary()
        assert "ObservationData" in ran_engine.observations.summary()
        assert "EmulatorBank" in ran_engine.emulator_bank.summary()
        assert ran_engine.parameter_space.plot_bounds(
            reference=ran_engine.parameter_space) is not None
        assert ran_engine.observations.plot_targets() is not None
        plt.close("all")

    def test_save_diagnostics(self, ran_engine, tmp_path):
        # engine.save_diagnostics delegates to each IterationResult.save
        ran_engine.save_diagnostics(str(tmp_path))
        wave_dirs = sorted(tmp_path.glob("wave*"))
        assert wave_dirs, "no wave directories were written"
        w = wave_dirs[0]
        assert (w / "samples.csv").exists()
        assert (w / "simulation_results.csv").exists()
        assert (w / "metrics.json").exists()

    def test_result_save(self, ran_engine, tmp_path):
        # IterationResult.save writes one wave's artifacts
        result = ran_engine.results[-1]
        wave_dir = result.save(str(tmp_path), all_results=ran_engine.results)
        from pathlib import Path
        assert Path(wave_dir).exists()
        assert (Path(wave_dir) / "samples.csv").exists()


class TestRepr:
    def test_repr(self, parameter_bounds, observations):
        engine = make(parameter_bounds, observations)
        text = repr(engine)
        assert "HistoryMatching(" in text
        assert "state=initialized" in text
