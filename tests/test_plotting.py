"""
Tests for the plotting and display API (historymatching.plotting and the
plot_*/summary methods on the engine, results, and domain objects).

Plots are exercised on the Agg backend; we assert that each call returns the
expected Matplotlib object(s) without error rather than inspecting pixels.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import historymatching as hm
from historymatching import plotting


@pytest.fixture(autouse=True)
def _close_figs():
    """Close all figures after each test to avoid leaking state."""
    yield
    plt.close("all")


@pytest.fixture
def samples():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "a": rng.normal(0.6, 0.05, 200),
        "b": rng.normal(0.3, 0.1, 200),
        "c": rng.uniform(0, 1, 200),
    })


@pytest.fixture
def bounds():
    return {"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (0.0, 1.0)}


# ── Module-level functions ───────────────────────────────────────────────────
class TestPlottingFunctions:
    def test_plot_convergence(self):
        ax = plotting.plot_convergence([1, 2, 3], [1.0, 0.3, 0.02])
        assert isinstance(ax, plt.Axes)

    def test_plot_marginals(self, samples, bounds):
        axes = plotting.plot_marginals(samples, truth={"a": 0.6}, bounds=bounds)
        assert len(axes) >= 3

    def test_plot_pairplot(self, samples, bounds):
        axes = plotting.plot_pairplot(samples, truth={"a": 0.6, "b": 0.3},
                                      bounds=bounds)
        assert axes.shape == (3, 3)

    def test_plot_pairplot_single_param(self, samples):
        axes = plotting.plot_pairplot(samples[["a"]])
        assert axes.shape == (1, 1)

    def test_plot_pairplot_caps_params(self):
        rng = np.random.default_rng(1)
        wide = pd.DataFrame({f"p{i}": rng.uniform(0, 1, 50) for i in range(12)})
        axes = plotting.plot_pairplot(wide, max_params=4)
        assert axes.shape == (4, 4)

    def test_plot_ensemble_fan(self):
        rng = np.random.default_rng(2)
        ax = plotting.plot_ensemble_fan(rng.normal(10, 2, (15, 30)),
                                        observed=np.linspace(5, 12, 30))
        assert isinstance(ax, plt.Axes)

    def test_plot_ensemble_fan_rejects_1d(self):
        with pytest.raises(ValueError):
            plotting.plot_ensemble_fan(np.arange(10))

    def test_plot_zscores_vs_targets(self):
        rng = np.random.default_rng(3)
        waves = [
            {"iteration": 1,
             "sim_results": pd.DataFrame({"x": rng.normal(150, 30, 80)}),
             "selected_features": ["x"]},
            {"iteration": 2,
             "sim_results": pd.DataFrame({"x": rng.normal(150, 10, 80)}),
             "selected_features": ["x"]},
        ]
        ax = plotting.plot_zscores_vs_targets(waves, {"x": (150, 20)})
        assert isinstance(ax, plt.Axes)

    def test_plot_constrained_dims(self, samples, bounds):
        axes = plotting.plot_constrained_dims(samples, bounds, n_top=2)
        assert len(axes) == 3  # spectrum + 2 components

    def test_plot_targets(self):
        ax = plotting.plot_targets({"x": (150, 20), "y": (500, 50)})
        assert isinstance(ax, plt.Axes)

    def test_plot_parameter_bounds(self, bounds):
        ax = plotting.plot_parameter_bounds({"a": (0.4, 0.8)}, reference=bounds)
        assert isinstance(ax, plt.Axes)

    def test_plot_emulator_quality(self):
        ax = plotting.plot_emulator_quality({"f1": {"r2_score": 0.9},
                                             "f2": {"r2_score": 0.5}})
        assert isinstance(ax, plt.Axes)

    def test_plot_predicted_vs_actual(self):
        rng = np.random.default_rng(4)
        ax = plotting.plot_predicted_vs_actual(rng.normal(0, 1, 40),
                                               rng.normal(0, 1, 40), r2=0.8)
        assert isinstance(ax, plt.Axes)

    def test_variance_reduction(self, samples, bounds):
        reduction, components, names = plotting.variance_reduction(samples, bounds)
        assert len(reduction) == 3
        assert components.shape == (3, 3)
        assert (0.0 <= reduction).all() and (reduction <= 1.0).all()

    def test_marginal_variance_reduction(self, samples, bounds):
        mvr = plotting.marginal_variance_reduction(samples, bounds)
        assert set(mvr) == {"a", "b", "c"}
        # 'a' was sampled much more tightly than 'c' relative to the prior
        assert mvr["a"] > mvr["c"]

    def test_plot_marginals_unknown_param_raises(self, samples):
        with pytest.raises(KeyError):
            plotting.plot_marginals(samples, params=["does_not_exist"])

    def test_no_numeric_columns_raises(self):
        df = pd.DataFrame({"label": ["a", "b", "c"]})
        with pytest.raises(ValueError):
            plotting.plot_pairplot(df)

    def test_plot_convergence_all_zero_fractions(self):
        # All-zero fractions must not raise or produce a degenerate axis.
        ax = plotting.plot_convergence([1, 2], [0.0, 0.0])
        lo, hi = ax.get_ylim()
        assert hi > lo

    def test_plot_emulator_quality_handles_none_r2(self):
        ax = plotting.plot_emulator_quality({"f1": {"r2_score": None},
                                             "f2": {}})  # missing key
        assert isinstance(ax, plt.Axes)

    def test_plot_predicted_vs_actual_empty_raises(self):
        with pytest.raises(ValueError):
            plotting.plot_predicted_vs_actual([], [])

    def test_plot_ensemble_fan_empty_raises(self):
        with pytest.raises(ValueError):
            plotting.plot_ensemble_fan(np.empty((0, 5)))

    def test_ax_is_reused(self, samples):
        fig, ax = plt.subplots()
        out = plotting.plot_convergence([1, 2], [1.0, 0.5], ax=ax)
        assert out is ax

    def test_top_level_reexports(self):
        for name in ["plot_convergence", "plot_pairplot", "plot_marginals",
                     "plot_ensemble_fan", "plot_zscores_vs_targets",
                     "plot_constrained_dims", "plot_targets",
                     "plot_parameter_bounds"]:
            assert hasattr(hm, name)


# ── Engine / result / domain-object methods on a real (small) run ────────────
@pytest.fixture(scope="module")
def fitted_engine(tmp_path_factory):
    rng = np.random.default_rng(0)

    def sim(samples_df):
        rows = []
        for _, r in samples_df.iterrows():
            rows.append({"f1": 2 * r["a"] + r["b"] + rng.normal(0, 0.05),
                         "f2": r["a"] * r["c"] + rng.normal(0, 0.05)})
        return pd.DataFrame(rows)

    b = hm.HistoryMatchingBuilder.from_data(
        {"a": (0, 1), "b": (0, 1), "c": (0, 1)},
        {"f1": (1.5, 0.2), "f2": (0.3, 0.1)})
    b.emulator_type = "gpr"
    b.n_samples = 60
    b.max_iterations = 2
    b.random_seed = 1
    b.feature_selection = ["f1", "f2"]
    b.output_dir = str(tmp_path_factory.mktemp("hm_plot_run"))
    engine = b.build()
    engine.set_simulation_function(sim)
    engine.run()
    return engine


class TestEngineDisplay:
    def test_summary(self, fitted_engine):
        s = fitted_engine.summary()
        assert "History Matching Summary" in s
        assert "NROY fraction per wave" in s

    def test_nroy_bounds(self, fitted_engine):
        bounds = fitted_engine.nroy_bounds()
        assert set(bounds) == {"a", "b", "c"}
        for lo, hi in bounds.values():
            assert lo <= hi

    def test_nroy_summary(self, fitted_engine):
        df = fitted_engine.nroy_summary()
        assert list(df.columns) == ["min", "max", "median", "q05", "q95", "reduction"]
        assert len(df) == 3

    def test_plot_convergence(self, fitted_engine):
        assert isinstance(fitted_engine.plot_convergence(), plt.Axes)

    def test_plot_nroy(self, fitted_engine):
        axes = fitted_engine.plot_nroy(truth={"a": 0.6, "b": 0.3})
        assert axes.shape[0] == axes.shape[1]

    def test_plot_marginals(self, fitted_engine):
        assert len(fitted_engine.plot_marginals()) >= 3

    def test_plot_zscores(self, fitted_engine):
        assert isinstance(fitted_engine.plot_zscores(), plt.Axes)

    def test_plot_constrained_dims(self, fitted_engine):
        assert len(fitted_engine.plot_constrained_dims(n_top=2)) == 3


class TestIterationResultDisplay:
    def test_summary_and_str(self, fitted_engine):
        r = fitted_engine.get_all_results()[-1]
        assert r.summary().startswith("Wave ")
        assert str(r) == r.summary()

    def test_quality_table(self, fitted_engine):
        r = fitted_engine.get_all_results()[-1]
        table = r.quality_table()
        assert list(table.columns) == ["r2", "mse", "n_train"]
        assert "f1" in table.index

    def test_plot_methods(self, fitted_engine):
        results = fitted_engine.get_all_results()
        r = results[-1]
        assert isinstance(r.plot_convergence(results), plt.Axes)
        assert isinstance(r.plot_emulator_quality(), plt.Axes)
        assert isinstance(r.plot_predicted_vs_actual("f1"), plt.Axes)


class TestEmulatorDisplay:
    def test_plot_predicted_vs_actual(self, fitted_engine):
        r = fitted_engine.get_all_results()[-1]
        em = r.get_emulator_for_feature("f1")
        assert isinstance(em.plot_predicted_vs_actual(), plt.Axes)

    def test_plot_diagnostics_returns_figs(self, fitted_engine):
        r = fitted_engine.get_all_results()[-1]
        em = r.get_emulator_for_feature("f1")
        figs = em.plot_diagnostics()
        assert isinstance(figs, list) and len(figs) >= 1
        assert all(isinstance(f, plt.Figure) for f in figs)

    def test_plot_methods_self_test(self):
        """plot_diagnostics / plot_predicted_vs_actual auto-run test() if needed."""
        from historymatching.emulators.linear import LinearModel
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.uniform(0, 1, 40), "b": rng.uniform(0, 1, 40)})
        y = pd.DataFrame({"out": 2 * X["a"] + X["b"]})
        em = LinearModel(X, y)
        em.train()
        assert not em.testing_complete
        figs = em.plot_diagnostics()          # should auto-test, not raise
        assert em.testing_complete
        assert isinstance(figs, list) and len(figs) >= 1

        em2 = LinearModel(X, y)
        em2.train()
        assert isinstance(em2.plot_predicted_vs_actual(), plt.Axes)
        assert em2.testing_complete


class TestDomainObjectDisplay:
    def test_parameter_space_summary(self, fitted_engine):
        assert "ParameterSpace" in fitted_engine.parameter_space.summary()
        assert fitted_engine.parameter_space._repr_html_()

    def test_parameter_space_plot_bounds(self, fitted_engine):
        ps = fitted_engine.parameter_space
        assert isinstance(ps.plot_bounds(reference=ps), plt.Axes)

    def test_observation_data_summary(self, fitted_engine):
        assert "ObservationData" in fitted_engine.observations.summary()
        assert fitted_engine.observations._repr_html_()

    def test_observation_data_plot_targets(self, fitted_engine):
        assert isinstance(fitted_engine.observations.plot_targets(), plt.Axes)

    def test_emulator_bank_summary(self, fitted_engine):
        assert "EmulatorBank" in fitted_engine.emulator_bank.summary()
