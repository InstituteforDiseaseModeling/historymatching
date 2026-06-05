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
        ax = plotting.plot_emulator_quality({"f1": {"r2": 0.9},
                                             "f2": {"r2": 0.5}})
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
        ax = plotting.plot_emulator_quality({"f1": {"r2": None},
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
