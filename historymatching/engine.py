"""
HistoryMatching — the main user-facing class for running a calibration.

Configure everything in one constructor call (parameter bounds, observations,
simulator function, and options), then call :meth:`HistoryMatching.run` for an
automated workflow or :meth:`HistoryMatching.step` / :meth:`commit_step` /
:meth:`revert_step` for interactive, wave-by-wave control.
"""

import logging
import pickle
import time as _time
import warnings
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import pandas as pd

from .emulator_bank import EmulatorBank
from .iteration_result import IterationResult
from .observation_data import ObservationData
from .parameter_space import ParameterSpace
from .emulators.factory import EmulatorFactory
from .feature_selection import FeatureSelectionStrategy, AutoFeatureSelection, ManualFeatureSelection
from .sampling import SamplingStrategy, SamplingStrategyFactory

logger = logging.getLogger(__name__)

# Show all parameters in pairplot when n_params <= this; otherwise show the top N most-constrained.
_PAIRPLOT_MAX_PARAMS = 15


def _compute_variance_reduction(
    nroy_samples: pd.DataFrame,
    parameter_space: "ParameterSpace",
) -> tuple:
    """
    PCA-based variance reduction analysis for NROY samples.

    Normalizes samples to the [0, 1]^d unit cube using prior bounds, fits PCA,
    and computes per-PC variance reduction relative to a uniform prior.

    Variance reduction per PC:
        reduction = 1 - (NROY_variance_along_PC / prior_variance)
        0.0 → direction as wide as the prior (unconstrained)
        1.0 → direction fully collapsed (fully constrained)

    Returns:
        reduction  : np.ndarray (n_params,), sorted most-constrained first
        components : np.ndarray (n_params, n_params), PCA loadings (rows), same order
        param_names: list[str], parameter names
    """
    import numpy as np
    from sklearn.decomposition import PCA

    param_names = parameter_space.get_parameter_names()
    X = np.empty((len(nroy_samples), len(param_names)))
    for j, name in enumerate(param_names):
        lo, hi = parameter_space.get_bounds(name)
        X[:, j] = (nroy_samples[name].to_numpy() - lo) / (hi - lo)

    prior_var = 1.0 / 12.0  # Uniform[0, 1] → variance = 1/12

    pca = PCA(n_components=len(param_names))
    pca.fit(X)

    reduction = np.clip(1.0 - pca.explained_variance_ / prior_var, 0.0, 1.0)
    order = np.argsort(reduction)[::-1]
    return reduction[order], pca.components_[order], param_names


def _marginal_variance_reduction(
    nroy_samples: pd.DataFrame,
    parameter_space: "ParameterSpace",
) -> dict:
    """
    Per-parameter marginal variance reduction vs uniform prior.

    Simpler than the PCA-based version: just compares the marginal variance of
    each parameter in the NROY cloud to the prior variance. Useful for ranking
    which parameters to show in a pairplot.

    Returns:
        dict mapping param_name → reduction in [0, 1]
    """
    import numpy as np

    prior_var = 1.0 / 12.0
    result = {}
    for name in parameter_space.get_parameter_names():
        lo, hi = parameter_space.get_bounds(name)
        x = (nroy_samples[name].to_numpy() - lo) / (hi - lo)
        nroy_var = float(np.var(x))
        result[name] = float(np.clip(1.0 - nroy_var / prior_var, 0.0, 1.0))
    return result


def _plot_constrained_dims(
    nroy_samples: pd.DataFrame,
    parameter_space: "ParameterSpace",
    wave_label: str,
    out_path: "Path",
    n_top: int = 5,
) -> None:
    """
    Save a constrained-directions diagnostic plot for a single HM wave.

    Top panel: variance reduction spectrum (most-constrained PCs first).
    Lower panels: loading bar charts for the top-N most-constrained PCs,
    showing |loading| as bar height and sign as colour (red = positive,
    blue = negative).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    reduction, components, param_names = _compute_variance_reduction(nroy_samples, parameter_space)
    n_params = len(param_names)
    n_top = min(n_top, n_params)

    fig, axes = plt.subplots(
        1 + n_top, 1,
        figsize=(max(10, n_params * 0.55), 4 + 2.5 * n_top),
        gridspec_kw={"height_ratios": [2.5] + [1.5] * n_top},
    )
    fig.suptitle(
        f"Constrained directions — {wave_label}\n"
        "Variance reduction = 1 − NROY_var / prior_var  "
        "(bar height = |loading|, red = positive, blue = negative)",
        fontsize=10,
    )

    # ── Spectrum ─────────────────────────────────────────────────────────────
    ax = axes[0]
    colors = ["firebrick" if r > 0.5 else "steelblue" for r in reduction]
    ax.bar(np.arange(n_params), reduction * 100, color=colors, edgecolor="none")
    ax.axhline(50, color="k", lw=0.8, ls="--", label="50% reduction")
    ax.set_ylabel("Variance reduction (%)")
    ax.set_xlabel("PC index (sorted most-constrained first)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.set_xticks(np.arange(n_params))
    ax.set_xticklabels([f"PC{i + 1}" for i in range(n_params)], fontsize=7, rotation=45)

    # ── Loadings for top-N constrained PCs ───────────────────────────────────
    for k in range(n_top):
        ax = axes[k + 1]
        loadings = components[k]
        bar_colors = ["firebrick" if v > 0 else "steelblue" for v in loadings]
        ax.bar(np.arange(n_params), np.abs(loadings), color=bar_colors, edgecolor="none")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("|Loading|")
        ax.set_title(f"PC{k + 1} — {reduction[k] * 100:.1f}% reduction", fontsize=9)
        ax.set_xticks(np.arange(n_params))
        ax.set_xticklabels(param_names, fontsize=6.5, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


class EngineState(Enum):
    """Possible states of the HistoryMatchingEngine."""

    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class IterationSnapshot:
    """Snapshot of engine state at a specific iteration."""

    iteration: int
    parameter_space: ParameterSpace
    emulator_bank: EmulatorBank
    result: Optional[IterationResult] = None
    next_samples: Optional[pd.DataFrame] = None  # Pre-computed samples for next iteration
    total_samples_generated: int = 0  # Total samples generated up to this iteration
    total_samples_accepted: int = 0   # Total samples accepted up to this iteration
    acceptance_rate: float = 1.0      # Cumulative acceptance rate up to this iteration
    metadata: dict[str, Any] = field(default_factory=dict)


class HistoryMatching:
    """
    Bayesian history matching — configure everything in one constructor call.

    Pass your parameter bounds, observations, and simulator ``function`` as plain
    arguments.  Friendly values (strings, dicts, lists) are accepted for the
    strategy options and turned into the underlying objects for you; sensible
    defaults are used for anything you omit.

    Examples:
        import historymatching as hm

        # The simulator takes a DataFrame of samples and returns one row of
        # outputs per sample.  Each output name must match an observation key.
        def run_sir(samples):
            rows = []
            for _, row in samples.iterrows():
                rows.append({'peak_incidence': simulate_peak(row['beta'], row['gamma'])})
            return rows   # a DataFrame is also accepted

        engine = hm.HistoryMatching(
            function=run_sir,
            bounds={'beta': (0.5, 3.0), 'gamma': (0.1, 1.0)},
            observations={'peak_incidence': (120.0, 50.0)},  # (mean, std)
            emulator_type='bayes_linear',
            n_samples=500,
            max_iterations=4,
        )

        # Automated execution
        results = engine.run()
        plausible = engine.get_nroy_samples()     # NROY = "Not Ruled Out Yet"

        # Interactive, step-by-step execution
        result = engine.step()             # run one wave
        engine.commit_step()               # accept it (or engine.revert_step())
        engine.feature_selection = ['different_output']   # reconfigure on the fly
        result = engine.step()

    The simulator ``function`` receives a pandas DataFrame of parameter samples
    (one row per sample) and may return either a DataFrame or a list of dicts
    (one dict per sample) mapping output names to values.
    """

    # Public progress counters; persisted/restored together by the checkpointer.
    _PROGRESS_ATTRS = (
        "current_iteration",
        "completed_iterations",
        "samples_generated",
        "samples_accepted",
        "emulators_trained",
        "acceptance_rate",
    )

    def __init__(
        self,
        function: Optional[Callable] = None,
        bounds: Union[dict, pd.DataFrame, ParameterSpace, None] = None,
        observations: Union[dict, pd.DataFrame, ObservationData, None] = None,
        *,
        sampling_strategy: Union[str, dict, SamplingStrategy] = "lhs",
        feature_selection: Union[str, list, dict, FeatureSelectionStrategy, None] = None,
        emulator_type: str = "bayes_linear",
        emulator_factory: Optional[EmulatorFactory] = None,
        emulator_bank: Optional[EmulatorBank] = None,
        n_samples: int = 1000,
        implausibility_threshold: float = 3.0,
        max_iterations: int = 10,
        random_seed: Optional[int] = None,
        auto_reduce_space: bool = False,
        oversample_factor: float = 1.1,
        max_batch_size: int = 10000,
        output_dir: Optional[str] = "./hm_output",
        run_name: Optional[str] = None,
        convergence_threshold: float = 0.0,
        nroy_method: str = "auto",
        nroy_options: Optional[dict] = None,
        max_candidate_factor: int = 1000,
    ):
        """
        Configure a history matching run.

        An "output" is a named scalar your simulator produces that you have an
        observed target for (e.g. ``'peak_infected'``).  Each wave trains an
        emulator for one or more outputs and rules out parameter regions whose
        emulated outputs are implausibly far from the observations.

        Args:
            function: The simulator.  A callable taking a DataFrame of parameter
                samples and returning a DataFrame (or list of dicts) of outputs,
                whose column/key names match the observation names.  May also be
                set later with ``engine.function = my_simulator``.
            bounds: The parameter space to search.  A dict mapping
                ``name -> (min, max)``, a DataFrame with ``parameter/minimum/maximum``
                columns, or a :class:`ParameterSpace`.
            observations: The target data.  A dict mapping ``output -> (mean, std)``
                (the second value is the standard deviation, not the variance),
                a DataFrame with ``feature/mean/std`` columns, or an
                :class:`ObservationData`.
            sampling_strategy: ``'lhs'`` (default) / ``'grid'`` / ``'random'``, a
                :class:`SamplingStrategy`, or a config dict (e.g. ``{'type': 'lhs',
                'criterion': 'center'}``).
            feature_selection: which outputs to emulate each wave.  A name or list
                of names (emulate exactly these), a config dict (e.g. ``{'method':
                'fano', 'max_features': 3}``), a :class:`FeatureSelectionStrategy`,
                or ``None`` for the automatic default (``method='mean_sq_z'`` —
                ranks outputs by mean squared z-score, i.e. how far each sits from
                its target in std units — one output per wave).
            emulator_type: ``'bayes_linear'`` (default) / ``'gpr'`` / ``'glm'`` / ``'linear'``.
            emulator_factory: a pre-built :class:`EmulatorFactory` (overrides
                ``emulator_type``; use it to pass emulator kwargs).
            emulator_bank: a pre-populated :class:`EmulatorBank` (for resuming).
            n_samples: parameter samples generated per wave.
            implausibility_threshold: implausibility cutoff, typically 2.5-4.0.
            max_iterations: maximum number of waves to run.
            random_seed: seed for reproducibility.
            auto_reduce_space: enable automatic parameter-space reduction.
            oversample_factor: oversampling factor for rejection filtering (>= 1.0).
            max_batch_size: max candidates per NROY sampling batch (>= 100).
            output_dir: where to auto-save waves, diagnostics, and checkpoints.
                Nothing is written until the first :meth:`run`/:meth:`step`; set
                ``output_dir=None`` to disable disk output entirely.
            run_name: subdirectory under ``output_dir`` (auto-generated if None).
            convergence_threshold: stop early once the plausible (NROY) fraction
                falls below this; ``0.0`` (default) disables early stopping.
            nroy_method: NROY sampler — ``'auto'`` (default) / ``'lhs'`` / ``'ray'``.
            nroy_options: dict of advanced options forwarded to the NROY sampler.
            max_candidate_factor: cap on candidates per wave as a multiple of
                ``n_samples`` (safety valve for near-empty NROY spaces).
        """
        # Core components — coerce friendly inputs into domain objects.
        self.parameter_space = self._coerce_parameter_space(bounds)
        self.observations = self._coerce_observations(observations)
        self._sampling_strategy = self._coerce_sampling(sampling_strategy)
        self._feature_selection_strategy = self._coerce_feature_selection(feature_selection)
        self._emulator_factory = self._coerce_emulator_factory(emulator_type, emulator_factory)
        self.emulator_bank = emulator_bank if emulator_bank is not None else EmulatorBank()

        # Workflow configuration
        self.n_samples = n_samples
        self.implausibility_threshold = implausibility_threshold
        self._max_iterations = max_iterations
        self.random_seed = random_seed
        self.auto_reduce_space = auto_reduce_space
        self.oversample_factor = oversample_factor
        self.max_batch_size = max_batch_size

        # NROY / convergence tuning (explicit, discoverable knobs)
        self.convergence_threshold = convergence_threshold
        self.nroy_method = nroy_method
        self.nroy_options = dict(nroy_options) if nroy_options else {}
        self.max_candidate_factor = max_candidate_factor

        # Engine state
        self.state = EngineState.INITIALIZED
        self._snapshots: list[IterationSnapshot] = []
        self._pending_result: Optional[IterationResult] = None
        self._pending_snapshot: Optional[IterationSnapshot] = None
        self._nroy_exhausted: bool = False

        # Progress counters (public, plain attributes — updated as waves run).
        self.current_iteration = 0
        self.completed_iterations: list[int] = []
        self.samples_generated = 0
        self.samples_accepted = 0
        self.emulators_trained = 0
        self.acceptance_rate = 1.0

        # Callbacks and hooks
        self._iteration_callbacks: list[Callable] = []
        self._progress_callbacks: list[Callable] = []

        # Simulator function (assign with ``engine.function = my_simulator``).
        self.function: Optional[Callable] = function

        # Output is created lazily on the first run()/step() so that merely
        # constructing a HistoryMatching writes nothing to disk.
        self._output_dir = output_dir
        self._run_name = run_name
        self.run_dir: Optional[Path] = None
        self._log_handler: Optional[logging.Handler] = None
        self._output_ready = False

        # Fail fast on invalid configuration rather than waiting until run().
        self.validate()

    def _ensure_output(self) -> None:
        """Create the run directory + file logging on first use (idempotent).

        Deferred from ``__init__`` so that constructing a HistoryMatching has no
        side effects; called at the start of :meth:`run`/:meth:`step`.
        """
        if self._output_ready or self._output_dir is None:
            self._output_ready = True
            return

        import datetime
        run_name = self._run_name or datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = Path(self._output_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Attach a single file handler to the top-level ``historymatching``
        # logger (sub-loggers propagate to it), replacing any handler a previous
        # engine attached so we don't leak handlers or duplicate log lines.
        pkg_logger = logging.getLogger('historymatching')
        for h in list(pkg_logger.handlers):
            if getattr(h, '_historymatching_run_handler', False):
                pkg_logger.removeHandler(h)
                h.close()
        fh = logging.FileHandler(self.run_dir / "log.txt")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        fh._historymatching_run_handler = True
        pkg_logger.addHandler(fh)
        pkg_logger.setLevel(logging.DEBUG)
        self._log_handler = fh
        self._output_ready = True

        # Configuration summary at the top of log.txt.
        param_names = self.parameter_space.get_parameter_names()
        obs_targets = self.observations.get_all_targets()
        logger.info(f"{'='*60}")
        logger.info("HISTORY MATCHING — CONFIGURATION")
        logger.info(f"{'='*60}")
        logger.info(f"  Emulator type:            {self.emulator_factory.get_default_type()}")
        logger.info(f"  Parameters:               {len(param_names)}")
        logger.info(f"  Observation targets:      {len(obs_targets)}")
        logger.info(f"  Samples per wave:         {self.n_samples}")
        logger.info(f"  Max iterations:           {self.max_iterations}")
        logger.info(f"  Implausibility threshold: {self.implausibility_threshold}")
        logger.info(f"  Auto space reduction:     {'enabled' if self.auto_reduce_space else 'disabled'}")
        logger.info(f"  Random seed:              {self.random_seed}")
        logger.info(f"  Oversample factor:        {self.oversample_factor}")
        logger.info(f"  Max batch size:           {self.max_batch_size}")
        logger.info(f"  Output directory:         {self.run_dir}")
        logger.info(f"  Run log:                  {self.run_dir / 'log.txt'}")
        logger.info(f"  Parameters: {param_names}")
        logger.info(f"  Targets: {list(obs_targets.keys())}")
        logger.info(f"{'='*60}")

    # ------------------------------------------------------------------ #
    # Coercion helpers — turn friendly constructor values into objects.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce_parameter_space(bounds) -> ParameterSpace:
        if bounds is None:
            raise ValueError(
                "bounds is required: pass a dict of {name: (min, max)}, "
                "a DataFrame with parameter/minimum/maximum columns, or a ParameterSpace."
            )
        if isinstance(bounds, ParameterSpace):
            return bounds
        return ParameterSpace(bounds)  # accepts dict or DataFrame

    @staticmethod
    def _coerce_observations(observations) -> ObservationData:
        if observations is None:
            raise ValueError(
                "observations is required: pass a dict of {feature: (mean, std)}, "
                "a DataFrame with feature/mean/std columns, or an ObservationData."
            )
        if isinstance(observations, ObservationData):
            return observations
        return ObservationData(observations)  # accepts dict or DataFrame

    @staticmethod
    def _coerce_sampling(sampling_strategy) -> SamplingStrategy:
        if sampling_strategy is None:
            return SamplingStrategyFactory.create("lhs")
        if isinstance(sampling_strategy, SamplingStrategy):
            return sampling_strategy
        if isinstance(sampling_strategy, str):
            return SamplingStrategyFactory.create(sampling_strategy)
        if isinstance(sampling_strategy, dict):
            opts = dict(sampling_strategy)  # copy so we don't mutate the caller's dict
            strategy_type = opts.pop("type", "lhs")
            return SamplingStrategyFactory.create(strategy_type, **opts)
        raise ValueError(
            f"Invalid sampling_strategy: {sampling_strategy!r}. Expected a name "
            f"('lhs'/'grid'/'random'), a config dict, or a SamplingStrategy."
        )

    @staticmethod
    def _coerce_feature_selection(feature_selection) -> FeatureSelectionStrategy:
        if feature_selection is None:
            return AutoFeatureSelection(method="mean_sq_z", max_features=1)
        if isinstance(feature_selection, FeatureSelectionStrategy):
            return feature_selection
        if isinstance(feature_selection, (str, list)):
            return ManualFeatureSelection(feature_selection)
        if isinstance(feature_selection, dict):
            return AutoFeatureSelection(
                method=feature_selection.get("method", "mean_sq_z"),
                threshold=feature_selection.get("threshold", None),
                max_features=feature_selection.get("max_features", 1),
                correlation_threshold=feature_selection.get("correlation_threshold", 0.8),
            )
        raise ValueError(
            f"Invalid feature_selection: {feature_selection!r}. Expected a feature "
            f"name or list of names, a config dict, or a FeatureSelectionStrategy."
        )

    @staticmethod
    def _coerce_emulator_factory(emulator_type, emulator_factory) -> EmulatorFactory:
        if emulator_factory is not None:
            return emulator_factory
        return EmulatorFactory(default_type=emulator_type or "bayes_linear")

    @property
    def results(self) -> list:
        """All committed :class:`IterationResult` objects, in order."""
        return [s.result for s in self._snapshots if s.result is not None]

    @property
    def parameters(self) -> list[str]:
        """Names of the parameters being calibrated."""
        return self.parameter_space.get_parameter_names()

    @property
    def outputs(self) -> list[str]:
        """Names of the observed outputs being matched."""
        return self.observations.get_feature_names()

    @staticmethod
    def plot_ensemble_fan(
        trajectories,
        observed=None,
        x=None,
        xlabel="Index",
        ylabel="Value",
        title="Ensemble vs observed",
        ax=None,
        show: bool = False,
        **kwargs,
    ):
        """Fan/spaghetti plot of an ensemble of trajectories vs observed data.

        Delegates to :func:`historymatching.plotting.plot_ensemble_fan`. Handy for
        eyeballing how well a set of plausible (NROY) parameter sets reproduces the
        data.

        Args:
            trajectories: 2-D array-like, shape ``(n_runs, n_timepoints)`` — one row
                per simulated trajectory.
            observed: Optional observed series of length ``n_timepoints``.
            x: Optional x-axis values (defaults to ``0..n_timepoints-1``).
            xlabel, ylabel, title: Axis labels / title.
            ax: Optional matplotlib Axes to draw into (a new figure is made if None).
            show: If True, call ``plt.show()`` before returning.
            **kwargs: Forwarded to the plotting function (e.g. ``ci``,
                ``show_members``, ``show_band``).

        Returns:
            The Matplotlib ``Axes``.
        """
        from . import plotting
        ax = plotting.plot_ensemble_fan(
            trajectories, observed=observed, x=x, ax=ax,
            xlabel=xlabel, ylabel=ylabel, title=title, **kwargs)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return ax

    # -- Reconfigurable options: assign friendly values, coerced like the constructor --
    @property
    def sampling_strategy(self) -> SamplingStrategy:
        """Sampling strategy. Assign a name/dict/strategy to change it (e.g. ``engine.sampling_strategy = 'grid'``)."""
        return self._sampling_strategy

    @sampling_strategy.setter
    def sampling_strategy(self, value) -> None:
        self._sampling_strategy = self._coerce_sampling(value)
        logger.info(f"Sampling strategy: {self._sampling_strategy.get_strategy_name()}")

    @property
    def feature_selection(self) -> FeatureSelectionStrategy:
        """Which outputs to emulate each wave. Assign a name/list/dict/strategy (e.g. ``engine.feature_selection = ['peak']``)."""
        return self._feature_selection_strategy

    @feature_selection.setter
    def feature_selection(self, value) -> None:
        self._feature_selection_strategy = self._coerce_feature_selection(value)
        logger.info(f"Feature selection: {self._feature_selection_strategy.get_strategy_name()}")

    @property
    def emulator_factory(self) -> EmulatorFactory:
        """The emulator factory. Assign an :class:`EmulatorFactory` for full control."""
        return self._emulator_factory

    @emulator_factory.setter
    def emulator_factory(self, value: EmulatorFactory) -> None:
        self._emulator_factory = value
        logger.info(f"Emulator factory: {value.get_default_type()}")

    @property
    def emulator_type(self) -> str:
        """Emulator type as a string. Assign ``'gpr'``/``'glm'``/``'linear'`` to change it."""
        return self._emulator_factory.get_default_type()

    @emulator_type.setter
    def emulator_type(self, value: str) -> None:
        self._emulator_factory = EmulatorFactory(default_type=value)
        logger.info(f"Emulator type: {value}")

    @property
    def max_iterations(self) -> int:
        """Maximum number of waves to run (you may raise it mid-run)."""
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        done = self.current_iteration
        if done and value <= done:
            raise ValueError(
                f"Cannot set max_iterations to {value}: {done} waves already completed. "
                f"New limit must be greater than {done}."
            )
        self._max_iterations = value
        if self.state == EngineState.COMPLETED and done < value:
            self.state = EngineState.PAUSED

    def __len__(self) -> int:
        """Number of committed waves."""
        return len(self._snapshots)

    def enumerate(self):
        """Iterate over committed waves as ``(iteration, result, samples)`` tuples.

        Example:
            for i, result, samples in engine.enumerate():
                print(i, result.nroy_fraction, len(samples))
        """
        for result in self.results:
            yield result.iteration, result, result.samples

    def add_iteration_callback(self, callback: Callable):
        """Add callback to be called after each iteration."""
        self._iteration_callbacks.append(callback)

    def add_progress_callback(self, callback: Callable):
        """Add callback to be called on progress updates."""
        self._progress_callbacks.append(callback)

    def validate(self):
        """
        Validate the engine's configuration.

        Checks that the required components are present and that all numeric and
        enumerated options are within their valid ranges.  Called automatically at
        the start of :meth:`run` and :meth:`step` (configuration attributes are
        public and may be changed after construction); may also be called directly.

        Raises:
            ValueError: If any required component is missing or an option is invalid.
        """
        if self.parameter_space is None:
            raise ValueError("Parameter space is required.")
        if self.observations is None:
            raise ValueError("Observations are required.")
        if self._sampling_strategy is None:
            raise ValueError("Sampling strategy is required.")
        if self._feature_selection_strategy is None:
            raise ValueError("Feature selection strategy is required.")
        if self._emulator_factory is None:
            raise ValueError("Emulator factory is required.")

        if self.n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")
        if self.implausibility_threshold <= 0:
            raise ValueError("Implausibility threshold must be positive")
        if self.oversample_factor < 1.0:
            raise ValueError("Oversample factor must be >= 1.0")
        if self.max_batch_size < 100:
            raise ValueError("Max batch size must be >= 100")

        if not (0.0 <= self.convergence_threshold <= 1.0):
            raise ValueError("Convergence threshold must be between 0.0 and 1.0")
        if self.nroy_method not in ('auto', 'lhs', 'ray'):
            raise ValueError(f"Unknown nroy_method '{self.nroy_method}'. Valid: ('auto', 'lhs', 'ray')")

    def step(self, features: Optional[list[str]] = None) -> IterationResult:
        """
        Execute a single history matching iteration.

        Args:
            features: Optional list of features to emulate (overrides strategy)

        Returns:
            IterationResult for this iteration

        Raises:
            RuntimeError: If engine is not in a valid state for stepping
            ValueError: If simulation function is not set or configuration is invalid
        """
        self.validate()
        self._ensure_output()

        if self.state not in [EngineState.INITIALIZED, EngineState.PAUSED]:
            if self.state == EngineState.RUNNING:
                raise RuntimeError(
                    f"Engine is currently running iteration {self.current_iteration + 1}. "
                    "Wait for it to complete before calling step() again."
                )
            elif self.state == EngineState.COMPLETED:
                raise RuntimeError(
                    f"Engine has completed all {self.max_iterations} iterations. "
                    "Use get_all_results() to access results or create a new engine instance to continue."
                )
            elif self.state == EngineState.ERROR:
                raise RuntimeError(
                    "Engine is in an error state from a previous operation. "
                    "Check the logs for details or create a new engine instance."
                )
            else:
                raise RuntimeError(
                    f"Engine is in state '{self.state.value}' and cannot execute step(). "
                    "If you have a pending iteration, use commit_step() to accept it or revert_step() to discard it."
                )

        if self.function is None:
            raise ValueError(
                "No simulator function has been configured. Before running waves, set "
                "engine.function = your_function (or pass function=... to the constructor). "
                "Your function takes a DataFrame of parameter samples and returns the outputs "
                "as a DataFrame or list of dicts."
            )

        if self.current_iteration >= self.max_iterations:
            raise RuntimeError(
                f"Maximum iterations limit reached ({self.max_iterations} waves completed). "
                f"To run more waves, raise the limit, e.g. "
                f"engine.max_iterations = {self.max_iterations + 5}, then step()/run() again."
            )

        wave_num = self.current_iteration + 1
        logger.info(f"{'='*60}")
        logger.info(f"WAVE {wave_num} STARTING")
        logger.info(f"{'='*60}")
        self.state = EngineState.RUNNING
        wave_t0 = _time.time()

        try:
            # ── Phase 1: Get samples ─────────────────────────────────────
            t0 = _time.time()
            if self.current_iteration == 0:
                samples = self._generate_plausible_samples()
                logger.info(f"[Wave {wave_num}] Phase 1/5 SAMPLING: generated {len(samples)} samples "
                            f"(acceptance rate: {self.acceptance_rate:.3f}) [{_time.time()-t0:.1f}s]")
            else:
                previous_snapshot = self._snapshots[self.current_iteration - 1]
                samples = previous_snapshot.next_samples
                if samples is None:
                    raise RuntimeError(
                        f"No pre-computed samples found from iteration {previous_snapshot.iteration}. "
                        f"This indicates an internal error - samples should have been computed during the previous step."
                    )
                logger.info(f"[Wave {wave_num}] Phase 1/5 SAMPLING: using {len(samples)} pre-computed samples "
                            f"from wave {previous_snapshot.iteration} [{_time.time()-t0:.1f}s]")

            # ── Phase 2: Run simulations ─────────────────────────────────
            t0 = _time.time()
            logger.info(f"[Wave {wave_num}] Phase 2/5 SIMULATION: running {len(samples)} simulations...")
            simulation_results = self._run_simulation(samples)
            logger.info(f"[Wave {wave_num}] Phase 2/5 SIMULATION: complete — {len(simulation_results.columns)} outputs "
                        f"[{_time.time()-t0:.1f}s]")

            # ── Phase 3: Select features ─────────────────────────────────
            t0 = _time.time()
            if features is None:
                selected_features = self._select_features(simulation_results)
            else:
                selected_features = features
            logger.info(f"[Wave {wave_num}] Phase 3/5 FEATURE SELECTION: {selected_features} [{_time.time()-t0:.1f}s]")

            # ── Phase 4: Train emulators ─────────────────────────────────
            t0 = _time.time()
            logger.info(f"[Wave {wave_num}] Phase 4/5 EMULATOR TRAINING: training {len(selected_features)} emulators...")
            emulators = self._create_emulators(samples, simulation_results, selected_features)
            logger.info(f"[Wave {wave_num}] Phase 4/5 EMULATOR TRAINING: complete [{_time.time()-t0:.1f}s]")

            # ── Phase 5: NROY sampling for next wave ─────────────────────
            t0 = _time.time()
            logger.info(f"[Wave {wave_num}] Phase 5/5 NROY SAMPLING: finding {self.n_samples} plausible candidates for next wave...")
            next_iteration_samples = self._compute_next_iteration_samples(emulators)
            logger.info(f"[Wave {wave_num}] Phase 5/5 NROY SAMPLING: found {len(next_iteration_samples)} candidates [{_time.time()-t0:.1f}s]")

            # Check: did we get enough samples for a meaningful next wave?
            min_samples = max(2 * len(self.parameter_space.get_parameter_names()), 20)
            if len(next_iteration_samples) < min_samples:
                feature_list = ', '.join(selected_features)
                logger.warning(
                    f"[Wave {wave_num}] NROY space collapsed: only {len(next_iteration_samples)} "
                    f"samples found (need {min_samples}+). This wave emulated [{feature_list}]. "
                    f"The model may be over-constrained — consider relaxing the implausibility "
                    f"threshold or increasing observation uncertainty (model discrepancy).")
                # Flag to stop after committing this wave
                self._nroy_exhausted = True

            # Determine parameter space for next iteration
            next_parameter_space = self._get_next_parameter_space(samples, emulators)

            # NROY fraction: what fraction of fresh LHS from the FULL prior pass
            # ALL emulators in the bank (including this wave's).  Cumulative —
            # must decrease monotonically as each wave adds constraints.
            nroy_fraction = getattr(self, '_last_nroy_fraction', 1.0)

            # Create iteration result
            iteration_result = IterationResult(
                iteration=self.current_iteration + 1,
                parameter_space=self.parameter_space,  # Current parameter space for this iteration
                samples=samples,
                simulation_results=simulation_results,
                emulated_outputs=selected_features,
                emulators=emulators,
                nroy_fraction=nroy_fraction,
                execution_time_seconds=_time.time() - wave_t0,
            )

            # Store pending changes (not committed yet)
            self._pending_result = iteration_result
            self._pending_snapshot = IterationSnapshot(
                iteration=self.current_iteration + 1,
                parameter_space=next_parameter_space,
                emulator_bank=self.emulator_bank.copy(),  # Copy current state
                result=iteration_result,
                next_samples=next_iteration_samples,  # Store pre-computed samples for next iteration
                total_samples_generated=self.samples_generated,
                total_samples_accepted=self.samples_accepted,
                acceptance_rate=self.acceptance_rate,
            )

            # Add emulators to pending snapshot's bank
            for feature, emulator in emulators.items():
                self._pending_snapshot.emulator_bank.add_emulator(iteration_result.iteration, feature, emulator)

            self.state = EngineState.PAUSED
            logger.info(f"[Wave {wave_num}] ALL PHASES COMPLETE [{_time.time()-wave_t0:.1f}s total]. Committing...")

            return iteration_result

        except Exception as e:
            self.state = EngineState.ERROR
            iteration_num = self.current_iteration + 1

            # Provide specific guidance based on the error type
            if "simulation" in str(e).lower() or "simulation_function" in str(e):
                error_msg = (
                    f"Simulation function failed during iteration {iteration_num}: {e}\n\n"
                    "Common causes and solutions:\n"
                    "  - Check that your simulation function can handle the generated parameter values\n"
                    "  - Ensure your simulation returns a DataFrame with the expected columns\n"
                    "  - Verify that parameter bounds are realistic for your model\n"
                    "  - Add error handling to your simulation function for edge cases"
                )
            elif "emulator" in str(e).lower():
                error_msg = (
                    f"Emulator creation/training failed during iteration {iteration_num}: {e}\n\n"
                    "Common causes and solutions:\n"
                    "  - Insufficient or invalid training data from simulation\n"
                    "  - Features with constant values or extreme outliers\n"
                    "  - Emulator type may be unsuitable for your data\n"
                    "  - Try a different emulator type or adjust emulator parameters"
                )
            elif "feature" in str(e).lower() or "selection" in str(e).lower():
                error_msg = (
                    f"Feature selection failed during iteration {iteration_num}: {e}\n\n"
                    "Common causes and solutions:\n"
                    "  - Mismatch between simulation outputs and observation features\n"
                    "  - Features with insufficient variation in simulation results\n"
                    "  - Check that your observation data contains the expected features\n"
                    "  - Verify simulation is producing all required outputs"
                )
            else:
                error_msg = f"Unexpected error during iteration {iteration_num}: {e}"

            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def commit_step(self) -> None:
        """
        Commit the pending iteration results.

        This makes the changes from the last step() permanent and advances
        the iteration counter.

        Raises:
            RuntimeError: If no pending iteration to commit
        """
        if self._pending_result is None or self._pending_snapshot is None:
            if self.state == EngineState.INITIALIZED:
                raise RuntimeError(
                    "No iteration has been executed yet. Call step() first to run an iteration, "
                    "then use commit_step() to accept the results."
                )
            elif self.state == EngineState.COMPLETED:
                raise RuntimeError(
                    "All iterations have been completed and committed. "
                    "Use get_all_results() to access the final results."
                )
            else:
                raise RuntimeError(
                    f"No pending iteration to commit (engine state: {self.state.value}). "
                    "Call step() first to execute an iteration that can be committed."
                )

        # Apply changes
        self.emulator_bank = self._pending_snapshot.emulator_bank

        # Update parameter space only if auto-reduction is enabled
        if self.auto_reduce_space:
            self.parameter_space = self._pending_snapshot.parameter_space

        # Update progress
        self.current_iteration += 1
        self.completed_iterations.append(self.current_iteration)
        self.samples_accepted += len(self._pending_result.samples)
        self.emulators_trained += len(self._pending_result.emulators)

        # Store snapshot
        self._snapshots.append(self._pending_snapshot)

        # Clear pending state
        committed_result = self._pending_result
        self._pending_result = None
        self._pending_snapshot = None

        # Update state
        if self.current_iteration >= self.max_iterations:
            self.state = EngineState.COMPLETED
        else:
            self.state = EngineState.PAUSED

        # Save wave output (emulators, diagnostics, checkpoint)
        self._save_wave_output(committed_result)

        # Call callbacks
        self._call_iteration_callbacks(committed_result)
        self._call_progress_callbacks()

        logger.info(f"[Wave {committed_result.iteration}] COMMITTED — diagnostics and checkpoint saved to {self.run_dir}")
        logger.info(f"{'='*60}")

    def revert_step(self) -> None:
        """
        Revert the pending iteration results.

        This discards the changes from the last step() and returns to
        the previous state.

        Raises:
            RuntimeError: If no pending iteration to revert
        """
        if self._pending_result is None:
            if self.state == EngineState.INITIALIZED:
                raise RuntimeError(
                    "No iteration has been executed yet. Call step() first to run an iteration "
                    "before attempting to revert it."
                )
            elif self.state == EngineState.COMPLETED:
                raise RuntimeError(
                    "All iterations have been completed. There are no pending results to revert. "
                    "Previous iterations were already committed."
                )
            else:
                raise RuntimeError(
                    f"No pending iteration to revert (engine state: {self.state.value}). "
                    "Call step() first to execute an iteration that can be reverted."
                )

        # Restore progress information from last committed snapshot
        if self._snapshots:
            last_snapshot = self._snapshots[-1]
            self.samples_generated = last_snapshot.total_samples_generated
            self.samples_accepted = last_snapshot.total_samples_accepted
            self.acceptance_rate = last_snapshot.acceptance_rate
        else:
            # No committed snapshots, reset to initial values
            self.samples_generated = 0
            self.samples_accepted = 0
            self.acceptance_rate = 1.0

        # Clear pending state
        reverted_iteration = self._pending_result.iteration
        self._pending_result = None
        self._pending_snapshot = None

        # Return to paused state
        self.state = EngineState.PAUSED

        logger.info(f"Iteration {reverted_iteration} reverted")

    def drop_emulator_from_pending(self, feature: str) -> None:
        """
        Remove a specific emulator from the pending iteration before committing.

        Call this after step() but before commit_step() to exclude an emulator
        whose diagnostics indicate a poor fit. The emulator will not be stored
        in the bank and will not contribute to implausibility filtering in
        future waves.

        The simulation data for this wave is unaffected — only the emulator is
        dropped. The remaining emulators are committed as normal.

        Args:
            feature: Name of the feature whose emulator should be dropped.

        Raises:
            RuntimeError: If called when no step is pending.
            KeyError: If the feature was not emulated in the pending step.

        Example:
            result = engine.step()

            metrics = result.get_emulator_quality_metrics()
            for output in result.emulated_outputs:
                print(f"{output}: R²={metrics[output]['r2']:.3f}")

            # Drop any emulator with a poor fit before committing
            engine.drop_emulator_from_pending('output_c')
            engine.commit_step()
        """
        if self._pending_snapshot is None:
            raise RuntimeError(
                "No pending iteration to modify. Call step() first, inspect the emulator "
                "diagnostics, then call drop_emulator_from_pending() before commit_step()."
            )

        iteration = self._pending_result.iteration
        if not self._pending_snapshot.emulator_bank.has_emulator(iteration, feature):
            available = list(self._pending_result.emulators.keys())
            raise KeyError(
                f"Feature '{feature}' was not emulated in the pending iteration. "
                f"Available features: {available}"
            )

        self._pending_snapshot.emulator_bank.remove_emulator(iteration, feature)
        logger.info(f"Emulator for '{feature}' removed from pending iteration {iteration}.")

    def get_status_summary(self) -> str:
        """
        Get a human-readable, multi-line summary of the current status.

        (To reconfigure mid-run, just assign to the matching attribute, e.g.
        ``engine.feature_selection = ['peak']`` or ``engine.max_iterations = 20``.)
        """
        summary = [
            "=== History Matching Status ===",
            f"State: {self.state.value}",
            f"Progress: {self.current_iteration}/{self.max_iterations} waves",
        ]

        if self.current_iteration > 0:
            summary.extend([
                f"Acceptance rate: {self.acceptance_rate:.1%}",
                f"Total samples generated: {self.samples_generated:,}",
                f"Total samples accepted: {self.samples_accepted:,}",
                f"Emulators trained: {self.emulators_trained}",
            ])

        if self._pending_result is not None:
            summary.append(f"[ACTION NEEDED] Pending wave {self._pending_result.iteration} - use commit_step() or revert_step()")

        if self.function is None:
            summary.append("[SET function] No simulator set - assign engine.function = your_function")
        else:
            summary.append("[OK] Simulator function configured")

        if self.state == EngineState.ERROR:
            summary.append("[ERROR] Engine is in an error state - check the logs for details")
        elif self.state == EngineState.COMPLETED:
            summary.append("[OK] All waves completed successfully")

        return "\n".join(summary)

    def run(self, auto_commit: bool = True, resume: bool = False) -> list[IterationResult]:
        """
        Run automated history matching workflow.

        Args:
            auto_commit: Whether to automatically commit each iteration
            resume: If True, load checkpoint from output_dir and continue.
                    If False (default), start fresh.  Raises if a checkpoint
                    exists and resume is False (to prevent accidental overwrites).

        Returns:
            List of IterationResult objects for all iterations

        Raises:
            ValueError: If simulation function is not set or configuration is invalid
        """
        self.validate()
        self._ensure_output()

        if self.function is None:
            raise ValueError(
                "Cannot start automated run: no simulator function has been configured. "
                "Set engine.function = your_function (or pass function=... to the constructor). "
                "Your function takes parameter samples (a DataFrame) and returns the outputs "
                "(a DataFrame or list of dicts).\n\n"
                "Example:\n"
                "  def my_simulation(params_df):\n"
                "      # Your simulation code here\n"
                "      return results_df\n"
                "  engine.function = my_simulation"
            )

        # Resume from checkpoint if requested
        if self.run_dir is not None:
            ckpt = self.run_dir / "checkpoint.pkl"
            if resume and ckpt.exists():
                logger.info(f"Resuming from checkpoint: {ckpt}")
                self._load_checkpoint_state(ckpt)
                logger.info(f"Resumed at wave {self.current_iteration}")
            elif not resume and ckpt.exists():
                logger.warning(
                    f"Checkpoint exists at {ckpt} but resume=False. "
                    f"Starting fresh (existing output will be overwritten)."
                )

        logger.info(f"Starting automated run with {self.max_iterations} max iterations")

        results = self.get_all_results()  # includes any resumed waves

        try:
            while self.current_iteration < self.max_iterations and self.state not in [EngineState.COMPLETED, EngineState.ERROR]:
                # Run iteration
                result = self.step()
                results.append(result)

                if auto_commit:
                    self.commit_step()
                    if getattr(self, '_nroy_exhausted', False):
                        logger.info("Stopping: NROY space exhausted after this wave.")
                        self.state = EngineState.COMPLETED
                        break
                else:
                    break  # Let user decide

                # Check convergence criteria
                if self._check_convergence():
                    logger.info("Convergence criteria met. Stopping early.")
                    break

            # Only set to COMPLETED if we actually finished all iterations
            if self.state != EngineState.ERROR and self.current_iteration >= self.max_iterations:
                self.state = EngineState.COMPLETED

            logger.info(f"Automated run completed. {len(results)} iterations executed.")

        except Exception as e:
            self.state = EngineState.ERROR
            failed_iteration = len(results) + 1

            error_msg = (
                f"Automated run failed at iteration {failed_iteration} of {self.max_iterations}: {e}\n\n"
                f"Progress before failure:\n"
                f"  - Completed iterations: {len(results)}\n"
                f"  - Total samples generated: {self.samples_generated:,}\n"
                f"  - Current acceptance rate: {self.acceptance_rate:.1%}\n\n"
                "You can:\n"
                "  - Fix the issue and restart with a new engine\n"
                "  - Use step-by-step execution (step/commit/revert) for more control\n"
                "  - Access partial results from completed iterations with get_all_results()"
            )

            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        return results

    def get_iteration_result(self, iteration: int) -> Optional[IterationResult]:
        """Get result for a specific iteration."""
        if iteration <= 0 or iteration > len(self._snapshots):
            return None
        return self._snapshots[iteration - 1].result

    def get_all_results(self) -> list[IterationResult]:
        """Get all committed iteration results."""
        return [snapshot.result for snapshot in self._snapshots if snapshot.result is not None]

    def print_emulator_quality_metrics(self, iteration: Optional[int] = None) -> dict:
        """Print and return per-feature emulator quality metrics for a wave.

        Args:
            iteration: Wave number to report (1-based).  Defaults to the most
                recently committed wave.

        Returns:
            Dict mapping feature name -> metrics dict (R², MSE, training size, ...).
        """
        if iteration is not None:
            result = self.get_iteration_result(iteration)
        else:
            committed = self.get_all_results()
            result = committed[-1] if committed else None

        if result is None:
            print("No completed iterations yet — run a wave first.")
            return {}

        metrics = result.get_emulator_quality_metrics()
        print(f"Emulator quality metrics (wave {result.iteration}):")
        for output, m in metrics.items():
            r2 = m.get("r2")
            mse = m.get("mse")
            n_train = m.get("n_train")
            r2_str = f"{r2:.3f}" if isinstance(r2, (int, float)) else "n/a"
            mse_str = f"{mse:.3g}" if isinstance(mse, (int, float)) else "n/a"
            n_str = f"{n_train}" if n_train is not None else "?"
            print(f"  {output:<24} R²={r2_str:<8} MSE={mse_str:<10} n={n_str}")
        return metrics

    def save_diagnostics(self, directory: str, verbose: bool = False) -> None:
        """Save every committed wave's artifacts under ``directory``.

        Writes one ``wave{N}/`` subdirectory per wave (samples, simulator
        outputs, pickled emulators, predicted-vs-actual + ARD diagnostic plots,
        a convergence figure, and ``metrics.json``) by calling
        :meth:`IterationResult.save` for each wave.  This is the manual
        equivalent of the per-wave output written automatically when
        ``output_dir`` is set.

        Args:
            directory: Directory to write into (created if needed).
            verbose: If True, print the path written for each wave.
        """
        all_results = self.results
        if not all_results:
            print("No completed waves yet — nothing to save.")
            return
        for result in all_results:
            result.save(directory, all_results=all_results)
            if verbose:
                print(f"  saved wave {result.iteration} to {directory}")

    def plot_nroy_parameters(
        self,
        samples: Optional[pd.DataFrame] = None,
        derived: Optional[dict] = None,
        true_parameters: Optional[dict] = None,
        bins: int = 25,
        fig_kwargs: Optional[dict] = None,
        show: bool = False,
    ):
        """Deprecated alias for :meth:`plot_nroy`.

        .. deprecated:: 2.0.1
            Use :meth:`plot_nroy` instead (``true_parameters=`` is now ``truth=``).
            Retained as a thin forwarder that returns ``(fig, axes)`` for
            backwards compatibility; ``fig_kwargs`` is no longer applied.
        """
        import warnings
        import numpy as np

        warnings.warn(
            "HistoryMatching.plot_nroy_parameters() is deprecated; use plot_nroy() "
            "instead (pass known values via `truth=` rather than `true_parameters=`).",
            DeprecationWarning, stacklevel=2,
        )
        axes = self.plot_nroy(samples=samples, truth=true_parameters,
                              derived=derived, bins=bins)
        fig = np.asarray(axes).flat[0].figure
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return fig, axes

    # ── Convenience plot/summary wrappers (delegate to historymatching.plotting) ──
    def _bounds_dict(self) -> dict:
        """``{parameter: (min, max)}`` from the current parameter space."""
        ps = self.parameter_space
        return {name: ps.get_bounds(name) for name in ps.get_parameter_names()}

    def _targets_dict(self) -> dict:
        """``{feature: (mean, std)}`` from the observations."""
        obs = self.observations
        return {f: obs.get_target_for_feature(f) for f in obs.get_feature_names()}

    def _nroy_for_plot(self, samples=None) -> pd.DataFrame:
        """NROY samples restricted to parameter columns, with a clear error when
        none are available (no waves run yet, or the NROY region collapsed)."""
        if samples is None:
            samples = self.get_nroy_samples()
        if samples is None or len(samples) == 0:
            raise ValueError(
                "No NROY samples to plot. Run at least one wave first; if you "
                "have, the NROY region may have collapsed — loosen the "
                "implausibility threshold or widen the observation uncertainty."
            )
        cols = [c for c in self.parameters if c in samples.columns]
        return samples[cols]

    def plot_convergence(self, *, ax=None, **kwargs):
        """Plot the NROY fraction per wave (delegates to
        :func:`historymatching.plotting.plot_convergence`)."""
        from . import plotting
        results = self.get_all_results()
        if not results:
            raise ValueError("No completed waves to plot. Run at least one wave first.")
        return plotting.plot_convergence(
            [r.iteration for r in results],
            [r.nroy_fraction for r in results],
            ax=ax, **kwargs)

    def plot_marginals(self, *, truth=None, axes=None, **kwargs):
        """Marginal histograms of the NROY samples (delegates to
        :func:`historymatching.plotting.plot_marginals`)."""
        from . import plotting
        return plotting.plot_marginals(
            self._nroy_for_plot(), truth=truth,
            bounds=self._bounds_dict(), axes=axes, **kwargs)

    def plot_nroy(self, *, samples=None, truth=None, derived=None, bins=25, axes=None, **kwargs):
        """Corner/pairplot of the NROY parameter cloud (delegates to
        :func:`historymatching.plotting.plot_pairplot`).

        Pass ``derived`` to overlay computed quantities, e.g.
        ``{'R0': lambda df: df['beta'] / df['gamma']}``.
        """
        from . import plotting
        return plotting.plot_pairplot(
            self._nroy_for_plot(samples), truth=truth, derived=derived, bins=bins,
            bounds=self._bounds_dict(), axes=axes, **kwargs)

    def plot_zscores(self, *, ax=None, **kwargs):
        """Standardised outputs vs targets across waves (delegates to
        :func:`historymatching.plotting.plot_zscores_vs_targets`)."""
        from . import plotting
        waves = [
            {"iteration": r.iteration,
             "sim_results": r.simulation_results,
             "selected_features": r.emulated_outputs}
            for r in self.get_all_results()
        ]
        if not waves:
            raise ValueError("No completed waves to plot. Run at least one wave first.")
        return plotting.plot_zscores_vs_targets(waves, self._targets_dict(), ax=ax, **kwargs)

    def plot_constrained_dims(self, *, n_top=5, axes=None, **kwargs):
        """Constrained-direction (variance-reduction) plot of the NROY cloud
        (delegates to :func:`historymatching.plotting.plot_constrained_dims`)."""
        from . import plotting
        return plotting.plot_constrained_dims(
            self._nroy_for_plot(), self._bounds_dict(),
            n_top=n_top, axes=axes, **kwargs)

    def nroy_summary(self) -> str:
        """Print and return a text summary of the current NROY region: number of
        waves and samples, latest NROY fraction, and each parameter's median with
        its 95% interval."""
        samples = self.get_nroy_samples()
        n = 0 if samples is None else len(samples)
        results = self.get_all_results()
        frac = results[-1].nroy_fraction if results else None
        lines = ["NROY summary", "-" * 48,
                 f"  waves run:     {len(results)}",
                 f"  NROY samples:  {n}"]
        if frac is not None:
            lines.append(f"  NROY fraction: {frac:.3%}")
        if n:
            lines.append(f"  {'parameter':<18}{'median':>12}   95% interval")
            for name in self.parameters:
                col = samples[name]
                lo, hi = col.quantile(0.025), col.quantile(0.975)
                lines.append(f"  {name:<18}{col.median():>12.4g}   [{lo:.4g}, {hi:.4g}]")
        text = "\n".join(lines)
        print(text)
        return text

    def get_nroy_samples(self, n: Optional[int] = None,
                         method: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Get plausible parameter samples — the calibration result.

        Returns the NROY ("Not Ruled Out Yet") samples: parameter sets that pass
        ALL committed emulators' implausibility checks.  By default returns the
        pre-computed set from the last wave (``n_samples`` of them).  Pass ``n``
        to draw a fresh, larger set filtered through the current emulator bank.
        No new simulations are run — only fast emulator predictions are used.

        Args:
            n: Number of NROY samples to return.  If None, returns the
               pre-computed set from the last committed wave.
            method: NROY sampling method: ``'auto'`` (LHS first, escalates
               to ray+importance if needed), ``'lhs'`` (pure rejection only),
               or None (uses engine default). For unbiased final samples
               (e.g. trajectory selection), use ``method='lhs'``.
            **kwargs: Extra options passed to ``generate_nroy_design()``:
               ``n_lines``, ``points_per_line`` (ray_resample);
               ``max_candidates`` (lhs); ``imp_scale``, ``maximin_reps``, etc.

        Returns:
            DataFrame of NROY samples, or empty DataFrame if no iterations committed.

        Example:
            results = engine.run()
            nroy = engine.get_nroy_samples()                    # cached from last wave
            nroy = engine.get_nroy_samples(10000)               # larger draw (default method)
            nroy = engine.get_nroy_samples(5000, method='lhs')  # unbiased for posterior
            nroy = engine.get_nroy_samples(5000, method='auto',
                                           n_lines=40, points_per_line=100)
        """
        if not self._snapshots:
            return pd.DataFrame()

        cached = self._snapshots[-1].next_samples
        if n is None and method is None:
            return cached
        if n is not None and method is None and n <= len(cached):
            return cached.head(n)

        from .nroy_sampling import generate_nroy_design

        n = n or self.n_samples
        method = method or self.nroy_method
        nroy_opts = {**self.nroy_options, **kwargs}

        return generate_nroy_design(
            n_points=n,
            parameter_space=self.parameter_space,
            emulator_bank=self.emulator_bank,
            observations=self.observations,
            threshold=self.implausibility_threshold,
            sampling_strategy=self.sampling_strategy,
            method=method,
            seed=self.random_seed,
            **nroy_opts,
        ).samples

    def get_pending_next_samples(self) -> Optional[pd.DataFrame]:
        """
        Get the proposed samples for the next iteration, if available.

        This allows inspection of pre-computed samples after step() but before commit_step().
        The samples are computed during step() execution and will be used for the next
        iteration if the current step is committed.

        Returns:
            DataFrame of proposed samples for next iteration, or None if no step is pending

        Example:
            result = engine.step()
            next_samples = engine.get_pending_next_samples()
            if next_samples is not None:
                print(f"Proposed {len(next_samples)} samples for next iteration")
                # Inspect the samples before deciding to commit
                engine.commit_step()  # or engine.revert_step()
        """
        if self._pending_snapshot is None:
            return None
        return self._pending_snapshot.next_samples

    def save_checkpoint(self, filepath: Path) -> None:
        """Save engine state to checkpoint file."""
        checkpoint_data = {
            "parameter_space": self.parameter_space,
            "observations": self.observations,
            "emulator_bank": self.emulator_bank,
            "progress": {name: getattr(self, name) for name in self._PROGRESS_ATTRS},
            "snapshots": self._snapshots,
            "max_iterations": self.max_iterations,
            "implausibility_threshold": self.implausibility_threshold,
            "n_samples": self.n_samples,
            "auto_reduce_space": self.auto_reduce_space,
            "oversample_factor": self.oversample_factor,
            "convergence_threshold": self.convergence_threshold,
            "nroy_method": self.nroy_method,
            "nroy_options": self.nroy_options,
            "max_candidate_factor": self.max_candidate_factor,
        }

        with open(filepath, "wb") as f:
            pickle.dump(checkpoint_data, f)

        logger.info(f"Checkpoint saved to {filepath}")

    @classmethod
    def load_checkpoint(cls, filepath: Path, sampling_strategy: SamplingStrategy, feature_selection: FeatureSelectionStrategy, emulator_factory: EmulatorFactory) -> "HistoryMatching":
        """Load engine state from checkpoint file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # Create engine with loaded data
        engine = cls(
            bounds=data["parameter_space"],
            observations=data["observations"],
            sampling_strategy=sampling_strategy,
            feature_selection=feature_selection,
            emulator_factory=emulator_factory,
            emulator_bank=data["emulator_bank"],
            output_dir=None,
            n_samples=data["n_samples"],
            implausibility_threshold=data["implausibility_threshold"],
            max_iterations=data["max_iterations"],
            auto_reduce_space=data.get("auto_reduce_space", False),
            oversample_factor=data.get("oversample_factor", 1.1),
            convergence_threshold=data.get("convergence_threshold", 0.0),
            nroy_method=data.get("nroy_method", "auto"),
            nroy_options=data.get("nroy_options"),
            max_candidate_factor=data.get("max_candidate_factor", 1000),
        )

        # Restore state
        engine._restore_progress(data["progress"])
        engine._snapshots = data["snapshots"]
        engine.state = EngineState.PAUSED

        logger.info(f"Engine loaded from checkpoint {filepath}")
        return engine

    # Internal methods

    def _load_checkpoint_state(self, filepath: Path) -> None:
        """Restore engine state from a checkpoint file (for resume)."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.emulator_bank = data["emulator_bank"]
        self._restore_progress(data["progress"])
        self._snapshots = data["snapshots"]
        self.state = EngineState.PAUSED
        logger.info(f"Loaded checkpoint: {len(self._snapshots)} waves completed")

    def _restore_progress(self, progress: dict) -> None:
        """Restore the public progress counters from a checkpoint dict."""
        for name in self._PROGRESS_ATTRS:
            if name in progress:
                setattr(self, name, progress[name])

    def _save_wave_output(self, result: IterationResult) -> None:
        """Save emulators, diagnostics, and checkpoint after committing a wave.

        Directory layout::

            {run_dir}/
              wave{N}/
                {feature}/
                  emulator.pkl
                  diagnostics.png
                  metrics.json
                convergence.png
                nroy_samples.csv
              checkpoint.pkl       # latest engine state (overwritten each wave)
              run_config.json      # written once on first wave
        """
        if self.run_dir is None:
            return

        import json as _json

        wave_dir = self.run_dir / f"wave{result.iteration}"
        wave_dir.mkdir(parents=True, exist_ok=True)

        # ── Per-feature: emulator + diagnostics ──────────────────────────
        for feature, emulator in result.emulators.items():
            feat_dir = wave_dir / feature
            feat_dir.mkdir(exist_ok=True)

            # Save emulator pickle
            try:
                import pickle
                with open(feat_dir / "emulator.pkl", "wb") as f:
                    pickle.dump(emulator, f)
            except Exception as e:
                logger.warning(f"Failed to save emulator for '{feature}': {e}")

            # Save diagnostics figure
            try:
                if not getattr(emulator, 'testing_complete', False):
                    emulator.test()
                import matplotlib
                emulator.plot_diagnostics()
                import matplotlib.pyplot as plt
                for i, fig_num in enumerate(plt.get_fignums()[-4:]):  # plot_diagnostics creates up to 4 figs
                    plt.figure(fig_num)
                    plt.savefig(feat_dir / f"diagnostics_{i}.png", dpi=100, bbox_inches='tight')
                    plt.close(fig_num)
            except Exception as e:
                logger.warning(f"Failed to save diagnostics for '{feature}': {e}")

            # Save metrics + hyperparameters
            try:
                metrics = result.get_emulator_quality_metrics().get(feature, {})
                try:
                    metrics['hyperparameters'] = emulator.get_hyperparameters()
                except Exception:
                    pass
                with open(feat_dir / "metrics.json", "w") as f:
                    _json.dump(metrics, f, indent=2, default=float)
            except Exception as e:
                logger.warning(f"Failed to save metrics for '{feature}': {e}")

        # ── Wave-level: convergence + NROY samples ───────────────────────
        try:
            all_results = self.get_all_results()
            if len(all_results) > 0:
                import matplotlib
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(7, 4))
                waves = [r.iteration for r in all_results]
                fracs = [r.nroy_fraction for r in all_results]
                ax.bar(waves, fracs, color='#3575b5', alpha=0.8, edgecolor='white')
                for w, frac in zip(waves, fracs):
                    label = f'{frac:.2%}' if frac < 0.01 else f'{frac:.1%}'
                    ax.annotate(label, (w, frac), textcoords='offset points',
                                xytext=(0, 5), ha='center', fontsize=8)
                ax.set_xlabel('Wave')
                ax.set_ylabel('Fraction of space remaining (NROY)')
                ax.set_title('Convergence')
                ax.set_yscale('log')
                ax.set_ylim(min(fracs) * 0.5, 1)
                ax.set_xticks(waves)
                fig.tight_layout()
                fig.savefig(wave_dir / "convergence.png", dpi=100, bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to save convergence plot: {e}")

        # ── Wave-level: z-scores vs ALL targets ─────────────────────────
        try:
            all_results = self.get_all_results()
            if len(all_results) > 0:
                import numpy as np
                import matplotlib
                import matplotlib.pyplot as plt

                targets = self.observations.get_all_targets()
                target_names = [k for k in targets if k in result.simulation_results.columns]

                if len(target_names) > 0 and len(all_results) > 0:
                    # Collect which features were emulated in which wave
                    emulated = {}
                    for r in all_results:
                        for feat in r.emulated_outputs:
                            emulated.setdefault(feat, []).append(r.iteration)

                    n_targets = len(target_names)
                    n_waves = len(all_results)
                    cmap = plt.get_cmap('plasma')
                    bar_width = 0.8 / n_waves

                    fig, ax = plt.subplots(figsize=(max(14, n_targets * 0.7), 7))
                    ymin_data, ymax_data = 0, 0

                    for wi, r in enumerate(all_results):
                        sims = r.simulation_results
                        for ti, key in enumerate(target_names):
                            if key not in sims.columns:
                                continue
                            obs_mean, obs_std = targets[key]
                            z = (sims[key].dropna() - obs_mean) / obs_std
                            x_pos = ti + (wi - n_waves / 2 + 0.5) * bar_width
                            color = cmap(wi / n_waves)
                            q05, q25, med, q75, q95 = np.percentile(z, [5, 25, 50, 75, 95])
                            ymin_data = min(ymin_data, q05)
                            ymax_data = max(ymax_data, q95)
                            ax.plot([x_pos, x_pos], [q05, q95],
                                    color=color, linewidth=1.2, alpha=0.5, solid_capstyle='round')
                            ax.plot([x_pos, x_pos], [q25, q75],
                                    color=color, linewidth=3.5, alpha=0.7, solid_capstyle='round')
                            ax.plot(x_pos, med, 'o', color=color, markersize=4, zorder=5)

                    for wi, r in enumerate(all_results):
                        ax.plot([], [], color=cmap(wi / n_waves), linewidth=3.5, label=f'Wave {r.iteration}')

                    ax.axhline(0, color='#d44d4d', lw=1.5, ls='--', alpha=0.7, label='Target')
                    ax.axhline(3.5, color='green', lw=0.8, ls=':', alpha=0.4)
                    ax.axhline(-3.5, color='green', lw=0.8, ls=':', alpha=0.4)
                    ax.axhspan(-3.5, 3.5, color='green', alpha=0.03)

                    margin = max(abs(ymin_data), abs(ymax_data)) * 1.15
                    if margin > 0:
                        ax.set_ylim(-margin, margin)

                    for ti, key in enumerate(target_names):
                        if key in emulated:
                            wlist = ','.join(str(w) for w in emulated[key])
                            ax.annotate(f'\u2605w{wlist}', (ti, -margin * 0.93), ha='center',
                                        fontsize=7, color='#2a7f3f', fontweight='bold')

                    ax.set_xticks(range(n_targets))
                    ax.set_xticklabels([k.replace('_', '\n') for k in target_names],
                                       fontsize=8, rotation=45, ha='right')
                    ax.set_ylabel('(Sim \u2212 Target) / Target \u03c3', fontsize=12)
                    ax.set_title('NROY z-scores across waves \u2014 thick=IQR, thin=5th\u201395th pctl, dot=median\n'
                                 'Green \u2605 = target was emulated in that wave', fontsize=13)
                    ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.18),
                              ncol=min(n_waves + 1, 8), framealpha=0.9)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(axis='y', alpha=0.2)
                    fig.tight_layout()
                    fig.savefig(wave_dir / "zscores_vs_targets.png", dpi=150, bbox_inches='tight')
                    plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to save z-scores plot: {e}")

        # ── Wave-level: constrained-directions diagnostic ────────────────
        try:
            snapshot = self._snapshots[-1]
            if snapshot.next_samples is not None and len(snapshot.next_samples) >= 10:
                _plot_constrained_dims(
                    nroy_samples=snapshot.next_samples,
                    parameter_space=self.parameter_space,
                    wave_label=f"wave{result.iteration}",
                    out_path=wave_dir / "constrained_dims.png",
                )
        except Exception as e:
            logger.warning(f"Failed to save constrained-dims plot: {e}")

        # ── Wave-level: parameter space pair plot ────────────────────────
        try:
            all_results = self.get_all_results()
            if len(all_results) >= 2:
                import matplotlib.pyplot as plt

                param_names = self.parameter_space.get_parameter_names()
                n_all = len(param_names)

                # Select parameters to show.  If the problem is small enough, show
                # everything.  Otherwise rank by marginal variance reduction — the
                # params whose NROY range has shrunk most relative to the prior.
                # This is better than ARD lengthscales, which reflect emulator
                # sensitivity to the current wave's target features rather than
                # overall constraint across all waves.
                snapshot = self._snapshots[-1]
                if n_all <= _PAIRPLOT_MAX_PARAMS:
                    sorted_params = param_names
                    subtitle_note = f"all {n_all} parameters"
                elif snapshot.next_samples is not None and len(snapshot.next_samples) >= 10:
                    reduction_map = _marginal_variance_reduction(
                        snapshot.next_samples, self.parameter_space
                    )
                    sorted_params = sorted(
                        reduction_map, key=reduction_map.get, reverse=True
                    )[:_PAIRPLOT_MAX_PARAMS]
                    subtitle_note = f"top {len(sorted_params)} most-constrained parameters"
                else:
                    sorted_params = param_names[:_PAIRPLOT_MAX_PARAMS]
                    subtitle_note = f"first {len(sorted_params)} parameters"

                n_pars = len(sorted_params)
                n_show = min(len(all_results), 3)
                show_indices = np.linspace(0, len(all_results) - 1, n_show, dtype=int)
                show_results = [all_results[i] for i in show_indices]

                cmap = plt.get_cmap('plasma')
                fig, axes = plt.subplots(n_pars, n_pars, figsize=(2.2 * n_pars, 2.2 * n_pars))
                if n_pars == 1:
                    axes = np.array([[axes]])

                for i, p1 in enumerate(sorted_params):
                    for j, p2 in enumerate(sorted_params):
                        ax = axes[i][j]
                        if i == j:
                            for si, r in enumerate(show_results):
                                if p1 in r.samples.columns:
                                    ax.hist(r.samples[p1], bins=25, density=True, alpha=0.5,
                                            color=cmap(si / n_show), edgecolor='none',
                                            label=f'W{r.iteration}')
                            if i == 0:
                                ax.legend(fontsize=6)
                        elif i > j:
                            for si, r in enumerate(show_results):
                                if p2 in r.samples.columns and p1 in r.samples.columns:
                                    alpha = 0.15 + 0.35 * (si / max(n_show - 1, 1))
                                    ax.scatter(r.samples[p2], r.samples[p1], s=2, alpha=alpha,
                                               color=cmap(si / n_show), edgecolors='none')
                        else:
                            ax.set_visible(False)

                        if j == 0 and i > 0:
                            ax.set_ylabel(p1.replace('_', '\n'), fontsize=6)
                        else:
                            ax.set_ylabel('')
                        if i == n_pars - 1:
                            ax.set_xlabel(p2.replace('_', '\n'), fontsize=6)
                        else:
                            ax.set_xlabel('')
                        ax.tick_params(labelsize=4)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)

                wave_labels = ' \u2192 '.join(str(r.iteration) for r in show_results)
                fig.suptitle(f'Parameter space: Waves {wave_labels}\n({subtitle_note})',
                             fontsize=13, fontweight='bold', y=1.02)
                fig.tight_layout()
                fig.savefig(wave_dir / "pairplot.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to save pair plot: {e}")

        # Save NROY samples for this wave (next iteration's candidates)
        try:
            snapshot = self._snapshots[-1]
            if snapshot.next_samples is not None:
                snapshot.next_samples.to_csv(wave_dir / "nroy_samples.csv", index=False)
        except Exception as e:
            logger.warning(f"Failed to save NROY samples: {e}")

        # ── Run-level: checkpoint + config ───────────────────────────────
        try:
            self.save_checkpoint(self.run_dir / "checkpoint.pkl")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

        # Run config (written once)
        config_path = self.run_dir / "run_config.json"
        if not config_path.exists():
            try:
                config = {
                    'parameters': self.parameter_space.get_parameter_names(),
                    'parameter_bounds': {
                        p: list(self.parameter_space.get_bounds(p))
                        for p in self.parameter_space.get_parameter_names()
                    },
                    'observations': self.observations.get_all_targets(),
                    'n_samples': self.n_samples,
                    'max_iterations': self.max_iterations,
                    'implausibility_threshold': self.implausibility_threshold,
                }
                with open(config_path, "w") as f:
                    _json.dump(config, f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Failed to save run config: {e}")

        logger.info(f"Wave {result.iteration} output saved to {wave_dir}")

    def _generate_plausible_samples(self) -> pd.DataFrame:
        """
        Generate plausible parameter samples by filtering through existing emulators.

        For first iteration, returns unfiltered samples.
        For subsequent iterations, proposes candidates and filters based on implausibility.
        """
        if not self.emulator_bank.has_emulators():
            # First iteration - no filtering needed
            samples = self.sampling_strategy.generate_samples(self.parameter_space, self.n_samples, seed=self.random_seed)
            self.samples_generated += len(samples)
            self.acceptance_rate = 1.0
            return samples

        # Subsequent iterations - adaptive rejection sampling loop
        plausible_samples = pd.DataFrame()
        total_candidates_generated = 0
        batch_num = 0
        batch_seed = self.random_seed  # Incremented each batch for fresh LHS draws
        max_candidate_factor = self.max_candidate_factor
        max_candidates = self.n_samples * max_candidate_factor
        last_pct_logged = -10
        t0 = _time.time()

        logger.info(f"  Plausible sampling: 0/{self.n_samples} (0%) — starting")

        # Initial batch size
        batch_size = min(int(self.n_samples * self.oversample_factor), self.max_batch_size)

        while len(plausible_samples) < self.n_samples:
            # Generate candidate samples (fresh seed each batch)
            candidates = self.sampling_strategy.generate_samples(self.parameter_space, batch_size, seed=batch_seed)
            if batch_seed is not None:
                batch_seed += 1

            # Filter candidates through existing emulators
            batch_plausible = self._filter_samples_by_implausibility(candidates)

            # Combine with existing plausible samples
            if len(batch_plausible) > 0:
                plausible_samples = pd.concat([plausible_samples, batch_plausible], ignore_index=True)

            # Update counters
            total_candidates_generated += len(candidates)
            batch_num += 1

            # Calculate acceptance rate for adaptive sizing
            current_acceptance_rate = len(plausible_samples) / total_candidates_generated

            # Log at every 10% milestone
            pct = int(100 * len(plausible_samples) / self.n_samples)
            if pct >= last_pct_logged + 10:
                last_pct_logged = pct - (pct % 10)
                elapsed = _time.time() - t0
                logger.info(
                    f"  Plausible sampling: {len(plausible_samples)}/{self.n_samples} ({pct}%) "
                    f"| {total_candidates_generated:,} tested | rate={current_acceptance_rate:.4%} [{elapsed:.0f}s]")

            logger.debug(
                f"  Sampling batch {batch_num}: {len(batch_plausible)}/{len(candidates)} accepted "
                f"| {len(plausible_samples)}/{self.n_samples} collected "
                f"| {total_candidates_generated:,} total candidates "
                f"| rate={current_acceptance_rate:.4%}"
            )

            # If we have enough samples, break
            if len(plausible_samples) >= self.n_samples:
                break

            # Safety valve: stop after generating too many candidates
            if total_candidates_generated >= max_candidates:
                logger.warning(
                    f"Reached candidate limit ({max_candidates:,}) with only "
                    f"{len(plausible_samples)}/{self.n_samples} plausible samples "
                    f"(acceptance rate: {current_acceptance_rate:.4%}).  "
                    f"Proceeding with {len(plausible_samples)} samples.  "
                    f"Adjust via the max_candidate_factor option (HistoryMatching(..., max_candidate_factor=N))."
                )
                break

            # Calculate next batch size adaptively
            remaining_needed = self.n_samples - len(plausible_samples)
            if current_acceptance_rate > 0:
                batch_size = min(int(self.oversample_factor * remaining_needed / current_acceptance_rate), self.max_batch_size)
            else:
                # If no samples accepted yet, increase batch size
                batch_size = min(batch_size * 2, self.max_batch_size)

        # Update global progress tracking
        self.samples_generated += total_candidates_generated
        self.acceptance_rate = len(plausible_samples) / total_candidates_generated if total_candidates_generated > 0 else 1.0

        # Return exactly the requested number of samples
        return plausible_samples.head(self.n_samples)

    def _build_fast_predictors(self, emulator_bank=None):
        """Build FastGPRPredictor objects for all GPR emulators in the bank.

        Returns a list of (FastGPRPredictor, obs_mean, obs_std, feature_name)
        tuples suitable for filter_nroy().  Non-GPR emulators are skipped
        (they fall back to the slow path).
        """
        from .emulators.fast_predict import FastGPRPredictor

        bank = emulator_bank or self.emulator_bank
        predictors = []

        for iteration in bank.get_all_iterations():
            emulators = bank.get_emulators_for_iteration(iteration)
            for feature_name, emulator in emulators.items():
                if not self.observations.has_feature(feature_name):
                    continue
                # Only works for GPR emulators (SE kernel with normalization)
                from .emulators.gpr import GPR
                if not isinstance(emulator, GPR):
                    continue
                try:
                    fast = FastGPRPredictor.from_emulator(emulator)
                    obs_mean, obs_std = self.observations.get_target_for_feature(feature_name)
                    predictors.append((fast, obs_mean, obs_std, feature_name))
                except Exception as e:
                    logger.warning(f"Could not build fast predictor for '{feature_name}': {e}")

        return predictors

    def _filter_samples_by_implausibility(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Filter candidate samples based on implausibility from existing emulators.

        Uses numba-accelerated short-circuit filtering when GPR emulators are
        available: points that fail one emulator are immediately dropped and
        never tested against the rest.  Falls back to the GPflow path for
        non-GPR emulators.
        """
        if not self.emulator_bank.has_emulators():
            return candidates

        return self._filter_fast(candidates, self.emulator_bank)

    def _filter_fast(self, candidates: pd.DataFrame, emulator_bank) -> pd.DataFrame:
        """Fast NROY filter using numba predictors with short-circuit evaluation.

        Falls back to the standard GPflow path for non-GPR emulators.
        """
        import numpy as np
        from .emulators.fast_predict import filter_nroy

        fast_predictors = self._build_fast_predictors(emulator_bank)

        if not fast_predictors:
            # No GPR emulators — fall back to standard (slow) path
            return self._filter_samples_slow(candidates, emulator_bank)

        # Fast path: numba short-circuit filter
        param_cols = self.parameter_space.get_parameter_names()
        X = candidates[param_cols].values.astype(np.float64)

        mask = filter_nroy(X, fast_predictors, threshold=self.implausibility_threshold)

        plausible = candidates[mask]
        logger.debug(
            f"Fast filter: {len(candidates)} → {len(plausible)} "
            f"({len(plausible)/len(candidates):.2%}) through {len(fast_predictors)} emulators"
        )
        return plausible

    def _filter_samples_slow(self, candidates: pd.DataFrame, emulator_bank) -> pd.DataFrame:
        """Implausibility filter with short-circuit evaluation for non-GPR emulators."""
        import numpy as np

        param_cols = self.parameter_space.get_parameter_names()
        mask = np.ones(len(candidates), dtype=bool)
        n_emulators = 0

        for iteration in reversed(emulator_bank.get_all_iterations()):
            emulators = emulator_bank.get_emulators_for_iteration(iteration)
            for feature_name, emulator in emulators.items():
                if mask.sum() == 0:
                    break
                if not self.observations.has_feature(feature_name):
                    continue
                try:
                    active = candidates.loc[mask, param_cols]
                    predictions = emulator.predict(active)
                    feature_impl = self.observations.calculate_implausibility(
                        feature_name, predictions.get_mean(), predictions.get_variance()
                    )
                    failures = feature_impl > self.implausibility_threshold
                    n_rejected = int(failures.sum())
                    mask[mask] &= ~failures.values
                    n_emulators += 1

                    logger.debug(
                        f"  {feature_name}: {len(active)} tested, "
                        f"{n_rejected} rejected, {mask.sum()} surviving"
                    )
                except Exception as e:
                    logger.warning(f"Implausibility calc failed for '{feature_name}': {e}")
                    continue

        plausible = candidates[mask]
        logger.debug(
            f"Slow filter: {len(candidates)} \u2192 {len(plausible)} "
            f"({len(plausible)/len(candidates):.2%}) through {n_emulators} emulators"
        )
        return plausible

    def _run_simulation(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Run the simulator and normalize its output to a DataFrame.

        The simulator receives a DataFrame of parameter samples and may return
        either a DataFrame or a list of dicts (one per sample); both are accepted.
        """
        output = self.function(samples)
        return self._coerce_simulation_output(output, len(samples))

    @staticmethod
    def _coerce_simulation_output(output, n_samples: int) -> pd.DataFrame:
        """Normalize a simulator's return value into a DataFrame of outputs."""
        if isinstance(output, pd.DataFrame):
            results = output
        elif isinstance(output, dict):
            # Dict of column-name -> array/list of per-sample values.
            results = pd.DataFrame(output)
        elif isinstance(output, list):
            # List of per-sample dicts (records), or a list of scalars.
            results = pd.DataFrame(output)
        else:
            try:
                results = pd.DataFrame(output)
            except Exception as e:
                raise TypeError(
                    "Simulation function must return a pandas DataFrame or a list of "
                    f"dicts (one per sample); got {type(output).__name__}."
                ) from e

        if len(results) != n_samples:
            raise ValueError(
                f"Simulation function returned {len(results)} rows for {n_samples} "
                f"input samples; it must return exactly one row (or dict) per sample."
            )
        return results.reset_index(drop=True)

    def _select_features(self, simulation_results: pd.DataFrame) -> list[str]:
        """Select features to emulate using configured strategy."""
        return self._feature_selection_strategy.select_features(simulation_results, self.observations, self.current_iteration + 1)

    def _create_emulators(self, samples: pd.DataFrame, simulation_results: pd.DataFrame, features: list[str]) -> dict[str, Any]:
        """Create and train emulators for selected features.

        Only parameter-space columns are passed to emulators — any extra
        columns (e.g. ``rand_seed``) added by the simulation function are
        excluded so they don't become spurious input dimensions.
        """
        param_cols = self.parameter_space.get_parameter_names()
        samples_clean = samples[param_cols]
        return self.emulator_factory.create_emulators_for_features(samples_clean, simulation_results, features)

    def _get_next_parameter_space(self, samples: pd.DataFrame, emulators: dict[str, Any]) -> ParameterSpace:
        """
        Determine parameter space for next iteration.

        By default, returns the original parameter space.
        If auto_reduce_space is enabled, constrains to plausible samples.
        """
        if not self.auto_reduce_space:
            return self.parameter_space  # Keep original space

        # Calculate implausibilities and constrain space
        implausibilities = []

        for feature_name, emulator in emulators.items():
            # Get predictions (only parameter columns, not metadata like rand_seed)
            param_cols = self.parameter_space.get_parameter_names()
            predictions = emulator.predict(samples[param_cols])

            # Calculate implausibility for this feature (vectorized)
            feature_implausibility = self.observations.calculate_implausibility(feature_name, predictions.get_mean(), predictions.get_variance())
            implausibilities.append(feature_implausibility)

        if not implausibilities:
            return self.parameter_space

        # Combine implausibilities (use maximum)
        combined_implausibility = pd.concat(implausibilities, axis=1).max(axis=1)

        # Find plausible samples
        plausible_mask = combined_implausibility <= self.implausibility_threshold
        plausible_samples = samples[plausible_mask]

        if len(plausible_samples) == 0:
            warnings.warn(
                f"No plausible samples found with implausibility threshold {self.implausibility_threshold}. "
                f"Parameter space will not be reduced this iteration. Consider:\n"
                f"  - Increasing the implausibility threshold (current: {self.implausibility_threshold})\n"
                f"  - Generating more samples per iteration (current: {self.n_samples})\n"
                f"  - Checking if your simulation is producing reasonable outputs\n"
                f"  - Reviewing your observation data for inconsistencies", stacklevel=2
            )
            return self.parameter_space

        # Create new parameter space constrained to plausible samples
        return self.parameter_space.constrain_to_samples(plausible_samples)

    def _compute_next_iteration_samples(self, current_emulators: dict[str, Any]) -> pd.DataFrame:
        """
        Compute plausible parameter samples for the next iteration.

        Args:
            current_emulators: Emulators created in the current iteration

        Returns:
            DataFrame of plausible samples for next iteration
        """
        # Create a temporary emulator bank with current emulators for filtering
        temp_bank = self.emulator_bank.copy()
        current_iteration = self.current_iteration + 1
        for feature, emulator in current_emulators.items():
            temp_bank.add_emulator(current_iteration, feature, emulator)

        from .nroy_sampling import generate_nroy_design

        nroy_method = self.nroy_method
        nroy_opts = self.nroy_options

        nroy_result = generate_nroy_design(
            n_points=self.n_samples,
            parameter_space=self.parameter_space,
            emulator_bank=temp_bank,
            observations=self.observations,
            threshold=self.implausibility_threshold,
            sampling_strategy=self.sampling_strategy,
            method=nroy_method,
            seed=self.random_seed,
            max_candidates=self.n_samples * self.max_candidate_factor,
            **nroy_opts,
        )

        self._update_nroy_stats(nroy_result.lhs_accepted, nroy_result.lhs_tested)
        return nroy_result.samples

    def _update_nroy_stats(self, n_plausible: int, n_generated: int) -> None:
        """Update NROY fraction and cumulative progress after rejection sampling."""
        self._last_nroy_fraction = (
            n_plausible / n_generated if n_generated > 0 else 1.0
        )
        self.samples_generated += n_generated
        self.samples_accepted += n_plausible
        self.acceptance_rate = (
            self.samples_accepted / self.samples_generated
            if self.samples_generated > 0 else 1.0
        )
        logger.debug(f"NROY fraction: {self._last_nroy_fraction:.6f}")

    def _create_snapshot(self) -> IterationSnapshot:
        """Create snapshot of current state."""
        return IterationSnapshot(iteration=self.current_iteration, parameter_space=self.parameter_space, emulator_bank=self.emulator_bank.copy())

    def _check_convergence(self) -> bool:
        """Check if convergence criteria are met.

        Returns True only when the acceptance rate (fraction of LHS candidates
        passing the emulator filter) drops below the configurable threshold.

        The threshold is set via the ``convergence_threshold`` option and
        defaults to 0.0, which disables early stopping; set it to a small
        positive value (e.g. 0.01) to stop once the NROY acceptance rate falls
        below that fraction.
        """
        threshold = self.convergence_threshold
        if threshold <= 0:
            return False  # Early stopping disabled

        rate = getattr(self, '_last_nroy_fraction', 1.0)
        if rate < threshold:
            logger.warning(
                f"NROY fraction ({rate:.4%}) fell below convergence threshold "
                f"({threshold:.2%}).  The NROY region may be very small — consider "
                f"relaxing the implausibility threshold or checking emulator quality.  "
                f"Stopping after {self.current_iteration} iterations."
            )
            return True
        return False

    def _call_iteration_callbacks(self, result: IterationResult):
        """Call registered iteration callbacks."""
        for callback in self._iteration_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.warning(f"Iteration callback failed: {e}")

    def _call_progress_callbacks(self):
        """Call registered progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def __repr__(self) -> str:
        """Config-revealing representation (leads with what is being calibrated)."""
        params = self.parameters
        outs = self.outputs
        sim = "set" if self.function is not None else "NOT SET"
        return (
            f"HistoryMatching(\n"
            f"  parameters={len(params)} {params},\n"
            f"  outputs={len(outs)} {outs},\n"
            f"  emulator={self.emulator_type}, simulator={sim},\n"
            f"  n_samples={self.n_samples}, implausibility_threshold={self.implausibility_threshold},\n"
            f"  state={self.state.value}, wave {self.current_iteration}/{self.max_iterations}, "
            f"acceptance_rate={self.acceptance_rate:.3f}\n"
            f")"
        )


# Backwards-compatible alias: the engine class was previously named
# ``HistoryMatchingEngine`` and configured via ``HistoryMatchingBuilder``.
HistoryMatchingEngine = HistoryMatching
