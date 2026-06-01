"""
HistoryMatchingEngine for interactive workflow execution.

Provides step-by-step execution with the ability to inspect results,
make adjustments, and revert changes if needed.
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
from .feature_selection import FeatureSelectionStrategy
from .sampling import SamplingStrategy
from . import plotting

logger = logging.getLogger(__name__)

# Show all parameters in pairplot when n_params <= this; otherwise show the top N most-constrained.
_PAIRPLOT_MAX_PARAMS = 15


def _bounds_from_space(parameter_space):
    """Helper: {name: (lo, hi)} bounds dict from a ParameterSpace."""
    return {p: parameter_space.get_bounds(p)
            for p in parameter_space.get_parameter_names()}


def _compute_variance_reduction(nroy_samples, parameter_space):
    """PCA variance reduction of an NROY cloud (delegates to plotting)."""
    return plotting.variance_reduction(nroy_samples, _bounds_from_space(parameter_space))


def _marginal_variance_reduction(nroy_samples, parameter_space):
    """Per-parameter marginal variance reduction (delegates to plotting)."""
    return plotting.marginal_variance_reduction(nroy_samples, _bounds_from_space(parameter_space))


def _plot_constrained_dims(nroy_samples, parameter_space, wave_label, out_path, n_top=5):
    """Save the constrained-directions diagnostic for one wave to disk."""
    import matplotlib.pyplot as plt
    axes = plotting.plot_constrained_dims(
        nroy_samples, _bounds_from_space(parameter_space),
        n_top=n_top, title=f"Constrained directions — {wave_label}")
    fig = axes[0].figure
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


@dataclass
class WorkflowProgress:
    """Progress tracking for history matching workflow."""

    current_iteration: int = 0
    completed_iterations: list[int] = field(default_factory=list)
    total_samples_generated: int = 0
    total_samples_accepted: int = 0
    total_emulators_trained: int = 0
    acceptance_rate: float = 1.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class HistoryMatchingEngine:
    """
    Interactive history matching engine with step-by-step execution.

    Supports both automated execution and interactive workflows where users
    can inspect results, make adjustments, and control progression.

    Examples:
        # Automated execution
        engine = builder.build()
        results = engine.run()

        # Interactive execution
        engine = builder.build()
        result = engine.step()  # Run one iteration
        if result.is_acceptable():
            engine.commit_step()  # Accept the iteration
        else:
            engine.revert_step()  # Reject and try different settings

        # Continue or modify strategy
        engine.update_feature_selection(['different_feature'])
        result = engine.step()
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        observations: ObservationData,
        sampling_strategy: SamplingStrategy,
        feature_selection_strategy: FeatureSelectionStrategy,
        emulator_factory: EmulatorFactory,
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
        **kwargs,
    ):
        """
        Initialize the history matching engine.

        Args:
            parameter_space: Parameter space definition
            observations: Observation data
            sampling_strategy: Strategy for parameter sampling
            feature_selection_strategy: Strategy for feature selection
            emulator_factory: Factory for creating emulators
            emulator_bank: Optional existing emulator bank
            n_samples: Number of samples per iteration
            implausibility_threshold: Threshold for parameter space reduction
            max_iterations: Maximum iterations to run
            random_seed: Random seed for reproducibility
            auto_reduce_space: Whether to automatically reduce parameter space
            oversample_factor: Factor for oversampling to account for filtering
            **kwargs: Additional settings
        """
        # Core components
        self.parameter_space = parameter_space  # Always keep original space
        self.observations = observations
        self.sampling_strategy = sampling_strategy
        self.feature_selection_strategy = feature_selection_strategy
        self.emulator_factory = emulator_factory
        self.emulator_bank = emulator_bank if emulator_bank is not None else EmulatorBank()

        # Workflow configuration
        self.n_samples = n_samples
        self.implausibility_threshold = implausibility_threshold
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.auto_reduce_space = auto_reduce_space
        self.oversample_factor = oversample_factor
        self.max_batch_size = max_batch_size

        # Engine state
        self._state = EngineState.INITIALIZED
        self._progress = WorkflowProgress()
        self._snapshots: list[IterationSnapshot] = []
        self._pending_result: Optional[IterationResult] = None
        self._pending_snapshot: Optional[IterationSnapshot] = None
        self._nroy_exhausted: bool = False

        # Callbacks and hooks
        self._iteration_callbacks: list[Callable] = []
        self._progress_callbacks: list[Callable] = []

        # Additional settings
        self.settings = kwargs

        # Simulation function (to be provided by user)
        self._simulation_function: Optional[Callable] = None

        # Output directory for checkpoints, emulators, and diagnostics
        self._run_dir: Optional[Path] = None
        if output_dir is not None:
            import datetime
            if run_name is None:
                run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
            self._run_dir = Path(output_dir) / run_name

        # Set up file logging to run directory — always works even if the
        # caller never configures Python logging (the logger level defaults
        # to WARNING, so we lower it here to let our messages through).
        if self._run_dir:
            self._run_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(self._run_dir / "log.txt")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
            logger.addHandler(fh)
            logger.setLevel(logging.DEBUG)
            # Attach file handler to sub-loggers
            for sub in ('emulators', 'feature_selection', 'nroy_sampling'):
                sub_logger = logging.getLogger(f'historymatching.{sub}')
                sub_logger.addHandler(fh)
                sub_logger.setLevel(logging.DEBUG)

        # Log full configuration summary at the top of log.txt
        param_names = parameter_space.get_parameter_names()
        obs_targets = observations.get_all_targets()
        logger.info(f"{'='*60}")
        logger.info(f"HISTORY MATCHING ENGINE — CONFIGURATION")
        logger.info(f"{'='*60}")
        logger.info(f"  Emulator type:          {emulator_factory.get_default_type()}")
        logger.info(f"  Parameters:             {len(param_names)}")
        logger.info(f"  Observation targets:    {len(obs_targets)}")
        logger.info(f"  Samples per wave:       {n_samples}")
        logger.info(f"  Max iterations:         {max_iterations}")
        logger.info(f"  Implausibility threshold: {implausibility_threshold}")
        logger.info(f"  Auto space reduction:   {'enabled' if auto_reduce_space else 'disabled'}")
        logger.info(f"  Random seed:            {random_seed}")
        logger.info(f"  Oversample factor:      {oversample_factor}")
        logger.info(f"  Max batch size:         {max_batch_size}")
        if self._run_dir:
            logger.info(f"  Output directory:       {self._run_dir}")
            logger.info(f"  Run log:                {self._run_dir / 'log.txt'}")
        logger.info(f"  Parameters: {param_names}")
        logger.info(f"  Targets: {list(obs_targets.keys())}")
        logger.info(f"{'='*60}")

    @property
    def state(self) -> EngineState:
        """Current engine state."""
        return self._state

    @property
    def progress(self) -> WorkflowProgress:
        """Current workflow progress."""
        return self._progress

    @property
    def current_iteration(self) -> int:
        """Current iteration number (0-based)."""
        return self._progress.current_iteration

    @property
    def acceptance_rate(self) -> float:
        """Current acceptance rate for sample filtering."""
        return self._progress.acceptance_rate

    @property
    def run_dir(self) -> Optional[Path]:
        """Output directory for this run, or None if disk output is disabled."""
        return self._run_dir

    def set_simulation_function(self, func: Callable[[pd.DataFrame], pd.DataFrame]):
        """
        Set the simulation function for generating model outputs.

        Args:
            func: Function that takes parameter samples DataFrame and returns results DataFrame
        """
        self._simulation_function = func
        logger.info("Simulation function configured")

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
        if self.sampling_strategy is None:
            raise ValueError("Sampling strategy is required.")
        if self.feature_selection_strategy is None:
            raise ValueError("Feature selection strategy is required.")
        if self.emulator_factory is None:
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

        convergence_threshold = self.settings.get('convergence_threshold')
        if convergence_threshold is not None and not (0.0 <= convergence_threshold <= 1.0):
            raise ValueError("Convergence threshold must be between 0.0 and 1.0")
        nroy_method = self.settings.get('nroy_method')
        if nroy_method is not None and nroy_method not in ('auto', 'lhs', 'ray'):
            raise ValueError(f"Unknown NROY method '{nroy_method}'. Valid: ('auto', 'lhs', 'ray')")

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

        if self._state not in [EngineState.INITIALIZED, EngineState.PAUSED]:
            if self._state == EngineState.RUNNING:
                raise RuntimeError(
                    f"Engine is currently running iteration {self._progress.current_iteration + 1}. "
                    "Wait for it to complete before calling step() again."
                )
            elif self._state == EngineState.COMPLETED:
                raise RuntimeError(
                    f"Engine has completed all {self.max_iterations} iterations. "
                    "Use get_all_results() to access results or create a new engine instance to continue."
                )
            elif self._state == EngineState.ERROR:
                raise RuntimeError(
                    "Engine is in an error state from a previous operation. "
                    "Check the logs for details or create a new engine instance."
                )
            else:
                raise RuntimeError(
                    f"Engine is in state '{self._state.value}' and cannot execute step(). "
                    "If you have a pending iteration, use commit_step() to accept it or revert_step() to discard it."
                )

        if self._simulation_function is None:
            raise ValueError(
                "No simulation function has been configured. Before running iterations, you must provide "
                "a simulation function using set_simulation_function(your_function). "
                "Your function should take a pandas DataFrame of parameter samples and return "
                "a DataFrame with simulation results."
            )

        if self._progress.current_iteration >= self.max_iterations:
            raise RuntimeError(
                f"Maximum iterations limit reached ({self.max_iterations} iterations completed). "
                f"To run more iterations, create a new engine with a higher max_iterations value, "
                f"or use engine.update_max_iterations({self.max_iterations + 5}) to extend the current run."
            )

        wave_num = self._progress.current_iteration + 1
        logger.info(f"{'='*60}")
        logger.info(f"WAVE {wave_num} STARTING")
        logger.info(f"{'='*60}")
        self._state = EngineState.RUNNING
        wave_t0 = _time.time()

        try:
            # ── Phase 1: Get samples ─────────────────────────────────────
            t0 = _time.time()
            if self._progress.current_iteration == 0:
                samples = self._generate_plausible_samples()
                logger.info(f"[Wave {wave_num}] Phase 1/5 SAMPLING: generated {len(samples)} samples "
                            f"(acceptance rate: {self._progress.acceptance_rate:.3f}) [{_time.time()-t0:.1f}s]")
            else:
                previous_snapshot = self._snapshots[self._progress.current_iteration - 1]
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
                iteration=self._progress.current_iteration + 1,
                parameter_space=self.parameter_space,  # Current parameter space for this iteration
                samples=samples,
                simulation_results=simulation_results,
                selected_features=selected_features,
                emulators=emulators,
                nroy_fraction=nroy_fraction,
                execution_time_seconds=0.0,  # TODO: Track actual execution time
            )

            # Store pending changes (not committed yet)
            self._pending_result = iteration_result
            self._pending_snapshot = IterationSnapshot(
                iteration=self._progress.current_iteration + 1,
                parameter_space=next_parameter_space,
                emulator_bank=self.emulator_bank.copy(),  # Copy current state
                result=iteration_result,
                next_samples=next_iteration_samples,  # Store pre-computed samples for next iteration
                total_samples_generated=self._progress.total_samples_generated,
                total_samples_accepted=self._progress.total_samples_accepted,
                acceptance_rate=self._progress.acceptance_rate,
            )

            # Add emulators to pending snapshot's bank
            for feature, emulator in emulators.items():
                self._pending_snapshot.emulator_bank.add_emulator(iteration_result.iteration, feature, emulator)

            self._state = EngineState.PAUSED
            logger.info(f"[Wave {wave_num}] ALL PHASES COMPLETE [{_time.time()-wave_t0:.1f}s total]. Committing...")

            return iteration_result

        except Exception as e:
            self._state = EngineState.ERROR
            iteration_num = self._progress.current_iteration + 1

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
            if self._state == EngineState.INITIALIZED:
                raise RuntimeError(
                    "No iteration has been executed yet. Call step() first to run an iteration, "
                    "then use commit_step() to accept the results."
                )
            elif self._state == EngineState.COMPLETED:
                raise RuntimeError(
                    "All iterations have been completed and committed. "
                    "Use get_all_results() to access the final results."
                )
            else:
                raise RuntimeError(
                    f"No pending iteration to commit (engine state: {self._state.value}). "
                    "Call step() first to execute an iteration that can be committed."
                )

        # Apply changes
        self.emulator_bank = self._pending_snapshot.emulator_bank

        # Update parameter space only if auto-reduction is enabled
        if self.auto_reduce_space:
            self.parameter_space = self._pending_snapshot.parameter_space

        # Update progress
        self._progress.current_iteration += 1
        self._progress.completed_iterations.append(self._progress.current_iteration)
        self._progress.total_samples_accepted += len(self._pending_result.samples)
        self._progress.total_emulators_trained += len(self._pending_result.emulators)

        # Store snapshot
        self._snapshots.append(self._pending_snapshot)

        # Clear pending state
        committed_result = self._pending_result
        self._pending_result = None
        self._pending_snapshot = None

        # Update state
        if self._progress.current_iteration >= self.max_iterations:
            self._state = EngineState.COMPLETED
        else:
            self._state = EngineState.PAUSED

        # Save wave output (emulators, diagnostics, checkpoint)
        self._save_wave_output(committed_result)

        # Call callbacks
        self._call_iteration_callbacks(committed_result)
        self._call_progress_callbacks()

        logger.info(f"[Wave {committed_result.iteration}] COMMITTED — diagnostics and checkpoint saved to {self._run_dir}")
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
            if self._state == EngineState.INITIALIZED:
                raise RuntimeError(
                    "No iteration has been executed yet. Call step() first to run an iteration "
                    "before attempting to revert it."
                )
            elif self._state == EngineState.COMPLETED:
                raise RuntimeError(
                    "All iterations have been completed. There are no pending results to revert. "
                    "Previous iterations were already committed."
                )
            else:
                raise RuntimeError(
                    f"No pending iteration to revert (engine state: {self._state.value}). "
                    "Call step() first to execute an iteration that can be reverted."
                )

        # Restore progress information from last committed snapshot
        if self._snapshots:
            last_snapshot = self._snapshots[-1]
            self._progress.total_samples_generated = last_snapshot.total_samples_generated
            self._progress.total_samples_accepted = last_snapshot.total_samples_accepted
            self._progress.acceptance_rate = last_snapshot.acceptance_rate
        else:
            # No committed snapshots, reset to initial values
            self._progress.total_samples_generated = 0
            self._progress.total_samples_accepted = 0
            self._progress.acceptance_rate = 1.0

        # Clear pending state
        reverted_iteration = self._pending_result.iteration
        self._pending_result = None
        self._pending_snapshot = None

        # Return to paused state
        self._state = EngineState.PAUSED

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

            # Inspect per-feature fit quality (R² / MSE / n_train)
            print(result.quality_table())

            # Drop any emulator with a poor fit before committing
            engine.drop_emulator_from_pending('feature_c')
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

    def update_feature_selection(self, features: Union[list[str], FeatureSelectionStrategy]):
        """
        Update feature selection strategy for next iteration.

        Args:
            features: List of feature names or new FeatureSelectionStrategy
        """
        if isinstance(features, list):
            from .feature_selection import ManualFeatureSelection

            self.feature_selection_strategy = ManualFeatureSelection(features)
        else:
            self.feature_selection_strategy = features

        logger.info(f"Feature selection strategy updated: {self.feature_selection_strategy.get_strategy_name()}")

    def update_sampling_strategy(self, strategy: SamplingStrategy):
        """Update sampling strategy for next iteration."""
        self.sampling_strategy = strategy
        logger.info(f"Sampling strategy updated: {strategy.get_strategy_name()}")

    def update_emulator_type(self, emulator_type: str, **kwargs):
        """Update emulator factory configuration."""
        self.emulator_factory = EmulatorFactory(default_type=emulator_type, **kwargs)
        logger.info(f"Emulator factory updated: {emulator_type}")

    def update_max_iterations(self, max_iterations: int):
        """
        Update the maximum number of iterations.

        Args:
            max_iterations: New maximum number of iterations

        Raises:
            ValueError: If new limit is less than current iteration
        """
        if max_iterations <= self._progress.current_iteration:
            raise ValueError(
                f"Cannot set max_iterations to {max_iterations} because "
                f"{self._progress.current_iteration} iterations have already been completed. "
                f"New limit must be greater than {self._progress.current_iteration}."
            )

        self.max_iterations = max_iterations

        # Update state if we were completed but now have room for more iterations
        if self._state == EngineState.COMPLETED and self._progress.current_iteration < max_iterations:
            self._state = EngineState.PAUSED

        logger.info(f"Maximum iterations updated to {max_iterations}")

    def get_status_summary(self) -> str:
        """
        Get a human-readable summary of the current engine status.

        Returns:
            Multi-line string describing the engine's current state
        """
        summary = [
            "=== History Matching Engine Status ===",
            f"State: {self._state.value}",
            f"Progress: {self._progress.current_iteration}/{self.max_iterations} iterations",
        ]

        if self._progress.current_iteration > 0:
            summary.extend([
                f"Acceptance rate: {self._progress.acceptance_rate:.1%}",
                f"Total samples generated: {self._progress.total_samples_generated:,}",
                f"Total samples accepted: {self._progress.total_samples_accepted:,}",
                f"Emulators trained: {self._progress.total_emulators_trained}",
            ])

        if self._pending_result is not None:
            summary.append(f"⚠️  Pending iteration {self._pending_result.iteration} - use commit_step() or revert_step()")

        if self._simulation_function is None:
            summary.append("❌ No simulation function set - use set_simulation_function()")
        else:
            summary.append("✅ Simulation function configured")

        if self._state == EngineState.ERROR:
            summary.append("❌ Engine is in error state - check logs for details")
        elif self._state == EngineState.COMPLETED:
            summary.append("✅ All iterations completed successfully")

        return "\n".join(summary)

    # ------------------------------------------------------------------ #
    # Post-run summaries and plots
    #
    # The plot_* methods return Matplotlib axes (or arrays of axes) so they
    # render inline in notebooks and can be further customised or saved.
    # They draw the same figures the engine writes to ``run_dir`` after each
    # wave — see :meth:`_save_wave_output`, which now calls these methods.
    # ------------------------------------------------------------------ #
    def _bounds(self) -> dict:
        """Return ``{name: (lo, hi)}`` for the current parameter space."""
        return _bounds_from_space(self.parameter_space)

    def nroy_bounds(self, samples: Optional[pd.DataFrame] = None) -> dict:
        """Per-parameter ``(min, max)`` range of the surviving NROY cloud.

        Args:
            samples: NROY samples to summarise; defaults to
                :meth:`get_nroy_samples`.

        Returns:
            ``{parameter: (min, max)}`` over the NROY samples.
        """
        samples = self.get_nroy_samples() if samples is None else samples
        return {p: (float(samples[p].min()), float(samples[p].max()))
                for p in self.parameter_space.get_parameter_names()
                if p in samples.columns}

    def nroy_summary(self, samples: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Per-parameter summary of the NROY cloud and its space reduction.

        Args:
            samples: NROY samples to summarise; defaults to
                :meth:`get_nroy_samples`.

        Returns:
            A DataFrame with one row per parameter and columns ``min``, ``max``,
            ``median``, ``q05``, ``q95``, and ``reduction`` — the factor by
            which the parameter's range shrank relative to the prior bounds.
        """
        samples = self.get_nroy_samples() if samples is None else samples
        rows = []
        for p in self.parameter_space.get_parameter_names():
            if p not in samples.columns:
                continue
            col = samples[p]
            lo, hi = self.parameter_space.get_bounds(p)
            width = (col.max() - col.min())
            rows.append({
                "parameter": p,
                "min": float(col.min()),
                "max": float(col.max()),
                "median": float(col.median()),
                "q05": float(col.quantile(0.05)),
                "q95": float(col.quantile(0.95)),
                "reduction": float((hi - lo) / width) if width > 0 else float("inf"),
            })
        return pd.DataFrame(rows).set_index("parameter")

    def summary(self, samples: Optional[pd.DataFrame] = None) -> str:
        """Human-readable summary of the completed run.

        Folds together the per-wave NROY fractions, the surviving parameter
        ranges and their space-reduction factors, and the engine's progress
        totals — the report users otherwise reassemble by hand after
        :meth:`run`.

        Args:
            samples: NROY samples for the parameter-range section; defaults to
                :meth:`get_nroy_samples`.

        Returns:
            A multi-line summary string.
        """
        results = self.get_all_results()
        lines = ["=== History Matching Summary ===",
                 f"Waves completed:   {len(results)}/{self.max_iterations}",
                 f"Emulators trained: {self._progress.total_emulators_trained}",
                 f"Samples generated: {self._progress.total_samples_generated:,}",
                 f"Samples accepted:  {self._progress.total_samples_accepted:,}",
                 f"Acceptance rate:   {self._progress.acceptance_rate:.3%}"]
        if results:
            lines.append("")
            lines.append("NROY fraction per wave:")
            for r in results:
                feats = ", ".join(r.selected_features)
                lines.append(f"  Wave {r.iteration}: {r.nroy_fraction:>8.3%}   features: {feats}")
        samples = self.get_nroy_samples() if samples is None else samples
        if samples is not None and len(samples) > 0:
            lines.append("")
            lines.append(f"Plausible (NROY) parameter ranges  [{len(samples)} samples]:")
            summary = self.nroy_summary(samples)
            for p, row in summary.iterrows():
                lines.append(f"  {p:<20} [{row['min']:.4g}, {row['max']:.4g}]"
                             f"   median {row['median']:.4g}   ({row['reduction']:.1f}× narrower)")
        return "\n".join(lines)

    def plot_convergence(self, *, ax=None, log: bool = True):
        """Plot the NROY fraction at each wave (the convergence diagnostic).

        Args:
            ax: Existing Matplotlib axes to draw into.
            log: Use a logarithmic y-axis (recommended).

        Returns:
            The Matplotlib ``Axes`` containing the plot.
        """
        results = self.get_all_results()
        if not results:
            raise RuntimeError("No committed waves yet — run at least one iteration first.")
        return plotting.plot_convergence(
            [r.iteration for r in results], [r.nroy_fraction for r in results],
            ax=ax, log=log)

    def plot_nroy(self, *, params=None, truth=None, samples=None, prior=None,
                  max_params: int = _PAIRPLOT_MAX_PARAMS, axes=None):
        """Corner plot of the non-implausible (NROY) parameter cloud.

        Marginals on the diagonal, pairwise scatter below.  This is the headline
        result of a run — the shape of the parameter region consistent with the
        observations.

        Args:
            params: Parameters to show; defaults to all (capped at
                ``max_params``, ranked by how much each was constrained).
            truth: Optional ``{name: value}`` of known true values, drawn as
                crosshairs (handy for synthetic-recovery checks).
            samples: NROY samples to plot; defaults to :meth:`get_nroy_samples`.
            prior: Optional background cloud (e.g. the first wave's samples) to
                show how much the region shrank.
            max_params: Cap on parameters shown.
            axes: Existing ``(p, p)`` axes array to draw into.

        Returns:
            The 2-D array of ``Axes``.
        """
        samples = self.get_nroy_samples() if samples is None else samples
        if samples is None or len(samples) == 0:
            raise RuntimeError("No NROY samples available — run at least one iteration first.")
        bounds = self._bounds()
        if params is None and len(bounds) > max_params and len(samples) >= 10:
            mvr = plotting.marginal_variance_reduction(samples, bounds)
            params = sorted(mvr, key=mvr.get, reverse=True)[:max_params]
        return plotting.plot_pairplot(samples, params=params, truth=truth,
                                      prior=prior, bounds=bounds,
                                      max_params=max_params, axes=axes)

    def plot_marginals(self, *, params=None, truth=None, samples=None,
                       prior=None, axes=None):
        """Plot a posterior marginal histogram for each parameter.

        Args:
            params: Parameters to show; defaults to all.
            truth: Optional ``{name: value}`` true values (dashed lines).
            samples: NROY samples; defaults to :meth:`get_nroy_samples`.
            prior: Optional background cloud to overlay.
            axes: Existing axes array to draw into.

        Returns:
            A flat array of ``Axes``.
        """
        samples = self.get_nroy_samples() if samples is None else samples
        if samples is None or len(samples) == 0:
            raise RuntimeError("No NROY samples available — run at least one iteration first.")
        return plotting.plot_marginals(samples, params=params, truth=truth,
                                       bounds=self._bounds(), prior=prior, axes=axes)

    def plot_zscores(self, *, ax=None):
        """Plot standardised simulation outputs against every target, by wave.

        Shows whether each emulated/observed feature is converging to its target
        within the implausibility band as waves progress.

        Args:
            ax: Existing axes to draw into.

        Returns:
            The Matplotlib ``Axes`` containing the plot.
        """
        results = self.get_all_results()
        if not results:
            raise RuntimeError("No committed waves yet — run at least one iteration first.")
        waves = [{"iteration": r.iteration, "sim_results": r.simulation_results,
                  "selected_features": r.selected_features} for r in results]
        return plotting.plot_zscores_vs_targets(
            waves, self.observations.get_all_targets(), ax=ax,
            threshold=self.implausibility_threshold)

    def plot_constrained_dims(self, *, samples=None, n_top: int = 5, axes=None):
        """Plot the parameter-space directions history matching constrained most.

        Args:
            samples: NROY samples; defaults to :meth:`get_nroy_samples`.
            n_top: Number of most-constrained principal components to detail.
            axes: Existing axes array to draw into.

        Returns:
            The array of ``Axes``.
        """
        samples = self.get_nroy_samples() if samples is None else samples
        if samples is None or len(samples) < 10:
            raise RuntimeError("Need at least 10 NROY samples for the constrained-directions plot.")
        return plotting.plot_constrained_dims(samples, self._bounds(),
                                              n_top=n_top, axes=axes)

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

        if self._simulation_function is None:
            raise ValueError(
                "Cannot start automated run: no simulation function has been configured. "
                "Use set_simulation_function(your_function) to provide a function that takes "
                "parameter samples (DataFrame) and returns simulation results (DataFrame).\n\n"
                "Example:\n"
                "  def my_simulation(params_df):\n"
                "      # Your simulation code here\n"
                "      return results_df\n"
                "  engine.set_simulation_function(my_simulation)"
            )

        # Resume from checkpoint if requested
        if self._run_dir is not None:
            ckpt = self._run_dir / "checkpoint.pkl"
            if resume and ckpt.exists():
                logger.info(f"Resuming from checkpoint: {ckpt}")
                self._load_checkpoint_state(ckpt)
                logger.info(f"Resumed at wave {self._progress.current_iteration}")
            elif not resume and ckpt.exists():
                logger.warning(
                    f"Checkpoint exists at {ckpt} but resume=False. "
                    f"Starting fresh (existing output will be overwritten)."
                )

        logger.info(f"Starting automated run with {self.max_iterations} max iterations")

        results = self.get_all_results()  # includes any resumed waves

        try:
            import time

            self._progress.start_time = time.time()

            while self._progress.current_iteration < self.max_iterations and self._state not in [EngineState.COMPLETED, EngineState.ERROR]:
                # Run iteration
                result = self.step()
                results.append(result)

                if auto_commit:
                    self.commit_step()
                    if getattr(self, '_nroy_exhausted', False):
                        logger.info("Stopping: NROY space exhausted after this wave.")
                        self._state = EngineState.COMPLETED
                        break
                else:
                    break  # Let user decide

                # Check convergence criteria
                if self._check_convergence():
                    logger.info("Convergence criteria met. Stopping early.")
                    break

            self._progress.end_time = time.time()

            # Only set to COMPLETED if we actually finished all iterations
            if self._state != EngineState.ERROR and self._progress.current_iteration >= self.max_iterations:
                self._state = EngineState.COMPLETED

            logger.info(f"Automated run completed. {len(results)} iterations executed.")

        except Exception as e:
            self._state = EngineState.ERROR
            failed_iteration = len(results) + 1

            error_msg = (
                f"Automated run failed at iteration {failed_iteration} of {self.max_iterations}: {e}\n\n"
                f"Progress before failure:\n"
                f"  - Completed iterations: {len(results)}\n"
                f"  - Total samples generated: {self._progress.total_samples_generated:,}\n"
                f"  - Current acceptance rate: {self._progress.acceptance_rate:.1%}\n\n"
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

    def get_nroy_samples(self, n: Optional[int] = None,
                         method: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Get NROY parameter samples — filtered through ALL committed emulators.

        By default returns the pre-computed samples from the last wave (size =
        ``samples_per_iteration``).  Pass ``n`` to draw a fresh, larger set
        filtered through the current emulator bank.  No new simulations are
        run — only emulator predictions are used.

        Args:
            n: Number of NROY samples to return.  If None, returns the
               pre-computed set from the last committed wave.
            method: NROY sampling method: ``'auto'`` (LHS first, escalates
               to ray+importance if needed), ``'lhs'`` (pure rejection only),
               or None (uses engine default). For unbiased final samples
               (e.g. trajectory selection), use ``method='lhs'``.
            **kwargs: Extra options passed to ``generate_nroy_design()``:
               ``n_lines``, ``points_per_line`` (ray);
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
        method = method or self.settings.get('nroy_method', 'auto')
        nroy_opts = {**self.settings.get('nroy_options', {}), **kwargs}

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

    def _get_nroy_samples_serial(self, n: int) -> pd.DataFrame:
        """Serial rejection sampling for NROY candidates."""
        plausible = pd.DataFrame()
        total_generated = 0
        batch_size = min(int(n * self.oversample_factor), self.max_batch_size)
        batch_seed = self.random_seed
        max_candidates = n * self.settings.get('max_candidate_factor', 1000)

        while len(plausible) < n:
            candidates = self.sampling_strategy.generate_samples(
                self.parameter_space, batch_size, seed=batch_seed,
            )
            if batch_seed is not None:
                batch_seed += 1

            batch_pass = self._filter_samples_by_implausibility(candidates)
            if len(batch_pass) > 0:
                plausible = pd.concat([plausible, batch_pass], ignore_index=True)

            total_generated += len(candidates)

            if total_generated >= max_candidates:
                logger.warning(
                    f"get_nroy_samples: reached candidate limit ({max_candidates:,}) "
                    f"with {len(plausible)}/{n} samples."
                )
                break

            remaining = n - len(plausible)
            rate = len(plausible) / total_generated if total_generated > 0 else 0.01
            if rate > 0:
                batch_size = min(int(self.oversample_factor * remaining / rate), self.max_batch_size)

        return plausible.head(n)

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
            "progress": self._progress,
            "snapshots": self._snapshots,
            "settings": self.settings,
            "max_iterations": self.max_iterations,
            "implausibility_threshold": self.implausibility_threshold,
            "n_samples": self.n_samples,
            "auto_reduce_space": self.auto_reduce_space,
            "oversample_factor": self.oversample_factor,
        }

        with open(filepath, "wb") as f:
            pickle.dump(checkpoint_data, f)

        logger.info(f"Checkpoint saved to {filepath}")

    @classmethod
    def load_checkpoint(cls, filepath: Path, sampling_strategy: SamplingStrategy, feature_selection_strategy: FeatureSelectionStrategy, emulator_factory: EmulatorFactory) -> "HistoryMatchingEngine":
        """Load engine state from checkpoint file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # Create engine with loaded data
        engine = cls(
            parameter_space=data["parameter_space"],
            observations=data["observations"],
            sampling_strategy=sampling_strategy,
            feature_selection_strategy=feature_selection_strategy,
            emulator_factory=emulator_factory,
            emulator_bank=data["emulator_bank"],
            n_samples=data["n_samples"],
            implausibility_threshold=data["implausibility_threshold"],
            max_iterations=data["max_iterations"],
            auto_reduce_space=data.get("auto_reduce_space", False),
            oversample_factor=data.get("oversample_factor", 2.0),
            **data["settings"],
        )

        # Restore state
        engine._progress = data["progress"]
        engine._snapshots = data["snapshots"]
        engine._state = EngineState.PAUSED

        logger.info(f"Engine loaded from checkpoint {filepath}")
        return engine

    # Internal methods

    def _load_checkpoint_state(self, filepath: Path) -> None:
        """Restore engine state from a checkpoint file (for resume)."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.emulator_bank = data["emulator_bank"]
        self._progress = data["progress"]
        self._snapshots = data["snapshots"]
        self._state = EngineState.PAUSED
        logger.info(f"Loaded checkpoint: {len(self._snapshots)} waves completed")

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
        if self._run_dir is None:
            return

        import json as _json

        wave_dir = self._run_dir / f"wave{result.iteration}"
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

            # Save diagnostics figures (plot_diagnostics returns the figures it
            # creates, so we save exactly those rather than scraping pyplot state)
            try:
                if not getattr(emulator, 'testing_complete', False):
                    emulator.test()
                import matplotlib.pyplot as plt
                figs = emulator.plot_diagnostics() or []
                for i, fig in enumerate(figs):
                    fig.savefig(feat_dir / f"diagnostics_{i}.png", dpi=100, bbox_inches='tight')
                    plt.close(fig)
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

        # ── Wave-level: convergence ──────────────────────────────────────
        try:
            import matplotlib.pyplot as plt
            if self.get_all_results():
                ax = self.plot_convergence()
                fig = ax.figure
                fig.savefig(wave_dir / "convergence.png", dpi=100, bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to save convergence plot: {e}")

        # ── Wave-level: z-scores vs ALL targets ─────────────────────────
        try:
            import matplotlib.pyplot as plt
            targets = self.observations.get_all_targets()
            if self.get_all_results() and any(
                    k in result.simulation_results.columns for k in targets):
                ax = self.plot_zscores()
                fig = ax.figure
                fig.savefig(wave_dir / "zscores_vs_targets.png", dpi=150, bbox_inches="tight")
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
            import matplotlib.pyplot as plt
            all_results = self.get_all_results()
            snapshot = self._snapshots[-1]
            nroy = snapshot.next_samples
            if len(all_results) >= 2:
                prior = all_results[0].samples  # first-wave (~prior) cloud
                # The NROY cloud can collapse to an empty/degenerate DataFrame in
                # over-constrained runs; fall back to the latest wave's actual
                # samples so the diagnostic is still written.
                if nroy is not None and len(nroy) > 0:
                    cloud, note = nroy, "(blue = current NROY, grey = wave 1)"
                else:
                    cloud = all_results[-1].samples
                    note = ("(NROY cloud empty/over-constrained; "
                            "blue = last wave samples, grey = wave 1)")
                axes = self.plot_nroy(samples=cloud, prior=prior)
                fig = axes[0, 0].figure
                fig.suptitle(
                    f"NROY parameter cloud after wave {result.iteration}\n{note}",
                    fontsize=13, fontweight="bold")
                fig.tight_layout()
                fig.savefig(wave_dir / "pairplot.png", dpi=150, bbox_inches="tight")
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
            self.save_checkpoint(self._run_dir / "checkpoint.pkl")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

        # Run config (written once)
        config_path = self._run_dir / "run_config.json"
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
            self._progress.total_samples_generated += len(samples)
            self._progress.acceptance_rate = 1.0
            return samples

        # Subsequent iterations - adaptive rejection sampling loop
        plausible_samples = pd.DataFrame()
        total_candidates_generated = 0
        batch_num = 0
        batch_seed = self.random_seed  # Incremented each batch for fresh LHS draws
        max_candidate_factor = self.settings.get('max_candidate_factor', 1000)
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
                    f"Adjust via builder.settings['max_candidate_factor'] = N (or engine.settings)."
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
        self._progress.total_samples_generated += total_candidates_generated
        self._progress.acceptance_rate = len(plausible_samples) / total_candidates_generated if total_candidates_generated > 0 else 1.0

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
        """Run simulation with parameter samples."""
        return self._simulation_function(samples)

    def _select_features(self, simulation_results: pd.DataFrame) -> list[str]:
        """Select features to emulate using configured strategy."""
        return self.feature_selection_strategy.select_features(simulation_results, self.observations, self._progress.current_iteration + 1)

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
        current_iteration = self._progress.current_iteration + 1
        for feature, emulator in current_emulators.items():
            temp_bank.add_emulator(current_iteration, feature, emulator)

        from .nroy_sampling import generate_nroy_design

        nroy_method = self.settings.get('nroy_method', 'auto')
        nroy_opts = self.settings.get('nroy_options', {})

        nroy_result = generate_nroy_design(
            n_points=self.n_samples,
            parameter_space=self.parameter_space,
            emulator_bank=temp_bank,
            observations=self.observations,
            threshold=self.implausibility_threshold,
            sampling_strategy=self.sampling_strategy,
            method=nroy_method,
            seed=self.random_seed,
            max_candidates=self.n_samples * self.settings.get('max_candidate_factor', 1000),
            **nroy_opts,
        )

        self._update_nroy_stats(nroy_result.lhs_accepted, nroy_result.lhs_tested)
        return nroy_result.samples

    def _compute_next_samples_serial(self, temp_bank: EmulatorBank) -> pd.DataFrame:
        """Serial rejection sampling for next-wave candidates (legacy, unused)."""
        plausible_samples = pd.DataFrame()
        total_candidates_generated = 0
        batch_num = 0
        batch_seed = self.random_seed
        max_candidate_factor = self.settings.get('max_candidate_factor', 1000)
        max_candidates = self.n_samples * max_candidate_factor
        batch_size = min(int(self.n_samples * self.oversample_factor), self.max_batch_size)
        last_pct_logged = -10  # track last percentage milestone logged
        t0 = _time.time()

        logger.info(f"  NROY sampling: 0/{self.n_samples} (0%) — starting")

        while len(plausible_samples) < self.n_samples:
            candidates = self.sampling_strategy.generate_samples(self.parameter_space, batch_size, seed=batch_seed)
            if batch_seed is not None:
                batch_seed += 1

            batch_plausible = self._filter_samples_with_bank(candidates, temp_bank)
            if len(batch_plausible) > 0:
                plausible_samples = pd.concat([plausible_samples, batch_plausible], ignore_index=True)

            total_candidates_generated += len(candidates)
            batch_num += 1
            current_acceptance_rate = len(plausible_samples) / total_candidates_generated

            # Log at every 10% milestone
            pct = int(100 * len(plausible_samples) / self.n_samples)
            if pct >= last_pct_logged + 10:
                last_pct_logged = pct - (pct % 10)
                elapsed = _time.time() - t0
                logger.info(
                    f"  NROY sampling: {len(plausible_samples)}/{self.n_samples} ({pct}%) "
                    f"| {total_candidates_generated:,} tested | rate={current_acceptance_rate:.4%} [{elapsed:.0f}s]")

            logger.debug(
                f"  Next-wave sampling batch {batch_num}: {len(batch_plausible)}/{len(candidates)} accepted "
                f"| {len(plausible_samples)}/{self.n_samples} collected "
                f"| {total_candidates_generated:,} total candidates "
                f"| rate={current_acceptance_rate:.4%}"
            )

            if len(plausible_samples) >= self.n_samples:
                break

            if total_candidates_generated >= max_candidates:
                logger.warning(
                    f"Reached candidate limit ({max_candidates:,}) with only "
                    f"{len(plausible_samples)}/{self.n_samples} plausible samples "
                    f"(acceptance rate: {current_acceptance_rate:.4%}).  "
                    f"Proceeding with {len(plausible_samples)} samples.  "
                    f"Adjust via builder.settings['max_candidate_factor'] = N (or engine.settings)."
                )
                break

            remaining_needed = self.n_samples - len(plausible_samples)
            if current_acceptance_rate > 0:
                batch_size = min(int(self.oversample_factor * remaining_needed / current_acceptance_rate), self.max_batch_size)
            else:
                batch_size = min(batch_size * 2, self.max_batch_size)

        self._update_nroy_stats(len(plausible_samples), total_candidates_generated)
        return plausible_samples.head(self.n_samples)

    def _update_nroy_stats(self, n_plausible: int, n_generated: int) -> None:
        """Update NROY fraction and cumulative progress after rejection sampling."""
        self._last_nroy_fraction = (
            n_plausible / n_generated if n_generated > 0 else 1.0
        )
        self._progress.total_samples_generated += n_generated
        self._progress.total_samples_accepted += n_plausible
        self._progress.acceptance_rate = (
            self._progress.total_samples_accepted / self._progress.total_samples_generated
            if self._progress.total_samples_generated > 0 else 1.0
        )
        logger.debug(f"NROY fraction: {self._last_nroy_fraction:.6f}")

    def _filter_samples_with_bank(self, candidates: pd.DataFrame, emulator_bank: EmulatorBank) -> pd.DataFrame:
        """Filter candidate samples using a specific emulator bank."""
        if not emulator_bank.has_emulators():
            return candidates

        return self._filter_fast(candidates, emulator_bank)

    def _create_snapshot(self) -> IterationSnapshot:
        """Create snapshot of current state."""
        return IterationSnapshot(iteration=self._progress.current_iteration, parameter_space=self.parameter_space, emulator_bank=self.emulator_bank.copy())

    def _check_convergence(self) -> bool:
        """Check if convergence criteria are met.

        Returns True only when the acceptance rate (fraction of LHS candidates
        passing the emulator filter) drops below the configurable threshold.

        The threshold is set via ``builder.convergence_threshold``
        and defaults to 0.01 (1%).  Setting the threshold to 0.0 disables early
        stopping entirely.
        """
        threshold = self.settings.get('convergence_threshold', 0.0)
        if threshold <= 0:
            return False  # Early stopping disabled

        rate = getattr(self, '_last_nroy_fraction', 1.0)
        if rate < threshold:
            logger.warning(
                f"NROY fraction ({rate:.4%}) fell below convergence threshold "
                f"({threshold:.2%}).  The NROY region may be very small — consider "
                f"relaxing the implausibility threshold or checking emulator quality.  "
                f"Stopping after {self._progress.current_iteration} iterations."
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
                callback(self._progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def __repr__(self) -> str:
        """String representation of engine state."""
        return (
            f"HistoryMatchingEngine(\n"
            f"  state={self._state.value},\n"
            f"  iteration={self._progress.current_iteration}/{self.max_iterations},\n"
            f"  acceptance_rate={self._progress.acceptance_rate:.3f},\n"
            f"  emulators_trained={self._progress.total_emulators_trained},\n"
            f"  auto_reduce_space={self.auto_reduce_space}\n"
            f")"
        )
