"""
HistoryMatchingEngine for interactive workflow execution.

Provides step-by-step execution with the ability to inspect results,
make adjustments, and revert changes if needed.
"""

import logging
import pickle
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

logger = logging.getLogger(__name__)


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
        self._parameter_space = parameter_space  # Always keep original space
        self._observations = observations
        self._sampling_strategy = sampling_strategy
        self._feature_selection_strategy = feature_selection_strategy
        self._emulator_factory = emulator_factory
        self._emulator_bank = emulator_bank or EmulatorBank()

        # Workflow configuration
        self._n_samples = n_samples
        self._implausibility_threshold = implausibility_threshold
        self._max_iterations = max_iterations
        self._random_seed = random_seed
        self._auto_reduce_space = auto_reduce_space
        self._oversample_factor = oversample_factor
        self._max_batch_size = max_batch_size

        # Engine state
        self._state = EngineState.INITIALIZED
        self._progress = WorkflowProgress()
        self._snapshots: list[IterationSnapshot] = []
        self._pending_result: Optional[IterationResult] = None
        self._pending_snapshot: Optional[IterationSnapshot] = None

        # Callbacks and hooks
        self._iteration_callbacks: list[Callable] = []
        self._progress_callbacks: list[Callable] = []

        # Additional settings
        self._settings = kwargs

        # Simulation function (to be provided by user)
        self._simulation_function: Optional[Callable] = None

        # Output directory for checkpoints, emulators, and diagnostics
        self._run_dir: Optional[Path] = None
        if output_dir is not None:
            import datetime
            if run_name is None:
                run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
            self._run_dir = Path(output_dir) / run_name

        logger.info(f"HistoryMatchingEngine initialized with {len(parameter_space.get_parameter_names())} parameters")
        logger.info(f"Auto space reduction: {'enabled' if auto_reduce_space else 'disabled'}")
        if self._run_dir:
            logger.info(f"Output directory: {self._run_dir}")

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
    def parameter_space(self) -> ParameterSpace:
        """Original parameter space (never reduced unless explicitly enabled)."""
        return self._parameter_space

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

    def step(self, features: Optional[list[str]] = None) -> IterationResult:
        """
        Execute a single history matching iteration.

        Args:
            features: Optional list of features to emulate (overrides strategy)

        Returns:
            IterationResult for this iteration

        Raises:
            RuntimeError: If engine is not in a valid state for stepping
            ValueError: If simulation function is not set
        """
        if self._state not in [EngineState.INITIALIZED, EngineState.PAUSED]:
            if self._state == EngineState.RUNNING:
                raise RuntimeError(
                    f"Engine is currently running iteration {self._progress.current_iteration + 1}. "
                    "Wait for it to complete before calling step() again."
                )
            elif self._state == EngineState.COMPLETED:
                raise RuntimeError(
                    f"Engine has completed all {self._max_iterations} iterations. "
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

        if self._progress.current_iteration >= self._max_iterations:
            raise RuntimeError(
                f"Maximum iterations limit reached ({self._max_iterations} iterations completed). "
                f"To run more iterations, create a new engine with a higher max_iterations value, "
                f"or use engine.update_max_iterations({self._max_iterations + 5}) to extend the current run."
            )

        logger.info(f"Starting iteration {self._progress.current_iteration + 1}")
        self._state = EngineState.RUNNING

        try:
            # Get samples for this iteration
            if self._progress.current_iteration == 0:
                # First iteration - generate samples from parameter space box
                samples = self._generate_plausible_samples()
                logger.info(f"Generated {len(samples)} samples for first iteration (acceptance rate: {self._progress.acceptance_rate:.3f})")
            else:
                # Use pre-computed samples from previous iteration's snapshot
                previous_snapshot = self._snapshots[self._progress.current_iteration - 1]
                samples = previous_snapshot.next_samples
                if samples is None:
                    raise RuntimeError(
                        f"No pre-computed samples found from iteration {previous_snapshot.iteration}. "
                        f"This indicates an internal error - samples should have been computed during the previous step."
                    )
                logger.info(f"Using {len(samples)} pre-computed samples from iteration {previous_snapshot.iteration}")

            # Run simulation
            simulation_results = self._run_simulation(samples)
            logger.debug(f"Simulation completed with {len(simulation_results.columns)} outputs")

            # Select features to emulate
            if features is None:
                selected_features = self._select_features(simulation_results)
            else:
                selected_features = features

            logger.info(f"Selected features for emulation: {selected_features}")

            # Create emulators
            emulators = self._create_emulators(samples, simulation_results, selected_features)
            logger.debug(f"Created {len(emulators)} emulators")

            # Compute samples for next iteration (for user inspection before commit)
            next_iteration_samples = self._compute_next_iteration_samples(emulators)
            logger.info(f"Computed {len(next_iteration_samples)} samples for next iteration")

            # Determine parameter space for next iteration
            next_parameter_space = self._get_next_parameter_space(samples, emulators)

            # NROY fraction: what fraction of fresh LHS from the FULL prior pass
            # ALL emulators in the bank (including this wave's).  Cumulative —
            # must decrease monotonically as each wave adds constraints.
            nroy_fraction = getattr(self, '_last_nroy_fraction', 1.0)

            # Create iteration result
            iteration_result = IterationResult(
                iteration=self._progress.current_iteration + 1,
                parameter_space=self._parameter_space,  # Current parameter space for this iteration
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
                emulator_bank=self._emulator_bank.copy(),  # Copy current state
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
            logger.info(f"Iteration {iteration_result.iteration} completed. Awaiting commit or revert.")

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
        self._emulator_bank = self._pending_snapshot.emulator_bank

        # Update parameter space only if auto-reduction is enabled
        if self._auto_reduce_space:
            self._parameter_space = self._pending_snapshot.parameter_space

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
        if self._progress.current_iteration >= self._max_iterations:
            self._state = EngineState.COMPLETED
        else:
            self._state = EngineState.PAUSED

        # Save wave output (emulators, diagnostics, checkpoint)
        self._save_wave_output(committed_result)

        # Call callbacks
        self._call_iteration_callbacks(committed_result)
        self._call_progress_callbacks()

        logger.info(f"Iteration {committed_result.iteration} committed.")

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

            for f in result.selected_features:
                metrics = result.get_emulator_quality_metrics()
                print(f"{f}: R²={metrics[f]['r2']:.3f}")

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

            self._feature_selection_strategy = ManualFeatureSelection(features)
        else:
            self._feature_selection_strategy = features

        logger.info(f"Feature selection strategy updated: {self._feature_selection_strategy.get_strategy_name()}")

    def update_sampling_strategy(self, strategy: SamplingStrategy):
        """Update sampling strategy for next iteration."""
        self._sampling_strategy = strategy
        logger.info(f"Sampling strategy updated: {strategy.get_strategy_name()}")

    def update_emulator_type(self, emulator_type: str, **kwargs):
        """Update emulator factory configuration."""
        self._emulator_factory = EmulatorFactory(default_type=emulator_type, **kwargs)
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

        self._max_iterations = max_iterations

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
            f"Progress: {self._progress.current_iteration}/{self._max_iterations} iterations",
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
            ValueError: If simulation function is not set
        """
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

        logger.info(f"Starting automated run with {self._max_iterations} max iterations")

        results = self.get_all_results()  # includes any resumed waves

        try:
            import time

            self._progress.start_time = time.time()

            while self._progress.current_iteration < self._max_iterations and self._state not in [EngineState.COMPLETED, EngineState.ERROR]:
                # Run iteration
                result = self.step()
                results.append(result)

                if auto_commit:
                    self.commit_step()
                else:
                    break  # Let user decide

                # Check convergence criteria
                if self._check_convergence():
                    logger.info("Convergence criteria met. Stopping early.")
                    break

            self._progress.end_time = time.time()

            # Only set to COMPLETED if we actually finished all iterations
            if self._state != EngineState.ERROR and self._progress.current_iteration >= self._max_iterations:
                self._state = EngineState.COMPLETED

            logger.info(f"Automated run completed. {len(results)} iterations executed.")

        except Exception as e:
            self._state = EngineState.ERROR
            failed_iteration = len(results) + 1

            error_msg = (
                f"Automated run failed at iteration {failed_iteration} of {self._max_iterations}: {e}\n\n"
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

    def get_nroy_samples(self, n: Optional[int] = None) -> pd.DataFrame:
        """
        Get NROY parameter samples — filtered through ALL committed emulators.

        By default returns the pre-computed samples from the last wave (size =
        ``samples_per_iteration``).  Pass ``n`` to draw a fresh, larger set by
        rejection-sampling from the full prior against the current emulator bank.
        No new simulations are run — only emulator predictions are used.

        Args:
            n: Number of NROY samples to return.  If None, returns the
               pre-computed set from the last committed wave.

        Returns:
            DataFrame of NROY samples, or empty DataFrame if no iterations committed.

        Example:
            results = engine.run()
            nroy = engine.get_nroy_samples()             # default size
            nroy = engine.get_nroy_samples(10000)         # larger draw
        """
        if not self._snapshots:
            return pd.DataFrame()

        cached = self._snapshots[-1].next_samples
        if n is None or n <= len(cached):
            return cached.head(n) if n is not None else cached

        return self._get_nroy_samples_serial(n)

    def _get_nroy_samples_serial(self, n: int) -> pd.DataFrame:
        """Serial rejection sampling for NROY candidates."""
        plausible = pd.DataFrame()
        total_generated = 0
        batch_size = min(int(n * self._oversample_factor), self._max_batch_size)
        batch_seed = self._random_seed
        max_candidates = n * self._settings.get('max_candidate_factor', 1000)

        while len(plausible) < n:
            candidates = self._sampling_strategy.generate_samples(
                self._parameter_space, batch_size, seed=batch_seed,
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
                batch_size = min(int(self._oversample_factor * remaining / rate), self._max_batch_size)

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
            "parameter_space": self._parameter_space,
            "observations": self._observations,
            "emulator_bank": self._emulator_bank,
            "progress": self._progress,
            "snapshots": self._snapshots,
            "settings": self._settings,
            "max_iterations": self._max_iterations,
            "implausibility_threshold": self._implausibility_threshold,
            "n_samples": self._n_samples,
            "auto_reduce_space": self._auto_reduce_space,
            "oversample_factor": self._oversample_factor,
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

        self._emulator_bank = data["emulator_bank"]
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

            # Save diagnostics figure
            try:
                if not getattr(emulator, 'testing_complete', False):
                    emulator.test()
                import matplotlib
                matplotlib.use('Agg')
                emulator.plot_diagnostics()
                import matplotlib.pyplot as plt
                for i, fig_num in enumerate(plt.get_fignums()[-4:]):  # plot_diagnostics creates up to 4 figs
                    plt.figure(fig_num)
                    plt.savefig(feat_dir / f"diagnostics_{i}.png", dpi=100, bbox_inches='tight')
                    plt.close(fig_num)
            except Exception as e:
                logger.warning(f"Failed to save diagnostics for '{feature}': {e}")

            # Save metrics
            try:
                metrics = result.get_emulator_quality_metrics().get(feature, {})
                with open(feat_dir / "metrics.json", "w") as f:
                    _json.dump(metrics, f, indent=2, default=float)
            except Exception as e:
                logger.warning(f"Failed to save metrics for '{feature}': {e}")

        # ── Wave-level: convergence + NROY samples ───────────────────────
        try:
            all_results = self.get_all_results()
            if len(all_results) > 0:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(7, 4))
                waves = [r.iteration for r in all_results]
                fracs = [r.nroy_fraction for r in all_results]
                ax.bar(waves, fracs, color='#3575b5', alpha=0.8, edgecolor='white')
                for w, frac in zip(waves, fracs):
                    ax.annotate(f'{frac:.1%}', (w, frac), textcoords='offset points',
                                xytext=(0, 5), ha='center', fontsize=9)
                ax.set_xlabel('Wave')
                ax.set_ylabel('NROY fraction')
                ax.set_title('Convergence')
                ax.set_ylim(0, 1)
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
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                targets = self._observations.get_all_targets()
                target_names = [k for k in targets if k in result.simulation_results.columns]

                if len(target_names) > 0 and len(all_results) > 0:
                    # Collect which features were emulated in which wave
                    emulated = {}
                    for r in all_results:
                        for feat in r.selected_features:
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

        # ── Wave-level: parameter space pair plot ────────────────────────
        try:
            all_results = self.get_all_results()
            if len(all_results) >= 2:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                param_names = self._parameter_space.get_parameter_names()

                # Pick top 8 most relevant params by shortest ARD lengthscale
                # across all emulators in this wave
                ls_min = {}
                for feat, emul in result.emulators.items():
                    model = getattr(emul, 'model', None)
                    if model is None or not hasattr(model, 'kernel'):
                        continue
                    try:
                        ls = model.kernel.lengthscales.numpy()
                        if ls.ndim == 0:
                            continue
                        names = (list(emul.X_train_df.columns) if hasattr(emul, 'X_train_df')
                                 else param_names[:len(ls)])
                        for n, v in zip(names, ls):
                            if n not in ls_min or v < ls_min[n]:
                                ls_min[n] = v
                    except Exception:
                        pass

                if len(ls_min) > 0:
                    sorted_params = sorted(ls_min, key=ls_min.get)[:8]
                else:
                    sorted_params = param_names[:8]

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
                fig.suptitle(f'Parameter space: Waves {wave_labels}\n'
                             f'({n_pars} most relevant parameters by ARD)',
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
            self.save_checkpoint(self._run_dir / "checkpoint.pkl")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

        # Run config (written once)
        config_path = self._run_dir / "run_config.json"
        if not config_path.exists():
            try:
                config = {
                    'parameters': self._parameter_space.get_parameter_names(),
                    'parameter_bounds': {
                        p: list(self._parameter_space.get_bounds(p))
                        for p in self._parameter_space.get_parameter_names()
                    },
                    'observations': self._observations.get_all_targets(),
                    'n_samples': self._n_samples,
                    'max_iterations': self._max_iterations,
                    'implausibility_threshold': self._implausibility_threshold,
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
        if not self._emulator_bank.has_emulators():
            # First iteration - no filtering needed
            samples = self._sampling_strategy.generate_samples(self._parameter_space, self._n_samples, seed=self._random_seed)
            self._progress.total_samples_generated += len(samples)
            self._progress.acceptance_rate = 1.0
            return samples

        # Subsequent iterations - adaptive rejection sampling loop
        plausible_samples = pd.DataFrame()
        total_candidates_generated = 0
        batch_num = 0
        batch_seed = self._random_seed  # Incremented each batch for fresh LHS draws
        max_candidate_factor = self._settings.get('max_candidate_factor', 1000)
        max_candidates = self._n_samples * max_candidate_factor

        # Initial batch size
        batch_size = min(int(self._n_samples * self._oversample_factor), self._max_batch_size)

        while len(plausible_samples) < self._n_samples:
            # Generate candidate samples (fresh seed each batch)
            candidates = self._sampling_strategy.generate_samples(self._parameter_space, batch_size, seed=batch_seed)
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

            logger.info(
                f"  Sampling batch {batch_num}: {len(batch_plausible)}/{len(candidates)} accepted "
                f"| {len(plausible_samples)}/{self._n_samples} collected "
                f"| {total_candidates_generated:,} total candidates "
                f"| rate={current_acceptance_rate:.4%}"
            )

            # If we have enough samples, break
            if len(plausible_samples) >= self._n_samples:
                break

            # Safety valve: stop after generating too many candidates
            if total_candidates_generated >= max_candidates:
                logger.warning(
                    f"Reached candidate limit ({max_candidates:,}) with only "
                    f"{len(plausible_samples)}/{self._n_samples} plausible samples "
                    f"(acceptance rate: {current_acceptance_rate:.4%}).  "
                    f"Proceeding with {len(plausible_samples)} samples.  "
                    f"Adjust with .with_setting('max_candidate_factor', N)."
                )
                break

            # Calculate next batch size adaptively
            remaining_needed = self._n_samples - len(plausible_samples)
            if current_acceptance_rate > 0:
                batch_size = min(int(self._oversample_factor * remaining_needed / current_acceptance_rate), self._max_batch_size)
            else:
                # If no samples accepted yet, increase batch size
                batch_size = min(batch_size * 2, self._max_batch_size)

        # Update global progress tracking
        self._progress.total_samples_generated += total_candidates_generated
        self._progress.acceptance_rate = len(plausible_samples) / total_candidates_generated if total_candidates_generated > 0 else 1.0

        # Return exactly the requested number of samples
        return plausible_samples.head(self._n_samples)

    def _build_fast_predictors(self, emulator_bank=None):
        """Build FastGPRPredictor objects for all GPR emulators in the bank.

        Returns a list of (FastGPRPredictor, obs_mean, obs_std, feature_name)
        tuples suitable for filter_nroy().  Non-GPR emulators are skipped
        (they fall back to the slow path).
        """
        from .emulators.fast_predict import FastGPRPredictor

        bank = emulator_bank or self._emulator_bank
        predictors = []

        for iteration in bank.get_all_iterations():
            emulators = bank.get_emulators_for_iteration(iteration)
            for feature_name, emulator in emulators.items():
                if not self._observations.has_feature(feature_name):
                    continue
                # Only works for GPR emulators (SE kernel with normalization)
                from .emulators.gpr import GPR
                if not isinstance(emulator, GPR):
                    continue
                try:
                    fast = FastGPRPredictor.from_emulator(emulator)
                    obs_mean, obs_std = self._observations.get_target_for_feature(feature_name)
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
        if not self._emulator_bank.has_emulators():
            return candidates

        return self._filter_fast(candidates, self._emulator_bank)

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
        param_cols = self._parameter_space.get_parameter_names()
        X = candidates[param_cols].values.astype(np.float64)

        mask = filter_nroy(X, fast_predictors, threshold=self._implausibility_threshold)

        plausible = candidates[mask]
        logger.debug(
            f"Fast filter: {len(candidates)} → {len(plausible)} "
            f"({len(plausible)/len(candidates):.2%}) through {len(fast_predictors)} emulators"
        )
        return plausible

    def _filter_samples_slow(self, candidates: pd.DataFrame, emulator_bank) -> pd.DataFrame:
        """Standard GPflow-based implausibility filter (fallback for non-GPR emulators)."""
        sample_implausibilities = []
        param_cols = self._parameter_space.get_parameter_names()
        candidates_clean = candidates[param_cols]

        for iteration in reversed(emulator_bank.get_all_iterations()):
            emulators = emulator_bank.get_emulators_for_iteration(iteration)
            for feature_name, emulator in emulators.items():
                try:
                    if not self._observations.has_feature(feature_name):
                        continue
                    predictions = emulator.predict(candidates_clean)
                    feature_implausibility = self._observations.calculate_implausibility(
                        feature_name, predictions.get_mean(), predictions.get_variance()
                    )
                    sample_implausibilities.append(feature_implausibility)
                except Exception as e:
                    logger.warning(f"Implausibility calc failed for '{feature_name}': {e}")
                    continue

        if not sample_implausibilities:
            return candidates

        combined = pd.concat(sample_implausibilities, axis=1).max(axis=1)
        combined.index = candidates.index
        return candidates[combined <= self._implausibility_threshold]

    def _run_simulation(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Run simulation with parameter samples."""
        return self._simulation_function(samples)

    def _select_features(self, simulation_results: pd.DataFrame) -> list[str]:
        """Select features to emulate using configured strategy."""
        return self._feature_selection_strategy.select_features(simulation_results, self._observations, self._progress.current_iteration + 1)

    def _create_emulators(self, samples: pd.DataFrame, simulation_results: pd.DataFrame, features: list[str]) -> dict[str, Any]:
        """Create and train emulators for selected features.

        Only parameter-space columns are passed to emulators — any extra
        columns (e.g. ``rand_seed``) added by the simulation function are
        excluded so they don't become spurious input dimensions.
        """
        param_cols = self._parameter_space.get_parameter_names()
        samples_clean = samples[param_cols]
        return self._emulator_factory.create_emulators_for_features(samples_clean, simulation_results, features)

    def _get_next_parameter_space(self, samples: pd.DataFrame, emulators: dict[str, Any]) -> ParameterSpace:
        """
        Determine parameter space for next iteration.

        By default, returns the original parameter space.
        If auto_reduce_space is enabled, constrains to plausible samples.
        """
        if not self._auto_reduce_space:
            return self._parameter_space  # Keep original space

        # Calculate implausibilities and constrain space
        implausibilities = []

        for feature_name, emulator in emulators.items():
            # Get predictions (only parameter columns, not metadata like rand_seed)
            param_cols = self._parameter_space.get_parameter_names()
            predictions = emulator.predict(samples[param_cols])

            # Calculate implausibility for this feature (vectorized)
            feature_implausibility = self._observations.calculate_implausibility(feature_name, predictions.get_mean(), predictions.get_variance())
            implausibilities.append(feature_implausibility)

        if not implausibilities:
            return self._parameter_space

        # Combine implausibilities (use maximum)
        combined_implausibility = pd.concat(implausibilities, axis=1).max(axis=1)

        # Find plausible samples
        plausible_mask = combined_implausibility <= self._implausibility_threshold
        plausible_samples = samples[plausible_mask]

        if len(plausible_samples) == 0:
            warnings.warn(
                f"No plausible samples found with implausibility threshold {self._implausibility_threshold}. "
                f"Parameter space will not be reduced this iteration. Consider:\n"
                f"  - Increasing the implausibility threshold (current: {self._implausibility_threshold})\n"
                f"  - Generating more samples per iteration (current: {self._n_samples})\n"
                f"  - Checking if your simulation is producing reasonable outputs\n"
                f"  - Reviewing your observation data for inconsistencies", stacklevel=2
            )
            return self._parameter_space

        # Create new parameter space constrained to plausible samples
        return self._parameter_space.constrain_to_samples(plausible_samples)

    def _compute_next_iteration_samples(self, current_emulators: dict[str, Any]) -> pd.DataFrame:
        """
        Compute plausible parameter samples for the next iteration.

        Args:
            current_emulators: Emulators created in the current iteration

        Returns:
            DataFrame of plausible samples for next iteration
        """
        # Create a temporary emulator bank with current emulators for filtering
        temp_bank = self._emulator_bank.copy()
        current_iteration = self._progress.current_iteration + 1
        for feature, emulator in current_emulators.items():
            temp_bank.add_emulator(current_iteration, feature, emulator)

        return self._compute_next_samples_serial(temp_bank)

    def _compute_next_samples_serial(self, temp_bank: EmulatorBank) -> pd.DataFrame:
        """Serial rejection sampling for next-wave candidates."""
        plausible_samples = pd.DataFrame()
        total_candidates_generated = 0
        batch_num = 0
        batch_seed = self._random_seed
        max_candidate_factor = self._settings.get('max_candidate_factor', 1000)
        max_candidates = self._n_samples * max_candidate_factor
        batch_size = min(int(self._n_samples * self._oversample_factor), self._max_batch_size)

        while len(plausible_samples) < self._n_samples:
            candidates = self._sampling_strategy.generate_samples(self._parameter_space, batch_size, seed=batch_seed)
            if batch_seed is not None:
                batch_seed += 1

            batch_plausible = self._filter_samples_with_bank(candidates, temp_bank)
            if len(batch_plausible) > 0:
                plausible_samples = pd.concat([plausible_samples, batch_plausible], ignore_index=True)

            total_candidates_generated += len(candidates)
            batch_num += 1
            current_acceptance_rate = len(plausible_samples) / total_candidates_generated

            logger.info(
                f"  Next-wave sampling batch {batch_num}: {len(batch_plausible)}/{len(candidates)} accepted "
                f"| {len(plausible_samples)}/{self._n_samples} collected "
                f"| {total_candidates_generated:,} total candidates "
                f"| rate={current_acceptance_rate:.4%}"
            )

            if len(plausible_samples) >= self._n_samples:
                break

            if total_candidates_generated >= max_candidates:
                logger.warning(
                    f"Reached candidate limit ({max_candidates:,}) with only "
                    f"{len(plausible_samples)}/{self._n_samples} plausible samples "
                    f"(acceptance rate: {current_acceptance_rate:.4%}).  "
                    f"Proceeding with {len(plausible_samples)} samples.  "
                    f"Adjust with .with_setting('max_candidate_factor', N)."
                )
                break

            remaining_needed = self._n_samples - len(plausible_samples)
            if current_acceptance_rate > 0:
                batch_size = min(int(self._oversample_factor * remaining_needed / current_acceptance_rate), self._max_batch_size)
            else:
                batch_size = min(batch_size * 2, self._max_batch_size)

        self._update_nroy_stats(len(plausible_samples), total_candidates_generated)
        return plausible_samples.head(self._n_samples)

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
        logger.debug(f"NROY fraction: {self._last_nroy_fraction:.3f}")

    def _filter_samples_with_bank(self, candidates: pd.DataFrame, emulator_bank: EmulatorBank) -> pd.DataFrame:
        """Filter candidate samples using a specific emulator bank."""
        if not emulator_bank.has_emulators():
            return candidates

        return self._filter_fast(candidates, emulator_bank)

    def _create_snapshot(self) -> IterationSnapshot:
        """Create snapshot of current state."""
        return IterationSnapshot(iteration=self._progress.current_iteration, parameter_space=self._parameter_space, emulator_bank=self._emulator_bank.copy())

    def _check_convergence(self) -> bool:
        """Check if convergence criteria are met.

        Returns True only when the acceptance rate (fraction of LHS candidates
        passing the emulator filter) drops below the configurable threshold.

        The threshold is set via ``HistoryMatchingBuilder.with_convergence_threshold()``
        and defaults to 0.01 (1%).  Setting the threshold to 0.0 disables early
        stopping entirely.
        """
        threshold = self._settings.get('convergence_threshold', 0.0)
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
            f"  iteration={self._progress.current_iteration}/{self._max_iterations},\n"
            f"  acceptance_rate={self._progress.acceptance_rate:.3f},\n"
            f"  emulators_trained={self._progress.total_emulators_trained},\n"
            f"  auto_reduce_space={self._auto_reduce_space}\n"
            f")"
        )
