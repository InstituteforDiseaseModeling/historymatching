"""
HistoryMatchingEngine for interactive workflow execution.

Provides step-by-step execution with the ability to inspect results,
make adjustments, and revert changes if needed.
"""

from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path
import pickle

try:
    from ..domain.parameter_space import ParameterSpace
    from ..domain.observation_data import ObservationData
    from ..domain.emulator_bank import EmulatorBank
    from ..domain.iteration_result import IterationResult
    from ..strategies.sampling import SamplingStrategy
    from ..strategies.feature_selection import FeatureSelectionStrategy
    from ..strategies.emulator_factory import EmulatorFactory
except ImportError:
    # For standalone testing
    from history_matching.domain.parameter_space import ParameterSpace
    from history_matching.domain.observation_data import ObservationData
    from history_matching.domain.emulator_bank import EmulatorBank
    from history_matching.domain.iteration_result import IterationResult
    from history_matching.strategies.sampling import SamplingStrategy
    from history_matching.strategies.feature_selection import FeatureSelectionStrategy
    from history_matching.strategies.emulator_factory import EmulatorFactory

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
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowProgress:
    """Progress tracking for history matching workflow."""
    current_iteration: int = 0
    completed_iterations: List[int] = field(default_factory=list)
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
    
    def __init__(self,
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
                 oversample_factor: float = 2.0,
                 **kwargs):
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
        
        # Engine state
        self._state = EngineState.INITIALIZED
        self._progress = WorkflowProgress()
        self._snapshots: List[IterationSnapshot] = []
        self._pending_result: Optional[IterationResult] = None
        self._pending_snapshot: Optional[IterationSnapshot] = None
        
        # Callbacks and hooks
        self._iteration_callbacks: List[Callable] = []
        self._progress_callbacks: List[Callable] = []
        
        # Additional settings
        self._settings = kwargs
        
        # Simulation function (to be provided by user)
        self._simulation_function: Optional[Callable] = None
        
        logger.info(f"HistoryMatchingEngine initialized with {len(parameter_space.get_parameter_names())} parameters")
        logger.info(f"Auto space reduction: {'enabled' if auto_reduce_space else 'disabled'}")
    
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
    
    def step(self, features: Optional[List[str]] = None) -> IterationResult:
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
            raise RuntimeError(f"Cannot step in state {self._state}. Use commit_step() or revert_step() first.")
        
        if self._simulation_function is None:
            raise ValueError("Simulation function must be set before running iterations. Use set_simulation_function().")
        
        if self._progress.current_iteration >= self._max_iterations:
            raise RuntimeError(f"Maximum iterations ({self._max_iterations}) reached")
        
        logger.info(f"Starting iteration {self._progress.current_iteration + 1}")
        self._state = EngineState.RUNNING
        
        try:
            # Generate plausible parameter samples
            samples = self._generate_plausible_samples()
            logger.info(f"Generated {len(samples)} plausible samples (acceptance rate: {self._progress.acceptance_rate:.3f})")
            
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
            
            # Determine parameter space for next iteration
            next_parameter_space = self._get_next_parameter_space(samples, emulators)
            
            # Calculate non-implausible points and fraction
            # For now, use all samples as non-implausible (will be filtered in next iteration)
            non_implausible_points = samples.copy()
            non_implausible_fraction = 1.0  # TODO: Calculate actual implausibility filtering
            
            # Create iteration result
            iteration_result = IterationResult(
                iteration=self._progress.current_iteration + 1,
                parameter_space=self._parameter_space,  # Current parameter space for this iteration
                samples=samples,
                simulation_results=simulation_results,
                selected_features=selected_features,
                emulators=emulators,
                non_implausible_points=non_implausible_points,
                non_implausible_fraction=non_implausible_fraction,
                execution_time_seconds=0.0  # TODO: Track actual execution time
            )
            
            # Store pending changes (not committed yet)
            self._pending_result = iteration_result
            self._pending_snapshot = IterationSnapshot(
                iteration=self._progress.current_iteration + 1,
                parameter_space=next_parameter_space,
                emulator_bank=self._emulator_bank.copy(),  # Copy current state
                result=iteration_result
            )
            
            # Add emulators to pending snapshot's bank
            for feature, emulator in emulators.items():
                self._pending_snapshot.emulator_bank.add_emulator(
                    feature, emulator, iteration_result.iteration
                )
            
            self._state = EngineState.PAUSED
            logger.info(f"Iteration {iteration_result.iteration} completed. Awaiting commit or revert.")
            
            return iteration_result
            
        except Exception as e:
            self._state = EngineState.ERROR
            logger.error(f"Error in iteration {self._progress.current_iteration + 1}: {e}")
            raise
    
    def commit_step(self) -> None:
        """
        Commit the pending iteration results.
        
        This makes the changes from the last step() permanent and advances
        the iteration counter.
        
        Raises:
            RuntimeError: If no pending iteration to commit
        """
        if self._pending_result is None or self._pending_snapshot is None:
            raise RuntimeError("No pending iteration to commit. Run step() first.")
        
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
            raise RuntimeError("No pending iteration to revert. Run step() first.")
        
        # Clear pending state
        reverted_iteration = self._pending_result.iteration
        self._pending_result = None
        self._pending_snapshot = None
        
        # Return to paused state
        self._state = EngineState.PAUSED
        
        logger.info(f"Iteration {reverted_iteration} reverted")
    
    def update_feature_selection(self, features: Union[List[str], FeatureSelectionStrategy]):
        """
        Update feature selection strategy for next iteration.
        
        Args:
            features: List of feature names or new FeatureSelectionStrategy
        """
        if isinstance(features, list):
            from ..strategies.feature_selection import ManualFeatureSelection
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
    
    def run(self, auto_commit: bool = True) -> List[IterationResult]:
        """
        Run automated history matching workflow.
        
        Args:
            auto_commit: Whether to automatically commit each iteration
            
        Returns:
            List of IterationResult objects for all iterations
            
        Raises:
            ValueError: If simulation function is not set
        """
        if self._simulation_function is None:
            raise ValueError("Simulation function must be set before running. Use set_simulation_function().")
        
        logger.info(f"Starting automated run with {self._max_iterations} max iterations")
        
        results = []
        
        try:
            import time
            self._progress.start_time = time.time()
            
            while (self._progress.current_iteration < self._max_iterations and 
                   self._state not in [EngineState.COMPLETED, EngineState.ERROR]):
                
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
            
            if self._state != EngineState.ERROR:
                self._state = EngineState.COMPLETED
            
            logger.info(f"Automated run completed. {len(results)} iterations executed.")
            
        except Exception as e:
            self._state = EngineState.ERROR
            logger.error(f"Error in automated run: {e}")
            raise
        
        return results
    
    def get_iteration_result(self, iteration: int) -> Optional[IterationResult]:
        """Get result for a specific iteration."""
        if iteration <= 0 or iteration > len(self._snapshots):
            return None
        return self._snapshots[iteration - 1].result
    
    def get_all_results(self) -> List[IterationResult]:
        """Get all committed iteration results."""
        return [snapshot.result for snapshot in self._snapshots if snapshot.result is not None]
    
    def save_checkpoint(self, filepath: Path) -> None:
        """Save engine state to checkpoint file."""
        checkpoint_data = {
            'parameter_space': self._parameter_space,
            'observations': self._observations,
            'emulator_bank': self._emulator_bank,
            'progress': self._progress,
            'snapshots': self._snapshots,
            'settings': self._settings,
            'max_iterations': self._max_iterations,
            'implausibility_threshold': self._implausibility_threshold,
            'n_samples': self._n_samples,
            'auto_reduce_space': self._auto_reduce_space,
            'oversample_factor': self._oversample_factor
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Checkpoint saved to {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: Path, 
                       sampling_strategy: SamplingStrategy,
                       feature_selection_strategy: FeatureSelectionStrategy,
                       emulator_factory: EmulatorFactory) -> 'HistoryMatchingEngine':
        """Load engine state from checkpoint file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create engine with loaded data
        engine = cls(
            parameter_space=data['parameter_space'],
            observations=data['observations'],
            sampling_strategy=sampling_strategy,
            feature_selection_strategy=feature_selection_strategy,
            emulator_factory=emulator_factory,
            emulator_bank=data['emulator_bank'],
            n_samples=data['n_samples'],
            implausibility_threshold=data['implausibility_threshold'],
            max_iterations=data['max_iterations'],
            auto_reduce_space=data.get('auto_reduce_space', False),
            oversample_factor=data.get('oversample_factor', 2.0),
            **data['settings']
        )
        
        # Restore state
        engine._progress = data['progress']
        engine._snapshots = data['snapshots']
        engine._state = EngineState.PAUSED
        
        logger.info(f"Engine loaded from checkpoint {filepath}")
        return engine
    
    # Internal methods
    
    def _generate_plausible_samples(self) -> pd.DataFrame:
        """
        Generate plausible parameter samples by filtering through existing emulators.
        
        For first iteration, returns unfiltered samples.
        For subsequent iterations, proposes candidates and filters based on implausibility.
        """
        if not self._emulator_bank.has_emulators():
            # First iteration - no filtering needed
            samples = self._sampling_strategy.generate_samples(
                self._parameter_space, 
                self._n_samples,
                seed=self._random_seed
            )
            self._progress.total_samples_generated += len(samples)
            self._progress.acceptance_rate = 1.0
            return samples
        
        # Subsequent iterations - propose and filter
        n_candidates = int(self._n_samples * self._oversample_factor)
        candidates = self._sampling_strategy.generate_samples(
            self._parameter_space,
            n_candidates,
            seed=self._random_seed
        )
        
        # Filter candidates through existing emulators
        plausible_samples = self._filter_samples_by_implausibility(candidates)
        
        # Update acceptance rate
        self._progress.total_samples_generated += len(candidates)
        self._progress.acceptance_rate = len(plausible_samples) / len(candidates)
        
        # Return requested number of samples
        if len(plausible_samples) >= self._n_samples:
            return plausible_samples.head(self._n_samples)
        else:
            warnings.warn(f"Only {len(plausible_samples)} plausible samples found "
                         f"(requested {self._n_samples}). Consider increasing oversample_factor.")
            return plausible_samples
    
    def _filter_samples_by_implausibility(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Filter candidate samples based on implausibility from existing emulators."""
        if not self._emulator_bank.has_emulators():
            return candidates
        
        # Calculate implausibility for each candidate sample
        sample_implausibilities = []
        
        # Get all emulators from the bank
        all_emulators = self._emulator_bank.get_all_emulators()
        
        for feature_name, emulator in all_emulators.items():
            try:
                # Get predictions from emulator
                predictions = emulator.predict(candidates)
                
                # Calculate implausibility for this feature
                feature_implausibility = self._observations.calculate_implausibility(
                    feature_name, 
                    predictions['value'], 
                    predictions.get('variance', None)
                )
                
                sample_implausibilities.append(feature_implausibility)
                
            except Exception as e:
                logger.warning(f"Failed to calculate implausibility for feature {feature_name}: {e}")
                continue
        
        if not sample_implausibilities:
            logger.warning("No valid implausibility calculations. Returning all candidates.")
            return candidates
        
        # Combine implausibilities (use maximum across features)
        combined_implausibility = pd.concat(sample_implausibilities, axis=1).max(axis=1)
        
        # Filter to plausible samples
        plausible_mask = combined_implausibility <= self._implausibility_threshold
        plausible_samples = candidates[plausible_mask]
        
        logger.debug(f"Filtered {len(candidates)} candidates to {len(plausible_samples)} plausible samples")
        
        return plausible_samples
    
    def _run_simulation(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Run simulation with parameter samples."""
        return self._simulation_function(samples)
    
    def _select_features(self, simulation_results: pd.DataFrame) -> List[str]:
        """Select features to emulate using configured strategy."""
        return self._feature_selection_strategy.select_features(
            simulation_results, 
            self._observations, 
            self._progress.current_iteration + 1
        )
    
    def _create_emulators(self, samples: pd.DataFrame, 
                         simulation_results: pd.DataFrame,
                         features: List[str]) -> Dict[str, Any]:
        """Create and train emulators for selected features."""
        return self._emulator_factory.create_emulators_for_features(
            samples, simulation_results, features
        )
    
    def _get_next_parameter_space(self, samples: pd.DataFrame, 
                                 emulators: Dict[str, Any]) -> ParameterSpace:
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
            # Get predictions
            predictions = emulator.predict(samples)
            
            # Calculate implausibility for this feature
            feature_implausibility = self._observations.calculate_implausibility(
                feature_name, predictions['value'], predictions.get('variance', None)
            )
            implausibilities.append(feature_implausibility)
        
        if not implausibilities:
            return self._parameter_space
        
        # Combine implausibilities (use maximum)
        combined_implausibility = pd.concat(implausibilities, axis=1).max(axis=1)
        
        # Find plausible samples
        plausible_mask = combined_implausibility <= self._implausibility_threshold
        plausible_samples = samples[plausible_mask]
        
        if len(plausible_samples) == 0:
            warnings.warn("No plausible samples found. Parameter space not reduced.")
            return self._parameter_space
        
        # Create new parameter space constrained to plausible samples
        return self._parameter_space.constrain_to_samples(plausible_samples)
    
    def _create_snapshot(self) -> IterationSnapshot:
        """Create snapshot of current state."""
        return IterationSnapshot(
            iteration=self._progress.current_iteration,
            parameter_space=self._parameter_space,
            emulator_bank=self._emulator_bank.copy()
        )
    
    def _check_convergence(self) -> bool:
        """Check if convergence criteria are met."""
        # Simple convergence check based on acceptance rate
        return self._progress.acceptance_rate < 0.01  # Less than 1% of samples accepted
    
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
        return (f"HistoryMatchingEngine(\n"
                f"  state={self._state.value},\n"
                f"  iteration={self._progress.current_iteration}/{self._max_iterations},\n"
                f"  acceptance_rate={self._progress.acceptance_rate:.3f},\n"
                f"  emulators_trained={self._progress.total_emulators_trained},\n"
                f"  auto_reduce_space={self._auto_reduce_space}\n"
                f")")