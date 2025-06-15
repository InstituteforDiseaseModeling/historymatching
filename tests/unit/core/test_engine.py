"""
Unit tests for HistoryMatchingEngine.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile

from history_matching.core.engine import HistoryMatchingEngine, EngineState, WorkflowProgress
from history_matching.domain.parameter_space import ParameterSpace
from history_matching.domain.observation_data import ObservationData
from history_matching.domain.emulator_bank import EmulatorBank
from history_matching.domain.iteration_result import IterationResult
from history_matching.strategies.sampling import LatinHypercubeSampling, RandomSampling
from history_matching.strategies.feature_selection import ManualFeatureSelection, AutoFeatureSelection
from history_matching.strategies.emulator_factory import EmulatorFactory
from history_matching.emulators.base import BaseEmulator


class MockEmulator(BaseEmulator):
    """Mock emulator for testing that inherits from BaseEmulator."""
    
    def __init__(self, feature_name='mock_feature'):
        super().__init__()  # Initialize BaseEmulator
        self.feature_name = feature_name
        self.training_complete = True
    
    def predict(self, X):
        """Return mock predictions with get_mean() and get_variance() methods."""
        n_samples = len(X)
        predictions = pd.DataFrame({
            'mean': np.random.normal(0, 1, n_samples),
            'var': np.ones(n_samples) * 0.1
        })
        
        # Add get_mean and get_variance methods that the engine expects
        def get_mean():
            return predictions['mean']
        
        def get_variance():
            return predictions['var']
        
        predictions.get_mean = get_mean
        predictions.get_variance = get_variance
        
        return predictions
    
    def train(self):
        self.training_complete = True


@pytest.fixture
def parameter_space():
    """Create test parameter space."""
    return ParameterSpace({
        'param1': (0.0, 1.0),
        'param2': (-1.0, 1.0),
        'param3': (10.0, 20.0)
    })

@pytest.fixture
def observations():
    """Create test observations."""
    return ObservationData({
        'output1': (5.0, 1.0),  # (mean, variance)
        'output2': (10.0, 4.0),
        'output3': (0.5, 0.01)
    })

@pytest.fixture
def mock_simulation_function():
    """Create mock simulation function."""
    def simulate(samples):
        n_samples = len(samples)
        return pd.DataFrame({
            'output1': np.random.normal(5, 1, n_samples),
            'output2': np.random.normal(10, 2, n_samples),
            'output3': np.random.normal(0.5, 0.1, n_samples),
            'extra_output': np.random.normal(0, 1, n_samples)
        })
    return simulate

@pytest.fixture
def basic_engine(parameter_space, observations):
    """Create basic engine for testing."""
    return HistoryMatchingEngine(
        parameter_space=parameter_space,
        observations=observations,
        sampling_strategy=RandomSampling(),
        feature_selection_strategy=ManualFeatureSelection(['output1']),
        emulator_factory=EmulatorFactory('linear'),
        n_samples=50  # Small for testing
    )


class TestHistoryMatchingEngine:
    """Test HistoryMatchingEngine functionality."""
    
    def test_engine_initialization(self, parameter_space, observations):
        """Test basic engine initialization."""
        engine = HistoryMatchingEngine(
            parameter_space=parameter_space,
            observations=observations,
            sampling_strategy=RandomSampling(),
            feature_selection_strategy=ManualFeatureSelection(['output1']),
            emulator_factory=EmulatorFactory('linear')
        )
        
        assert engine.state == EngineState.INITIALIZED
        assert engine.current_iteration == 0
        assert engine.parameter_space is parameter_space
        assert engine._observations is observations
        assert engine._n_samples == 1000  # default
        assert engine._auto_reduce_space is False  # default
        assert engine._oversample_factor == 1.1  # default
    
    def test_engine_initialization_with_options(self, parameter_space, observations):
        """Test engine initialization with custom options."""
        engine = HistoryMatchingEngine(
            parameter_space=parameter_space,
            observations=observations,
            sampling_strategy=RandomSampling(),
            feature_selection_strategy=ManualFeatureSelection(['output1']),
            emulator_factory=EmulatorFactory('linear'),
            n_samples=500,
            auto_reduce_space=True,
            oversample_factor=3.0,
            max_iterations=5,
            random_seed=42
        )
        
        assert engine._n_samples == 500
        assert engine._auto_reduce_space is True
        assert engine._oversample_factor == 3.0
        assert engine._max_iterations == 5
        assert engine._random_seed == 42
    
    def test_set_simulation_function(self, basic_engine, mock_simulation_function):
        """Test setting simulation function."""
        basic_engine.set_simulation_function(mock_simulation_function)
        assert basic_engine._simulation_function is mock_simulation_function
    
    def test_step_without_simulation_function(self, basic_engine):
        """Test stepping without simulation function raises error."""
        with pytest.raises(ValueError, match="No simulation function has been configured"):
            basic_engine.step()
    
    def test_step_invalid_state(self, basic_engine, mock_simulation_function):
        """Test stepping in invalid state."""
        basic_engine.set_simulation_function(mock_simulation_function)
        basic_engine._state = EngineState.RUNNING
        
        with pytest.raises(RuntimeError, match="Engine is currently running iteration"):
            basic_engine.step()
    
    def test_first_iteration_step(self, basic_engine, mock_simulation_function):
        """Test first iteration step."""
        basic_engine.set_simulation_function(mock_simulation_function)
        
        # Mock emulator factory to return mock emulators
        with patch.object(basic_engine._emulator_factory, 'create_emulators_for_features') as mock_create:
            mock_emulator = MockEmulator('output1')
            mock_create.return_value = {'output1': mock_emulator}
            
            result = basic_engine.step()
        
        assert isinstance(result, IterationResult)
        assert result.iteration == 1
        assert len(result.samples) == 50
        assert 'output1' in result.selected_features
        assert basic_engine.state == EngineState.PAUSED
        assert basic_engine._pending_result is result
    
    def test_commit_step(self, basic_engine, mock_simulation_function):
        """Test committing a step."""
        basic_engine.set_simulation_function(mock_simulation_function)
        
        # Run step
        with patch.object(basic_engine._emulator_factory, 'create_emulators_for_features') as mock_create:
            mock_emulator = MockEmulator('output1')
            mock_create.return_value = {'output1': mock_emulator}
            
            result = basic_engine.step()
        
        # Commit step
        basic_engine.commit_step()
        
        assert basic_engine.current_iteration == 1
        assert basic_engine._pending_result is None
        assert basic_engine.state == EngineState.PAUSED
        assert len(basic_engine._snapshots) == 1
        assert basic_engine.progress.total_samples_accepted == 50
    
    def test_revert_step(self, basic_engine, mock_simulation_function):
        """Test reverting a step."""
        basic_engine.set_simulation_function(mock_simulation_function)
        
        # Run step
        with patch.object(basic_engine._emulator_factory, 'create_emulators_for_features') as mock_create:
            mock_emulator = MockEmulator('output1')
            mock_create.return_value = {'output1': mock_emulator}
            
            result = basic_engine.step()
        
        # Revert step
        basic_engine.revert_step()
        
        assert basic_engine.current_iteration == 0
        assert basic_engine._pending_result is None
        assert basic_engine.state == EngineState.PAUSED
        assert len(basic_engine._snapshots) == 0
    
    def test_commit_without_pending_step(self, basic_engine):
        """Test committing without pending step raises error."""
        with pytest.raises(RuntimeError, match="No iteration has been executed yet"):
            basic_engine.commit_step()
    
    def test_revert_without_pending_step(self, basic_engine):
        """Test reverting without pending step raises error."""
        with pytest.raises(RuntimeError, match="No iteration has been executed yet"):
            basic_engine.revert_step()
    
    def test_second_iteration_with_precomputed_samples(self, basic_engine, mock_simulation_function):
        """Test that second iteration uses pre-computed samples from first iteration."""
        basic_engine.set_simulation_function(mock_simulation_function)
        
        # First iteration - mock to include next sample computation
        with patch.object(basic_engine._emulator_factory, 'create_emulators_for_features') as mock_create:
            mock_emulator = MockEmulator('output1')
            mock_create.return_value = {'output1': mock_emulator}
            
            # Mock the compute next iteration samples to return filtered samples
            with patch.object(basic_engine, '_compute_next_iteration_samples') as mock_compute:
                mock_compute.return_value = pd.DataFrame({
                    'param1': [0.5] * 30,
                    'param2': [0.0] * 30,
                    'param3': [15.0] * 30
                })
                
                result1 = basic_engine.step()
                basic_engine.commit_step()
                
                # Should have computed next iteration samples during first step
                assert mock_compute.call_count == 1
        
        # Second iteration should use pre-computed samples (no filtering at start)
        with patch.object(basic_engine, '_filter_samples_by_implausibility') as mock_filter:
            with patch.object(basic_engine._emulator_factory, 'create_emulators_for_features') as mock_create:
                mock_emulator2 = MockEmulator('output1')
                mock_create.return_value = {'output1': mock_emulator2}
                
                result2 = basic_engine.step()
        
        # Should NOT have called filtering during second step (uses pre-computed samples)
        assert mock_filter.call_count == 0
        # Should use the pre-computed samples (30 samples from first iteration)
        assert len(result2.samples) == 30
    
    def test_update_strategies(self, basic_engine):
        """Test updating strategies."""
        # Update feature selection
        basic_engine.update_feature_selection(['output1', 'output2'])
        assert isinstance(basic_engine._feature_selection_strategy, ManualFeatureSelection)
        
        # Update with strategy object
        auto_strategy = AutoFeatureSelection(method='var')
        basic_engine.update_feature_selection(auto_strategy)
        assert basic_engine._feature_selection_strategy is auto_strategy
        
        # Update sampling strategy
        new_sampling = LatinHypercubeSampling()
        basic_engine.update_sampling_strategy(new_sampling)
        assert basic_engine._sampling_strategy is new_sampling
        
        # Update emulator type
        basic_engine.update_emulator_type('gpr', kernel='rbf')
        assert basic_engine._emulator_factory.get_default_type() == 'gpr'
    
    def test_automated_run(self, basic_engine, mock_simulation_function):
        """Test automated run."""
        basic_engine.set_simulation_function(mock_simulation_function)
        basic_engine._max_iterations = 3
        
        with patch.object(basic_engine._emulator_factory, 'create_emulators_for_features') as mock_create:
            mock_emulator = MockEmulator('output1')
            mock_create.return_value = {'output1': mock_emulator}
            
            results = basic_engine.run()
        
        assert len(results) == 3
        assert basic_engine.current_iteration == 3
        assert basic_engine.state == EngineState.COMPLETED
        assert all(isinstance(r, IterationResult) for r in results)
    
    def test_automated_run_no_auto_commit(self, basic_engine, mock_simulation_function):
        """Test automated run without auto-commit."""
        basic_engine.set_simulation_function(mock_simulation_function)
        
        with patch.object(basic_engine._emulator_factory, 'create_emulators_for_features') as mock_create:
            mock_emulator = MockEmulator('output1')
            mock_create.return_value = {'output1': mock_emulator}
            
            results = basic_engine.run(auto_commit=False)
        
        assert len(results) == 1  # Only one iteration
        assert basic_engine.current_iteration == 0  # Not committed
        assert basic_engine.state == EngineState.PAUSED
    
    def test_max_iterations_limit(self, basic_engine, mock_simulation_function):
        """Test maximum iterations limit."""
        basic_engine.set_simulation_function(mock_simulation_function)
        basic_engine._max_iterations = 2
        
        with patch.object(basic_engine._emulator_factory, 'create_emulators_for_features') as mock_create:
            mock_emulator = MockEmulator('output1')
            mock_create.return_value = {'output1': mock_emulator}
            
            # Run two iterations
            basic_engine.step()
            basic_engine.commit_step()
            basic_engine.step()
            basic_engine.commit_step()
            
            # Third iteration should fail
            with pytest.raises(RuntimeError, match="Engine has completed all .* iterations"):
                basic_engine.step()
    
    def test_get_iteration_results(self, basic_engine, mock_simulation_function):
        """Test getting iteration results."""
        basic_engine.set_simulation_function(mock_simulation_function)
        
        with patch.object(basic_engine._emulator_factory, 'create_emulators_for_features') as mock_create:
            mock_emulator = MockEmulator('output1')
            mock_create.return_value = {'output1': mock_emulator}
            
            # Run and commit two iterations
            result1 = basic_engine.step()
            basic_engine.commit_step()
            result2 = basic_engine.step()
            basic_engine.commit_step()
        
        # Test getting specific iteration
        assert basic_engine.get_iteration_result(1) is result1
        assert basic_engine.get_iteration_result(2) is result2
        assert basic_engine.get_iteration_result(3) is None
        
        # Test getting all results
        all_results = basic_engine.get_all_results()
        assert len(all_results) == 2
        assert all_results[0] is result1
        assert all_results[1] is result2
    
    def test_callbacks(self, basic_engine, mock_simulation_function):
        """Test iteration and progress callbacks."""
        basic_engine.set_simulation_function(mock_simulation_function)
        
        # Add callbacks
        iteration_callback = MagicMock()
        progress_callback = MagicMock()
        basic_engine.add_iteration_callback(iteration_callback)
        basic_engine.add_progress_callback(progress_callback)
        
        with patch.object(basic_engine._emulator_factory, 'create_emulators_for_features') as mock_create:
            mock_emulator = MockEmulator('output1')
            mock_create.return_value = {'output1': mock_emulator}
            
            result = basic_engine.step()
            basic_engine.commit_step()
        
        # Callbacks should have been called
        iteration_callback.assert_called_once_with(result)
        progress_callback.assert_called_once()
    
    def test_space_reduction_disabled_by_default(self, basic_engine, mock_simulation_function):
        """Test that space reduction is disabled by default."""
        basic_engine.set_simulation_function(mock_simulation_function)
        
        initial_space = basic_engine.parameter_space
        
        with patch.object(basic_engine._emulator_factory, 'create_emulators_for_features') as mock_create:
            mock_emulator = MockEmulator('output1')
            mock_create.return_value = {'output1': mock_emulator}
            
            result = basic_engine.step()
            basic_engine.commit_step()
        
        # Parameter space should remain the same
        assert basic_engine.parameter_space is initial_space
    
    def test_space_reduction_enabled(self, parameter_space, observations, mock_simulation_function):
        """Test space reduction when enabled."""
        engine = HistoryMatchingEngine(
            parameter_space=parameter_space,
            observations=observations,
            sampling_strategy=RandomSampling(),
            feature_selection_strategy=ManualFeatureSelection(['output1']),
            emulator_factory=EmulatorFactory('linear'),
            n_samples=50,
            auto_reduce_space=True  # Enable space reduction
        )
        
        engine.set_simulation_function(mock_simulation_function)
        
        with patch.object(engine._emulator_factory, 'create_emulators_for_features') as mock_create:
            # Use our MockEmulator with custom prediction
            mock_emulator = MockEmulator('output1')
            # Override the predict method to return high values that will be implausible
            def custom_predict(X):
                n_samples = len(X)
                predictions = pd.DataFrame({
                    'mean': [100.0] * n_samples,  # Very different from target (5.0)
                    'var': [0.1] * n_samples
                })
                # Add get_mean and get_variance methods
                predictions.get_mean = lambda: predictions['mean']
                predictions.get_variance = lambda: predictions['var']
                return predictions
            mock_emulator.predict = custom_predict
            mock_create.return_value = {'output1': mock_emulator}
            
            with patch.object(engine, '_observations') as mock_obs:
                # Mock implausibility calculation to return high values for some samples
                def mock_calculate_implausibility(feature_name, means, variances):
                    n_samples = len(means)
                    return pd.Series([1.0] * (n_samples // 2) + [5.0] * (n_samples - n_samples // 2))
                
                mock_obs.calculate_implausibility.side_effect = mock_calculate_implausibility
                
                result = engine.step()
                engine.commit_step()
        
        # This test mainly checks that the space reduction logic is called
        # The actual reduction depends on the mock implementation
        assert engine.current_iteration == 1
    
    def test_checkpoint_save_load(self, basic_engine, mock_simulation_function):
        """Test saving and loading checkpoints."""
        basic_engine.set_simulation_function(mock_simulation_function)
        
        # Run an iteration
        with patch.object(basic_engine._emulator_factory, 'create_emulators_for_features') as mock_create:
            mock_emulator = MockEmulator('output1')
            mock_create.return_value = {'output1': mock_emulator}
            
            result = basic_engine.step()
            basic_engine.commit_step()
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            checkpoint_path = Path(f.name)
        
        try:
            basic_engine.save_checkpoint(checkpoint_path)
            
            # Load checkpoint
            loaded_engine = HistoryMatchingEngine.load_checkpoint(
                checkpoint_path,
                sampling_strategy=RandomSampling(),
                feature_selection_strategy=ManualFeatureSelection(['output1']),
                emulator_factory=EmulatorFactory('linear')
            )
            
            # Check that state was restored
            assert loaded_engine.current_iteration == basic_engine.current_iteration
            assert loaded_engine.state == EngineState.PAUSED
            assert len(loaded_engine._snapshots) == len(basic_engine._snapshots)
            
        finally:
            checkpoint_path.unlink()  # Clean up
    
    def test_engine_repr(self, basic_engine):
        """Test string representation."""
        repr_str = repr(basic_engine)
        
        assert 'HistoryMatchingEngine' in repr_str
        assert 'state=initialized' in repr_str
        assert 'iteration=0' in repr_str
        assert 'auto_reduce_space=False' in repr_str


class TestWorkflowProgress:
    """Test WorkflowProgress tracking."""
    
    def test_progress_initialization(self):
        """Test progress initialization."""
        progress = WorkflowProgress()
        
        assert progress.current_iteration == 0
        assert progress.completed_iterations == []
        assert progress.total_samples_generated == 0
        assert progress.total_samples_accepted == 0
        assert progress.acceptance_rate == 1.0
    
    def test_progress_updates(self, basic_engine, mock_simulation_function):
        """Test that progress is updated correctly."""
        basic_engine.set_simulation_function(mock_simulation_function)
        
        with patch.object(basic_engine._emulator_factory, 'create_emulators_for_features') as mock_create:
            mock_emulator = MockEmulator('output1')
            mock_create.return_value = {'output1': mock_emulator}
            
            # First iteration
            result1 = basic_engine.step()
            basic_engine.commit_step()
            
            progress = basic_engine.progress
            assert progress.current_iteration == 1
            assert progress.completed_iterations == [1]
            assert progress.total_samples_accepted == 50
            
            # Second iteration
            result2 = basic_engine.step()
            basic_engine.commit_step()
            
            assert progress.current_iteration == 2
            assert progress.completed_iterations == [1, 2]
            assert progress.total_samples_accepted == 100


class TestSampleFiltering:
    """Test sample filtering functionality."""
    
    def test_first_iteration_no_filtering(self, basic_engine, mock_simulation_function):
        """Test that first iteration doesn't filter samples."""
        basic_engine.set_simulation_function(mock_simulation_function)
        
        with patch.object(basic_engine._emulator_factory, 'create_emulators_for_features') as mock_create:
            mock_emulator = MockEmulator('output1')
            mock_create.return_value = {'output1': mock_emulator}
            
            result = basic_engine.step()
        
        # First iteration should have acceptance rate of 1.0 (no filtering)
        assert basic_engine.acceptance_rate == 1.0
        assert len(result.samples) == 50
    
    def test_sample_filtering_logic(self, basic_engine):
        """Test sample filtering logic."""
        # Create mock samples
        candidates = pd.DataFrame({
            'param1': [0.1, 0.5, 0.9],
            'param2': [0.0, 0.5, -0.5],
            'param3': [12.0, 15.0, 18.0]
        })
        
        # Mock emulator bank with emulators
        mock_emulator = MockEmulator('output1')
        predictions = pd.DataFrame({
            'mean': [5.0, 10.0, 1.0],  # Different predictions
            'var': [0.1, 0.1, 0.1]
        })
        predictions.get_mean = lambda: predictions['mean']
        predictions.get_variance = lambda: predictions['var']
        mock_emulator.predict = MagicMock(return_value=predictions)
        
        basic_engine._emulator_bank.add_emulator(1, 'output1', mock_emulator)
        
        # Mock observations to return specific implausibilities
        with patch.object(basic_engine._observations, 'calculate_implausibility') as mock_calc:
            mock_calc.return_value = pd.Series([1.0, 2.0, 5.0])  # Last one is implausible
            
            filtered = basic_engine._filter_samples_by_implausibility(candidates)
        
        # Should filter out the last sample (implausibility > threshold)
        assert len(filtered) == 2
        assert filtered.index.tolist() == [0, 1]
    
    def test_adaptive_sampling_during_next_sample_computation(self, basic_engine, mock_simulation_function):
        """Test that adaptive sampling works during next iteration sample computation."""
        basic_engine.set_simulation_function(mock_simulation_function)
        basic_engine._n_samples = 50  # Request samples
        
        # Mock first iteration to test adaptive sampling during next sample computation
        with patch.object(basic_engine._emulator_factory, 'create_emulators_for_features') as mock_create:
            mock_emulator = MockEmulator('output1')
            mock_create.return_value = {'output1': mock_emulator}
            
            # Mock the filtering within _filter_samples_with_bank to simulate low acceptance rate
            call_count = 0
            def mock_filter_side_effect(candidates, emulator_bank):
                nonlocal call_count
                call_count += 1
                # Return 10 samples from each batch to simulate low acceptance rate
                n_return = min(10, len(candidates))
                return candidates.head(n_return)
            
            with patch.object(basic_engine, '_filter_samples_with_bank', side_effect=mock_filter_side_effect):
                result1 = basic_engine.step()
                
                # Check that adaptive sampling occurred during next sample computation
                # (multiple calls to _filter_samples_with_bank means multiple batches were tried)
                assert call_count > 1
                
                # The pending snapshot should have the computed next samples
                next_samples = basic_engine.get_pending_next_samples()
                assert next_samples is not None
                assert len(next_samples) == 50  # Should eventually get requested number
                
                # Commit the step
                basic_engine.commit_step()
        
        # The acceptance rate should reflect the filtering that happened during next sample computation
        # Note: The acceptance rate is calculated based on the first iteration's box sampling (100%)
        # plus the filtering during next sample computation
        assert basic_engine.acceptance_rate < 1.0  # Should show filtering happened