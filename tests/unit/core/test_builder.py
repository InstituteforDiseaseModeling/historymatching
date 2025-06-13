"""
Unit tests for HistoryMatchingBuilder.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from history_matching.core.builder import HistoryMatchingBuilder, quick_setup, advanced_setup
from history_matching.domain.parameter_space import ParameterSpace
from history_matching.domain.observation_data import ObservationData
from history_matching.domain.emulator_bank import EmulatorBank
from history_matching.strategies.sampling import LatinHypercubeSampling, RandomSampling
from history_matching.strategies.feature_selection import ManualFeatureSelection, AutoFeatureSelection
from history_matching.strategies.emulator_factory import EmulatorFactory
from history_matching.core.engine import HistoryMatchingEngine


class TestHistoryMatchingBuilder:
    """Test HistoryMatchingBuilder functionality."""
    
    @pytest.fixture
    def sample_parameter_bounds(self):
        """Sample parameter bounds for testing."""
        return {
            'param1': (0.0, 1.0),
            'param2': (-5.0, 5.0),
            'param3': (10.0, 100.0)
        }
    
    @pytest.fixture
    def sample_observations(self):
        """Sample observations for testing."""
        return {
            'output1': (25.0, 5.0),  # (target, std)
            'output2': (100.0, 10.0),
            'output3': (0.5, 0.1)
        }
    
    @pytest.fixture
    def sample_parameter_df(self):
        """Sample parameter space DataFrame."""
        return pd.DataFrame({
            'parameter': ['param1', 'param2', 'param3'],
            'minimum': [0.0, -5.0, 10.0],
            'maximum': [1.0, 5.0, 100.0]
        })
    
    @pytest.fixture
    def sample_observations_df(self):
        """Sample observations DataFrame."""
        return pd.DataFrame({
            'feature': ['output1', 'output2', 'output3'],
            'mean': [25.0, 100.0, 0.5],
            'variance': [25.0, 100.0, 0.01]  # std^2
        })
    
    def test_builder_initialization(self):
        """Test basic builder initialization."""
        builder = HistoryMatchingBuilder()
        
        # Should have None for required components
        assert builder._parameter_space is None
        assert builder._observations is None
        
        # Should have defaults for configuration
        assert builder._n_samples == 1000
        assert builder._implausibility_threshold == 3.0
        assert builder._max_iterations == 10
        assert builder._random_seed is None
    
    def test_from_data_constructor(self, sample_parameter_bounds, sample_observations):
        """Test from_data class method constructor."""
        builder = HistoryMatchingBuilder.from_data(
            parameter_bounds=sample_parameter_bounds,
            observations=sample_observations
        )
        
        # Check that domain objects were created
        assert isinstance(builder._parameter_space, ParameterSpace)
        assert isinstance(builder._observations, ObservationData)
        
        # Check parameter space
        param_names = builder._parameter_space.get_parameter_names()
        assert set(param_names) == set(sample_parameter_bounds.keys())
        
        # Check observations
        obs_features = builder._observations.get_feature_names()
        assert set(obs_features) == set(sample_observations.keys())
    
    def test_from_data_with_kwargs(self, sample_parameter_bounds, sample_observations):
        """Test from_data with additional configuration."""
        builder = HistoryMatchingBuilder.from_data(
            parameter_bounds=sample_parameter_bounds,
            observations=sample_observations,
            n_samples=500,
            max_iterations=5,
            sampling_strategy='grid',
            emulator_type='linear'
        )
        
        assert builder._n_samples == 500
        assert builder._max_iterations == 5
        
        # Preview should show configured values
        config = builder.preview_configuration()
        assert 'Grid Sampling' in config['sampling_strategy']
        assert config['emulator_type'] == 'linear'
    
    def test_from_dataframes_constructor(self, sample_parameter_df, sample_observations_df):
        """Test from_dataframes class method constructor."""
        builder = HistoryMatchingBuilder.from_dataframes(
            parameter_space_df=sample_parameter_df,
            observations_df=sample_observations_df
        )
        
        # Check that domain objects were created
        assert isinstance(builder._parameter_space, ParameterSpace)
        assert isinstance(builder._observations, ObservationData)
        
        # Check parameter space
        param_names = builder._parameter_space.get_parameter_names()
        assert set(param_names) == set(sample_parameter_df['parameter'])
        
        # Check observations  
        obs_features = builder._observations.get_feature_names()
        assert set(obs_features) == set(sample_observations_df['feature'])
    
    def test_from_existing_constructor(self, sample_parameter_bounds, sample_observations):
        """Test from_existing class method constructor."""
        # Create domain objects
        parameter_space = ParameterSpace(sample_parameter_bounds)
        observations = ObservationData({
            name: (target, std**2) for name, (target, std) in sample_observations.items()
        })
        
        builder = HistoryMatchingBuilder.from_existing(
            parameter_space=parameter_space,
            observations=observations
        )
        
        assert builder._parameter_space is parameter_space
        assert builder._observations is observations
    
    def test_with_sampling_strategy_string(self, sample_parameter_bounds, sample_observations):
        """Test configuring sampling strategy by string."""
        builder = (HistoryMatchingBuilder
                  .from_data(sample_parameter_bounds, sample_observations)
                  .with_sampling_strategy('grid'))
        
        config = builder.preview_configuration()
        assert 'Grid Sampling' in config['sampling_strategy']
    
    def test_with_sampling_strategy_object(self, sample_parameter_bounds, sample_observations):
        """Test configuring sampling strategy with object."""
        strategy = RandomSampling()
        
        builder = (HistoryMatchingBuilder
                  .from_data(sample_parameter_bounds, sample_observations)
                  .with_sampling_strategy(strategy))
        
        assert builder._sampling_strategy is strategy
    
    def test_with_sampling_strategy_dict(self, sample_parameter_bounds, sample_observations):
        """Test configuring sampling strategy with dict."""
        builder = (HistoryMatchingBuilder
                  .from_data(sample_parameter_bounds, sample_observations)
                  .with_sampling_strategy({'type': 'lhs', 'criterion': 'center'}))
        
        config = builder.preview_configuration()
        assert 'center' in config['sampling_strategy']
    
    def test_with_feature_selection_list(self, sample_parameter_bounds, sample_observations):
        """Test configuring feature selection with list."""
        builder = (HistoryMatchingBuilder
                  .from_data(sample_parameter_bounds, sample_observations)
                  .with_feature_selection(['output1', 'output2']))
        
        assert isinstance(builder._feature_selection_strategy, ManualFeatureSelection)
        
        config = builder.preview_configuration()
        assert 'Manual Selection' in config['feature_selection_strategy']
    
    def test_with_feature_selection_string(self, sample_parameter_bounds, sample_observations):
        """Test configuring feature selection with single string."""
        builder = (HistoryMatchingBuilder
                  .from_data(sample_parameter_bounds, sample_observations)
                  .with_feature_selection('output1'))
        
        assert isinstance(builder._feature_selection_strategy, ManualFeatureSelection)
    
    def test_with_feature_selection_object(self, sample_parameter_bounds, sample_observations):
        """Test configuring feature selection with object."""
        strategy = AutoFeatureSelection(method='var', max_features=2)
        
        builder = (HistoryMatchingBuilder
                  .from_data(sample_parameter_bounds, sample_observations)
                  .with_feature_selection(strategy))
        
        assert builder._feature_selection_strategy is strategy
    
    def test_with_feature_selection_dict(self, sample_parameter_bounds, sample_observations):
        """Test configuring feature selection with dict."""
        builder = (HistoryMatchingBuilder
                  .from_data(sample_parameter_bounds, sample_observations)
                  .with_feature_selection({
                      'method': 'var',
                      'max_features': 2,
                      'threshold': 0.5
                  }))
        
        assert isinstance(builder._feature_selection_strategy, AutoFeatureSelection)
        assert builder._feature_selection_strategy.method == 'var'
        assert builder._feature_selection_strategy.max_features == 2
        assert builder._feature_selection_strategy.threshold == 0.5
    
    def test_with_emulator_type(self, sample_parameter_bounds, sample_observations):
        """Test configuring emulator type."""
        builder = (HistoryMatchingBuilder
                  .from_data(sample_parameter_bounds, sample_observations)
                  .with_emulator_type('linear', test_fraction=0.3))
        
        assert isinstance(builder._emulator_factory, EmulatorFactory)
        assert builder._emulator_factory.get_default_type() == 'linear'
        assert builder._emulator_factory.get_default_kwargs()['test_fraction'] == 0.3
    
    def test_with_emulator_factory(self, sample_parameter_bounds, sample_observations):
        """Test configuring custom emulator factory."""
        factory = EmulatorFactory('gpr', kernel='rbf')
        
        builder = (HistoryMatchingBuilder
                  .from_data(sample_parameter_bounds, sample_observations)
                  .with_emulator_factory(factory))
        
        assert builder._emulator_factory is factory
    
    def test_workflow_parameters(self, sample_parameter_bounds, sample_observations):
        """Test configuring workflow parameters."""
        builder = (HistoryMatchingBuilder
                  .from_data(sample_parameter_bounds, sample_observations)
                  .with_samples_per_iteration(2000)
                  .with_max_iterations(20)
                  .with_implausibility_threshold(2.5)
                  .with_random_seed(42))
        
        assert builder._n_samples == 2000
        assert builder._max_iterations == 20
        assert builder._implausibility_threshold == 2.5
        assert builder._random_seed == 42
    
    def test_space_reduction_options(self, sample_parameter_bounds, sample_observations):
        """Test space reduction configuration."""
        builder = (HistoryMatchingBuilder
                  .from_data(sample_parameter_bounds, sample_observations)
                  .with_space_reduction(True)
                  .with_oversample_factor(5.0))
        
        assert builder._settings['auto_reduce_space'] is True
        assert builder._settings['oversample_factor'] == 5.0
    
    def test_with_emulator_bank(self, sample_parameter_bounds, sample_observations):
        """Test configuring existing emulator bank."""
        bank = EmulatorBank()
        
        builder = (HistoryMatchingBuilder
                  .from_data(sample_parameter_bounds, sample_observations)
                  .with_emulator_bank(bank))
        
        assert builder._emulator_bank is bank
    
    def test_with_setting(self, sample_parameter_bounds, sample_observations):
        """Test adding custom settings."""
        builder = (HistoryMatchingBuilder
                  .from_data(sample_parameter_bounds, sample_observations)
                  .with_setting('custom_param', 'custom_value')
                  .with_setting('debug', True))
        
        assert builder._settings['custom_param'] == 'custom_value'
        assert builder._settings['debug'] is True
    
    def test_parameter_validation(self, sample_parameter_bounds, sample_observations):
        """Test parameter validation."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        
        # Test invalid values
        with pytest.raises(ValueError):
            builder.with_samples_per_iteration(0)
        
        with pytest.raises(ValueError):
            builder.with_max_iterations(-1)
        
        with pytest.raises(ValueError):
            builder.with_implausibility_threshold(0)
        
        with pytest.raises(ValueError):
            builder.with_oversample_factor(0.5)
    
    def test_build_success(self, sample_parameter_bounds, sample_observations):
        """Test successful engine building."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        
        engine = builder.build()
        
        assert isinstance(engine, HistoryMatchingEngine)
        assert engine.parameter_space is not None
        assert engine._observations is not None
    
    def test_build_missing_components(self):
        """Test building without required components."""
        builder = HistoryMatchingBuilder()
        
        # Missing parameter space
        with pytest.raises(ValueError, match="Parameter space is required"):
            builder.build()
    
    def test_build_with_defaults(self, sample_parameter_bounds, sample_observations):
        """Test that defaults are applied when building."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        
        # Don't configure strategies explicitly
        engine = builder.build()
        
        # Should have default strategies
        assert engine._sampling_strategy is not None
        assert engine._feature_selection_strategy is not None
        assert engine._emulator_factory is not None
        assert engine._emulator_bank is not None
    
    def test_preview_configuration(self, sample_parameter_bounds, sample_observations):
        """Test configuration preview."""
        builder = (HistoryMatchingBuilder
                  .from_data(sample_parameter_bounds, sample_observations)
                  .with_sampling_strategy('grid')
                  .with_emulator_type('linear')
                  .with_samples_per_iteration(500))
        
        config = builder.preview_configuration()
        
        assert config['parameter_space']['n_parameters'] == 3
        assert config['observations']['n_features'] == 3
        assert 'Grid Sampling' in config['sampling_strategy']
        assert config['emulator_type'] == 'linear'
        assert config['workflow_settings']['n_samples'] == 500
    
    def test_builder_repr(self, sample_parameter_bounds, sample_observations):
        """Test string representation."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        
        repr_str = repr(builder)
        
        assert 'HistoryMatchingBuilder' in repr_str
        assert 'parameters=3' in repr_str
        assert 'features=3' in repr_str
    
    def test_method_chaining(self, sample_parameter_bounds, sample_observations):
        """Test that all methods support chaining."""
        engine = (HistoryMatchingBuilder
                 .from_data(sample_parameter_bounds, sample_observations)
                 .with_sampling_strategy('lhs')
                 .with_feature_selection(['output1'])
                 .with_emulator_type('gpr')
                 .with_samples_per_iteration(1500)
                 .with_max_iterations(15)
                 .with_implausibility_threshold(2.8)
                 .with_random_seed(123)
                 .with_space_reduction(True)
                 .with_oversample_factor(3.0)
                 .build())
        
        assert isinstance(engine, HistoryMatchingEngine)
        assert engine._n_samples == 1500
        assert engine._max_iterations == 15
        assert engine._implausibility_threshold == 2.8
        assert engine._random_seed == 123
        assert engine._auto_reduce_space is True
        assert engine._oversample_factor == 3.0


class TestConvenienceFunctions:
    """Test convenience functions for quick setup."""
    
    def test_quick_setup(self):
        """Test quick_setup convenience function."""
        parameter_bounds = {
            'param1': (0, 1),
            'param2': (-1, 1)
        }
        observations = {
            'output1': (5.0, 1.0),
            'output2': (10.0, 2.0)
        }
        
        engine = quick_setup(
            parameter_bounds=parameter_bounds,
            observations=observations,
            n_samples=800
        )
        
        assert isinstance(engine, HistoryMatchingEngine)
        assert engine._n_samples == 800
        assert len(engine.parameter_space.get_parameter_names()) == 2
        assert len(engine._observations.get_feature_names()) == 2
    
    def test_advanced_setup(self):
        """Test advanced_setup convenience function."""
        parameter_df = pd.DataFrame({
            'parameter': ['param1', 'param2'],
            'minimum': [0, -1],
            'maximum': [1, 1]
        })
        
        observations_df = pd.DataFrame({
            'feature': ['output1', 'output2'],
            'mean': [5.0, 10.0],
            'variance': [1.0, 4.0]
        })
        
        engine = advanced_setup(
            parameter_space_df=parameter_df,
            observations_df=observations_df,
            sampling_strategy='grid',
            emulator_type='linear',
            max_iterations=8
        )
        
        assert isinstance(engine, HistoryMatchingEngine)
        assert engine._max_iterations == 8
        assert 'Grid' in engine._sampling_strategy.get_strategy_name()
        assert engine._emulator_factory.get_default_type() == 'linear'


class TestBuilderEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_sampling_strategy_type(self):
        """Test invalid sampling strategy type."""
        builder = HistoryMatchingBuilder()
        
        with pytest.raises(ValueError):
            builder.with_sampling_strategy(123)  # Invalid type
    
    def test_invalid_feature_selection_type(self):
        """Test invalid feature selection type."""
        builder = HistoryMatchingBuilder()
        
        with pytest.raises(ValueError):
            builder.with_feature_selection(123)  # Invalid type
    
    def test_empty_parameter_bounds(self):
        """Test with empty parameter bounds."""
        with pytest.raises(ValueError):
            HistoryMatchingBuilder.from_data({}, {'output': (1, 0.1)})
    
    def test_empty_observations(self):
        """Test with empty observations."""
        with pytest.raises(ValueError):
            HistoryMatchingBuilder.from_data({'param': (0, 1)}, {})
    
    def test_inconsistent_data_formats(self):
        """Test with inconsistent data formats."""
        # Parameter bounds with wrong tuple size
        with pytest.raises((ValueError, TypeError)):
            HistoryMatchingBuilder.from_data(
                {'param': (0,)},  # Missing max value
                {'output': (1, 0.1)}
            )
    
    def test_build_configuration_conflicts(self):
        """Test building with conflicting configurations."""
        builder = HistoryMatchingBuilder.from_data(
            {'param': (0, 1)},
            {'output': (1, 0.1)}
        )
        
        # These shouldn't cause conflicts, just test they work
        engine = (builder
                 .with_feature_selection(['output'])  # Manual selection
                 .with_feature_selection({'method': 'fano'})  # Override with auto
                 .build())
        
        assert isinstance(engine._feature_selection_strategy, AutoFeatureSelection)