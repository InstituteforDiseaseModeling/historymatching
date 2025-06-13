"""
Unit tests for feature selection strategies.
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from unittest.mock import patch, MagicMock

from history_matching.domain.observation_data import ObservationData
from history_matching.strategies.feature_selection import (
    FeatureSelectionStrategy,
    ManualFeatureSelection,
    AutoFeatureSelection,
    InteractiveFeatureSelection,
    MultiFeatureSelection
)


class TestFeatureSelectionStrategy:
    """Test the abstract FeatureSelectionStrategy base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that FeatureSelectionStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FeatureSelectionStrategy()
    
    def test_subclass_must_implement_abstract_methods(self):
        """Test that subclasses must implement abstract methods."""
        
        class IncompleteSelector(FeatureSelectionStrategy):
            def select_features(self, simulation_results, observations, iteration=1):
                return ['feature1']
        
        with pytest.raises(TypeError):
            IncompleteSelector()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample simulation results and observations."""
        simulation_results = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [0.5, 1.5, 2.5, 3.5],
            'feature3': [10.0, 20.0, 30.0, 40.0],
            'feature4': [0.1, 0.2, 0.3, 0.4]
        })
        
        observations = ObservationData({
            'feature1': (2.5, 0.5**2),  # (mean, variance)
            'feature2': (2.0, 0.3**2),
            'feature3': (25.0, 5.0**2)
        })
        
        return simulation_results, observations
    
    def test_validate_features_all_valid(self, sample_data):
        """Test validation when all features are valid."""
        
        class TestSelector(FeatureSelectionStrategy):
            def select_features(self, simulation_results, observations, iteration=1):
                return ['feature1', 'feature2']
            
            def get_strategy_name(self):
                return "Test Selector"
        
        selector = TestSelector()
        simulation_results, observations = sample_data
        
        valid_features = selector.validate_features(
            ['feature1', 'feature2'], simulation_results, observations
        )
        
        assert valid_features == ['feature1', 'feature2']
    
    def test_validate_features_missing_from_simulation(self, sample_data):
        """Test validation when feature missing from simulation results."""
        
        class TestSelector(FeatureSelectionStrategy):
            def select_features(self, simulation_results, observations, iteration=1):
                return ['feature1']
            
            def get_strategy_name(self):
                return "Test Selector"
        
        selector = TestSelector()
        simulation_results, observations = sample_data
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            valid_features = selector.validate_features(
                ['feature1', 'missing_feature'], simulation_results, observations
            )
            
            assert len(w) == 1
            assert "not found in simulation results" in str(w[0].message)
            assert valid_features == ['feature1']
    
    def test_validate_features_missing_from_observations(self, sample_data):
        """Test validation when feature missing from observations."""
        
        class TestSelector(FeatureSelectionStrategy):
            def select_features(self, simulation_results, observations, iteration=1):
                return ['feature1']
            
            def get_strategy_name(self):
                return "Test Selector"
        
        selector = TestSelector()
        simulation_results, observations = sample_data
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            valid_features = selector.validate_features(
                ['feature1', 'feature4'], simulation_results, observations
            )
            
            assert len(w) == 1
            assert "not found in observations" in str(w[0].message)
            assert valid_features == ['feature1']
    
    def test_validate_features_no_valid_features(self, sample_data):
        """Test validation when no features are valid."""
        
        class TestSelector(FeatureSelectionStrategy):
            def select_features(self, simulation_results, observations, iteration=1):
                return []
            
            def get_strategy_name(self):
                return "Test Selector"
        
        selector = TestSelector()
        simulation_results, observations = sample_data
        
        with pytest.raises(ValueError, match="No valid features selected"):
            selector.validate_features(
                ['missing1', 'missing2'], simulation_results, observations
            )


class TestManualFeatureSelection:
    """Test Manual Feature Selection strategy."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample simulation results and observations."""
        simulation_results = pd.DataFrame({
            'infections': [100, 200, 300],
            'deaths': [10, 20, 30],
            'hospitalizations': [50, 100, 150]
        })
        
        observations = ObservationData({
            'infections': (200, 50**2),  # (mean, variance)
            'deaths': (20, 5**2),
            'hospitalizations': (100, 25**2)
        })
        
        return simulation_results, observations
    
    def test_initialization_single_feature(self):
        """Test initialization with single feature."""
        selector = ManualFeatureSelection('infections')
        assert selector.selected_features == ['infections']
    
    def test_initialization_multiple_features(self):
        """Test initialization with multiple features."""
        selector = ManualFeatureSelection(['infections', 'deaths'])
        assert selector.selected_features == ['infections', 'deaths']
    
    def test_initialization_empty_features(self):
        """Test initialization with empty features raises error."""
        with pytest.raises(ValueError, match="Must provide at least one feature"):
            ManualFeatureSelection([])
    
    def test_select_features(self, sample_data):
        """Test feature selection."""
        simulation_results, observations = sample_data
        
        selector = ManualFeatureSelection(['infections', 'deaths'])
        selected = selector.select_features(simulation_results, observations, iteration=1)
        
        assert selected == ['infections', 'deaths']
    
    def test_select_features_with_invalid(self, sample_data):
        """Test feature selection with some invalid features."""
        simulation_results, observations = sample_data
        
        selector = ManualFeatureSelection(['infections', 'invalid_feature'])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            selected = selector.select_features(simulation_results, observations, iteration=1)
            
            assert len(w) == 1
            assert selected == ['infections']
    
    def test_get_strategy_name(self):
        """Test strategy name generation."""
        selector = ManualFeatureSelection(['infections', 'deaths'])
        assert selector.get_strategy_name() == "Manual Selection (2 features)"


class TestAutoFeatureSelection:
    """Test Automatic Feature Selection strategy."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample simulation results and observations with varying statistics."""
        np.random.seed(42)  # For reproducible test data
        simulation_results = pd.DataFrame({
            'high_var': np.random.normal(0, 10, 100),      # High variance
            'low_var': np.random.normal(0, 1, 100),        # Low variance
            'high_mean': np.random.normal(100, 5, 100),    # High mean
            'low_mean': np.random.normal(1, 2, 100),       # Low mean
            'correlated': np.random.normal(0, 10, 100),    # Will be correlated with high_var
        })
        # Make correlated feature actually correlated
        simulation_results['correlated'] = simulation_results['high_var'] * 0.9 + np.random.normal(0, 1, 100)
        
        observations = ObservationData({
            'high_var': (0, 5**2),  # (mean, variance)
            'low_var': (0, 1**2),
            'high_mean': (100, 10**2),
            'low_mean': (1, 2**2),
            'correlated': (0, 5**2)
        })
        
        return simulation_results, observations
    
    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        selector = AutoFeatureSelection()
        assert selector.method == 'fano'
        assert selector.threshold is None
        assert selector.cooldown_period == 5
        assert selector.correlation_threshold == 0.8
        assert selector.max_features == 1
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        selector = AutoFeatureSelection(
            method='var',
            threshold=0.5,
            cooldown_period=3,
            correlation_threshold=0.7,
            max_features=2
        )
        assert selector.method == 'var'
        assert selector.threshold == 0.5
        assert selector.cooldown_period == 3
        assert selector.correlation_threshold == 0.7
        assert selector.max_features == 2
    
    @patch('history_matching.strategies.feature_selection.features.Statistics.fano')
    def test_select_features_fano_method(self, mock_fano, sample_data):
        """Test feature selection using fano method."""
        simulation_results, observations = sample_data
        
        # Mock fano statistics
        mock_fano.return_value = pd.DataFrame({
            'fano': pd.Series([10.0, 1.0, 2.0, 0.5, 9.0], 
                             index=['high_var', 'low_var', 'high_mean', 'low_mean', 'correlated'])
        })
        
        selector = AutoFeatureSelection(method='fano', max_features=1)
        selected = selector.select_features(simulation_results, observations, iteration=1)
        
        mock_fano.assert_called_once()
        assert len(selected) == 1
        assert selected[0] == 'high_var'  # Should select highest fano factor
    
    def test_select_features_variance_method(self, sample_data):
        """Test feature selection using variance method."""
        simulation_results, observations = sample_data
        
        selector = AutoFeatureSelection(method='var', max_features=1)
        selected = selector.select_features(simulation_results, observations, iteration=1)
        
        assert len(selected) == 1
        # Should select the feature with highest variance
        variances = simulation_results.var()
        expected_feature = variances.idxmax()
        assert selected[0] == expected_feature
    
    def test_select_features_mean_method(self, sample_data):
        """Test feature selection using mean method."""
        simulation_results, observations = sample_data
        
        selector = AutoFeatureSelection(method='mean', max_features=1)
        selected = selector.select_features(simulation_results, observations, iteration=1)
        
        assert len(selected) == 1
        # Should select the feature with highest absolute mean
        abs_means = simulation_results.mean().abs()
        expected_feature = abs_means.idxmax()
        assert selected[0] == expected_feature
    
    def test_select_features_unknown_method_fallback(self, sample_data):
        """Test fallback to variance when unknown method is used."""
        simulation_results, observations = sample_data
        
        selector = AutoFeatureSelection(method='unknown_method', max_features=1)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            selected = selector.select_features(simulation_results, observations, iteration=1)
            
            assert len(w) == 1
            assert "Falling back to variance" in str(w[0].message)
            assert len(selected) == 1
    
    def test_select_features_with_threshold(self, sample_data):
        """Test feature selection with threshold filtering."""
        simulation_results, observations = sample_data
        
        # Set a high threshold that only high variance features can meet
        high_threshold = simulation_results.var().quantile(0.8)
        selector = AutoFeatureSelection(method='var', threshold=high_threshold, max_features=2)
        selected = selector.select_features(simulation_results, observations, iteration=1)
        
        # Should only select features above threshold
        for feature in selected:
            assert simulation_results[feature].var() >= high_threshold
    
    def test_select_features_correlation_filtering(self, sample_data):
        """Test that highly correlated features are filtered out."""
        simulation_results, observations = sample_data
        
        # Reset history to ensure clean test
        AutoFeatureSelection._global_history.clear()
        
        selector = AutoFeatureSelection(method='var', max_features=3, correlation_threshold=0.5)
        selected = selector.select_features(simulation_results, observations, iteration=1)
        
        # Check that no two selected features are highly correlated
        for i, feat1 in enumerate(selected):
            for j, feat2 in enumerate(selected[i+1:], i+1):
                correlation = simulation_results[feat1].corr(simulation_results[feat2])
                assert abs(correlation) < 0.5
    
    def test_select_features_history_tracking(self, sample_data):
        """Test that history tracking prevents immediate re-selection."""
        simulation_results, observations = sample_data
        
        # Reset history
        AutoFeatureSelection._global_history.clear()
        
        selector = AutoFeatureSelection(method='var', max_features=1)
        
        # First selection
        selected1 = selector.select_features(simulation_results, observations, iteration=1)
        
        # Second selection should avoid the first selected feature
        selected2 = selector.select_features(simulation_results, observations, iteration=2)
        
        assert selected1 != selected2
        assert selected1[0] not in selected2
    
    def test_select_features_no_common_features(self):
        """Test behavior when no common features exist."""
        simulation_results = pd.DataFrame({'sim_only': [1, 2, 3]})
        observations = ObservationData({'obs_only': (1, 0.1**2)})  # (mean, variance)
        
        selector = AutoFeatureSelection()
        
        with pytest.raises(ValueError, match="No common features"):
            selector.select_features(simulation_results, observations, iteration=1)
    
    def test_reset_history(self):
        """Test resetting selection history."""
        selector = AutoFeatureSelection()
        
        # Add some history
        AutoFeatureSelection._global_history.extend(['feature1', 'feature2'])
        assert len(AutoFeatureSelection._global_history) == 2
        
        # Reset history
        selector.reset_history()
        assert len(AutoFeatureSelection._global_history) == 0
    
    def test_get_strategy_name(self):
        """Test strategy name generation."""
        selector = AutoFeatureSelection(method='var', max_features=2)
        assert selector.get_strategy_name() == "Auto Selection (method=var, max=2)"


class TestInteractiveFeatureSelection:
    """Test Interactive Feature Selection strategy."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample simulation results and observations."""
        simulation_results = pd.DataFrame({
            'infections': [100, 200, 300],
            'deaths': [10, 20, 30],
            'hospitalizations': [50, 100, 150]
        })
        
        observations = ObservationData({
            'infections': (200, 50**2),  # (mean, variance)
            'deaths': (20, 5**2),
            'hospitalizations': (100, 25**2)
        })
        
        return simulation_results, observations
    
    def test_initialization_default_fallback(self):
        """Test initialization with default fallback strategy."""
        selector = InteractiveFeatureSelection()
        assert isinstance(selector.fallback_strategy, AutoFeatureSelection)
    
    def test_initialization_custom_fallback(self):
        """Test initialization with custom fallback strategy."""
        fallback = ManualFeatureSelection(['infections'])
        selector = InteractiveFeatureSelection(fallback_strategy=fallback)
        assert selector.fallback_strategy is fallback
    
    @patch('sys.stdin.isatty', return_value=False)
    def test_select_features_non_interactive(self, mock_isatty, sample_data):
        """Test feature selection in non-interactive environment."""
        simulation_results, observations = sample_data
        
        fallback = ManualFeatureSelection(['infections'])
        selector = InteractiveFeatureSelection(fallback_strategy=fallback)
        
        selected = selector.select_features(simulation_results, observations, iteration=1)
        
        assert selected == ['infections']
        mock_isatty.assert_called_once()
    
    @patch('sys.stdin.isatty', return_value=True)
    @patch('builtins.input', return_value='auto')
    def test_select_features_interactive_auto(self, mock_input, mock_isatty, sample_data):
        """Test interactive selection with 'auto' input."""
        simulation_results, observations = sample_data
        
        fallback = ManualFeatureSelection(['infections'])
        selector = InteractiveFeatureSelection(fallback_strategy=fallback)
        
        selected = selector.select_features(simulation_results, observations, iteration=1)
        
        assert selected == ['infections']
        mock_input.assert_called_once()
    
    @patch('sys.stdin.isatty', return_value=True)
    @patch('builtins.input', return_value='infections, deaths')
    def test_select_features_interactive_manual(self, mock_input, mock_isatty, sample_data):
        """Test interactive selection with manual input."""
        simulation_results, observations = sample_data
        
        selector = InteractiveFeatureSelection()
        selected = selector.select_features(simulation_results, observations, iteration=1)
        
        assert set(selected) == {'infections', 'deaths'}
        mock_input.assert_called_once()
    
    @patch('sys.stdin.isatty', return_value=True)
    @patch('builtins.input', side_effect=['', 'invalid_feature', 'infections'])
    def test_select_features_interactive_retry(self, mock_input, mock_isatty, sample_data):
        """Test interactive selection with retry on invalid input."""
        simulation_results, observations = sample_data
        
        selector = InteractiveFeatureSelection()
        selected = selector.select_features(simulation_results, observations, iteration=1)
        
        assert selected == ['infections']
        assert mock_input.call_count == 3
    
    @patch('sys.stdin.isatty', side_effect=OSError("Not available"))
    def test_select_features_exception_fallback(self, mock_isatty, sample_data):
        """Test fallback when interactive selection fails."""
        simulation_results, observations = sample_data
        
        fallback = ManualFeatureSelection(['infections'])
        selector = InteractiveFeatureSelection(fallback_strategy=fallback)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            selected = selector.select_features(simulation_results, observations, iteration=1)
            
            assert len(w) == 1
            assert "Interactive selection not available" in str(w[0].message)
            assert selected == ['infections']
    
    def test_get_strategy_name(self):
        """Test strategy name generation."""
        fallback = ManualFeatureSelection(['infections'])
        selector = InteractiveFeatureSelection(fallback_strategy=fallback)
        
        expected = "Interactive Selection (fallback: Manual Selection (1 features))"
        assert selector.get_strategy_name() == expected


class TestMultiFeatureSelection:
    """Test Multi-Feature Selection strategy."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample simulation results and observations."""
        np.random.seed(42)
        simulation_results = pd.DataFrame({
            'feature1': np.random.normal(0, 5, 50),
            'feature2': np.random.normal(0, 3, 50),
            'feature3': np.random.normal(0, 1, 50),
            'feature4': np.random.normal(0, 2, 50),
        })
        
        observations = ObservationData({
            'feature1': (0, 1**2),  # (mean, variance)
            'feature2': (0, 1**2),
            'feature3': (0, 1**2),
            'feature4': (0, 1**2)
        })
        
        return simulation_results, observations
    
    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        selector = MultiFeatureSelection()
        assert selector.auto_selector.max_features == 2
        assert selector.auto_selector.method == 'fano'
        assert selector.auto_selector.correlation_threshold == 0.5
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        selector = MultiFeatureSelection(n_features=3, method='var', correlation_threshold=0.7)
        assert selector.auto_selector.max_features == 3
        assert selector.auto_selector.method == 'var'
        assert selector.auto_selector.correlation_threshold == 0.7
    
    def test_select_features_multiple(self, sample_data):
        """Test selection of multiple features."""
        simulation_results, observations = sample_data
        
        # Reset history for clean test
        AutoFeatureSelection._global_history.clear()
        
        selector = MultiFeatureSelection(n_features=2, method='var')
        selected = selector.select_features(simulation_results, observations, iteration=1)
        
        assert len(selected) <= 2  # May be fewer due to correlation filtering
        assert len(selected) > 0
        
        # All selected features should be valid
        for feature in selected:
            assert feature in simulation_results.columns
            assert observations.has_feature(feature)
    
    def test_get_strategy_name(self):
        """Test strategy name generation."""
        selector = MultiFeatureSelection(n_features=3, method='var')
        expected = "Multi Selection (method=var, max=3)"
        assert selector.get_strategy_name() == expected


class TestFeatureSelectionIntegration:
    """Integration tests for feature selection strategies."""
    
    @pytest.fixture
    def realistic_data(self):
        """Create realistic simulation results and observations."""
        np.random.seed(42)
        
        # Simulate epidemic data with different characteristics
        n_samples = 100
        simulation_results = pd.DataFrame({
            'infections': np.random.poisson(1000, n_samples),      # Count data, high variance
            'deaths': np.random.poisson(50, n_samples),           # Count data, lower variance
            'hospitalizations': np.random.poisson(200, n_samples), # Count data, medium variance
            'incidence_rate': np.random.beta(2, 5, n_samples),    # Rate data, bounded [0,1]
            'reproduction_number': np.random.gamma(2, 0.5, n_samples), # Continuous, positive
        })
        
        observations = ObservationData({
            'infections': (1000, 100**2),  # (mean, variance)
            'deaths': (50, 10**2),
            'hospitalizations': (200, 50**2),
            'incidence_rate': (0.3, 0.1**2),
            'reproduction_number': (1.0, 0.3**2)
        })
        
        return simulation_results, observations
    
    def test_all_strategies_work_with_realistic_data(self, realistic_data):
        """Test that all strategies work with realistic data."""
        simulation_results, observations = realistic_data
        
        strategies = [
            ManualFeatureSelection(['infections']),
            AutoFeatureSelection(method='var', max_features=1),
            MultiFeatureSelection(n_features=2, method='var'),
            InteractiveFeatureSelection(ManualFeatureSelection(['deaths']))
        ]
        
        for strategy in strategies:
            selected = strategy.select_features(simulation_results, observations, iteration=1)
            
            # Basic validation
            assert isinstance(selected, list)
            assert len(selected) > 0
            assert all(isinstance(f, str) for f in selected)
            
            # All features should exist in both datasets
            for feature in selected:
                assert feature in simulation_results.columns
                assert observations.has_feature(feature)
    
    def test_feature_selection_consistency(self, realistic_data):
        """Test that feature selection is consistent across calls with same parameters."""
        simulation_results, observations = realistic_data
        
        # Manual selection should be completely consistent
        manual_selector = ManualFeatureSelection(['infections', 'deaths'])
        selected1 = manual_selector.select_features(simulation_results, observations, iteration=1)
        selected2 = manual_selector.select_features(simulation_results, observations, iteration=2)
        assert selected1 == selected2
        
        # Auto selection should be consistent within same iteration (no history effects)
        AutoFeatureSelection._global_history.clear()
        auto_selector = AutoFeatureSelection(method='var', max_features=1)
        selected1 = auto_selector.select_features(simulation_results, observations, iteration=1)
        
        AutoFeatureSelection._global_history.clear()
        selected2 = auto_selector.select_features(simulation_results, observations, iteration=1)
        assert selected1 == selected2
    
    def test_feature_selection_with_edge_cases(self):
        """Test feature selection with edge cases."""
        # Single feature case
        single_feature_results = pd.DataFrame({'only_feature': [1, 2, 3, 4, 5]})
        single_feature_obs = ObservationData({'only_feature': (3, 1**2)})  # (mean, variance)
        
        selector = AutoFeatureSelection(max_features=2)
        selected = selector.select_features(single_feature_results, single_feature_obs, iteration=1)
        assert selected == ['only_feature']
        
        # All features identical values
        identical_results = pd.DataFrame({
            'feat1': [1.0] * 10,
            'feat2': [1.0] * 10,
            'feat3': [1.0] * 10
        })
        identical_obs = ObservationData({
            'feat1': (1, 0.1**2),  # (mean, variance)
            'feat2': (1, 0.1**2),
            'feat3': (1, 0.1**2)
        })
        
        selector = AutoFeatureSelection(method='var', max_features=2)
        selected = selector.select_features(identical_results, identical_obs, iteration=1)
        assert len(selected) == 1  # Should pick one when all are identical