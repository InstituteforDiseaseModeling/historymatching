"""
Unit tests for feature selection strategies.
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from unittest.mock import patch, MagicMock
import historymatching as hm



class TestFeatureSelectionStrategy:
    """Test the abstract FeatureSelectionStrategy base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that FeatureSelectionStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            hm.FeatureSelectionStrategy()
    
    def test_subclass_must_implement_abstract_methods(self):
        """Test that subclasses must implement abstract methods."""
        
        class IncompleteSelector(hm.FeatureSelectionStrategy):
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
        
        observations = hm.ObservationData({
            'feature1': (2.5, 0.5),  # (mean, std)
            'feature2': (2.0, 0.3),
            'feature3': (25.0, 5.0)
        })
        
        return simulation_results, observations
    
    def test_validate_features_all_valid(self, sample_data):
        """Test validation when all features are valid."""
        
        class TestSelector(hm.FeatureSelectionStrategy):
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
        
        class TestSelector(hm.FeatureSelectionStrategy):
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
        
        class TestSelector(hm.FeatureSelectionStrategy):
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
        
        class TestSelector(hm.FeatureSelectionStrategy):
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
        
        observations = hm.ObservationData({
            'infections': (200, 50),  # (mean, std)
            'deaths': (20, 5),
            'hospitalizations': (100, 25)
        })
        
        return simulation_results, observations
    
    def test_initialization_single_feature(self):
        """Test initialization with single feature."""
        selector = hm.ManualFeatureSelection('infections')
        assert selector.selected_features == ['infections']
    
    def test_initialization_multiple_features(self):
        """Test initialization with multiple features."""
        selector = hm.ManualFeatureSelection(['infections', 'deaths'])
        assert selector.selected_features == ['infections', 'deaths']
    
    def test_initialization_empty_features(self):
        """Test initialization with empty features raises error."""
        with pytest.raises(ValueError, match="Must provide at least one feature"):
            hm.ManualFeatureSelection([])
    
    def test_select_features(self, sample_data):
        """Test feature selection."""
        simulation_results, observations = sample_data
        
        selector = hm.ManualFeatureSelection(['infections', 'deaths'])
        selected = selector.select_features(simulation_results, observations, iteration=1)
        
        assert selected == ['infections', 'deaths']
    
    def test_select_features_with_invalid(self, sample_data):
        """Test feature selection with some invalid features."""
        simulation_results, observations = sample_data
        
        selector = hm.ManualFeatureSelection(['infections', 'invalid_feature'])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            selected = selector.select_features(simulation_results, observations, iteration=1)
            
            assert len(w) == 1
            assert selected == ['infections']
    
    def test_get_strategy_name(self):
        """Test strategy name generation."""
        selector = hm.ManualFeatureSelection(['infections', 'deaths'])
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
        
        observations = hm.ObservationData({
            'high_var': (0, 5),  # (mean, std)
            'low_var': (0, 1),
            'high_mean': (100, 10),
            'low_mean': (1, 2),
            'correlated': (0, 5)
        })
        
        return simulation_results, observations
    
    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        selector = hm.AutoFeatureSelection()
        assert selector.method == 'mean_sq_z'
        assert selector.threshold is None
        assert selector.cooldown_period == 1
        assert selector.correlation_threshold == 0.8
        assert selector.max_features == 1
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        selector = hm.AutoFeatureSelection(
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
    
    def test_select_features_fano_method(self, sample_data):
        """Test feature selection using fano method.

        Fano is computed on z-scores (normalized by observation uncertainty)
        so that features on different scales are comparable.  The feature with
        the highest z-score Fano factor should be selected — it has the most
        spread relative to the target tolerance.
        """
        simulation_results, observations = sample_data

        selector = hm.AutoFeatureSelection(method='fano', max_features=1)
        selected = selector.select_features(simulation_results, observations, iteration=1)

        # Should select one feature
        assert len(selected) == 1

        # Compute expected z-score Fano factors manually
        expected_fano = {}
        for feature in simulation_results.columns:
            if observations.has_feature(feature):
                obs_mean, obs_std = observations.get_target_for_feature(feature)
                z = (simulation_results[feature] - obs_mean) / obs_std
                z_mean = z.mean()
                z_var = z.var()
                if abs(z_mean) > 1e-10:
                    expected_fano[feature] = z_var / abs(z_mean)
                else:
                    expected_fano[feature] = z_var

        # The selected feature should have the highest finite fano factor
        finite_fano = {k: v for k, v in expected_fano.items() if np.isfinite(v)}
        if finite_fano:
            expected_best = max(finite_fano.keys(), key=lambda k: finite_fano[k])
            assert selected[0] == expected_best
    
    def test_select_features_variance_method(self, sample_data):
        """Test feature selection using variance method (on z-scores)."""
        simulation_results, observations = sample_data

        selector = hm.AutoFeatureSelection(method='var', max_features=1)
        selected = selector.select_features(simulation_results, observations, iteration=1)

        assert len(selected) == 1
        # Should select the feature with highest z-score variance
        z_vars = {}
        for feature in simulation_results.columns:
            if observations.has_feature(feature):
                obs_mean, obs_std = observations.get_target_for_feature(feature)
                z_vars[feature] = ((simulation_results[feature] - obs_mean) / obs_std).var()
        expected_feature = max(z_vars, key=z_vars.get)
        assert selected[0] == expected_feature
    
    def test_select_features_mean_method(self, sample_data):
        """Test feature selection using mean method (on z-scores)."""
        simulation_results, observations = sample_data

        selector = hm.AutoFeatureSelection(method='mean', max_features=1)
        selected = selector.select_features(simulation_results, observations, iteration=1)

        assert len(selected) == 1
        # Should select the feature with highest absolute z-score mean (most biased)
        z_means = {}
        for feature in simulation_results.columns:
            if observations.has_feature(feature):
                obs_mean, obs_std = observations.get_target_for_feature(feature)
                z_means[feature] = abs(((simulation_results[feature] - obs_mean) / obs_std).mean())
        expected_feature = max(z_means, key=z_means.get)
        assert selected[0] == expected_feature
    
    def test_select_features_unknown_method_fallback(self, sample_data):
        """Test fallback to variance when unknown method is used."""
        simulation_results, observations = sample_data
        
        selector = hm.AutoFeatureSelection(method='unknown_method', max_features=1)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            selected = selector.select_features(simulation_results, observations, iteration=1)
            
            assert len(w) == 1
            assert "Falling back to variance" in str(w[0].message)
            assert len(selected) == 1
    
    def test_select_features_with_threshold(self, sample_data):
        """Test feature selection with threshold filtering (on z-scores)."""
        simulation_results, observations = sample_data

        # Compute z-score variances to set a meaningful threshold
        z_vars = {}
        for feature in simulation_results.columns:
            if observations.has_feature(feature):
                obs_mean, obs_std = observations.get_target_for_feature(feature)
                z_vars[feature] = ((simulation_results[feature] - obs_mean) / obs_std).var()

        high_threshold = pd.Series(z_vars).quantile(0.8)
        selector = hm.AutoFeatureSelection(method='var', threshold=high_threshold, max_features=2)
        selected = selector.select_features(simulation_results, observations, iteration=1)

        # Should only select features whose z-score variance is above threshold
        for feature in selected:
            assert z_vars[feature] >= high_threshold
    
    def test_select_features_correlation_filtering(self, sample_data):
        """Test that highly correlated features are filtered out."""
        simulation_results, observations = sample_data
        
        # Reset history to ensure clean test
        selector = hm.AutoFeatureSelection(method='var', max_features=3, correlation_threshold=0.5)
        selected = selector.select_features(simulation_results, observations, iteration=1)
        
        # Check that no two selected features are highly correlated
        for i, feat1 in enumerate(selected):
            for j, feat2 in enumerate(selected[i+1:], i+1):
                correlation = simulation_results[feat1].corr(simulation_results[feat2])
                assert abs(correlation) < 0.5
    
    def test_select_features_history_tracking(self, sample_data):
        """Test that history tracking prevents immediate re-selection."""
        simulation_results, observations = sample_data
        
        selector = hm.AutoFeatureSelection(method='var', max_features=1)
        
        # First selection
        selected1 = selector.select_features(simulation_results, observations, iteration=1)
        
        # Second selection should avoid the first selected feature (using same selector instance)
        selected2 = selector.select_features(simulation_results, observations, iteration=2)
        
        assert selected1 != selected2
        assert selected1[0] not in selected2
    
    def test_select_features_no_common_features(self):
        """Test behavior when no common features exist."""
        simulation_results = pd.DataFrame({'sim_only': [1, 2, 3]})
        observations = hm.ObservationData({'obs_only': (1, 0.1)})  # (mean, std)
        
        selector = hm.AutoFeatureSelection()

        # The error names the mismatch so a beginner can fix it.
        with pytest.raises(ValueError, match="None of the simulator's output columns match"):
            selector.select_features(simulation_results, observations, iteration=1)
    
    def test_reset_history(self):
        """Test resetting selection history."""
        selector = hm.AutoFeatureSelection()
        
        # Add some history to the instance
        selector.history.extend(['feature1', 'feature2'])
        assert len(selector.history) == 2
        
        # Reset history
        selector.reset_history()
        assert len(selector.history) == 0
    
    def test_get_strategy_name(self):
        """Test strategy name generation."""
        selector = hm.AutoFeatureSelection(method='var', max_features=2)
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
        
        observations = hm.ObservationData({
            'infections': (200, 50),  # (mean, std)
            'deaths': (20, 5),
            'hospitalizations': (100, 25)
        })
        
        return simulation_results, observations
    
    def test_initialization_default_fallback(self):
        """Test initialization with default fallback strategy."""
        selector = hm.InteractiveFeatureSelection()
        assert isinstance(selector.fallback_strategy, hm.AutoFeatureSelection)
    
    def test_initialization_custom_fallback(self):
        """Test initialization with custom fallback strategy."""
        fallback = hm.ManualFeatureSelection(['infections'])
        selector = hm.InteractiveFeatureSelection(fallback_strategy=fallback)
        assert selector.fallback_strategy is fallback
    
    @patch('sys.stdin.isatty', return_value=False)
    def test_select_features_non_interactive(self, mock_isatty, sample_data):
        """Test feature selection in non-interactive environment."""
        simulation_results, observations = sample_data
        
        fallback = hm.ManualFeatureSelection(['infections'])
        selector = hm.InteractiveFeatureSelection(fallback_strategy=fallback)
        
        selected = selector.select_features(simulation_results, observations, iteration=1)
        
        assert selected == ['infections']
        mock_isatty.assert_called_once()
    
    @patch('sys.stdin.isatty', return_value=True)
    @patch('builtins.input', return_value='auto')
    def test_select_features_interactive_auto(self, mock_input, mock_isatty, sample_data):
        """Test interactive selection with 'auto' input."""
        simulation_results, observations = sample_data
        
        fallback = hm.ManualFeatureSelection(['infections'])
        selector = hm.InteractiveFeatureSelection(fallback_strategy=fallback)
        
        selected = selector.select_features(simulation_results, observations, iteration=1)
        
        assert selected == ['infections']
        mock_input.assert_called_once()
    
    @patch('sys.stdin.isatty', return_value=True)
    @patch('builtins.input', return_value='infections, deaths')
    def test_select_features_interactive_manual(self, mock_input, mock_isatty, sample_data):
        """Test interactive selection with manual input."""
        simulation_results, observations = sample_data
        
        selector = hm.InteractiveFeatureSelection()
        selected = selector.select_features(simulation_results, observations, iteration=1)
        
        assert set(selected) == {'infections', 'deaths'}
        mock_input.assert_called_once()
    
    @patch('sys.stdin.isatty', return_value=True)
    @patch('builtins.input', side_effect=['', 'invalid_feature', 'infections'])
    def test_select_features_interactive_retry(self, mock_input, mock_isatty, sample_data):
        """Test interactive selection with retry on invalid input."""
        simulation_results, observations = sample_data
        
        selector = hm.InteractiveFeatureSelection()
        selected = selector.select_features(simulation_results, observations, iteration=1)
        
        assert selected == ['infections']
        assert mock_input.call_count == 3
    
    @patch('sys.stdin.isatty', side_effect=OSError("Not available"))
    def test_select_features_exception_fallback(self, mock_isatty, sample_data):
        """Test fallback when interactive selection fails."""
        simulation_results, observations = sample_data
        
        fallback = hm.ManualFeatureSelection(['infections'])
        selector = hm.InteractiveFeatureSelection(fallback_strategy=fallback)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            selected = selector.select_features(simulation_results, observations, iteration=1)
            
            assert len(w) == 1
            assert "Interactive selection not available" in str(w[0].message)
            assert selected == ['infections']
    
    def test_get_strategy_name(self):
        """Test strategy name generation."""
        fallback = hm.ManualFeatureSelection(['infections'])
        selector = hm.InteractiveFeatureSelection(fallback_strategy=fallback)
        
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
        
        observations = hm.ObservationData({
            'feature1': (0, 1.0),  # (mean, std)
            'feature2': (0, 1.0),
            'feature3': (0, 1.0),
            'feature4': (0, 1.0)
        })
        
        return simulation_results, observations
    
    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        selector = hm.MultiFeatureSelection()
        assert selector.auto_selector.max_features == 2
        assert selector.auto_selector.method == 'mean_sq_z'
        assert selector.auto_selector.correlation_threshold == 0.5
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        selector = hm.MultiFeatureSelection(n_features=3, method='var', correlation_threshold=0.7)
        assert selector.auto_selector.max_features == 3
        assert selector.auto_selector.method == 'var'
        assert selector.auto_selector.correlation_threshold == 0.7
    
    def test_select_features_multiple(self, sample_data):
        """Test selection of multiple features."""
        simulation_results, observations = sample_data
        
        # Reset history for clean test
        selector = hm.MultiFeatureSelection(n_features=2, method='var')
        selected = selector.select_features(simulation_results, observations, iteration=1)
        
        assert len(selected) <= 2  # May be fewer due to correlation filtering
        assert len(selected) > 0
        
        # All selected features should be valid
        for feature in selected:
            assert feature in simulation_results.columns
            assert observations.has_feature(feature)
    
    def test_get_strategy_name(self):
        """Test strategy name generation."""
        selector = hm.MultiFeatureSelection(n_features=3, method='var')
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
        
        observations = hm.ObservationData({
            'infections': (1000, 100),  # (mean, std)
            'deaths': (50, 10),
            'hospitalizations': (200, 50),
            'incidence_rate': (0.3, 0.1),
            'reproduction_number': (1.0, 0.3)
        })
        
        return simulation_results, observations
    
    def test_all_strategies_work_with_realistic_data(self, realistic_data):
        """Test that all strategies work with realistic data."""
        simulation_results, observations = realistic_data
        
        strategies = [
            hm.ManualFeatureSelection(['infections']),
            hm.AutoFeatureSelection(method='var', max_features=1),
            hm.MultiFeatureSelection(n_features=2, method='var'),
            hm.InteractiveFeatureSelection(hm.ManualFeatureSelection(['deaths']))
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
        manual_selector = hm.ManualFeatureSelection(['infections', 'deaths'])
        selected1 = manual_selector.select_features(simulation_results, observations, iteration=1)
        selected2 = manual_selector.select_features(simulation_results, observations, iteration=2)
        assert selected1 == selected2
        
        # Auto selection should be consistent with same starting conditions
        auto_selector1 = hm.AutoFeatureSelection(method='var', max_features=1)
        selected1 = auto_selector1.select_features(simulation_results, observations, iteration=1)
        
        auto_selector2 = hm.AutoFeatureSelection(method='var', max_features=1)
        selected2 = auto_selector2.select_features(simulation_results, observations, iteration=1)
        assert selected1 == selected2
    
    def test_feature_selection_with_edge_cases(self):
        """Test feature selection with edge cases."""
        # Single feature case
        single_feature_results = pd.DataFrame({'only_feature': [1, 2, 3, 4, 5]})
        single_feature_obs = hm.ObservationData({'only_feature': (3, 1.0)})  # (mean, std)
        
        selector = hm.AutoFeatureSelection(max_features=2)
        selected = selector.select_features(single_feature_results, single_feature_obs, iteration=1)
        assert selected == ['only_feature']
        
        # All features identical values
        identical_results = pd.DataFrame({
            'feat1': [1.0] * 10,
            'feat2': [1.0] * 10,
            'feat3': [1.0] * 10
        })
        identical_obs = hm.ObservationData({
            'feat1': (1, 0.1),  # (mean, std)
            'feat2': (1, 0.1),
            'feat3': (1, 0.1)
        })
        
        selector = hm.AutoFeatureSelection(method='var', max_features=2)
        selected = selector.select_features(identical_results, identical_obs, iteration=1)
        # When all features have identical variance (0), selector picks up to max_features
        # since they all tie in ranking
        assert len(selected) <= 2
        assert len(selected) >= 1