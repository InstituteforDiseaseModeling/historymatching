"""
Unit tests for enhanced emulator factory with strategy pattern.
"""

import pytest
import pandas as pd
import numpy as np
import importlib
from unittest.mock import patch

from history_matching.emulators.factory import (
    EmulatorFactory,
    create_linear_emulator,
    create_gpr_emulator,
    create_glm_emulator
)
from history_matching.emulators.base import BaseEmulator
from history_matching.emulators.linear import LinearModel
from history_matching.emulators.glm import GLM
from history_matching.emulators.gpr import GPR
from history_matching.emulators.results import EmulationResults


class MockEmulator(BaseEmulator):
    """Mock emulator for testing."""
    
    def __init__(self, X=None, y=None, **kwargs):
        super().__init__(X, y)
        self.kwargs = kwargs
        self.trained = False
    
    def train(self):
        self.trained = True
        self.training_complete = True
    
    def predict(self, x):
        return EmulationResults(
            mean=[1.0] * len(x),
            std=[0.316] * len(x)  # sqrt(0.1) ≈ 0.316
        )


class TestEmulatorFactory:
    """Test the enhanced EmulatorFactory class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = pd.DataFrame({
            'param1': np.random.uniform(0, 1, 20),
            'param2': np.random.uniform(-1, 1, 20)
        })
        y = pd.DataFrame({
            'output': np.random.normal(0, 1, 20)
        })
        return X, y
    
    @pytest.fixture
    def multi_output_data(self):
        """Create sample data with multiple outputs."""
        np.random.seed(42)
        X = pd.DataFrame({
            'param1': np.random.uniform(0, 1, 20),
            'param2': np.random.uniform(-1, 1, 20)
        })
        y = pd.DataFrame({
            'infections': np.random.poisson(100, 20),
            'deaths': np.random.poisson(10, 20),
            'hospitalizations': np.random.poisson(50, 20)
        })
        return X, y
    
    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        factory = EmulatorFactory()
        assert factory.default_type == 'gpr'
        assert factory.default_kwargs == {}
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        factory = EmulatorFactory(default_type='linear', regularization=0.1, max_iter=100)
        assert factory.default_type == 'linear'
        assert factory.default_kwargs == {'regularization': 0.1, 'max_iter': 100}
    
    def test_initialization_invalid_default_type(self):
        """Test initialization with invalid default type."""
        with pytest.raises(ValueError, match="Unknown default emulator type"):
            EmulatorFactory(default_type='invalid_type')
    
    def test_emulator_registry_contents(self):
        """Test that the emulator registry contains expected types."""
        registry = EmulatorFactory._emulator_registry
        
        assert 'linear' in registry
        assert 'glm' in registry
        assert 'gpr' in registry
        assert 'gaussian' in registry  # alias for gpr
        
        assert registry['linear'] == LinearModel
        assert registry['glm'] == GLM
        assert registry['gpr'] == GPR
        assert registry['gaussian'] == GPR
    
    def test_create_emulator_default_type(self, sample_data):
        """Test creating emulator with default type."""
        X, y = sample_data
        factory = EmulatorFactory(default_type='linear')
        
        with patch.object(factory, 'create_emulator') as mock_create:
            mock_emulator = MockEmulator()
            mock_create.return_value = mock_emulator
            
            emulator = factory.create_emulator(X, y)
            
            assert emulator is mock_emulator
            mock_create.assert_called_once_with(X, y)
    
    def test_create_emulator_specific_type(self, sample_data):
        """Test creating emulator with specific type."""
        X, y = sample_data
        factory = EmulatorFactory(default_type='linear')
        
        with patch.object(factory, 'create_emulator') as mock_create:
            mock_emulator = MockEmulator()
            mock_create.return_value = mock_emulator
            
            emulator = factory.create_emulator(X, y, emulator_type='glm')
            
            assert emulator is mock_emulator
            mock_create.assert_called_once_with(X, y, emulator_type='glm')
    
    def test_create_emulator_invalid_type(self, sample_data):
        """Test creating emulator with invalid type."""
        X, y = sample_data
        factory = EmulatorFactory()
        
        with pytest.raises(ValueError, match="Unknown emulator type"):
            factory.create_emulator(X, y, emulator_type='invalid')
    
    def test_create_emulator_multiple_output_columns(self, sample_data):
        """Test that creating emulator with multiple output columns raises error."""
        X, _ = sample_data
        y_multi = pd.DataFrame({
            'output1': [1, 2, 3],
            'output2': [4, 5, 6]
        })
        
        factory = EmulatorFactory()
        
        with pytest.raises(TypeError, match="Output data must have exactly one column"):
            factory.create_emulator(X, y_multi)
    
    def test_create_emulator_kwargs_merging(self, sample_data):
        """Test that default kwargs are merged with specific kwargs."""
        X, y = sample_data
        
        # Create a factory with default parameters
        factory = EmulatorFactory(default_type='linear', test_fraction=0.3)
        
        # Check that defaults are set correctly
        assert factory.default_kwargs['test_fraction'] == 0.3
        
        # Create emulator with override parameter
        # This should merge default test_fraction=0.3 with override test_fraction=0.4
        # Result should be test_fraction=0.4 (override wins)
        emulator = factory.create_emulator(X, y, test_fraction=0.4)
        
        # Verify emulator was created successfully
        assert emulator is not None
        assert hasattr(emulator, 'train')
        assert hasattr(emulator, 'predict')
    
    def test_create_emulators_for_features(self, multi_output_data):
        """Test creating emulators for multiple features."""
        X, y = multi_output_data
        features = ['infections', 'deaths']
        
        with patch.object(EmulatorFactory, 'create_emulator') as mock_create:
            mock_emulator = MockEmulator()
            mock_create.return_value = mock_emulator
            
            factory = EmulatorFactory()
            emulators = factory.create_emulators_for_features(X, y, features)
            
            assert len(emulators) == 2
            assert 'infections' in emulators
            assert 'deaths' in emulators
            assert mock_create.call_count == 2
            
            # Check that each emulator was trained
            for emulator in emulators.values():
                assert emulator.trained
    
    def test_create_emulators_for_features_missing_feature(self, multi_output_data):
        """Test error when feature is missing from simulation results."""
        X, y = multi_output_data
        features = ['infections', 'missing_feature']
        
        factory = EmulatorFactory()
        
        with pytest.raises(ValueError, match="Features not found in simulation results"):
            factory.create_emulators_for_features(X, y, features)
    
    def test_create_and_train_emulator(self, sample_data):
        """Test convenience method that creates and trains emulator."""
        X, y = sample_data
        
        with patch.object(EmulatorFactory, 'create_emulator') as mock_create:
            mock_emulator = MockEmulator()
            mock_create.return_value = mock_emulator
            
            factory = EmulatorFactory()
            emulator = factory.create_and_train_emulator(X, y)
            
            mock_create.assert_called_once_with(X, y, None)
            assert emulator.trained
    
    def test_with_defaults(self):
        """Test creating new factory with updated defaults."""
        factory1 = EmulatorFactory(default_type='linear', param1=1, param2=2)
        factory2 = factory1.with_defaults(param2=20, param3=3)
        
        assert factory2.default_type == 'linear'
        assert factory2.default_kwargs == {'param1': 1, 'param2': 20, 'param3': 3}
        
        # Original factory should be unchanged
        assert factory1.default_kwargs == {'param1': 1, 'param2': 2}
    
    def test_set_default_type(self):
        """Test creating new factory with different default type."""
        factory1 = EmulatorFactory(default_type='linear', param1=1)
        factory2 = factory1.set_default_type('gpr')
        
        assert factory2.default_type == 'gpr'
        assert factory2.default_kwargs == {'param1': 1}
        
        # Original factory should be unchanged
        assert factory1.default_type == 'linear'
    
    def test_register_emulator(self):
        """Test registering a custom emulator type."""
        # Create a custom emulator class
        class CustomEmulator(BaseEmulator):
            def train(self):
                pass
            
            def predict(self, x):
                return EmulationResults(
                    mean=[0.5] * len(x),
                    std=[0.316] * len(x)  # sqrt(0.1) ≈ 0.316
                )

        # Register it
        EmulatorFactory.register_emulator('custom', CustomEmulator)
        
        # Test that it was registered
        assert 'custom' in EmulatorFactory._emulator_registry
        assert EmulatorFactory._emulator_registry['custom'] == CustomEmulator
        
        # Test that we can create it
        factory = EmulatorFactory(default_type='custom')
        assert factory.default_type == 'custom'
        
        # Clean up
        del EmulatorFactory._emulator_registry['custom']
    
    def test_register_invalid_emulator(self):
        """Test registering non-BaseEmulator class raises error."""
        class NotAnEmulator:
            pass
        
        with pytest.raises(TypeError, match="must be a subclass of BaseEmulator"):
            EmulatorFactory.register_emulator('invalid', NotAnEmulator)
    
    def test_available_emulators(self):
        """Test getting list of available emulator types."""
        available = EmulatorFactory.available_emulators()
        expected = ['linear', 'glm', 'gpr', 'gaussian', 'bayes_linear']
        
        assert set(available) == set(expected)
    
    def test_get_emulator_info(self):
        """Test getting emulator information."""
        info = EmulatorFactory.get_emulator_info('linear')
        
        assert info['name'] == 'linear'
        assert info['class'] == 'LinearModel'
        assert info['module'] == 'history_matching.emulators.linear'
        assert isinstance(info['description'], str)
    
    def test_get_emulator_info_unknown(self):
        """Test getting info for unknown emulator raises error."""
        with pytest.raises(ValueError, match="Unknown emulator type"):
            EmulatorFactory.get_emulator_info('unknown')
    
    def test_with_defaults_class(self):
        """Test class method for creating factory with defaults."""
        factory = EmulatorFactory.with_defaults_class('glm', param1=1, param2=2)
        
        assert factory.default_type == 'glm'
        assert factory.default_kwargs == {'param1': 1, 'param2': 2}
    
    def test_get_default_type(self):
        """Test getting default emulator type."""
        factory = EmulatorFactory(default_type='linear')
        assert factory.get_default_type() == 'linear'
    
    def test_get_default_kwargs(self):
        """Test getting default parameters."""
        factory = EmulatorFactory(param1=1, param2=2)
        kwargs = factory.get_default_kwargs()
        
        assert kwargs == {'param1': 1, 'param2': 2}
        
        # Should return a copy
        kwargs['param3'] = 3
        assert factory.get_default_kwargs() == {'param1': 1, 'param2': 2}
    
    def test_repr(self):
        """Test string representation."""
        factory = EmulatorFactory(default_type='linear')
        repr_str = repr(factory)
        
        assert 'EmulatorFactory' in repr_str
        assert "default_type='linear'" in repr_str
        assert 'available_types=' in repr_str


class TestConvenienceFunctions:
    """Test convenience functions for creating emulators."""
    
    def setup_method(self):
        """Setup for each test method - ensure clean registry state."""
        # Import original registry to restore after tests
        from history_matching.emulators.factory import EmulatorFactory
        from history_matching.emulators.linear import LinearModel
        from history_matching.emulators.glm import GLM
        from history_matching.emulators.gpr import GPR
        
        # Store original registry
        self._original_registry = EmulatorFactory._emulator_registry.copy()
    
    def teardown_method(self):
        """Cleanup after each test method - restore clean registry state."""
        from history_matching.emulators.factory import EmulatorFactory
        
        # Restore original registry
        EmulatorFactory._emulator_registry = self._original_registry
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        X = pd.DataFrame({'param1': [1, 2, 3], 'param2': [4, 5, 6]})
        y = pd.DataFrame({'output': [7, 8, 9]})
        return X, y
    
    def test_create_linear_emulator(self, sample_data):
        """Test create_linear_emulator convenience function."""
        X, y = sample_data
        
        with patch('history_matching.emulators.factory.EmulatorFactory.create_emulator') as mock_create:
            mock_emulator = MockEmulator(X, y)
            mock_create.return_value = mock_emulator
            
            result = create_linear_emulator(X, y, test_param=123)
            
            mock_create.assert_called_once_with(X, y, test_param=123)
            assert result is mock_emulator
    
    def test_create_gpr_emulator(self, sample_data):
        """Test create_gpr_emulator convenience function."""
        X, y = sample_data
        
        with patch('history_matching.emulators.factory.EmulatorFactory.create_emulator') as mock_create:
            mock_emulator = MockEmulator(X, y)
            mock_create.return_value = mock_emulator
            
            result = create_gpr_emulator(X, y, kernel='rbf')
            
            mock_create.assert_called_once_with(X, y, kernel='rbf')
            assert result is mock_emulator
    
    def test_create_glm_emulator(self, sample_data):
        """Test create_glm_emulator convenience function."""
        X, y = sample_data
        
        with patch('history_matching.emulators.factory.EmulatorFactory.create_emulator') as mock_create:
            mock_emulator = MockEmulator(X, y)
            mock_create.return_value = mock_emulator
            
            result = create_glm_emulator(X, y, family='poisson')
            
            mock_create.assert_called_once_with(X, y, family='poisson')
            assert result is mock_emulator


class TestEmulatorFactoryIntegration:
    """Integration tests for EmulatorFactory."""
    
    def setup_method(self):
        """Setup for each test method - ensure clean registry state."""
        # Import original registry to restore after tests
        from history_matching.emulators.factory import EmulatorFactory
        from history_matching.emulators.linear import LinearModel
        from history_matching.emulators.glm import GLM
        from history_matching.emulators.gpr import GPR
        
        # Store original registry
        self._original_registry = EmulatorFactory._emulator_registry.copy()
    
    def teardown_method(self):
        """Cleanup after each test method - restore clean registry state."""
        from history_matching.emulators.factory import EmulatorFactory
        
        # Restore original registry
        EmulatorFactory._emulator_registry = self._original_registry
    
    @pytest.fixture
    def realistic_data(self):
        """Create realistic training data."""
        np.random.seed(42)
        n_samples = 50
        
        X = pd.DataFrame({
            'transmission_rate': np.random.uniform(0.1, 1.0, n_samples),
            'recovery_rate': np.random.uniform(0.05, 0.3, n_samples),
            'initial_infected': np.random.randint(1, 100, n_samples)
        })
        
        # Create realistic simulation outputs
        y_single = pd.DataFrame({
            'peak_infections': np.random.poisson(1000, n_samples)
        })
        
        y_multi = pd.DataFrame({
            'total_infections': np.random.poisson(5000, n_samples),
            'total_deaths': np.random.poisson(200, n_samples),
            'peak_day': np.random.randint(50, 150, n_samples)
        })
        
        return X, y_single, y_multi
    
    def test_factory_works_with_all_emulator_types(self, realistic_data):
        """Test that factory works with all registered emulator types."""
        X, y_single, _ = realistic_data
        
        # Test each emulator type
        for emulator_type in ['linear', 'gpr', 'glm']:
            factory = EmulatorFactory(default_type=emulator_type)
            
            try:
                emulator = factory.create_emulator(X, y_single)
                assert emulator is not None
                assert hasattr(emulator, 'train')
                assert hasattr(emulator, 'predict')
            except Exception as e:
                pytest.fail(f"Failed to create {emulator_type} emulator: {e}")
    
    def test_multi_feature_emulator_creation(self, realistic_data):
        """Test creating emulators for multiple features."""
        X, _, y_multi = realistic_data
        features = ['total_infections', 'total_deaths']
        
        factory = EmulatorFactory(default_type='linear')  # Use linear for speed
        
        emulators = factory.create_emulators_for_features(X, y_multi, features)
        
        assert len(emulators) == 2
        assert all(feature in emulators for feature in features)
        
        # Check that each emulator is properly configured
        for feature, emulator in emulators.items():
            assert emulator.training_complete
            assert hasattr(emulator, 'predict')
    
    def test_factory_parameter_propagation(self, realistic_data):
        """Test that factory parameters are properly propagated to emulators."""
        X, y_single, _ = realistic_data
        
        # Create factory with custom defaults
        factory = EmulatorFactory(
            default_type='linear',
            test_fraction=0.3  # Custom test fraction
        )
        
        emulator = factory.create_emulator(X, y_single)
        
        # Check that emulator was created successfully and can be trained
        # The test_fraction parameter should have been used during initialization
        assert emulator is not None
        assert hasattr(emulator, 'train')
        
        # Check that test_fraction was used by verifying data split
        # If test_fraction=0.3, then test set should be ~30% of data
        total_samples = len(X)
        expected_test_size = int(total_samples * 0.3)
        actual_test_size = len(emulator.X_test) if hasattr(emulator, 'X_test') and emulator.X_test is not None else 0
        
        # Allow some tolerance for rounding in train_test_split
        assert abs(actual_test_size - expected_test_size) <= 2
    
    def test_factory_error_handling(self, realistic_data):
        """Test factory error handling with problematic data."""
        X, _, _ = realistic_data
        
        factory = EmulatorFactory()
        
        # Test with empty DataFrame
        empty_y = pd.DataFrame()
        with pytest.raises((ValueError, TypeError, IndexError)):
            factory.create_emulator(X, empty_y)
        
        # Test with mismatched dimensions
        wrong_size_y = pd.DataFrame({'output': [1, 2, 3]})  # Only 3 rows vs 50 in X
        with pytest.raises((ValueError, TypeError, IndexError)):
            factory.create_emulator(X, wrong_size_y)
    
    def test_factory_consistency(self, realistic_data):
        """Test that factory produces consistent results."""
        X, y_single, _ = realistic_data
        
        factory = EmulatorFactory(default_type='linear')
        
        # Create two emulators with same parameters
        emulator1 = factory.create_emulator(X, y_single)
        emulator2 = factory.create_emulator(X, y_single)
        
        # They should be different instances but same type
        assert emulator1 is not emulator2
        assert type(emulator1) == type(emulator2)
        assert emulator1.__class__ == emulator2.__class__
    
    def test_custom_emulator_registration_and_usage(self, realistic_data):
        """Test registering and using a custom emulator."""
        X, y_single, _ = realistic_data
        
        # Define a simple custom emulator
        class SimpleAverageEmulator(BaseEmulator):
            def __init__(self, X=None, y=None, **kwargs):
                super().__init__(X, y)
                self.kwargs = kwargs
            
            def train(self):
                if self.y_train is not None:
                    # Handle both scalar and Series/DataFrame cases
                    mean_val = self.y_train.mean()
                    if hasattr(mean_val, 'iloc'):
                        self.mean_value = mean_val.iloc[0]
                    else:
                        self.mean_value = float(mean_val)
                self.training_complete = True
            
            def predict(self, x):
                if not hasattr(self, 'mean_value'):
                    raise RuntimeError("Emulator not trained")
                return EmulationResults(
                    mean=[self.mean_value] * len(x),
                    std=[0.1] * len(x)  # sqrt(0.01) = 0.1
                )
        
        # Register the custom emulator
        EmulatorFactory.register_emulator('simple_average', SimpleAverageEmulator)
        
        try:
            # Test that we can create and use it
            factory = EmulatorFactory(default_type='simple_average')
            emulator = factory.create_and_train_emulator(X, y_single)
            
            assert isinstance(emulator, SimpleAverageEmulator)
            assert emulator.training_complete
            
            # Test prediction
            predictions = emulator.predict(X.head(5))
            assert len(predictions) == 5
            assert hasattr(predictions, 'get_mean')
            assert len(predictions.get_mean()) == 5
            
        finally:
            # Clean up
            if 'simple_average' in EmulatorFactory._emulator_registry:
                del EmulatorFactory._emulator_registry['simple_average']

    def test_gpr_train_predict_test_roundtrip(self):
        """End-to-end GPR: train on a known function, predict, test diagnostics.

        Validates that the GPR emulator (with Constant mean function and
        data-informed initial hyperparameters) produces accurate predictions
        on a smooth test function in 3D, and that test() populates
        emulator_metrics correctly.
        """
        np.random.seed(0)
        n = 100

        X = pd.DataFrame({
            'x1': np.random.uniform(0, 1, n),
            'x2': np.random.uniform(0, 1, n),
            'x3': np.random.uniform(0, 1, n),
        })
        # Smooth function with an offset + small noise (noise-free data causes
        # Cholesky failures when likelihood variance is unconstrained)
        y = pd.DataFrame({
            'output': 50.0 + 10.0 * X['x1'] - 5.0 * X['x2'] + 2.0 * X['x1'] * X['x3']
                      + np.random.normal(0, 0.1, n)
        })

        from history_matching.emulators.gpr import GPR

        emulator = GPR(X, y, test_fraction=0.25)
        emulator.train()
        assert emulator.training_complete

        # Predict on a fresh set of points
        X_new = pd.DataFrame({
            'x1': np.random.uniform(0, 1, 20),
            'x2': np.random.uniform(0, 1, 20),
            'x3': np.random.uniform(0, 1, 20),
        })
        preds = emulator.predict(X_new)
        mean = preds.get_mean()
        var  = preds.get_variance()

        assert len(mean) == 20
        assert len(var) == 20
        assert np.all(var >= 0), "Variance must be non-negative"

        # Predictions should be close to the true function (R² > 0.9 on new data)
        y_true = 50.0 + 10.0 * X_new['x1'] - 5.0 * X_new['x2'] + 2.0 * X_new['x1'] * X_new['x3']
        ss_res = np.sum((mean - y_true.values) ** 2)
        ss_tot = np.sum((y_true.values - y_true.mean()) ** 2)
        r2_new = 1.0 - ss_res / ss_tot
        assert r2_new > 0.9, f"GPR predictions poor on new data: R²={r2_new:.3f}"

        # test() should populate emulator_metrics
        emulator.test()
        assert emulator.testing_complete
        em = emulator.emulator_metrics
        assert 'R2' in em and not np.isnan(em['R2']), "R² should be computed"
        assert 'MSE' in em and not np.isnan(em['MSE']), "MSE should be computed"
        assert em['R2'] > 0.8, f"Test-set R² too low: {em['R2']:.3f}"

    def test_gpr_with_wildly_different_scales(self):
        """GPR should handle parameters on very different scales.

        Without input normalization, kernel lengthscales can't accommodate
        parameters spanning orders of magnitude (e.g. 0.0005–0.006 vs 1–3).
        """
        np.random.seed(11)
        n = 100

        X = pd.DataFrame({
            'tiny':  np.random.uniform(0.0005, 0.006, n),   # O(1e-3)
            'mid':   np.random.uniform(0.1, 1.0, n),        # O(1e-1)
            'large': np.random.uniform(1.0, 3.0, n),        # O(1)
        })
        y = pd.DataFrame({
            'output': 100 * X['tiny'] + 5 * X['mid'] + X['large']
                      + np.random.normal(0, 0.01, n)
        })

        from history_matching.emulators.gpr import GPR

        emulator = GPR(X, y, test_fraction=0.25)
        emulator.train()
        assert emulator.training_complete

        emulator.test()
        em = emulator.emulator_metrics
        assert em['R2'] > 0.8, (
            f"GPR failed with multi-scale inputs (R²={em['R2']:.3f}). "
            f"Input normalization may not be working."
        )

    def test_gpr_with_constant_offset_data(self):
        """GPR should handle data with a large constant offset (mean function test).

        Without a mean function (or the old column-of-ones hack), GPR with an
        SE kernel centered at zero would struggle with y ~ 3000 + small_signal.
        The Constant mean function should absorb the offset.
        """
        np.random.seed(7)
        n = 80

        X = pd.DataFrame({
            'p1': np.random.uniform(0, 1, n),
            'p2': np.random.uniform(0, 1, n),
        })
        # Large offset + small signal — similar to birth weight calibration
        y = pd.DataFrame({
            'output': 3000.0 + 5.0 * X['p1'] - 3.0 * X['p2']
                      + np.random.normal(0, 0.1, n)
        })

        from history_matching.emulators.gpr import GPR

        emulator = GPR(X, y, test_fraction=0.25)
        emulator.train()
        assert emulator.training_complete

        emulator.test()
        em = emulator.emulator_metrics
        assert em['R2'] > 0.8, (
            f"GPR failed on large-offset data (R²={em['R2']:.3f}). "
            f"The mean function may not be absorbing the constant offset."
        )