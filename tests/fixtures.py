"""
Test fixtures and utilities for history matching tests.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Callable, Dict, Tuple

from historymatching.utils import PARAMETER_SPACE_COLUMNS, OBSERVATIONS_COLUMNS
import historymatching as hm


class TestDataFactory:
    """Factory for creating consistent test data."""
    
    @staticmethod
    def create_simple_parameter_space(n_params: int = 3) -> pd.DataFrame:
        """
        Create a simple parameter space for testing.
        
        Args:
            n_params: Number of parameters to create
            
        Returns:
            DataFrame with parameter space definition
        """
        param_names = [f"param_{chr(97 + i)}" for i in range(n_params)]  # param_a, param_b, etc.
        data = []
        
        for i, name in enumerate(param_names):
            min_val = i * 10.0  # 0, 10, 20, ...
            max_val = min_val + 10.0  # 10, 20, 30, ...
            data.append([name, min_val, max_val])
            
        return pd.DataFrame(data, columns=PARAMETER_SPACE_COLUMNS)
    
    @staticmethod
    def create_simple_observations(features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create simple observation data for testing.
        
        Args:
            features: List of feature names (default: ['output_1', 'output_2'])
            
        Returns:
            DataFrame with observations
        """
        if features is None:
            features = ['output_1', 'output_2']
            
        data = []
        for i, feature in enumerate(features):
            mean_val = 10.0 + i * 5.0  # 10, 15, 20, ...
            std_val = 1.0 + i * 0.5   # 1.0, 1.5, 2.0, ...
            data.append([feature, mean_val, std_val])
            
        return pd.DataFrame(data, columns=OBSERVATIONS_COLUMNS)
    
    @staticmethod
    def create_linear_model(coefficients: Optional[Dict[str, float]] = None) -> Callable:
        """
        Create a simple linear model for testing.
        
        Args:
            coefficients: Dict mapping parameter names to coefficients
            
        Returns:
            Model function that takes samples DataFrame and returns results DataFrame
        """
        if coefficients is None:
            coefficients = {'param_a': 1.0, 'param_b': 2.0, 'param_c': 0.5}
            
        def linear_model(samples_df: pd.DataFrame) -> pd.DataFrame:
            # Calculate linear combination
            output_1 = np.zeros(len(samples_df))
            output_2 = np.zeros(len(samples_df))
            
            for param, coeff in coefficients.items():
                if param in samples_df.columns:
                    output_1 += coeff * samples_df[param]
                    output_2 += coeff * samples_df[param] * 2  # Different scaling
                    
            # Add small amount of noise for realism
            noise_1 = np.random.normal(0, 0.1, len(samples_df))
            noise_2 = np.random.normal(0, 0.2, len(samples_df))
            
            return pd.DataFrame({
                'output_1': output_1 + noise_1,
                'output_2': output_2 + noise_2
            })
            
        return linear_model
    
    @staticmethod
    def create_sample_data(parameter_space_df: pd.DataFrame, n_samples: int = 100, 
                          seed: Optional[int] = 42) -> pd.DataFrame:
        """
        Create random sample data within parameter bounds.
        
        Args:
            parameter_space_df: Parameter space definition
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with parameter samples
        """
        if seed is not None:
            np.random.seed(seed)
            
        samples = {}
        
        for _, row in parameter_space_df.iterrows():
            param_name = row['parameter']
            min_val = row['minimum']
            max_val = row['maximum']
            
            samples[param_name] = np.random.uniform(min_val, max_val, n_samples)
            
        return pd.DataFrame(samples)
    
    @staticmethod
    def create_simulation_results(samples_df: pd.DataFrame, 
                                 model_func: Optional[Callable] = None) -> pd.DataFrame:
        """
        Create simulation results from samples using a model function.
        
        Args:
            samples_df: Parameter samples
            model_func: Model function (uses linear model if None)
            
        Returns:
            DataFrame with simulation results
        """
        if model_func is None:
            model_func = TestDataFactory.create_linear_model()
            
        return model_func(samples_df)


class MockEmulator(hm.BaseEmulator):
    """Mock emulator for testing purposes."""
    
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, 
                 r2_score: float = 0.9, mse: float = 0.1):
        """
        Initialize mock emulator.
        
        Args:
            X: Input data
            y: Output data
            r2_score: Mock R² score to return
            mse: Mock MSE to return
        """
        super().__init__(X, y)
        self.X = X
        self.y = y
        self._r2_score = r2_score
        self._mse = mse
        self.training_complete = False
        self.testing_complete = False
        
    def train(self):
        """Mock training method."""
        self.training_complete = True
        self.testing_complete = True
        
        # Set up mock train/test splits
        split_idx = int(0.75 * len(self.X))
        self.X_train = self.X.iloc[:split_idx]
        self.X_test = self.X.iloc[split_idx:]
        self.y_train = self.y.iloc[:split_idx]
        self.y_test = self.y.iloc[split_idx:]
        
        # Create mock predictions
        self.y_pred_test = self.y_test + np.random.normal(0, 0.1, len(self.y_test))
        
    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Mock prediction method.
        
        Args:
            X_test: Test input data
            
        Returns:
            DataFrame with predicted means and variances
        """
        n_points = len(X_test)
        
        # Simple mock predictions based on sum of parameters
        if len(X_test.columns) > 0:
            means = X_test.sum(axis=1).values
        else:
            means = np.ones(n_points)
            
        variances = np.full(n_points, 0.1)  # Constant variance
        
        out = pd.DataFrame({
            'mean': means,
            'variance': variances
        })
        
        return out
    
    def score(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        """Return mock R² score."""
        return self._r2_score
    
    def print_emulator_description(self):
        """Return mock emulator description."""
        return "Mock emulator for testing"


class TestAssertions:
    """Custom assertion methods for testing."""
    
    @staticmethod
    def assert_dataframe_structure(df: pd.DataFrame, expected_columns: List[str], 
                                  expected_types: Optional[Dict[str, type]] = None):
        """
        Assert DataFrame has expected structure.
        
        Args:
            df: DataFrame to check
            expected_columns: Expected column names
            expected_types: Expected column types (optional)
        """
        assert list(df.columns) == expected_columns, \
            f"Expected columns {expected_columns}, got {list(df.columns)}"
            
        if expected_types:
            for col, expected_type in expected_types.items():
                actual_type = df[col].dtype
                assert actual_type == expected_type, \
                    f"Column {col}: expected type {expected_type}, got {actual_type}"
    
    @staticmethod
    def assert_parameter_bounds_respected(samples_df: pd.DataFrame, 
                                        parameter_space_df: pd.DataFrame):
        """
        Assert all samples fall within parameter bounds.
        
        Args:
            samples_df: Parameter samples
            parameter_space_df: Parameter space definition
        """
        for _, row in parameter_space_df.iterrows():
            param_name = row['parameter']
            min_val = row['minimum']
            max_val = row['maximum']
            
            if param_name in samples_df.columns:
                param_samples = samples_df[param_name]
                assert param_samples.min() >= min_val, \
                    f"Parameter {param_name} below minimum: {param_samples.min()} < {min_val}"
                assert param_samples.max() <= max_val, \
                    f"Parameter {param_name} above maximum: {param_samples.max()} > {max_val}"
    
    @staticmethod
    def assert_emulator_trained(emulator: hm.BaseEmulator):
        """
        Assert emulator has been properly trained.
        
        Args:
            emulator: Emulator to check
        """
        assert hasattr(emulator, 'predict'), "Emulator must have predict method"
        assert emulator.training_complete, "Emulator training not complete"
        
        # Try a prediction to ensure it works
        if hasattr(emulator, 'X') and emulator.X is not None:
            X_test = emulator.X.iloc[:1]  # Use first row for test
            means, variances = emulator.predict(X_test)
            assert len(means) == 1, "Prediction should return one mean value"
            assert len(variances) == 1, "Prediction should return one variance value"
    
    @staticmethod
    def assert_observations_valid(observations_df: pd.DataFrame):
        """
        Assert observations DataFrame is valid.
        
        Args:
            observations_df: Observations to validate
        """
        assert all(col in observations_df.columns for col in OBSERVATIONS_COLUMNS), \
            f"Observations must have columns: {OBSERVATIONS_COLUMNS}"
            
        for _, row in observations_df.iterrows():
            var_val = row['variance']
            assert var_val > 0, f"Variance must be positive, got {var_val}"
            assert np.isfinite(var_val), f"Variance must be finite, got {var_val}"
            
            mean_val = row['mean']
            assert np.isfinite(mean_val), f"Mean must be finite, got {mean_val}"


class TestConstants:
    """Constants for testing."""
    
    # Default parameter space
    DEFAULT_PARAMETER_SPACE = TestDataFactory.create_simple_parameter_space(3)
    
    # Default observations
    DEFAULT_OBSERVATIONS = TestDataFactory.create_simple_observations(['output_1', 'output_2'])
    
    # Default sample size
    DEFAULT_N_SAMPLES = 50
    
    # Default random seed
    DEFAULT_SEED = 42