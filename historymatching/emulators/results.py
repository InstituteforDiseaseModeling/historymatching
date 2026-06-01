"""
EmulationResults class for standardized emulator output handling.
"""

from typing import Optional, Union, Tuple
import numpy as np
import pandas as pd
from scipy import stats


class EmulationResults:
    """
    Standardized container for emulator prediction results.
    
    Provides clean access to mean and standard deviation predictions,
    with optional additional data for emulator-specific outputs.
    """
    
    def __init__(self, 
                 mean: Union[np.ndarray, pd.Series],
                 std: Union[np.ndarray, pd.Series], 
                 additional_data: Optional[pd.DataFrame] = None):
        """
        Initialize emulation results.
        
        Args:
            mean: Predicted means (required)
            std: Predicted standard deviations (required) 
            additional_data: Optional DataFrame with other results (CI, etc.)
        """
        self._mean = pd.Series(mean) if not isinstance(mean, pd.Series) else mean
        self._std = pd.Series(std) if not isinstance(std, pd.Series) else std
        self._additional_data = additional_data
        self._validate_lengths()
    
    def _validate_lengths(self):
        """Validate that mean and std have the same length."""
        if len(self._mean) != len(self._std):
            raise ValueError(f"Mean and std must have same length. "
                           f"Got mean: {len(self._mean)}, std: {len(self._std)}")
        
        if self._additional_data is not None and len(self._additional_data) != len(self._mean):
            raise ValueError(f"Additional data must have same length as mean/std. "
                           f"Got additional_data: {len(self._additional_data)}, mean/std: {len(self._mean)}")
    
    def get_mean(self) -> pd.Series:
        """Get predicted means."""
        return self._mean
        
    def get_std(self) -> pd.Series:
        """Get predicted standard deviations."""
        return self._std
        
    def get_variance(self) -> pd.Series:
        """Get predicted variances (computed from std)."""
        return self._std ** 2
        
    def get_ci(self, confidence_level: float = 0.95) -> Tuple[pd.Series, pd.Series]:
        """
        Get confidence intervals assuming normal distribution.
        
        Args:
            confidence_level: Confidence level (0.95 = 95%)
        
        Returns:
            Tuple of (lower_bound, upper_bound) Series
        """
        if not 0 < confidence_level < 1:
            raise ValueError(f"Confidence level must be between 0 and 1, got {confidence_level}")
            
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin = z_score * self._std
        return self._mean - margin, self._mean + margin
        
    def get_additional_data(self) -> Optional[pd.DataFrame]:
        """Get additional emulator-specific data."""
        return self._additional_data
        
    def __len__(self) -> int:
        """Number of predictions."""
        return len(self._mean)
        
    def __repr__(self) -> str:
        """String representation."""
        return (f"EmulationResults(n_predictions={len(self)}, "
                f"has_additional_data={self._additional_data is not None})")