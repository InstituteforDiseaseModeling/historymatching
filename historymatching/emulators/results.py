"""
EmulationResults class for standardized emulator output handling.
"""

from typing import Optional, Tuple, Union

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
                 additional_data: Optional[pd.DataFrame] = None,
                 index: Optional[pd.Index] = None):
        """
        Initialize emulation results.
        
        Args:
            mean: Predicted means (required)
            std: Predicted standard deviations (required) 
            additional_data: Optional DataFrame with other results (CI, etc.)
            index: Index to label all predictions with — normally the index of the
                DataFrame that was predicted on. Defaults to the index of `mean`
                or `std` if either is a Series, otherwise a plain RangeIndex.

        Note:
            Mean, std, and additional_data are all forced onto a single shared
            index. Emulators mix numpy arrays (which get a fresh RangeIndex) and
            pandas Series (which carry the caller's labels), and mixing the two
            here is silently dangerous: downstream arithmetic like
            `mean / np.sqrt(variance)` aligns Series *by label*, so two
            same-length operands with different labels produce the union of both
            indexes instead of an elementwise result.
        """
        self._validate_lengths(mean, std, additional_data)

        if index is None:
            if isinstance(mean, pd.Series):
                index = mean.index
            elif isinstance(std, pd.Series):
                index = std.index
            else:
                index = pd.RangeIndex(len(mean))

        # Rebuild from raw values so alignment is positional, not by label.
        self._mean = pd.Series(np.asarray(mean).ravel(), index=index)
        self._std = pd.Series(np.asarray(std).ravel(), index=index)
        if additional_data is not None and not additional_data.index.equals(index):
            additional_data = additional_data.set_axis(index)
        self._additional_data = additional_data
    
    @staticmethod
    def _validate_lengths(mean, std, additional_data):
        """Validate that mean, std, and additional_data all have the same length."""
        if len(mean) != len(std):
            raise ValueError(f"Mean and std must have same length. "
                           f"Got mean: {len(mean)}, std: {len(std)}")
        
        if additional_data is not None and len(additional_data) != len(mean):
            raise ValueError(f"Additional data must have same length as mean/std. "
                           f"Got additional_data: {len(additional_data)}, mean/std: {len(mean)}")
    
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