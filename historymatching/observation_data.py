"""
ObservationData domain object for history matching.
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from .constants import OBSERVATIONS_COLUMNS


class ObservationData:
    """
    Encapsulates observational data and implausibility calculations.
    
    Represents the target observations that the model should match,
    including their means and variances. Provides methods for calculating
    implausibility metrics.
    """

    def __init__(self, observations: Union[pd.DataFrame, dict]):
        """
        Initialize observation data.

        Args:
            observations: DataFrame with columns ['feature', 'mean', 'std']
                         or dict mapping each observed output name to a
                         ``(mean, std)`` tuple.  The second value is the
                         standard deviation (sigma), *not* the variance.
        """
        if isinstance(observations, dict):
            # Convert dict to DataFrame, with a clear error for the common
            # mistake of passing a bare number instead of a (mean, std) pair.
            data = []
            for name, values in observations.items():
                if not isinstance(values, (tuple, list)) or len(values) != 2:
                    raise ValueError(
                        f"observations['{name}'] must be a (mean, std) tuple, got {values!r}. "
                        f"The second value is the standard deviation, not the variance. "
                        f"Example: observations={{'{name}': (120.0, 5.0)}}"
                    )
                data.append((name, values[0], values[1]))
            observations = pd.DataFrame(data, columns=OBSERVATIONS_COLUMNS)

        self._observations = self._validate_and_normalize(observations)

    def _validate_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize observations DataFrame."""
        # Check required columns
        if not all(col in df.columns for col in OBSERVATIONS_COLUMNS):
            raise ValueError(f"Observations must have columns: {OBSERVATIONS_COLUMNS}")

        # Check for empty observations
        if len(df) == 0:
            raise ValueError("Observations cannot be empty")

        # Create copy to avoid modifying original
        df = df.copy()

        # Set feature names as index for easy lookup
        if "feature" in df.columns:
            df = df.set_index("feature", drop=False)

        # Validate data
        for idx, row in df.iterrows():
            feature_name = row["feature"]
            mean_val, std_val = row["mean"], row["std"]

            if not np.isfinite(mean_val):
                raise ValueError(f"Feature '{feature_name}' has non-finite mean: {mean_val}")

            if not np.isfinite(std_val) or std_val <= 0:
                raise ValueError(f"Feature '{feature_name}' has invalid std: {std_val} (must be positive and finite)")

        # Check for duplicate feature names
        if df["feature"].duplicated().any():
            duplicates = df["feature"][df["feature"].duplicated()].tolist()
            raise ValueError(f"Duplicate feature names found: {duplicates}")

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of all observed feature names."""
        return self._observations["feature"].tolist()

    def get_target_for_feature(self, feature_name: str) -> Tuple[float, float]:
        """
        Get target mean and std for a specific feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Tuple of (mean, std) values
        """
        if feature_name not in self._observations.index:
            raise ValueError(f"Feature '{feature_name}' not found in observations")

        row = self._observations.loc[feature_name]
        return float(row["mean"]), float(row["std"])

    def get_all_targets(self) -> Dict[str, Tuple[float, float]]:
        """
        Get all feature targets as a dictionary.
        
        Returns:
            Dict mapping feature names to (mean, std) tuples
        """
        targets = {}
        for feature_name in self.get_feature_names():
            targets[feature_name] = self.get_target_for_feature(feature_name)
        return targets

    def plot_targets(self, *, ax=None, **kwargs):
        """Plot observation targets as means with ±1σ error bars (delegates to
        :func:`historymatching.plotting.plot_targets`)."""
        from . import plotting
        return plotting.plot_targets(self.get_all_targets(), ax=ax, **kwargs)

    def summary(self) -> str:
        """Human-readable summary: one line per observed feature (mean, std)."""
        names = self.get_feature_names()
        lines = [f"ObservationData: {len(names)} target(s)"]
        for n in names:
            mean, std = self.get_target_for_feature(n)
            lines.append(f"  {n:<20} mean={mean:g}, std={std:g}")
        return "\n".join(lines)

    def calculate_implausibility(self, feature_name: str, predicted_mean: Union[float, pd.Series],
                                predicted_variance: Union[float, pd.Series], model_discrepancy: float = 0.0) -> Union[float, pd.Series]:
        """
        Calculate implausibility metric for a single feature.
        
        The implausibility is calculated as:
        I = |predicted_mean - observed_mean| / sqrt(predicted_variance + observed_variance + model_discrepancy^2)
        
        Args:
            feature_name: Name of the feature
            predicted_mean: Predicted mean from emulator (scalar or Series)
            predicted_variance: Predicted variance from emulator (scalar or Series)
            model_discrepancy: Additional model uncertainty
            
        Returns:
            Implausibility value(s) (lower is more plausible) - scalar if inputs are scalar, Series if inputs are Series
        """
        if feature_name not in self._observations.index:
            raise ValueError(f"Feature '{feature_name}' not found in observations")

        observed_mean, observed_std = self.get_target_for_feature(feature_name)
        observed_variance = observed_std**2  # Convert std to variance for calculation

        # Calculate total variance (emulator + observation + model discrepancy)
        total_variance = predicted_variance + observed_variance + model_discrepancy**2

        if np.any(total_variance <= 0):
            raise ValueError(f"Total variance must be positive, got {total_variance}")

        # Calculate implausibility
        mean_difference = abs(predicted_mean - observed_mean)
        implausibility = mean_difference / np.sqrt(total_variance)

        # Return appropriate type based on input
        if isinstance(predicted_mean, pd.Series) or isinstance(predicted_variance, pd.Series):
            return implausibility
        else:
            return float(implausibility)

    def calculate_implausibilities(self, predictions: Dict[str, Tuple[float, float]],
                                  model_discrepancy: float = 0.0) -> Dict[str, float]:
        """
        Calculate implausibilities for multiple features.
        
        Args:
            predictions: Dict mapping feature names to (predicted_mean, predicted_variance) tuples
            model_discrepancy: Additional model uncertainty
            
        Returns:
            Dict mapping feature names to implausibility values
        """
        implausibilities = {}

        for feature_name, (pred_mean, pred_var) in predictions.items():
            if self.has_feature(feature_name):
                implausibilities[feature_name] = self.calculate_implausibility(
                    feature_name, pred_mean, pred_var, model_discrepancy
                )

        return implausibilities

    def calculate_maximum_implausibility(self, predictions: Dict[str, Tuple[float, float]],
                                       model_discrepancy: float = 0.0) -> float:
        """
        Calculate maximum implausibility across all features.
        
        Args:
            predictions: Dict mapping feature names to (predicted_mean, predicted_variance) tuples
            model_discrepancy: Additional model uncertainty
            
        Returns:
            Maximum implausibility value across all features
        """
        implausibilities = self.calculate_implausibilities(predictions, model_discrepancy)

        if not implausibilities:
            raise ValueError("No matching features found between predictions and observations")

        return max(implausibilities.values())

    def has_feature(self, feature_name: str) -> bool:
        """
        Check if a feature exists in observations.
        
        Args:
            feature_name: Name of the feature to check
            
        Returns:
            True if feature exists, False otherwise
        """
        return feature_name in self._observations.index

    def filter_features(self, feature_names: List[str]) -> "ObservationData":
        """
        Create new ObservationData with only specified features.
        
        Args:
            feature_names: List of feature names to keep
            
        Returns:
            New ObservationData instance with filtered features
        """
        # Check that all requested features exist
        missing_features = [f for f in feature_names if not self.has_feature(f)]
        if missing_features:
            raise ValueError(f"Features not found in observations: {missing_features}")

        # Filter observations
        filtered_obs = self._observations[self._observations["feature"].isin(feature_names)]

        # Reset index to get back to standard DataFrame format
        filtered_df = filtered_obs.reset_index(drop=True)

        return ObservationData(filtered_df)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return the underlying observations DataFrame.
        
        Returns:
            DataFrame with observations data
        """
        return self._observations.reset_index(drop=True).copy()

    def __len__(self) -> int:
        """Return number of observed features."""
        return len(self._observations)

    def __eq__(self, other) -> bool:
        """Check equality with another ObservationData."""
        if not isinstance(other, ObservationData):
            return False

        return self._observations.equals(other._observations)

    def __repr__(self) -> str:
        """String representation showing each observed (mean, std)."""
        names = self.get_feature_names()
        shown = names[:6]
        items = ", ".join(
            f"{n}=(mean={self.get_target_for_feature(n)[0]:g}, std={self.get_target_for_feature(n)[1]:g})"
            for n in shown
        )
        more = f", +{len(names) - len(shown)} more" if len(names) > len(shown) else ""
        return f"ObservationData({len(names)} observations: {items}{more})"
