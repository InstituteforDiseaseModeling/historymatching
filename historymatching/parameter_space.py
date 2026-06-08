"""
ParameterSpace domain object for history matching.
"""

from typing import List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd

from .constants import PARAMETER_SPACE_COLUMNS

logger = logging.getLogger(__name__)


class ParameterSpace:
    """
    Encapsulates parameter space information and operations.
    
    Represents the bounds and constraints for parameters in a history matching
    analysis. Supports operations like constraining the space based on samples
    and calculating volume reductions.
    """

    def __init__(self, parameters: Union[pd.DataFrame, dict]):
        """
        Initialize parameter space.
        
        Args:
            parameters: DataFrame with columns ['parameter', 'minimum', 'maximum']
                       or dict mapping parameter names to (min, max) tuples
        """
        if isinstance(parameters, dict):
            # Convert dict to DataFrame with validation
            data = []
            for name, bounds in parameters.items():
                if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                    raise ValueError(f"Parameter '{name}' bounds must be a tuple/list of (min, max), got: {bounds}")
                try:
                    min_val, max_val = bounds[0], bounds[1]
                    data.append((name, min_val, max_val))
                except (IndexError, TypeError) as e:
                    raise ValueError(f"Parameter '{name}' bounds must be a tuple/list of (min, max), got: {bounds}") from e
            parameters = pd.DataFrame(data, columns=PARAMETER_SPACE_COLUMNS)

        self._parameters = self._validate_and_normalize(parameters)
        self._original_bounds = self._parameters.copy()

    def _validate_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize parameter space DataFrame."""
        # Check required columns
        if not all(col in df.columns for col in PARAMETER_SPACE_COLUMNS):
            raise ValueError(f"Parameter space must have columns: {PARAMETER_SPACE_COLUMNS}")

        # Check for empty parameter space
        if len(df) == 0:
            raise ValueError("Parameter space cannot be empty")

        # Create copy to avoid modifying original
        df = df.copy()

        # Set parameter names as index for easy lookup
        if "parameter" in df.columns:
            df = df.set_index("parameter", drop=False)

        # Validate bounds
        for idx, row in df.iterrows():
            param_name = row["parameter"]
            min_val, max_val = row["minimum"], row["maximum"]

            if not np.isfinite(min_val) or not np.isfinite(max_val):
                raise ValueError(f"Parameter '{param_name}' has non-finite bounds: [{min_val}, {max_val}]")

            if min_val >= max_val:
                raise ValueError(f"Parameter '{param_name}' minimum ({min_val}) must be less than maximum ({max_val})")

        # Check for duplicate parameter names
        if df["parameter"].duplicated().any():
            duplicates = df["parameter"][df["parameter"].duplicated()].tolist()
            raise ValueError(f"Duplicate parameter names found: {duplicates}")

        return df

    def get_bounds(self, parameter_name: str) -> Tuple[float, float]:
        """
        Get bounds for a specific parameter.
        
        Args:
            parameter_name: Name of the parameter
            
        Returns:
            Tuple of (minimum, maximum) values
        """
        if parameter_name not in self._parameters.index:
            raise ValueError(f"Parameter '{parameter_name}' not found in parameter space")

        row = self._parameters.loc[parameter_name]
        return float(row["minimum"]), float(row["maximum"])

    def get_parameter_names(self) -> List[str]:
        """Get list of all parameter names."""
        return self._parameters["parameter"].tolist()

    def plot_bounds(self, *, reference=None, ax=None, **kwargs):
        """Plot each parameter's bounds as a horizontal range, optionally against
        a ``reference`` ParameterSpace to show shrinkage (delegates to
        :func:`historymatching.plotting.plot_parameter_bounds`)."""
        from . import plotting
        bounds = {n: self.get_bounds(n) for n in self.get_parameter_names()}
        ref = None
        if reference is not None:
            ref = {n: reference.get_bounds(n) for n in reference.get_parameter_names()}
        return plotting.plot_parameter_bounds(bounds, reference=ref, ax=ax, **kwargs)

    def summary(self) -> str:
        """Human-readable summary: one line per parameter with its bounds."""
        names = self.get_parameter_names()
        lines = [f"ParameterSpace: {len(names)} parameter(s)"]
        for n in names:
            lo, hi = self.get_bounds(n)
            lines.append(f"  {n:<20} [{lo:g}, {hi:g}]")
        return "\n".join(lines)

    def constrain_parameter(self, param_name: str, new_min: float, new_max: float) -> "ParameterSpace":
        """
        Create new ParameterSpace with updated bounds for one parameter.
        
        Args:
            param_name: Name of parameter to constrain
            new_min: New minimum value
            new_max: New maximum value
            
        Returns:
            New ParameterSpace instance with updated bounds
        """
        if param_name not in self._parameters.index:
            raise ValueError(f"Parameter '{param_name}' not found")

        if new_min >= new_max:
            raise ValueError(f"New minimum ({new_min}) must be less than maximum ({new_max})")

        # Get current bounds to ensure we're not expanding
        current_min, current_max = self.get_bounds(param_name)
        if new_min < current_min or new_max > current_max:
            raise ValueError(f"Cannot expand parameter space. Current bounds: [{current_min}, {current_max}], "
                           f"requested: [{new_min}, {new_max}]")

        # Create new parameter space
        new_params = self._parameters.copy()
        new_params.loc[param_name, "minimum"] = new_min
        new_params.loc[param_name, "maximum"] = new_max

        # Reset index to get back to standard DataFrame format
        new_params_df = new_params.reset_index(drop=True)

        return ParameterSpace(new_params_df)

    def constrain_to_samples(self, samples_df: pd.DataFrame, percentile: float = 95) -> "ParameterSpace":
        """
        Create new ParameterSpace constrained to sample bounds.
        
        Args:
            samples_df: DataFrame containing parameter samples
            percentile: Percentile to use for bounds (e.g., 95 means 2.5-97.5 range)
            
        Returns:
            New ParameterSpace constrained to sample bounds
        """
        if not self.validate_samples(samples_df):
            logger.warning(
                "Some samples fall slightly outside current parameter space bounds. "
                "This can happen due to floating-point precision in emulator filtering. "
                "Proceeding — new bounds will be clipped to the current space."
            )

        # Calculate percentile bounds
        lower_percentile = (100 - percentile) / 2
        upper_percentile = 100 - lower_percentile

        new_params = self._parameters.copy()

        for param_name in self.get_parameter_names():
            if param_name in samples_df.columns:
                param_samples = samples_df[param_name]
                new_min = np.percentile(param_samples, lower_percentile)
                new_max = np.percentile(param_samples, upper_percentile)

                # Ensure we don't expand beyond current bounds
                current_min, current_max = self.get_bounds(param_name)
                new_min = max(new_min, current_min)
                new_max = min(new_max, current_max)

                new_params.loc[param_name, "minimum"] = new_min
                new_params.loc[param_name, "maximum"] = new_max

        # Reset index to get back to standard DataFrame format
        new_params_df = new_params.reset_index(drop=True)

        return ParameterSpace(new_params_df)

    def volume_fraction_remaining(self, original_space: "ParameterSpace") -> float:
        """
        Calculate remaining parameter space volume as fraction of original.
        
        Args:
            original_space: Original ParameterSpace to compare against
            
        Returns:
            Fraction of volume remaining (0.0 to 1.0)
        """
        if set(self.get_parameter_names()) != set(original_space.get_parameter_names()):
            raise ValueError("Parameter spaces must have same parameters")

        current_volume = 1.0
        original_volume = 1.0

        for param_name in self.get_parameter_names():
            current_min, current_max = self.get_bounds(param_name)
            original_min, original_max = original_space.get_bounds(param_name)

            current_volume *= (current_max - current_min)
            original_volume *= (original_max - original_min)

        return current_volume / original_volume if original_volume > 0 else 0.0

    def sample_uniformly(self, n_samples: int, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate uniform random samples within parameter bounds.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with columns for each parameter
        """
        if seed is not None:
            np.random.seed(seed)

        samples = {}

        for param_name in self.get_parameter_names():
            min_val, max_val = self.get_bounds(param_name)
            samples[param_name] = np.random.uniform(min_val, max_val, n_samples)

        return pd.DataFrame(samples)

    def validate_samples(self, samples_df: pd.DataFrame) -> bool:
        """
        Check if all samples fall within parameter bounds.
        
        Args:
            samples_df: DataFrame containing parameter samples
            
        Returns:
            True if all samples are valid, False otherwise
        """
        for param_name in self.get_parameter_names():
            if param_name not in samples_df.columns:
                continue  # Skip missing parameters

            min_val, max_val = self.get_bounds(param_name)
            param_samples = samples_df[param_name]

            if param_samples.min() < min_val or param_samples.max() > max_val:
                return False

        return True

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return the underlying parameters DataFrame.
        
        Returns:
            DataFrame with parameter space definition
        """
        return self._parameters.reset_index(drop=True).copy()

    def __len__(self) -> int:
        """Return number of parameters."""
        return len(self._parameters)

    def __eq__(self, other) -> bool:
        """Check equality with another ParameterSpace."""
        if not isinstance(other, ParameterSpace):
            return False

        return self._parameters.equals(other._parameters)

    def __repr__(self) -> str:
        """String representation showing each parameter's bounds."""
        names = self.get_parameter_names()
        shown = names[:6]
        items = ", ".join(
            f"{n}=[{self.get_bounds(n)[0]:g}, {self.get_bounds(n)[1]:g}]" for n in shown
        )
        more = f", +{len(names) - len(shown)} more" if len(names) > len(shown) else ""
        return f"ParameterSpace({len(names)} parameters: {items}{more})"
