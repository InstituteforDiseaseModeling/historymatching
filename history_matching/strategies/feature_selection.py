"""
Feature selection strategy implementations for history matching.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np
import pandas as pd
import warnings

try:
    from ..domain.observation_data import ObservationData
    from .. import features
except ImportError:
    # For standalone testing
    from history_matching.domain.observation_data import ObservationData
    import history_matching.features as features


class FeatureSelectionStrategy(ABC):
    """
    Abstract base class for feature selection strategies.
    
    Feature selection strategies determine which model outputs (features)
    should be emulated in each history matching iteration.
    """
    
    @abstractmethod
    def select_features(self, simulation_results: pd.DataFrame, 
                       observations: ObservationData,
                       iteration: int = 1) -> List[str]:
        """
        Select features to emulate for this iteration.
        
        Args:
            simulation_results: DataFrame with simulation outputs
            observations: ObservationData containing target observations
            iteration: Current iteration number (1-based)
            
        Returns:
            List of feature names to emulate
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return human-readable name for this strategy."""
        pass
    
    def validate_features(self, features: List[str], 
                         simulation_results: pd.DataFrame,
                         observations: ObservationData) -> List[str]:
        """
        Validate selected features exist in both simulation results and observations.
        
        Args:
            features: List of feature names to validate
            simulation_results: DataFrame with simulation outputs
            observations: ObservationData containing target observations
            
        Returns:
            List of valid feature names
            
        Raises:
            ValueError: If no valid features remain
        """
        valid_features = []
        
        for feature in features:
            if feature not in simulation_results.columns:
                warnings.warn(f"Feature '{feature}' not found in simulation results")
                continue
                
            if not observations.has_feature(feature):
                warnings.warn(f"Feature '{feature}' not found in observations")
                continue
                
            valid_features.append(feature)
        
        if not valid_features:
            available_sim = list(simulation_results.columns)
            available_obs = observations.get_feature_names()
            common_features = list(set(available_sim) & set(available_obs))
            
            if common_features:
                raise ValueError(
                    f"No valid features selected. Available features in both "
                    f"simulation results and observations: {common_features}"
                )
            else:
                raise ValueError(
                    f"No common features between simulation results {available_sim} "
                    f"and observations {available_obs}"
                )
        
        return valid_features


class ManualFeatureSelection(FeatureSelectionStrategy):
    """
    Manual feature selection strategy.
    
    Returns a predefined list of features for each iteration.
    Useful when you know exactly which features to emulate.
    """
    
    def __init__(self, selected_features: Union[str, List[str]]):
        """
        Initialize manual feature selection.
        
        Args:
            selected_features: Feature name or list of feature names to select
        """
        if isinstance(selected_features, str):
            self.selected_features = [selected_features]
        else:
            self.selected_features = list(selected_features)
            
        if not self.selected_features:
            raise ValueError("Must provide at least one feature name")
    
    def select_features(self, simulation_results: pd.DataFrame, 
                       observations: ObservationData,
                       iteration: int = 1) -> List[str]:
        """Select predefined features."""
        return self.validate_features(
            self.selected_features, simulation_results, observations
        )
    
    def get_strategy_name(self) -> str:
        return f"Manual Selection ({len(self.selected_features)} features)"


class AutoFeatureSelection(FeatureSelectionStrategy):
    """
    Automatic feature selection strategy.
    
    Uses statistical metrics to automatically select the most informative
    features for emulation. Based on the existing feature_selection function.
    """
    
    def __init__(self, method: str = 'fano', threshold: Optional[float] = None,
                 cooldown_period: int = 5, correlation_threshold: float = 0.8,
                 max_features: int = 1):
        """
        Initialize automatic feature selection.
        
        Args:
            method: Statistical method for ranking features ('fano', 'var', 'mean', etc.)
            threshold: Minimum threshold value for the metric (optional)
            cooldown_period: Number of recent selections to track for avoiding repetition
            correlation_threshold: Maximum correlation allowed with recent selections
            max_features: Maximum number of features to select per iteration
        """
        self.method = method
        self.threshold = threshold
        self.cooldown_period = cooldown_period
        self.correlation_threshold = correlation_threshold
        self.max_features = max_features
        
        # History tracking (class-level to persist across iterations)
        if not hasattr(AutoFeatureSelection, '_global_history'):
            AutoFeatureSelection._global_history = []
    
    def select_features(self, simulation_results: pd.DataFrame, 
                       observations: ObservationData,
                       iteration: int = 1) -> List[str]:
        """Select features automatically using statistical metrics."""
        
        # Filter to features that exist in both simulation results and observations
        common_features = []
        for feature in simulation_results.columns:
            if observations.has_feature(feature):
                common_features.append(feature)
        
        if not common_features:
            raise ValueError("No common features between simulation results and observations")
        
        # Create subset with only common features
        feature_data = simulation_results[common_features]
        
        # Calculate feature statistics
        try:
            if self.method == 'fano':
                stats_df = features.Statistics.fano(feature_data)
                metric_values = stats_df['fano']
            elif self.method == 'var':
                metric_values = feature_data.var()
            elif self.method == 'mean':
                metric_values = feature_data.mean().abs()  # Use absolute mean
            elif self.method == 'std':
                metric_values = feature_data.std()
            else:
                # Try to get method from Statistics class
                if hasattr(features.Statistics, self.method):
                    stats_method = getattr(features.Statistics, self.method)
                    stats_df = stats_method(feature_data)
                    # Assume the method returns a DataFrame with method name as column
                    if self.method in stats_df.columns:
                        metric_values = stats_df[self.method]
                    else:
                        # Take the first column if method name not found
                        metric_values = stats_df.iloc[:, 0]
                else:
                    raise ValueError(f"Unknown feature selection method: {self.method}")
        except Exception as e:
            warnings.warn(f"Failed to calculate {self.method} statistics: {e}. "
                         f"Falling back to variance.")
            metric_values = feature_data.var()
        
        # Apply threshold if specified
        if self.threshold is not None:
            valid_mask = metric_values >= self.threshold
            if not valid_mask.any():
                warnings.warn(f"No features meet threshold {self.threshold}. "
                             f"Using best available feature.")
            else:
                metric_values = metric_values[valid_mask]
                feature_data = feature_data.loc[:, valid_mask]
        
        # Rank features by metric (descending order for most metrics)
        ranked_features = metric_values.abs().sort_values(ascending=False)
        
        selected_features = []
        
        # Select features that meet criteria
        for feature_name in ranked_features.index:
            if len(selected_features) >= self.max_features:
                break
                
            # Check if feature value is finite
            if not np.isfinite(ranked_features[feature_name]):
                continue
            
            # Check against history (cooldown period)
            if feature_name in AutoFeatureSelection._global_history:
                continue
            
            # Check correlation with already selected features
            reject_due_to_correlation = False
            for selected_feature in selected_features:
                correlation = feature_data[feature_name].corr(
                    feature_data[selected_feature], method='pearson'
                )
                if abs(correlation) >= self.correlation_threshold:
                    reject_due_to_correlation = True
                    break
            
            # Check correlation with recently selected features
            if not reject_due_to_correlation:
                for recent_feature in AutoFeatureSelection._global_history:
                    if recent_feature in feature_data.columns:
                        correlation = feature_data[feature_name].corr(
                            feature_data[recent_feature], method='pearson'
                        )
                        if abs(correlation) >= self.correlation_threshold:
                            reject_due_to_correlation = True
                            break
            
            if not reject_due_to_correlation:
                selected_features.append(feature_name)
        
        # If no features selected, take the best one regardless of criteria
        if not selected_features:
            best_feature = ranked_features.index[0]
            selected_features = [best_feature]
            warnings.warn(f"No features met selection criteria. Using best feature: {best_feature}")
        
        # Update global history
        AutoFeatureSelection._global_history.extend(selected_features)
        
        # Maintain history size
        while (len(AutoFeatureSelection._global_history) > self.cooldown_period or
               len(AutoFeatureSelection._global_history) >= len(common_features)):
            AutoFeatureSelection._global_history.pop(0)
        
        return selected_features
    
    def get_strategy_name(self) -> str:
        return f"Auto Selection (method={self.method}, max={self.max_features})"
    
    def reset_history(self):
        """Reset the selection history (useful for testing or restarting)."""
        AutoFeatureSelection._global_history.clear()


class InteractiveFeatureSelection(FeatureSelectionStrategy):
    """
    Interactive feature selection strategy.
    
    Prompts the user to select features interactively. Falls back to
    automatic selection when running in non-interactive environments.
    """
    
    def __init__(self, fallback_strategy: Optional[FeatureSelectionStrategy] = None):
        """
        Initialize interactive feature selection.
        
        Args:
            fallback_strategy: Strategy to use when interaction is not possible
        """
        self.fallback_strategy = fallback_strategy or AutoFeatureSelection()
    
    def select_features(self, simulation_results: pd.DataFrame, 
                       observations: ObservationData,
                       iteration: int = 1) -> List[str]:
        """Select features interactively or fall back to automatic selection."""
        
        # Get available features
        common_features = []
        for feature in simulation_results.columns:
            if observations.has_feature(feature):
                common_features.append(feature)
        
        if not common_features:
            raise ValueError("No common features between simulation results and observations")
        
        # Try interactive selection
        try:
            import sys
            
            # Check if we're in an interactive environment
            if sys.stdin.isatty():
                print(f"\nIteration {iteration}: Select features to emulate")
                print(f"Available features: {common_features}")
                
                while True:
                    user_input = input("Enter feature name(s) separated by commas (or 'auto' for automatic): ").strip()
                    
                    if user_input.lower() == 'auto':
                        return self.fallback_strategy.select_features(
                            simulation_results, observations, iteration
                        )
                    
                    # Parse user input
                    selected = [f.strip() for f in user_input.split(',') if f.strip()]
                    
                    if not selected:
                        print("Please enter at least one feature name.")
                        continue
                    
                    # Validate features
                    try:
                        valid_features = self.validate_features(
                            selected, simulation_results, observations
                        )
                        return valid_features
                    except ValueError as e:
                        print(f"Error: {e}")
                        print("Please try again.")
                        continue
            else:
                # Not interactive, use fallback
                return self.fallback_strategy.select_features(
                    simulation_results, observations, iteration
                )
                
        except (ImportError, OSError, KeyboardInterrupt):
            # Fall back to automatic selection
            warnings.warn("Interactive selection not available. Using automatic selection.")
            return self.fallback_strategy.select_features(
                simulation_results, observations, iteration
            )
    
    def get_strategy_name(self) -> str:
        return f"Interactive Selection (fallback: {self.fallback_strategy.get_strategy_name()})"


class MultiFeatureSelection(FeatureSelectionStrategy):
    """
    Multi-feature selection strategy.
    
    Selects multiple features per iteration using automatic methods.
    This is a convenience wrapper around AutoFeatureSelection with max_features > 1.
    """
    
    def __init__(self, n_features: int = 2, method: str = 'fano', 
                 correlation_threshold: float = 0.5):
        """
        Initialize multi-feature selection.
        
        Args:
            n_features: Number of features to select per iteration
            method: Statistical method for ranking features
            correlation_threshold: Maximum correlation allowed between selected features
        """
        self.auto_selector = AutoFeatureSelection(
            method=method,
            max_features=n_features,
            correlation_threshold=correlation_threshold
        )
    
    def select_features(self, simulation_results: pd.DataFrame, 
                       observations: ObservationData,
                       iteration: int = 1) -> List[str]:
        """Select multiple features automatically."""
        return self.auto_selector.select_features(simulation_results, observations, iteration)
    
    def get_strategy_name(self) -> str:
        return self.auto_selector.get_strategy_name().replace("Auto", "Multi")