"""
Feature selection strategy implementations for history matching.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import logging
import numpy as np
import pandas as pd
import warnings

logger = logging.getLogger(__name__)

from .observation_data import ObservationData


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
    
    def __init__(self, method: str = 'mean_sq_z', threshold: Optional[float] = None,
                 cooldown_period: int = 1, correlation_threshold: float = 0.8,
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
        
        # History tracking (instance-level for clean encapsulation)
        self.history = []
    
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
            raise ValueError(
                "None of the simulator's output columns match the observed features.\n"
                f"  Simulator returned: {list(simulation_results.columns)}\n"
                f"  Observations expect: {observations.get_feature_names()}\n"
                "Rename your simulator outputs (or your observation keys) so at least one matches."
            )
        
        # Create subset with only common features
        feature_data = simulation_results[common_features]
        
        # Normalize to z-scores before computing metrics so that features
        # on different scales (e.g. birth weight in grams vs rates in 0-1)
        # are comparable.  Without this, Fano/var always picks the feature
        # with the largest physical units.
        z_data = pd.DataFrame(index=feature_data.index)
        for feature in common_features:
            obs_mean, obs_std = observations.get_target_for_feature(feature)
            if obs_std > 0:
                z_data[feature] = (feature_data[feature] - obs_mean) / obs_std
            else:
                z_data[feature] = feature_data[feature] - obs_mean

        # Calculate feature statistics on z-scores
        try:
            if self.method == 'fano':
                # Fano factor on z-scores: variance / |mean|
                # High Fano = high spread relative to bias → emulator can help most
                means = z_data.mean()
                variances = z_data.var()
                metric_values = pd.Series(index=means.index, dtype=float)
                for feature in means.index:
                    if abs(means[feature]) > 1e-10:
                        metric_values[feature] = variances[feature] / abs(means[feature])
                    else:
                        metric_values[feature] = variances[feature]  # Pure variance when unbiased
            elif self.method == 'var':
                metric_values = z_data.var()
            elif self.method == 'mean':
                metric_values = z_data.mean().abs()
            elif self.method == 'std':
                metric_values = z_data.std()
            elif self.method in ('mean_sq_z', 'msz'):
                # Mean squared z-score: E[z²] = bias² + variance
                # Combines distance-from-target and spread in one metric.
                # Scale-invariant (z-scores). No tuning parameters.
                metric_values = (z_data ** 2).mean()
            else:
                raise ValueError(f"Unknown feature selection method: {self.method}. "
                                 f"Supported: 'mean_sq_z', 'fano', 'var', 'mean', 'std'")
        except Exception as e:
            warnings.warn(f"Failed to calculate {self.method} statistics: {e}. "
                         f"Falling back to variance of z-scores.")
            metric_values = z_data.var()
        
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

        # Log top candidates and cooldown state
        top_n = min(5, len(ranked_features))
        top_str = ", ".join(f"{f}={ranked_features[f]:.2f}" for f in ranked_features.index[:top_n])
        logger.info(f"  Feature ranking ({self.method}): {top_str}")
        if self.history:
            logger.info(f"  Cooldown (skip): {self.history}")

        selected_features = []

        # Select features that meet criteria
        for feature_name in ranked_features.index:
            if len(selected_features) >= self.max_features:
                break
                
            # Check if feature value is finite
            if not np.isfinite(ranked_features[feature_name]):
                continue
            
            # Check against history (cooldown period)
            if feature_name in self.history:
                continue
            
            # Check correlation with already selected features (on z-scores)
            reject_due_to_correlation = False
            for selected_feature in selected_features:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        correlation = z_data[feature_name].corr(
                            z_data[selected_feature], method='pearson'
                        )
                    # If correlation is NaN (e.g., constant data), treat as no correlation
                    if pd.isna(correlation):
                        correlation = 0.0
                    if abs(correlation) >= self.correlation_threshold:
                        reject_due_to_correlation = True
                        break
                except Exception:
                    # If correlation calculation fails, assume no correlation
                    continue
            
            # Check correlation with recently selected features (on z-scores)
            if not reject_due_to_correlation:
                for recent_feature in self.history:
                    if recent_feature in z_data.columns:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", RuntimeWarning)
                                correlation = z_data[feature_name].corr(
                                    z_data[recent_feature], method='pearson'
                                )
                            # If correlation is NaN (e.g., constant data), treat as no correlation
                            if pd.isna(correlation):
                                correlation = 0.0
                            if abs(correlation) >= self.correlation_threshold:
                                reject_due_to_correlation = True
                                break
                        except Exception:
                            # If correlation calculation fails, assume no correlation
                            continue
            
            if not reject_due_to_correlation:
                selected_features.append(feature_name)
        
        # If no features selected, take the best one regardless of criteria
        if not selected_features:
            best_feature = ranked_features.index[0]
            selected_features = [best_feature]
            logger.warning(f"  No features met selection criteria — forcing best: {best_feature}")

        for f in selected_features:
            score = ranked_features.get(f, float('nan'))
            mean_z = z_data[f].mean() if f in z_data.columns else float('nan')
            std_z = z_data[f].std() if f in z_data.columns else float('nan')
            logger.info(f"  SELECTED: {f} (score={score:.2f}, mean_z={mean_z:.2f}, std_z={std_z:.2f})")
        
        # Update instance history
        self.history.extend(selected_features)
        
        # Maintain history size
        while (len(self.history) > self.cooldown_period or
               len(self.history) >= len(common_features)):
            self.history.pop(0)
        
        return selected_features
    
    def get_strategy_name(self) -> str:
        return f"Auto Selection (method={self.method}, max={self.max_features})"
    
    def reset_history(self):
        """Reset the selection history (useful for testing or restarting)."""
        self.history.clear()


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
            raise ValueError(
                "None of the simulator's output columns match the observed features.\n"
                f"  Simulator returned: {list(simulation_results.columns)}\n"
                f"  Observations expect: {observations.get_feature_names()}\n"
                "Rename your simulator outputs (or your observation keys) so at least one matches."
            )
        
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
    
    def __init__(self, n_features: int = 2, method: str = 'mean_sq_z',
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