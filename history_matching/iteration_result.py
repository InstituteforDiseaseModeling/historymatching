"""
IterationResult domain object for history matching.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import os
import json
import numpy as np
import pandas as pd

from .parameter_space import ParameterSpace
from .observation_data import ObservationData
from .emulator_bank import EmulatorBank
from .emulators.base import BaseEmulator


@dataclass(frozen=True)
class IterationResult:
    """
    Immutable result object from a single history matching iteration.
    
    Contains all data and results from one iteration, including samples,
    simulation results, trained emulators, and analysis results.
    """
    
    iteration: int
    parameter_space: ParameterSpace
    samples: pd.DataFrame
    simulation_results: pd.DataFrame
    selected_features: List[str]
    emulators: Dict[str, BaseEmulator]
    non_implausible_points: pd.DataFrame
    non_implausible_fraction: float
    execution_time_seconds: float
    
    def get_emulator_for_feature(self, feature: str) -> BaseEmulator:
        """
        Get emulator for a specific feature.
        
        Args:
            feature: Feature name
            
        Returns:
            Emulator instance
            
        Raises:
            KeyError: If feature not found
        """
        if feature not in self.emulators:
            available_features = list(self.emulators.keys())
            raise KeyError(f"Feature '{feature}' not found. Available features: {available_features}")
            
        return self.emulators[feature]
        
    def plot_emulator_diagnostics(self, feature: str, **kwargs):
        """
        Plot diagnostics for a specific emulator.
        
        Args:
            feature: Feature name
            **kwargs: Additional arguments passed to emulator's plot method
        """
        emulator = self.get_emulator_for_feature(feature)
        
        if hasattr(emulator, 'plot_diagnostics'):
            emulator.plot_diagnostics(**kwargs)
        elif hasattr(emulator, 'plot'):
            emulator.plot(**kwargs)
        else:
            print(f"Emulator for feature '{feature}' does not support plotting")
            
    def calculate_space_reduction(self, original_space: ParameterSpace) -> float:
        """
        Calculate parameter space reduction factor.
        
        Args:
            original_space: Original parameter space before constraints
            
        Returns:
            Space reduction factor (higher means more reduction)
        """
        if self.non_implausible_fraction == 0:
            return float('inf')  # Complete reduction
            
        # Calculate volume-based reduction if we have bounds
        try:
            volume_fraction = self.parameter_space.volume_fraction_remaining(original_space)
            return 1.0 / volume_fraction if volume_fraction > 0 else float('inf')
        except:
            # Fall back to point-based reduction
            return 1.0 / self.non_implausible_fraction
            
    def get_implausibility_scores(self, observations: ObservationData, 
                                 model_discrepancy: float = 0.0) -> pd.DataFrame:
        """
        Calculate implausibility scores for all sample points.
        
        Args:
            observations: ObservationData instance for comparison
            model_discrepancy: Additional model uncertainty
            
        Returns:
            DataFrame with samples and their implausibility scores
        """
        # Start with samples DataFrame
        result_df = self.samples.copy()
        
        # Add implausibility scores for each feature
        for feature in self.selected_features:
            if feature in self.emulators and observations.has_feature(feature):
                emulator = self.emulators[feature]
                
                # Get predictions for all samples
                predictions = emulator.predict(self.samples)
                
                # Calculate implausibilities (vectorized)
                implausibilities = observations.calculate_implausibility(
                    feature, predictions.get_mean(), predictions.get_variance(), model_discrepancy
                )
                    
                result_df[f'implausibility_{feature}'] = implausibilities
                
        # Calculate maximum implausibility across all features
        impl_columns = [col for col in result_df.columns if col.startswith('implausibility_')]
        if impl_columns:
            result_df['max_implausibility'] = result_df[impl_columns].max(axis=1)
            
        return result_df
        
    def get_emulator_quality_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get quality metrics for all trained emulators.
        
        Returns:
            Dict mapping feature names to their quality metrics
        """
        metrics = {}
        
        for feature, emulator in self.emulators.items():
            feature_metrics = {}

            # Ensure emulator has been tested (lazy — test() is not called
            # by the engine automatically, only train() is).
            if hasattr(emulator, 'test') and not getattr(emulator, 'testing_complete', False):
                try:
                    emulator.test()
                except Exception:
                    pass  # Testing may fail; metrics will remain absent

            # Pull metrics from emulator_metrics dict (populated by test())
            em_metrics = getattr(emulator, 'emulator_metrics', {})
            if 'R2' in em_metrics:
                feature_metrics['r2_score'] = float(em_metrics['R2'])
            if 'MSE' in em_metrics:
                feature_metrics['mse'] = float(em_metrics['MSE'])

            # Try to get training data size
            if hasattr(emulator, 'X_train') and emulator.X_train is not None:
                feature_metrics['training_size'] = len(emulator.X_train)

            # Store completion status
            feature_metrics['training_complete'] = getattr(emulator, 'training_complete', False)
            feature_metrics['testing_complete'] = getattr(emulator, 'testing_complete', False)
            
            metrics[feature] = feature_metrics
            
        return metrics
        
    def summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for this iteration.
        
        Returns:
            Dict with key metrics and statistics
        """
        emulator_metrics = self.get_emulator_quality_metrics()
        
        # Calculate average R² if available
        r2_scores = [metrics.get('r2_score') for metrics in emulator_metrics.values() 
                    if 'r2_score' in metrics]
        avg_r2 = np.mean(r2_scores) if r2_scores else None
        
        return {
            'iteration': self.iteration,
            'n_samples': len(self.samples),
            'n_features': len(self.selected_features),
            'selected_features': self.selected_features,
            'non_implausible_points': len(self.non_implausible_points),
            'non_implausible_fraction': self.non_implausible_fraction,
            'execution_time_seconds': self.execution_time_seconds,
            'parameter_count': len(self.parameter_space),
            'average_emulator_r2': avg_r2,
            'emulator_metrics': emulator_metrics
        }
        
    def export_emulators(self, directory_path: str):
        """
        Save emulators to specified directory.
        
        Args:
            directory_path: Directory to save emulators
        """
        # Create emulator bank and add this iteration's emulators
        bank = EmulatorBank()
        for feature, emulator in self.emulators.items():
            bank.add_emulator(self.iteration, feature, emulator)
            
        # Save to directory
        bank.save_to_directory(directory_path)
        
    def export_results(self, file_path: str, format: str = 'json'):
        """
        Export key results to file.
        
        Args:
            file_path: Path for output file
            format: Output format ('json' or 'csv')
        """
        summary = self.summary_statistics()
        
        if format.lower() == 'json':
            # Convert numpy types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
                
            # Deep convert all values
            def deep_convert(d):
                if isinstance(d, dict):
                    return {k: deep_convert(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [deep_convert(v) for v in d]
                else:
                    return convert_types(d)
                    
            summary_clean = deep_convert(summary)
            
            with open(file_path, 'w') as f:
                json.dump(summary_clean, f, indent=2)
                
        elif format.lower() == 'csv':
            # Flatten the summary for CSV
            flat_data = []
            
            # Basic metrics
            basic_metrics = {k: v for k, v in summary.items() 
                           if not isinstance(v, (dict, list))}
            flat_data.append(basic_metrics)
            
            # Save as single-row CSV
            df = pd.DataFrame(flat_data)
            df.to_csv(file_path, index=False)
            
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'")
            
    def export_samples_and_results(self, directory_path: str):
        """
        Export samples and simulation results to CSV files.
        
        Args:
            directory_path: Directory to save files
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            
        # Export samples
        samples_file = os.path.join(directory_path, f"iteration_{self.iteration}_samples.csv")
        self.samples.to_csv(samples_file, index=False)
        
        # Export simulation results
        results_file = os.path.join(directory_path, f"iteration_{self.iteration}_simulation_results.csv")
        self.simulation_results.to_csv(results_file, index=False)
        
        # Export non-implausible points
        if not self.non_implausible_points.empty:
            non_impl_file = os.path.join(directory_path, f"iteration_{self.iteration}_non_implausible.csv")
            self.non_implausible_points.to_csv(non_impl_file, index=False)
            
    def __post_init__(self):
        """Validate the iteration result after creation."""
        # Validate iteration number
        if self.iteration < 1:
            raise ValueError(f"Iteration must be >= 1, got {self.iteration}")
            
        # Validate data consistency
        if len(self.samples) != len(self.simulation_results):
            raise ValueError("Samples and simulation results must have same length")
            
        # Validate fractions
        if not 0.0 <= self.non_implausible_fraction <= 1.0:
            raise ValueError(f"Non-implausible fraction must be between 0 and 1, got {self.non_implausible_fraction}")
            
        # Validate features
        for feature in self.selected_features:
            if feature not in self.emulators:
                raise ValueError(f"Selected feature '{feature}' missing from emulators")
                
        # Validate execution time
        if self.execution_time_seconds < 0:
            raise ValueError(f"Execution time must be non-negative, got {self.execution_time_seconds}")
            
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (f"IterationResult(iteration={self.iteration}, "
                f"features={self.selected_features}, "
                f"non_implausible_fraction={self.non_implausible_fraction:.3f})")
                
    def __repr__(self) -> str:
        """Developer string representation."""
        return (f"IterationResult(iteration={self.iteration}, "
                f"n_samples={len(self.samples)}, "
                f"features={len(self.selected_features)}, "
                f"non_implausible_fraction={self.non_implausible_fraction:.3f})")