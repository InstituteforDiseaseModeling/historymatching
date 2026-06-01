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
from . import plotting


@dataclass(frozen=True)
class IterationResult:
    """
    Immutable result object from a single history matching iteration.

    Contains all data and results from one iteration, including samples,
    simulation results, trained emulators, and convergence diagnostics.

    The NROY set is not stored here — it is defined implicitly by the
    emulator bank.  To obtain NROY samples, generate fresh candidates
    and filter them through the engine's emulator bank.  The final wave's
    ``samples`` + ``simulation_results`` can be fed directly into
    trajectory selection (likelihood + resampling).
    """

    iteration: int
    parameter_space: ParameterSpace
    samples: pd.DataFrame
    simulation_results: pd.DataFrame
    selected_features: List[str]
    emulators: Dict[str, BaseEmulator]
    nroy_fraction: float
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
        
    def plot_emulator_diagnostics(self, feature: str):
        """
        Plot full diagnostics for a specific emulator (predicted-vs-actual,
        residuals, error distributions).

        The emulator is tested automatically if it has not been already.

        Args:
            feature: Feature name

        Returns:
            The list of Matplotlib figures created, or ``None`` if the emulator
            does not support plotting.
        """
        emulator = self.get_emulator_for_feature(feature)

        if hasattr(emulator, 'plot_diagnostics'):
            if hasattr(emulator, 'testing_complete') and not emulator.testing_complete:
                emulator.test()
            return emulator.plot_diagnostics()
        else:
            print(f"Emulator for feature '{feature}' does not support plotting")
            return None
            
    def calculate_space_reduction(self, original_space: ParameterSpace) -> float:
        """
        Calculate parameter space reduction factor.

        Args:
            original_space: Original parameter space before constraints

        Returns:
            Space reduction factor (higher means more reduction)
        """
        if self.nroy_fraction == 0:
            return float('inf')  # Complete reduction

        # Calculate volume-based reduction if we have bounds
        try:
            volume_fraction = self.parameter_space.volume_fraction_remaining(original_space)
            return 1.0 / volume_fraction if volume_fraction > 0 else float('inf')
        except:
            # Fall back to point-based reduction
            return 1.0 / self.nroy_fraction
            
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
            'nroy_fraction': self.nroy_fraction,
            'execution_time_seconds': self.execution_time_seconds,
            'parameter_count': len(self.parameter_space),
            'average_emulator_r2': avg_r2,
            'emulator_metrics': emulator_metrics
        }

    def summary(self) -> str:
        """Human-readable one-block summary of this wave.

        Returns:
            A short multi-line string with the sample count, NROY fraction,
            emulated features, and emulator count — the per-wave report users
            otherwise reassemble by hand after :meth:`step`.
        """
        avg_r2 = self.summary_statistics().get('average_emulator_r2')
        r2_str = f"{avg_r2:.3f}" if avg_r2 is not None else "n/a"
        return (f"Wave {self.iteration}: "
                f"{len(self.samples)} samples | "
                f"NROY fraction {self.nroy_fraction:.1%} | "
                f"features {self.selected_features} | "
                f"{len(self.emulators)} emulators (avg R²={r2_str})")

    def quality_table(self) -> pd.DataFrame:
        """Per-feature emulator quality as a tidy table.

        Returns:
            A DataFrame indexed by feature with columns ``r2``, ``mse`` and
            ``n_train`` — a sortable, copy-pasteable view of
            :meth:`get_emulator_quality_metrics`.  Renders nicely in notebooks.
        """
        metrics = self.get_emulator_quality_metrics()
        rows = []
        for feature in self.selected_features:
            m = metrics.get(feature, {})
            rows.append({
                "feature": feature,
                "r2": m.get("r2_score", float("nan")),
                "mse": m.get("mse", float("nan")),
                "n_train": m.get("training_size", None),
            })
        return pd.DataFrame(rows).set_index("feature")

    def plot_convergence(self, all_results: list, *, ax=None, log: bool = True):
        """Plot the NROY fraction across waves up to and including this one.

        Args:
            all_results: List of :class:`IterationResult` objects (with
                ``.iteration`` and ``.nroy_fraction``).
            ax: Existing Matplotlib axes to draw into.
            log: Use a logarithmic y-axis.

        Returns:
            The Matplotlib ``Axes`` containing the plot.
        """
        return plotting.plot_convergence(
            [r.iteration for r in all_results],
            [r.nroy_fraction for r in all_results], ax=ax, log=log)

    def plot_emulator_quality(self, *, ax=None):
        """Bar chart of per-feature emulator R² for this wave.

        Args:
            ax: Existing Matplotlib axes to draw into.

        Returns:
            The Matplotlib ``Axes`` containing the plot.
        """
        return plotting.plot_emulator_quality(self.get_emulator_quality_metrics(), ax=ax)

    def plot_predicted_vs_actual(self, feature: str, *, ax=None):
        """Predicted-vs-actual scatter for one feature's emulator.

        Args:
            feature: Feature whose emulator to plot.
            ax: Existing Matplotlib axes to draw into.

        Returns:
            The Matplotlib ``Axes`` containing the plot.
        """
        emulator = self.get_emulator_for_feature(feature)
        return emulator.plot_predicted_vs_actual(ax=ax, title=f"{feature}: predicted vs actual")

    def plot_all_emulator_diagnostics(self):
        """Show full diagnostic figures for every emulator trained this wave.

        Calls each emulator's :meth:`plot_diagnostics` (predicted-vs-actual,
        residuals, error distributions).  Returns nothing; figures render
        inline.
        """
        for feature, emulator in self.emulators.items():
            if not getattr(emulator, 'testing_complete', False):
                try:
                    emulator.test()
                except Exception:
                    continue
            print(f"── {feature} ──")
            emulator.plot_diagnostics()

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.summary()


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
        
            
    def save_diagnostics(self, fig_dir: str, all_results: Optional[list] = None) -> None:
        """Save per-wave emulator diagnostic figures and metrics summary.

        Creates ``{fig_dir}/wave{N}/`` with:

        Per emulator (``{feature}_diagnostics.png``):
          - **Predicted vs actual** scatter with 1:1 line and R²/MSE annotation
          - **ARD lengthscales** bar chart (GPR only)

        Wave-level summaries:
          - ``convergence.png`` — NROY fraction across waves
          - ``metrics.json`` — R², MSE, training size, ARD lengthscales

        For z-scores-vs-targets and pair plots, use the engine's auto-output
        (``builder.output_dir``), which has access to observations.

        Args:
            fig_dir: Directory to save figures into (created if needed).
            all_results: List of all IterationResult objects so far (including
                this one).  Needed for the convergence plot.
        """
        import matplotlib
        import matplotlib.pyplot as plt

        wave_dir = os.path.join(fig_dir, f"wave{self.iteration}")
        os.makedirs(wave_dir, exist_ok=True)

        all_metrics = self.get_emulator_quality_metrics()

        for feature, emulator in self.emulators.items():
            # Ensure tested
            if not getattr(emulator, 'testing_complete', False):
                try:
                    emulator.test()
                except Exception:
                    continue

            # Extract ARD lengthscales (if GPR)
            ard_ls = None
            ard_names = None
            model = getattr(emulator, 'model', None)
            if model is not None and hasattr(model, 'kernel'):
                try:
                    ls = model.kernel.lengthscales.numpy()
                    if ls.ndim > 0:
                        ard_names = (list(emulator.X_train_df.columns)
                                     if hasattr(emulator, 'X_train_df') else
                                     [f"dim_{i}" for i in range(len(ls))])
                        ard_ls = ls
                        ls_dict = {n: float(v) for n, v in zip(ard_names, ls)}
                        all_metrics.setdefault(feature, {})['ard_lengthscales'] = ls_dict
                except Exception:
                    pass

            has_ard = ard_ls is not None
            ncols = 2 if has_ard else 1
            fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5))
            if ncols == 1:
                axes = [axes]

            # ── Panel 1: Predicted vs actual ──────────────────────────
            em = getattr(emulator, 'emulator_metrics', {})
            n_train = len(emulator.X_train) if emulator.X_train is not None else None
            plotting.plot_predicted_vs_actual(
                emulator.y_test.flatten(), emulator.y_test_pred.flatten(),
                ax=axes[0], r2=em.get('R2'), mse=em.get('MSE'),
                n_train=n_train, title="Predicted vs actual")

            # ── Panel 2: ARD lengthscales ─────────────────────────────
            if has_ard:
                ax2 = axes[1]
                order = np.argsort(ard_ls)
                sorted_names = [ard_names[i] for i in order]
                sorted_ls = ard_ls[order]
                colors = ['#d44d4d' if v < np.median(ard_ls) else '#888888'
                          for v in sorted_ls]
                ax2.barh(range(len(sorted_ls)), sorted_ls, color=colors, height=0.7)
                ax2.set_yticks(range(len(sorted_ls)))
                ax2.set_yticklabels([n.replace('_', '\n') for n in sorted_names],
                                    fontsize=6)
                ax2.set_xlabel('Lengthscale (shorter = more relevant)', fontsize=8)
                ax2.set_title('ARD Lengthscales', fontsize=9)

            for ax in axes:
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)

            fig.suptitle(f"Wave {self.iteration} — {feature}",
                         fontsize=11, fontweight='bold', y=1.02)
            fig.tight_layout()
            path = os.path.join(wave_dir, f"{feature}_diagnostics.png")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        # ── Wave summary: NROY convergence ────────────────────────────
        if all_results is not None and len(all_results) > 0:
            ax = self.plot_convergence(all_results)
            fig = ax.figure
            fig.tight_layout()
            fig.savefig(os.path.join(wave_dir, "convergence.png"),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

        # ── Save metrics JSON ─────────────────────────────────────────
        metrics_path = os.path.join(wave_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2, default=float)

    def __post_init__(self):
        """Validate the iteration result after creation."""
        # Validate iteration number
        if self.iteration < 1:
            raise ValueError(f"Iteration must be >= 1, got {self.iteration}")
            
        # Validate data consistency
        if len(self.samples) != len(self.simulation_results):
            raise ValueError("Samples and simulation results must have same length")
            
        # Validate fractions
        if not 0.0 <= self.nroy_fraction <= 1.0:
            raise ValueError(f"NROY fraction must be between 0 and 1, got {self.nroy_fraction}")
            
        # Validate features
        for feature in self.selected_features:
            if feature not in self.emulators:
                raise ValueError(f"Selected feature '{feature}' missing from emulators")
                
        # Validate execution time
        if self.execution_time_seconds < 0:
            raise ValueError(f"Execution time must be non-negative, got {self.execution_time_seconds}")
            
    def __repr__(self) -> str:
        """Developer string representation."""
        return (f"IterationResult(iteration={self.iteration}, "
                f"n_samples={len(self.samples)}, "
                f"features={len(self.selected_features)}, "
                f"nroy_fraction={self.nroy_fraction:.3f})")