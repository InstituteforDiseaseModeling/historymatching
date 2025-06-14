"""
HistoryMatchingBuilder for streamlined configuration and setup.

Provides a clean, intuitive API for configuring history matching workflows
without requiring deep knowledge of the underlying components.
"""

from typing import Optional, Union, Dict, Any, List
import pandas as pd
import warnings

try:
    from ..domain.parameter_space import ParameterSpace
    from ..domain.observation_data import ObservationData
    from ..domain.emulator_bank import EmulatorBank
    from ..strategies.sampling import SamplingStrategy, SamplingStrategyFactory
    from ..strategies.feature_selection import FeatureSelectionStrategy, AutoFeatureSelection, ManualFeatureSelection
    from ..strategies.emulator_factory import EmulatorFactory
    from .engine import HistoryMatchingEngine
except ImportError:
    # For standalone testing
    from history_matching.domain.parameter_space import ParameterSpace
    from history_matching.domain.observation_data import ObservationData
    from history_matching.domain.emulator_bank import EmulatorBank
    from history_matching.strategies.sampling import SamplingStrategy, SamplingStrategyFactory
    from history_matching.strategies.feature_selection import FeatureSelectionStrategy, AutoFeatureSelection, ManualFeatureSelection
    from history_matching.strategies.emulator_factory import EmulatorFactory
    from history_matching.core.engine import HistoryMatchingEngine


class HistoryMatchingBuilder:
    """
    Builder for creating HistoryMatchingEngine instances with streamlined configuration.
    
    Provides intuitive constructor options and smart defaults to make history matching
    accessible without requiring deep knowledge of internal components.
    
    Examples:
        # Quick start with minimal configuration
        builder = HistoryMatchingBuilder.from_data(
            parameter_bounds={'param1': (0, 1), 'param2': (-1, 1)},
            observations={'output1': (10.0, 1.0)}  # (target, std)
        )
        engine = builder.build()
        
        # More control with custom strategies
        builder = HistoryMatchingBuilder.from_dataframes(
            parameter_space_df=param_df,
            observations_df=obs_df,
            sampling_strategy='lhs',
            feature_selection=['output1', 'output2']
        )
        engine = builder.with_emulator_type('gpr').build()
    """
    
    def __init__(self):
        """Initialize empty builder. Use class methods for construction."""
        self._parameter_space: Optional[ParameterSpace] = None
        self._observations: Optional[ObservationData] = None
        self._sampling_strategy: Optional[SamplingStrategy] = None
        self._feature_selection_strategy: Optional[FeatureSelectionStrategy] = None
        self._emulator_factory: Optional[EmulatorFactory] = None
        self._emulator_bank: Optional[EmulatorBank] = None
        
        # Configuration options
        self._n_samples: int = 1000
        self._implausibility_threshold: float = 3.0
        self._max_iterations: int = 10
        self._random_seed: Optional[int] = None
        
        # Additional settings
        self._settings: Dict[str, Any] = {}
    
    @classmethod
    def from_data(cls, 
                  parameter_bounds: Dict[str, tuple], 
                  observations: Dict[str, tuple],
                  **kwargs) -> 'HistoryMatchingBuilder':
        """
        Create builder from simple data structures.
        
        Args:
            parameter_bounds: Dict mapping parameter names to (min, max) tuples
            observations: Dict mapping feature names to (target, std) tuples
            **kwargs: Additional configuration options
            
        Returns:
            Configured HistoryMatchingBuilder instance
        """
        builder = cls()
        
        # Create parameter space
        builder._parameter_space = ParameterSpace(parameter_bounds)
        
        # Create observations (std is used directly now)
        builder._observations = ObservationData(observations)
        
        # Apply additional configuration
        builder._apply_kwargs(kwargs)
        
        return builder
    
    @classmethod 
    def from_dataframes(cls,
                       parameter_space_df: pd.DataFrame,
                       observations_df: pd.DataFrame,
                       **kwargs) -> 'HistoryMatchingBuilder':
        """
        Create builder from pandas DataFrames.
        
        Args:
            parameter_space_df: DataFrame with parameter space definition
            observations_df: DataFrame with observation data
            **kwargs: Additional configuration options
            
        Returns:
            Configured HistoryMatchingBuilder instance
        """
        builder = cls()
        
        # Create domain objects
        builder._parameter_space = ParameterSpace(parameter_space_df)
        builder._observations = ObservationData(observations_df)
        
        # Apply additional configuration
        builder._apply_kwargs(kwargs)
        
        return builder
    
    @classmethod
    def from_existing(cls,
                     parameter_space: ParameterSpace,
                     observations: ObservationData,
                     **kwargs) -> 'HistoryMatchingBuilder':
        """
        Create builder from existing domain objects.
        
        Args:
            parameter_space: Pre-configured ParameterSpace
            observations: Pre-configured ObservationData
            **kwargs: Additional configuration options
            
        Returns:
            Configured HistoryMatchingBuilder instance
        """
        builder = cls()
        builder._parameter_space = parameter_space
        builder._observations = observations
        
        # Apply additional configuration
        builder._apply_kwargs(kwargs)
        
        return builder
    
    def _apply_kwargs(self, kwargs: Dict[str, Any]):
        """Apply keyword arguments to builder configuration."""
        # Sampling configuration
        if 'sampling_strategy' in kwargs:
            self.with_sampling_strategy(kwargs['sampling_strategy'])
        
        # Feature selection configuration
        if 'feature_selection' in kwargs:
            self.with_feature_selection(kwargs['feature_selection'])
        
        # Emulator configuration
        if 'emulator_type' in kwargs:
            self.with_emulator_type(kwargs['emulator_type'])
        
        # Workflow parameters
        if 'n_samples' in kwargs:
            self.with_samples_per_iteration(kwargs['n_samples'])
        
        if 'max_iterations' in kwargs:
            self.with_max_iterations(kwargs['max_iterations'])
        
        if 'implausibility_threshold' in kwargs:
            self.with_implausibility_threshold(kwargs['implausibility_threshold'])
        
        if 'random_seed' in kwargs:
            self.with_random_seed(kwargs['random_seed'])
        
        if 'auto_reduce_space' in kwargs:
            self.with_space_reduction(kwargs['auto_reduce_space'])
        
        if 'oversample_factor' in kwargs:
            self.with_oversample_factor(kwargs['oversample_factor'])
        
        # Store additional settings
        excluded_keys = {
            'sampling_strategy', 'feature_selection', 'emulator_type',
            'n_samples', 'max_iterations', 'implausibility_threshold', 'random_seed',
            'auto_reduce_space', 'oversample_factor'
        }
        self._settings.update({k: v for k, v in kwargs.items() if k not in excluded_keys})
    
    def with_sampling_strategy(self, strategy: Union[str, SamplingStrategy, Dict[str, Any]]) -> 'HistoryMatchingBuilder':
        """
        Configure sampling strategy.
        
        Args:
            strategy: Strategy name, strategy instance, or dict with type and parameters
            
        Returns:
            Self for method chaining
        """
        if isinstance(strategy, str):
            self._sampling_strategy = SamplingStrategyFactory.create(strategy)
        elif isinstance(strategy, SamplingStrategy):
            self._sampling_strategy = strategy
        elif isinstance(strategy, dict):
            strategy_type = strategy.pop('type', 'lhs')
            self._sampling_strategy = SamplingStrategyFactory.create(strategy_type, **strategy)
        else:
            raise ValueError(f"Invalid sampling strategy type: {type(strategy)}")
        
        return self
    
    def with_feature_selection(self, selection: Union[str, List[str], FeatureSelectionStrategy, Dict[str, Any]]) -> 'HistoryMatchingBuilder':
        """
        Configure feature selection strategy.
        
        Args:
            selection: Feature list, strategy name, strategy instance, or config dict
            
        Returns:
            Self for method chaining
        """
        if isinstance(selection, (str, list)):
            # Manual feature selection
            self._feature_selection_strategy = ManualFeatureSelection(selection)
        elif isinstance(selection, FeatureSelectionStrategy):
            self._feature_selection_strategy = selection
        elif isinstance(selection, dict):
            # Automatic feature selection with configuration
            method = selection.get('method', 'fano')
            max_features = selection.get('max_features', 1)
            threshold = selection.get('threshold', None)
            correlation_threshold = selection.get('correlation_threshold', 0.8)
            
            self._feature_selection_strategy = AutoFeatureSelection(
                method=method,
                threshold=threshold,
                max_features=max_features,
                correlation_threshold=correlation_threshold
            )
        else:
            raise ValueError(f"Invalid feature selection type: {type(selection)}")
        
        return self
    
    def with_emulator_type(self, emulator_type: str, **kwargs) -> 'HistoryMatchingBuilder':
        """
        Configure emulator factory.
        
        Args:
            emulator_type: Type of emulator ('linear', 'gpr', 'glm')
            **kwargs: Additional parameters for emulator factory
            
        Returns:
            Self for method chaining
        """
        self._emulator_factory = EmulatorFactory(default_type=emulator_type, **kwargs)
        return self
    
    def with_emulator_factory(self, factory: EmulatorFactory) -> 'HistoryMatchingBuilder':
        """
        Use custom emulator factory.
        
        Args:
            factory: Pre-configured EmulatorFactory instance
            
        Returns:
            Self for method chaining
        """
        self._emulator_factory = factory
        return self
    
    def with_samples_per_iteration(self, n_samples: int) -> 'HistoryMatchingBuilder':
        """
        Set number of samples per iteration.
        
        Args:
            n_samples: Number of samples to generate per iteration
            
        Returns:
            Self for method chaining
        """
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        self._n_samples = n_samples
        return self
    
    def with_max_iterations(self, max_iterations: int) -> 'HistoryMatchingBuilder':
        """
        Set maximum number of iterations.
        
        Args:
            max_iterations: Maximum iterations to run
            
        Returns:
            Self for method chaining
        """
        if max_iterations <= 0:
            raise ValueError("Max iterations must be positive")
        self._max_iterations = max_iterations
        return self
    
    def with_implausibility_threshold(self, threshold: float) -> 'HistoryMatchingBuilder':
        """
        Set implausibility threshold for parameter space reduction.
        
        Args:
            threshold: Implausibility threshold (typically 2.5-4.0)
            
        Returns:
            Self for method chaining
        """
        if threshold <= 0:
            raise ValueError("Implausibility threshold must be positive")
        self._implausibility_threshold = threshold
        return self
    
    def with_random_seed(self, seed: int) -> 'HistoryMatchingBuilder':
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed
            
        Returns:
            Self for method chaining
        """
        self._random_seed = seed
        return self
    
    def with_space_reduction(self, auto_reduce: bool) -> 'HistoryMatchingBuilder':
        """
        Enable or disable automatic parameter space reduction.
        
        Args:
            auto_reduce: Whether to automatically reduce parameter space based on implausibility
            
        Returns:
            Self for method chaining
        """
        self._settings['auto_reduce_space'] = auto_reduce
        return self
    
    def with_oversample_factor(self, factor: float) -> 'HistoryMatchingBuilder':
        """
        Set oversampling factor for sample filtering.
        
        Args:
            factor: Factor by which to oversample when filtering (e.g., 2.0 means generate 2x samples)
            
        Returns:
            Self for method chaining
        """
        if factor < 1.0:
            raise ValueError("Oversample factor must be >= 1.0")
        self._settings['oversample_factor'] = factor
        return self
    
    def with_emulator_bank(self, bank: EmulatorBank) -> 'HistoryMatchingBuilder':
        """
        Use existing emulator bank (for resuming workflows).
        
        Args:
            bank: Pre-configured EmulatorBank with existing emulators
            
        Returns:
            Self for method chaining
        """
        self._emulator_bank = bank
        return self
    
    def with_setting(self, key: str, value: Any) -> 'HistoryMatchingBuilder':
        """
        Add custom setting.
        
        Args:
            key: Setting name
            value: Setting value
            
        Returns:
            Self for method chaining
        """
        self._settings[key] = value
        return self
    
    def build(self) -> 'HistoryMatchingEngine':
        """
        Build the HistoryMatchingEngine with current configuration.
        
        Returns:
            Configured HistoryMatchingEngine ready for execution
            
        Raises:
            ValueError: If required components are missing
        """
        # Validate required components
        if self._parameter_space is None:
            raise ValueError("Parameter space is required. Use from_data(), from_dataframes(), or from_existing().")
        
        if self._observations is None:
            raise ValueError("Observations are required. Use from_data(), from_dataframes(), or from_existing().")
        
        # Apply smart defaults
        self._apply_defaults()
        
        # Create engine
        engine = HistoryMatchingEngine(
            parameter_space=self._parameter_space,
            observations=self._observations,
            sampling_strategy=self._sampling_strategy,
            feature_selection_strategy=self._feature_selection_strategy,
            emulator_factory=self._emulator_factory,
            emulator_bank=self._emulator_bank,
            n_samples=self._n_samples,
            implausibility_threshold=self._implausibility_threshold,
            max_iterations=self._max_iterations,
            random_seed=self._random_seed,
            **self._settings
        )
        
        return engine
    
    def _apply_defaults(self):
        """Apply smart defaults for unspecified components."""
        # Default sampling strategy
        if self._sampling_strategy is None:
            self._sampling_strategy = SamplingStrategyFactory.create('lhs')
            warnings.warn("No sampling strategy specified. Using Latin Hypercube Sampling.")
        
        # Default feature selection strategy
        if self._feature_selection_strategy is None:
            self._feature_selection_strategy = AutoFeatureSelection(method='fano', max_features=1)
            warnings.warn("No feature selection specified. Using automatic selection with Fano factor.")
        
        # Default emulator factory
        if self._emulator_factory is None:
            self._emulator_factory = EmulatorFactory(default_type='gpr')
            warnings.warn("No emulator type specified. Using Gaussian Process Regression (GPR).")
        
        # Default emulator bank
        if self._emulator_bank is None:
            self._emulator_bank = EmulatorBank()
    
    def preview_configuration(self) -> Dict[str, Any]:
        """
        Preview the current configuration without building.
        
        Returns:
            Dict with configuration summary
        """
        # Apply defaults temporarily for preview
        temp_sampling = self._sampling_strategy or SamplingStrategyFactory.create('lhs')
        temp_feature_selection = self._feature_selection_strategy or AutoFeatureSelection()
        temp_emulator_factory = self._emulator_factory or EmulatorFactory()
        
        return {
            'parameter_space': {
                'parameters': self._parameter_space.get_parameter_names() if self._parameter_space else None,
                'n_parameters': len(self._parameter_space.get_parameter_names()) if self._parameter_space else 0
            },
            'observations': {
                'features': self._observations.get_feature_names() if self._observations else None,
                'n_features': len(self._observations.get_feature_names()) if self._observations else 0
            },
            'sampling_strategy': temp_sampling.get_strategy_name(),
            'feature_selection_strategy': temp_feature_selection.get_strategy_name(),
            'emulator_type': temp_emulator_factory.get_default_type(),
            'workflow_settings': {
                'n_samples': self._n_samples,
                'max_iterations': self._max_iterations,
                'implausibility_threshold': self._implausibility_threshold,
                'random_seed': self._random_seed
            },
            'additional_settings': self._settings
        }
    
    def __repr__(self) -> str:
        """String representation of builder state."""
        config = self.preview_configuration()
        return (f"HistoryMatchingBuilder(\n"
                f"  parameters={config['parameter_space']['n_parameters']}, "
                f"features={config['observations']['n_features']},\n"
                f"  sampling='{config['sampling_strategy']}',\n"
                f"  feature_selection='{config['feature_selection_strategy']}',\n"
                f"  emulator='{config['emulator_type']}',\n"
                f"  n_samples={config['workflow_settings']['n_samples']}\n"
                f")")


# Convenience functions for quick setup
def quick_setup(parameter_bounds: Dict[str, tuple], 
               observations: Dict[str, tuple],
               **kwargs) -> HistoryMatchingEngine:
    """
    Quick setup function for simple history matching workflows.
    
    Args:
        parameter_bounds: Dict mapping parameter names to (min, max) tuples
        observations: Dict mapping feature names to (target, std) tuples
        **kwargs: Additional configuration options
        
    Returns:
        Ready-to-use HistoryMatchingEngine
    """
    return HistoryMatchingBuilder.from_data(parameter_bounds, observations, **kwargs).build()


def advanced_setup(parameter_space_df: pd.DataFrame,
                  observations_df: pd.DataFrame,
                  sampling_strategy: str = 'lhs',
                  emulator_type: str = 'gpr',
                  **kwargs) -> HistoryMatchingEngine:
    """
    Advanced setup function with more control options.
    
    Args:
        parameter_space_df: DataFrame with parameter space definition
        observations_df: DataFrame with observation data
        sampling_strategy: Sampling strategy name
        emulator_type: Emulator type name
        **kwargs: Additional configuration options
        
    Returns:
        Ready-to-use HistoryMatchingEngine
    """
    return (HistoryMatchingBuilder
            .from_dataframes(parameter_space_df, observations_df, **kwargs)
            .with_sampling_strategy(sampling_strategy)
            .with_emulator_type(emulator_type)
            .build())