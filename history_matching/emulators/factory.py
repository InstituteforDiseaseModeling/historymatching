"""
Emulator factory with strategy pattern for history matching.
"""

from typing import Dict, Type, Optional, Any, List
import time as _time
import pandas as pd
import logging

from .base import BaseEmulator
from .linear import LinearModel
from .glm import GLM
from .gpr import GPR
from .bayes_linear import BayesLinear

logger = logging.getLogger(__name__)


class EmulatorFactory:
    """
    Factory for creating and managing emulator instances.
    
    Provides a registry-based approach for creating emulators by type,
    with support for custom emulator registration and default parameters.
    """
    
    # Registry of available emulator types
    _emulator_registry: Dict[str, Type[BaseEmulator]] = {
        'linear': LinearModel,
        'glm': GLM,
        'gpr': GPR,
        'gaussian': GPR,  # alias for backward compatibility
        'bayes_linear': BayesLinear,
    }
    
    def __init__(self, default_type: str = 'gpr', **default_kwargs):
        """
        Initialize emulator factory.
        
        Args:
            default_type: Default emulator type to use
            **default_kwargs: Default parameters to pass to emulators
        """
        self.default_type = default_type.lower()
        self.default_kwargs = default_kwargs
        
        # Validate default type
        if self.default_type not in self._emulator_registry:
            available = list(self._emulator_registry.keys())
            raise ValueError(f"Unknown default emulator type: '{default_type}'. "
                           f"Available types: {available}")
    
    def create_emulator(self, X: pd.DataFrame, y: pd.DataFrame, 
                       emulator_type: Optional[str] = None, **kwargs) -> BaseEmulator:
        """
        Create a single emulator instance.
        
        Args:
            X: Input data (parameter samples)
            y: Output data (single feature/column)
            emulator_type: Type of emulator to create (uses default if None)
            **kwargs: Additional parameters for emulator constructor
            
        Returns:
            Configured emulator instance (not yet trained)
            
        Raises:
            ValueError: If emulator type is unknown
            TypeError: If y has multiple columns
        """
        # Use default type if not specified
        emulator_type = (emulator_type or self.default_type).lower()
        
        # Validate emulator type
        if emulator_type not in self._emulator_registry:
            available = list(self._emulator_registry.keys())
            raise ValueError(f"Unknown emulator type: '{emulator_type}'. "
                           f"Available types: {available}")
        
        # Validate input data
        if y.shape[1] != 1:
            raise TypeError(f"Output data must have exactly one column, got {y.shape[1]} columns. "
                          f"Use create_emulators_for_features() for multiple features.")
        
        # Merge default kwargs with specific kwargs (specific kwargs take precedence)
        emulator_kwargs = {**self.default_kwargs, **kwargs}
        
        # Get emulator class and create instance
        emulator_class = self._emulator_registry[emulator_type]
        
        try:
            emulator = emulator_class(X, y, **emulator_kwargs)
            logger.debug(f"Created {emulator_type} emulator for feature: {y.columns[0]}")
            return emulator
        except Exception as e:
            logger.error(f"Failed to create {emulator_type} emulator: {e}")
            raise
    
    def create_emulators_for_features(self, samples: pd.DataFrame, 
                                    simulation_results: pd.DataFrame,
                                    features: List[str],
                                    emulator_type: Optional[str] = None,
                                    **kwargs) -> Dict[str, BaseEmulator]:
        """
        Create emulators for multiple features.
        
        Args:
            samples: Parameter samples DataFrame
            simulation_results: Simulation results DataFrame
            features: List of feature names to create emulators for
            emulator_type: Type of emulator to create (uses default if None)
            **kwargs: Additional parameters for emulator constructors
            
        Returns:
            Dict mapping feature names to trained emulator instances
            
        Raises:
            ValueError: If any feature is not found in simulation_results
        """
        emulators = {}
        
        # Validate features exist in simulation results
        missing_features = [f for f in features if f not in simulation_results.columns]
        if missing_features:
            available = list(simulation_results.columns)
            raise ValueError(f"Features not found in simulation results: {missing_features}. "
                           f"Available features: {available}")
        
        # Create emulator for each feature
        for i, feature in enumerate(features, 1):
            try:
                # Extract single feature as DataFrame
                y_data = simulation_results[[feature]]

                # Create and train emulator
                logger.info(f"  Emulator {i}/{len(features)} [{feature}]: creating ({len(samples)} samples, "
                            f"{len(samples.columns)} params, type={emulator_type or self.default_type})...")
                emulator = self.create_emulator(samples, y_data, emulator_type, **kwargs)
                t0 = _time.time()
                logger.info(f"  Emulator {i}/{len(features)} [{feature}]: training...")
                emulator.train()
                elapsed = _time.time() - t0

                emulators[feature] = emulator
                logger.info(f"  Emulator {i}/{len(features)} [{feature}]: trained [{elapsed:.1f}s]")

            except Exception as e:
                logger.error(f"Failed to create emulator for feature '{feature}': {e}")
                raise
        
        return emulators
    
    def create_and_train_emulator(self, X: pd.DataFrame, y: pd.DataFrame,
                                 emulator_type: Optional[str] = None, **kwargs) -> BaseEmulator:
        """
        Create and immediately train an emulator.
        
        Convenience method that combines create_emulator() and train().
        
        Args:
            X: Input data (parameter samples)
            y: Output data (single feature/column)
            emulator_type: Type of emulator to create
            **kwargs: Additional parameters for emulator constructor
            
        Returns:
            Trained emulator instance
        """
        emulator = self.create_emulator(X, y, emulator_type, **kwargs)
        emulator.train()
        return emulator
    
    def with_defaults(self, **kwargs) -> 'EmulatorFactory':
        """
        Create new factory with updated default parameters.
        
        Args:
            **kwargs: Parameters to update in defaults
            
        Returns:
            New EmulatorFactory instance with updated defaults
        """
        new_defaults = {**self.default_kwargs, **kwargs}
        return EmulatorFactory(self.default_type, **new_defaults)
    
    def set_default_type(self, emulator_type: str) -> 'EmulatorFactory':
        """
        Create new factory with different default emulator type.
        
        Args:
            emulator_type: New default emulator type
            
        Returns:
            New EmulatorFactory instance with updated default type
        """
        return EmulatorFactory(emulator_type, **self.default_kwargs)
    
    @classmethod
    def register_emulator(cls, name: str, emulator_class: Type[BaseEmulator]):
        """
        Register a custom emulator type.
        
        Args:
            name: Name to register the emulator under
            emulator_class: BaseEmulator subclass to register
            
        Raises:
            TypeError: If emulator_class is not a BaseEmulator subclass
        """
        if not issubclass(emulator_class, BaseEmulator):
            raise TypeError(f"Emulator class must be a subclass of BaseEmulator, "
                          f"got {emulator_class}")
        
        cls._emulator_registry[name.lower()] = emulator_class
        logger.info(f"Registered custom emulator: {name} -> {emulator_class.__name__}")
    
    @classmethod
    def available_emulators(cls) -> List[str]:
        """
        Get list of available emulator types.
        
        Returns:
            List of registered emulator type names
        """
        return list(cls._emulator_registry.keys())
    
    @classmethod
    def get_emulator_info(cls, emulator_type: str) -> Dict[str, Any]:
        """
        Get information about an emulator type.
        
        Args:
            emulator_type: Name of the emulator type
            
        Returns:
            Dict with emulator information
            
        Raises:
            ValueError: If emulator type is unknown
        """
        emulator_type_lower = emulator_type.lower()
        
        if emulator_type_lower not in cls._emulator_registry:
            available = list(cls._emulator_registry.keys())
            raise ValueError(f"Unknown emulator type: '{emulator_type}'. "
                           f"Available types: {available}")
        
        emulator_class = cls._emulator_registry[emulator_type_lower]
        
        return {
            'name': emulator_type,
            'class': emulator_class.__name__,
            'module': emulator_class.__module__,
            'description': emulator_class.__doc__.strip() if emulator_class.__doc__ else "No description available"
        }
    
    @classmethod
    def with_defaults_class(cls, default_type: str = 'gpr', **default_kwargs) -> 'EmulatorFactory':
        """
        Class method to create factory with specific defaults.
        
        Args:
            default_type: Default emulator type
            **default_kwargs: Default parameters
            
        Returns:
            EmulatorFactory instance with specified defaults
        """
        return cls(default_type, **default_kwargs)
    
    def get_default_type(self) -> str:
        """Get the default emulator type."""
        return self.default_type
    
    def get_default_kwargs(self) -> Dict[str, Any]:
        """Get the default parameters."""
        return self.default_kwargs.copy()
    
    def __repr__(self) -> str:
        """String representation of the factory."""
        available_types = self.available_emulators()
        return (f"EmulatorFactory(default_type='{self.default_type}', "
                f"available_types={available_types})")


# Convenience functions for quick emulator creation
def create_linear_emulator(X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> BaseEmulator:
    """
    Create a linear emulator.
    
    Args:
        X: Input data
        y: Output data
        **kwargs: Additional parameters
        
    Returns:
        Linear emulator instance
    """
    factory = EmulatorFactory('linear')
    return factory.create_emulator(X, y, **kwargs)


def create_gpr_emulator(X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> BaseEmulator:
    """
    Create a GPR emulator.
    
    Args:
        X: Input data
        y: Output data
        **kwargs: Additional parameters
        
    Returns:
        GPR emulator instance
    """
    factory = EmulatorFactory('gpr')
    return factory.create_emulator(X, y, **kwargs)


def create_glm_emulator(X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> BaseEmulator:
    """
    Create a GLM emulator.

    Args:
        X: Input data
        y: Output data
        **kwargs: Additional parameters

    Returns:
        GLM emulator instance
    """
    factory = EmulatorFactory('glm')
    return factory.create_emulator(X, y, **kwargs)


def create_bayes_linear_emulator(X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> BaseEmulator:
    """
    Create a Bayes Linear emulator.

    Args:
        X: Input data
        y: Output data
        **kwargs: Additional parameters (e.g. nugget)

    Returns:
        BayesLinear emulator instance
    """
    factory = EmulatorFactory('bayes_linear')
    return factory.create_emulator(X, y, **kwargs)