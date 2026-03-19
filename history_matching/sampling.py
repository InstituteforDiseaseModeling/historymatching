"""
Sampling strategy implementations for history matching.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Type
import numpy as np
import pandas as pd

from .parameter_space import ParameterSpace
from itertools import product

from scipy.stats.qmc import LatinHypercube as _LHSEngine


def _lhs(n_parameters: int, samples: int, seed: Optional[int] = None) -> "np.ndarray":
    """Drop-in replacement for pyDOE2.lhs using scipy.stats.qmc."""
    return _LHSEngine(d=n_parameters, seed=seed).random(n=samples)


class SamplingStrategy(ABC):
    """
    Abstract base class for parameter space sampling strategies.
    
    Sampling strategies generate parameter samples within a given parameter space
    for use in history matching iterations.
    """
    
    @abstractmethod
    def generate_samples(self, parameter_space: ParameterSpace, n_samples: int, 
                        seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate parameter samples within the given parameter space.
        
        Args:
            parameter_space: ParameterSpace defining the bounds
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility (optional)
            
        Returns:
            DataFrame with columns for each parameter and rows for each sample
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return human-readable name for this strategy."""
        pass
    
    def validate_parameters(self, **kwargs):
        """
        Validate strategy-specific parameters.
        
        Override in subclasses to add parameter validation.
        
        Raises:
            ValueError: If parameters are invalid
        """
        pass


class LatinHypercubeSampling(SamplingStrategy):
    """
    Latin Hypercube Sampling strategy.
    
    Generates samples using Latin Hypercube Sampling, which ensures
    good space-filling properties by dividing each parameter dimension
    into equally-sized intervals.
    """
    
    def __init__(self, criterion: str = 'maximin', iterations: int = 5):
        """
        Initialize Latin Hypercube sampling strategy.
        
        Args:
            criterion: Optimization criterion ('center', 'maximin', 'centermaximin', 'correlation')
            iterations: Number of optimization iterations
        """
        self.criterion = criterion
        self.iterations = iterations
        self.validate_parameters(criterion=criterion, iterations=iterations)
        
    def validate_parameters(self, **kwargs):
        """Validate LHS parameters."""
        criterion = kwargs.get('criterion', self.criterion)
        iterations = kwargs.get('iterations', self.iterations)
        
        valid_criteria = ['center', 'maximin', 'centermaximin', 'correlation']
        if criterion not in valid_criteria:
            raise ValueError(f"Invalid criterion '{criterion}'. Must be one of: {valid_criteria}")
            
        if iterations < 1:
            raise ValueError(f"Iterations must be >= 1, got {iterations}")
    
    def generate_samples(self, parameter_space: ParameterSpace, n_samples: int,
                        seed: Optional[int] = None) -> pd.DataFrame:
        """Generate Latin Hypercube samples."""
        # Convert ParameterSpace to legacy DataFrame format for existing sampler
        parameter_space_df = parameter_space.to_dataframe()

        # Generate Latin Hypercube samples directly
        n_parameters = parameter_space_df.shape[0]

        # Generate Latin Hypercube Samples in the unit hypercube [0, 1]
        lhs_samples = _lhs(n_parameters, n_samples, seed=seed)
        
        # Scale the samples to the ranges defined in parameter_space
        scaled_samples = np.zeros_like(lhs_samples)
        for i, (min_val, max_val) in enumerate(zip(parameter_space_df['minimum'], parameter_space_df['maximum'])):
            scaled_samples[:, i] = lhs_samples[:, i] * (max_val - min_val) + min_val
        
        # Create a DataFrame for the samples, using the parameter names as columns
        samples = pd.DataFrame(scaled_samples, columns=parameter_space_df['parameter'])
        
        return samples
    
    def get_strategy_name(self) -> str:
        return f"Latin Hypercube Sampling (criterion={self.criterion})"


class GridSampling(SamplingStrategy):
    """
    Grid sampling strategy.
    
    Generates samples on a regular grid in parameter space, providing
    systematic coverage of the space.
    """
    
    def __init__(self, samples_per_dimension: Optional[int] = None):
        """
        Initialize grid sampling strategy.
        
        Args:
            samples_per_dimension: Number of samples per dimension (optional)
                                 If None, calculated from total n_samples
        """
        self.samples_per_dimension = samples_per_dimension
        self.validate_parameters(samples_per_dimension=samples_per_dimension)
        
    def validate_parameters(self, **kwargs):
        """Validate grid sampling parameters."""
        samples_per_dim = kwargs.get('samples_per_dimension', self.samples_per_dimension)
        
        if samples_per_dim is not None and samples_per_dim < 1:
            raise ValueError(f"samples_per_dimension must be >= 1, got {samples_per_dim}")
    
    def generate_samples(self, parameter_space: ParameterSpace, n_samples: int, 
                        seed: Optional[int] = None) -> pd.DataFrame:
        """Generate grid samples."""
        # Convert ParameterSpace to legacy DataFrame format for existing sampler
        parameter_space_df = parameter_space.to_dataframe()
        
        # Generate grid samples directly
        n_parameters = parameter_space_df.shape[0]
        n_steps = max(1, int(n_samples ** (1 / n_parameters)))
        
        # Create a list of evenly spaced values for each parameter using n_steps
        grid_ranges = [np.linspace(row['minimum'], row['maximum'], n_steps) 
                      for _, row in parameter_space_df.iterrows()]
        
        # Generate the Cartesian product of the grid ranges (all combinations of values)
        grid_samples = list(product(*grid_ranges))
        
        # Convert the list of grid samples to a DataFrame
        samples = pd.DataFrame(grid_samples, columns=parameter_space_df['parameter'])
        
        return samples
    
    def get_strategy_name(self) -> str:
        if self.samples_per_dimension:
            return f"Grid Sampling ({self.samples_per_dimension} per dimension)"
        return "Grid Sampling"


class RandomSampling(SamplingStrategy):
    """
    Random sampling strategy.
    
    Generates uniformly random samples within the parameter space bounds.
    """
    
    def __init__(self):
        """Initialize random sampling strategy."""
        pass
    
    def generate_samples(self, parameter_space: ParameterSpace, n_samples: int, 
                        seed: Optional[int] = None) -> pd.DataFrame:
        """Generate random samples."""
        if seed is not None:
            np.random.seed(seed)
            
        # Convert ParameterSpace to legacy DataFrame format for existing sampler
        parameter_space_df = parameter_space.to_dataframe()
        
        # Generate random samples directly
        samples = pd.DataFrame()
        
        for entry in parameter_space_df.itertuples():
            points = np.random.default_rng().uniform(entry.minimum, entry.maximum, n_samples)
            samples[entry.parameter] = points
        
        return samples
    
    def get_strategy_name(self) -> str:
        return "Random Sampling"


class SamplingStrategyFactory:
    """
    Factory for creating sampling strategy instances.
    
    Provides a registry-based approach for creating sampling strategies
    by name, with support for custom strategy registration.
    """
    
    _strategies: Dict[str, Type[SamplingStrategy]] = {
        'lhs': LatinHypercubeSampling,
        'latin_hypercube': LatinHypercubeSampling,
        'grid': GridSampling,
        'random': RandomSampling,
        'uniform': RandomSampling,  # alias
    }
    
    @classmethod
    def create(cls, strategy_name: str, **kwargs) -> SamplingStrategy:
        """
        Create a sampling strategy by name.
        
        Args:
            strategy_name: Name of the strategy to create
            **kwargs: Strategy-specific parameters
            
        Returns:
            Configured SamplingStrategy instance
            
        Raises:
            ValueError: If strategy name is unknown
        """
        strategy_name_lower = strategy_name.lower()
        
        if strategy_name_lower not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(f"Unknown sampling strategy: '{strategy_name}'. "
                           f"Available strategies: {available}")
        
        strategy_class = cls._strategies[strategy_name_lower]
        return strategy_class(**kwargs)
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[SamplingStrategy]):
        """
        Register a custom sampling strategy.
        
        Args:
            name: Name to register the strategy under
            strategy_class: SamplingStrategy subclass to register
            
        Raises:
            TypeError: If strategy_class is not a SamplingStrategy subclass
        """
        if not issubclass(strategy_class, SamplingStrategy):
            raise TypeError(f"Strategy class must be a subclass of SamplingStrategy, "
                          f"got {strategy_class}")
        
        cls._strategies[name.lower()] = strategy_class
    
    @classmethod
    def available_strategies(cls) -> list[str]:
        """
        Get list of available strategy names.
        
        Returns:
            List of registered strategy names
        """
        return list(cls._strategies.keys())
    
    @classmethod
    def get_strategy_info(cls, strategy_name: str) -> Dict[str, str]:
        """
        Get information about a strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dict with strategy information
            
        Raises:
            ValueError: If strategy name is unknown
        """
        if strategy_name.lower() not in cls._strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
            
        strategy_class = cls._strategies[strategy_name.lower()]
        
        return {
            'name': strategy_name,
            'class': strategy_class.__name__,
            'description': strategy_class.__doc__.strip() if strategy_class.__doc__ else "No description available"
        }