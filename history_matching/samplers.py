"""Parameter space point samplers."""
from itertools import product
import numpy as np
import pandas as pd
from pyDOE import lhs




def get_samples( parameter_space: pd.DataFrame, n_samples: int = 8, method='lhs' ) -> pd.DataFrame:
    """
    Generate samples based on the selected method.

    Args:
        parameter_space: a DataFrame with columns 'parameter', 'minimum', and 'maximum'
        n_samples: number of samples
        method: name of the method to use. It can be:
                'lhs'  : Latin Hypercube Sampling;
                'grid'   : Grid of samples in the parameter space;
                'random' : Random samples.

    Returns:
        a DataFrame with one row per sample, and one column per parameter
    """
    # Map sampling methods to their corresponding functions
    sampling_methods = { 'lhs'   : lhs_sampler,
                         'random': random,
                         'grid'  : grid,
                        }
    
    # Draw samples
    if method in sampling_methods:
        new_candidates = sampling_methods[method]( parameter_space, n_samples )
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    return new_candidates




def lhs_sampler(parameter_space: pd.DataFrame, n_samples: int = 8) -> pd.DataFrame:
    """
    Generate a Latin hypercube sample of points in parameter space.

    Args:
        parameter_space: a DataFrame with columns 'parameter', 'minimum', and 'maximum'
        n_samples: number of samples

    Returns:
        a DataFrame with one row per sample, and one column per parameter
    """
    # Extract the number of parameters
    n_parameters = parameter_space.shape[0]
    
    # Generate Latin Hypercube Samples in the unit hypercube [0, 1]
    lhs_samples = lhs( n_parameters, samples=n_samples )
    
    # Scale the samples to the ranges defined in parameter_space
    scaled_samples = np.zeros_like(lhs_samples)
    for i, ( min_val, max_val ) in enumerate( zip(parameter_space['minimum'], parameter_space['maximum']) ):
        scaled_samples[:, i] = lhs_samples[:, i] * (max_val - min_val) + min_val
    
    # Create a DataFrame for the samples, using the parameter names as columns
    samples = pd.DataFrame(scaled_samples, columns=parameter_space['parameter'])
    return samples




def grid(parameter_space: pd.DataFrame, n_samples: int = 16) -> pd.DataFrame:
    """
    Generate a grid of samples in parameter space.

    Args:
        parameter_space: a DataFrame with columns 'parameter', 'minimum', and 'maximum'
        n_samples: number of samples

    Returns:
        a DataFrame with one row per sample, and one column per parameter
    """
    # Calculate the number of steps per parameter (round to nearest integer)
    n_parameters = parameter_space.shape[0]
    n_steps = max( 1, int( n_samples ** (1 / n_parameters) ) )
    
    # Create a list of evenly spaced values for each parameter using n_steps
    grid_ranges = [ np.linspace(row['minimum'], row['maximum'], n_steps) 
                                for _, row in parameter_space.iterrows() ]
    
    # Generate the Cartesian product of the grid ranges (all combinations of values)
    grid_samples = list( product(*grid_ranges) )
    
    # Convert the list of grid samples to a DataFrame
    samples = pd.DataFrame( grid_samples, columns=parameter_space['parameter'] )
    return samples




def random(parameter_space: pd.DataFrame, n_samples: int = 16) -> pd.DataFrame:
    """
    Generate a random sample of points in parameter space.

    Args:
        parameter_space: a DataFrame with columns 'parameter', 'minimum', and 'maximum'
        n_samples: number of samples

    Returns:
        a DataFrame with one row per sample, and one column per parameter
    """

    samples = pd.DataFrame()

    for entry in parameter_space.itertuples():
        points = np.random.default_rng().uniform(entry.minimum, entry.maximum, n_samples)
        samples[entry.parameter] = points

    return samples
