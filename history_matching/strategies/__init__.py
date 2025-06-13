"""
Strategy patterns for history matching.

This module contains strategy implementations for the key configurable
components of the history matching algorithm, including sampling methods,
feature selection algorithms, and emulator factories.
"""

from .sampling import (
    SamplingStrategy,
    LatinHypercubeSampling, 
    GridSampling,
    RandomSampling,
    SamplingStrategyFactory
)

from .feature_selection import (
    FeatureSelectionStrategy,
    ManualFeatureSelection,
    AutoFeatureSelection,
    InteractiveFeatureSelection
)

from .emulator_factory import EmulatorFactory

__all__ = [
    # Sampling strategies
    'SamplingStrategy',
    'LatinHypercubeSampling',
    'GridSampling', 
    'RandomSampling',
    'SamplingStrategyFactory',
    
    # Feature selection strategies
    'FeatureSelectionStrategy',
    'ManualFeatureSelection',
    'AutoFeatureSelection',
    'InteractiveFeatureSelection',
    
    # Emulator factory
    'EmulatorFactory'
]