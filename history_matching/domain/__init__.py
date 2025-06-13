"""
Domain objects for history matching.

This module contains the core domain objects that encapsulate
the business logic and data structures for history matching.
"""

from .parameter_space import ParameterSpace
from .observation_data import ObservationData
from .emulator_bank import EmulatorBank
from .iteration_result import IterationResult

__all__ = [
    'ParameterSpace',
    'ObservationData', 
    'EmulatorBank',
    'IterationResult'
]