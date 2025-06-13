__version__ = "1.0.0"

# History Matching Library - Object-Oriented API

# Core components for building and running history matching workflows  
from .core.builder import HistoryMatchingBuilder, quick_setup, advanced_setup  # noqa: F401 isort: skip
from .core.engine import HistoryMatchingEngine  # noqa: F401 isort: skip

# Domain objects for representing history matching concepts
from .domain.parameter_space import ParameterSpace  # noqa: F401 isort: skip
from .domain.observation_data import ObservationData  # noqa: F401 isort: skip
from .domain.emulator_bank import EmulatorBank  # noqa: F401 isort: skip
from .domain.iteration_result import IterationResult  # noqa: F401 isort: skip

# Strategy components for extensible algorithms
from .strategies.sampling import SamplingStrategyFactory  # noqa: F401 isort: skip
from .strategies.feature_selection import AutoFeatureSelection, ManualFeatureSelection  # noqa: F401 isort: skip
from .strategies.emulator_factory import EmulatorFactory  # noqa: F401 isort: skip

# Internal utilities (used by OOP components, not part of public API)
from .utils import PARAMETER_SPACE_COLUMNS, OBSERVATIONS_COLUMNS  # noqa: F401 isort: skip
