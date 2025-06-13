__version__ = "0.9.0"

# Import the following to make them local to history_matching module.

# New Object-Oriented API (recommended)
from .core.builder import HistoryMatchingBuilder, quick_setup, advanced_setup  # noqa: F401 isort: skip
from .core.engine import HistoryMatchingEngine  # noqa: F401 isort: skip

# Domain objects
from .domain.parameter_space import ParameterSpace  # noqa: F401 isort: skip
from .domain.observation_data import ObservationData  # noqa: F401 isort: skip
from .domain.emulator_bank import EmulatorBank  # noqa: F401 isort: skip
from .domain.iteration_result import IterationResult  # noqa: F401 isort: skip

# Strategy components
from .strategies.sampling import SamplingStrategyFactory  # noqa: F401 isort: skip
from .strategies.feature_selection import AutoFeatureSelection, ManualFeatureSelection  # noqa: F401 isort: skip
from .strategies.emulator_factory import EmulatorFactory  # noqa: F401 isort: skip

# Legacy configuration-based API (deprecated)
from .config import Config  # noqa: F401 isort: skip
from .constrict import next_point_generation  # noqa: F401 isort: skip
from .samplers import grid as grid_sampler, lhs as latin_hypercube_sampler, random as random_sampler  # noqa: F401 isort: skip
from .step import do_step, reduce_space  # noqa: F401 isort: skip
from .utils import mean_and_variance_for_observations, features_from_observations  # noqa: F401 isort: skip
from .utils import PARAMETER_SPACE_COLUMNS, OBSERVATIONS_COLUMNS  # noqa: F401 isort: skip
