__version__ = "1.0.0"

from .engine import HistoryMatching, HistoryMatchingEngine  # noqa: F401 isort: skip
from .parameter_space import ParameterSpace  # noqa: F401 isort: skip
from .observation_data import ObservationData  # noqa: F401 isort: skip
from .emulator_bank import EmulatorBank  # noqa: F401 isort: skip
from .iteration_result import IterationResult  # noqa: F401 isort: skip
from .sampling import SamplingStrategyFactory  # noqa: F401 isort: skip
from .feature_selection import AutoFeatureSelection, ManualFeatureSelection  # noqa: F401 isort: skip
from .emulators.factory import EmulatorFactory  # noqa: F401 isort: skip
from .utils import PARAMETER_SPACE_COLUMNS, OBSERVATIONS_COLUMNS  # noqa: F401 isort: skip

# ``HistoryMatch`` is accepted as a short alias for ``HistoryMatching``.
HistoryMatch = HistoryMatching
