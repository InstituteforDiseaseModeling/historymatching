__version__ = "1.0.0"

from .builder import HistoryMatchingBuilder  # noqa: F401 isort: skip
from .engine import HistoryMatchingEngine  # noqa: F401 isort: skip
from .parameter_space import ParameterSpace  # noqa: F401 isort: skip
from .observation_data import ObservationData  # noqa: F401 isort: skip
from .emulator_bank import EmulatorBank  # noqa: F401 isort: skip
from .iteration_result import IterationResult  # noqa: F401 isort: skip
from .sampling import SamplingStrategyFactory  # noqa: F401 isort: skip
from .feature_selection import AutoFeatureSelection, ManualFeatureSelection  # noqa: F401 isort: skip
from .emulators.factory import EmulatorFactory  # noqa: F401 isort: skip
from .nroy_sampling import generate_nroy_design, NROYResult  # noqa: F401 isort: skip
from .utils import PARAMETER_SPACE_COLUMNS, OBSERVATIONS_COLUMNS  # noqa: F401 isort: skip

# Plotting helpers — also available as the ``historymatching.plotting`` module.
from . import plotting  # noqa: F401 isort: skip
from .plotting import (  # noqa: F401 isort: skip
    plot_convergence,
    plot_marginals,
    plot_pairplot,
    plot_ensemble_fan,
    plot_zscores_vs_targets,
    plot_constrained_dims,
    plot_targets,
    plot_parameter_bounds,
    plot_emulator_quality,
    plot_predicted_vs_actual,
)
