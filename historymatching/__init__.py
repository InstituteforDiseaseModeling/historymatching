"""History Matching — Bayesian calibration of simulation models.

The entry point is :class:`HistoryMatching`: configure a run with plain
arguments (parameter bounds, observations, and your simulator ``function``),
then call :meth:`~HistoryMatching.run`.

    import historymatching as hm

    match = hm.HistoryMatching(
        function=my_simulator,
        parameter_bounds={'beta': (0.1, 0.5), 'gamma': (0.01, 0.1)},
        observations={'peak_infected': (150.0, 20.0)},  # (mean, std)
    )
    results = match.run()
    plausible = match.get_nroy_samples()
"""

__version__ = "1.0.0"

from .engine import HistoryMatching  # noqa: F401 isort: skip
from .iteration_result import IterationResult  # noqa: F401 isort: skip
from .parameter_space import ParameterSpace  # noqa: F401 isort: skip
from .observation_data import ObservationData  # noqa: F401 isort: skip
from .feature_selection import AutoFeatureSelection, ManualFeatureSelection  # noqa: F401 isort: skip
from .emulators.factory import EmulatorFactory  # noqa: F401 isort: skip
from .emulator_bank import EmulatorBank  # noqa: F401 isort: skip
from .sampling import SamplingStrategyFactory  # noqa: F401 isort: skip
from .plotting import plot_ensemble_fan  # noqa: F401 isort: skip

__all__ = [
    "HistoryMatching",
    "IterationResult",
    "ParameterSpace",
    "ObservationData",
    "AutoFeatureSelection",
    "ManualFeatureSelection",
    "EmulatorFactory",
    "EmulatorBank",
    "SamplingStrategyFactory",
    "plot_ensemble_fan",
]
