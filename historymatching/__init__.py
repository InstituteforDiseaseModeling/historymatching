"""History Matching — Bayesian calibration of simulation models.

The entry point is :class:`HistoryMatching`: configure a run with plain
arguments (your simulator ``function``, parameter bounds, and observations),
then call :meth:`~HistoryMatching.run`.

    import historymatching as hm

    engine = hm.HistoryMatching(
        function=my_simulator,
        bounds={'beta': (0.1, 0.5), 'gamma': (0.01, 0.1)},
        observations={'peak_infected': (150.0, 20.0)},  # (mean, std)
    )
    results = engine.run()
    plausible = engine.get_nroy_samples()

All the public building blocks are available at the top level (e.g.
``hm.ParameterSpace``, ``hm.RandomSampling``, ``hm.GPR``) so you rarely need to
import from submodules.
"""

__version__ = "1.0.0"

from .engine import HistoryMatching, EngineState  # noqa: F401 isort: skip
from .iteration_result import IterationResult  # noqa: F401 isort: skip
from .parameter_space import ParameterSpace  # noqa: F401 isort: skip
from .observation_data import ObservationData  # noqa: F401 isort: skip
from .feature_selection import (  # noqa: F401 isort: skip
    FeatureSelectionStrategy,
    AutoFeatureSelection,
    ManualFeatureSelection,
    InteractiveFeatureSelection,
    MultiFeatureSelection,
)
from .sampling import (  # noqa: F401 isort: skip
    SamplingStrategy,
    LatinHypercubeSampling,
    GridSampling,
    RandomSampling,
    SamplingStrategyFactory,
)
from .emulator_bank import EmulatorBank  # noqa: F401 isort: skip
from .emulators import (  # noqa: F401 isort: skip
    BaseEmulator,
    LinearModel,
    GLM,
    GPR,
    BayesLinear,
    EmulatorFactory,
)
from .emulators.results import EmulationResults  # noqa: F401 isort: skip
from .nroy_sampling import generate_nroy_design, NROYResult  # noqa: F401 isort: skip

__all__ = [
    "HistoryMatching",
    "EngineState",
    "IterationResult",
    "ParameterSpace",
    "ObservationData",
    "FeatureSelectionStrategy",
    "AutoFeatureSelection",
    "ManualFeatureSelection",
    "InteractiveFeatureSelection",
    "MultiFeatureSelection",
    "SamplingStrategy",
    "LatinHypercubeSampling",
    "GridSampling",
    "RandomSampling",
    "SamplingStrategyFactory",
    "EmulatorBank",
    "BaseEmulator",
    "LinearModel",
    "GLM",
    "GPR",
    "BayesLinear",
    "EmulatorFactory",
    "EmulationResults",
    "generate_nroy_design",
    "NROYResult",
]
