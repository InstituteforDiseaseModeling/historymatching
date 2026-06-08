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

__version__ = "2.0.1"
__versiondate__ = "2026-06-07"

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
# Imported after its dependencies above: engine.py does `import historymatching as hm`
# and resolves hm.ParameterSpace etc. at class-definition time, so those names must
# already be bound here. (This is why engine is not imported first.)
from .engine import HistoryMatching, EngineState  # noqa: F401 isort: skip
from .emulators.results import EmulationResults  # noqa: F401 isort: skip
from .nroy_sampling import generate_nroy_design, NROYResult  # noqa: F401 isort: skip

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
    # Plotting helpers
    "plotting",
    "plot_convergence",
    "plot_marginals",
    "plot_pairplot",
    "plot_ensemble_fan",
    "plot_zscores_vs_targets",
    "plot_constrained_dims",
    "plot_targets",
    "plot_parameter_bounds",
    "plot_emulator_quality",
    "plot_predicted_vs_actual",
]
