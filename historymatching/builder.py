"""
HistoryMatchingBuilder for streamlined configuration and setup.

Provides a clean, intuitive API for configuring history matching workflows
without requiring deep knowledge of the underlying components.

Configuration is done via plain public attributes (e.g. ``builder.n_samples = 500``);
``build()`` validates the configuration and constructs a HistoryMatchingEngine.
"""

from typing import Optional, Union, Dict, Any, List
import pandas as pd
import warnings

from .parameter_space import ParameterSpace
from .observation_data import ObservationData
from .emulator_bank import EmulatorBank
from .sampling import SamplingStrategy, SamplingStrategyFactory
from .feature_selection import FeatureSelectionStrategy, AutoFeatureSelection, ManualFeatureSelection
from .emulators.factory import EmulatorFactory
from .engine import HistoryMatchingEngine


# Valid NROY sampling methods (shared with validate()).
_VALID_NROY_METHODS = ('auto', 'lhs', 'ray')


class HistoryMatchingBuilder:
    """
    Builder for creating HistoryMatchingEngine instances with streamlined configuration.

    Configure the builder by assigning to its public attributes, then call
    :meth:`build`.  Friendly values (strings, lists, dicts) are accepted for the
    strategy attributes and coerced into the underlying objects at build time.

    Examples:
        # Quick start with minimal configuration
        builder = HistoryMatchingBuilder.from_data(
            parameter_bounds={'param1': (0, 1), 'param2': (-1, 1)},
            observations={'output1': (10.0, 1.0)}  # (target, std)
        )
        engine = builder.build()

        # More control via plain attribute assignment
        builder = HistoryMatchingBuilder.from_dataframes(
            parameter_space_df=param_df,
            observations_df=obs_df,
        )
        builder.emulator_type = 'gpr'
        builder.n_samples = 500
        builder.feature_selection = ['output1', 'output2']
        engine = builder.build()

    Attributes:
        parameter_space: ParameterSpace (set via the ``from_*`` constructors).
        observations: ObservationData (set via the ``from_*`` constructors).
        sampling_strategy: ``'lhs'`` / ``'grid'`` / ``'random'``, a SamplingStrategy
            instance, or a config dict.  ``None`` -> Latin Hypercube default.
        feature_selection: feature name list, strategy name, FeatureSelectionStrategy
            instance, or config dict.  ``None`` -> automatic mean-sq-z default.
        emulator_type: ``'gpr'`` (default) / ``'glm'`` / ``'linear'``.
        emulator_factory: a pre-built EmulatorFactory (overrides ``emulator_type``).
        emulator_bank: a pre-populated EmulatorBank (for resuming workflows).
        n_samples: samples generated per iteration (default 1000).
        implausibility_threshold: implausibility cutoff, typically 2.5-4.0 (default 3.0).
        max_iterations: maximum iterations to run (default 10).
        random_seed: seed for reproducibility (default None).
        output_dir: directory for checkpoints/diagnostics; ``None`` disables disk output.
        run_name: subdirectory under ``output_dir`` (auto-generated if None).
        auto_reduce_space: enable automatic parameter-space reduction (engine default if None).
        oversample_factor: oversampling factor for filtering, >= 1.0 (engine default if None).
        max_batch_size: max candidates per NROY sampling batch, >= 100 (engine default if None).
        convergence_threshold: acceptance-rate floor for early stopping, in [0, 1].
        nroy_method: ``'auto'`` / ``'lhs'`` / ``'ray'``.
        nroy_options: dict of NROY tuning options passed to ``generate_nroy_design()``.
        settings: dict of additional custom settings forwarded to the engine.
    """

    def __init__(self):
        """Initialize a builder with default configuration. Use the ``from_*`` class methods to supply data."""
        # Domain objects (populated by the from_* constructors)
        self.parameter_space: Optional[ParameterSpace] = None
        self.observations: Optional[ObservationData] = None

        # Strategy configuration — friendly values, coerced at build()
        self.sampling_strategy: Union[str, SamplingStrategy, Dict[str, Any], None] = None
        self.feature_selection: Union[str, List[str], FeatureSelectionStrategy, Dict[str, Any], None] = None
        self.emulator_type: Optional[str] = None
        self.emulator_factory: Optional[EmulatorFactory] = None
        self.emulator_bank: Optional[EmulatorBank] = None

        # Workflow configuration
        self.n_samples: int = 1000
        self.implausibility_threshold: float = 3.0
        self.max_iterations: int = 10
        self.random_seed: Optional[int] = None

        # Output / checkpoint
        self.output_dir: Optional[str] = "./hm_output"
        self.run_name: Optional[str] = None  # auto-generated if None

        # Optional engine knobs (None -> use the engine's own default)
        self.auto_reduce_space: Optional[bool] = None
        self.oversample_factor: Optional[float] = None
        self.max_batch_size: Optional[int] = None
        self.convergence_threshold: Optional[float] = None
        self.nroy_method: Optional[str] = None
        self.nroy_options: Optional[Dict[str, Any]] = None

        # Additional custom settings forwarded verbatim to the engine
        self.settings: Dict[str, Any] = {}

    @classmethod
    def from_data(cls,
                  parameter_bounds: Dict[str, tuple],
                  observations: Dict[str, tuple],
                  **kwargs) -> 'HistoryMatchingBuilder':
        """
        Create builder from simple data structures.

        Args:
            parameter_bounds: Dict mapping parameter names to (min, max) tuples
            observations: Dict mapping feature names to (target, std) tuples
            **kwargs: Additional configuration options (set as builder attributes)

        Returns:
            Configured HistoryMatchingBuilder instance
        """
        builder = cls()
        builder.parameter_space = ParameterSpace(parameter_bounds)
        builder.observations = ObservationData(observations)
        builder._apply_kwargs(kwargs)
        return builder

    @classmethod
    def from_dataframes(cls,
                       parameter_space_df: pd.DataFrame,
                       observations_df: pd.DataFrame,
                       **kwargs) -> 'HistoryMatchingBuilder':
        """
        Create builder from pandas DataFrames.

        Args:
            parameter_space_df: DataFrame with parameter space definition
            observations_df: DataFrame with observation data
            **kwargs: Additional configuration options (set as builder attributes)

        Returns:
            Configured HistoryMatchingBuilder instance
        """
        builder = cls()
        builder.parameter_space = ParameterSpace(parameter_space_df)
        builder.observations = ObservationData(observations_df)
        builder._apply_kwargs(kwargs)
        return builder

    @classmethod
    def from_existing(cls,
                     parameter_space: ParameterSpace,
                     observations: ObservationData,
                     **kwargs) -> 'HistoryMatchingBuilder':
        """
        Create builder from existing domain objects.

        Args:
            parameter_space: Pre-configured ParameterSpace
            observations: Pre-configured ObservationData
            **kwargs: Additional configuration options (set as builder attributes)

        Returns:
            Configured HistoryMatchingBuilder instance
        """
        builder = cls()
        builder.parameter_space = parameter_space
        builder.observations = observations
        builder._apply_kwargs(kwargs)
        return builder

    # Keyword-argument names recognised by the from_* constructors and mapped
    # directly onto builder attributes; anything else is stored in ``settings``.
    _KNOWN_KWARGS = (
        'sampling_strategy', 'feature_selection', 'emulator_type', 'emulator_factory',
        'emulator_bank', 'n_samples', 'max_iterations', 'implausibility_threshold',
        'random_seed', 'output_dir', 'run_name', 'auto_reduce_space', 'oversample_factor',
        'max_batch_size', 'convergence_threshold', 'nroy_method', 'nroy_options',
    )

    def _apply_kwargs(self, kwargs: Dict[str, Any]):
        """Apply keyword arguments to builder configuration.

        Recognised keys are set as attributes; everything else goes into ``settings``.
        """
        for key, value in kwargs.items():
            if key in self._KNOWN_KWARGS:
                setattr(self, key, value)
            else:
                self.settings[key] = value

    # ------------------------------------------------------------------ #
    # Coercion helpers — turn friendly config values into objects.
    # ------------------------------------------------------------------ #
    def _coerce_sampling(self) -> SamplingStrategy:
        strategy = self.sampling_strategy
        if strategy is None:
            warnings.warn("No sampling strategy specified. Using Latin Hypercube Sampling.")
            return SamplingStrategyFactory.create('lhs')
        if isinstance(strategy, str):
            return SamplingStrategyFactory.create(strategy)
        if isinstance(strategy, SamplingStrategy):
            return strategy
        if isinstance(strategy, dict):
            opts = dict(strategy)  # copy so we don't mutate the caller's dict
            strategy_type = opts.pop('type', 'lhs')
            return SamplingStrategyFactory.create(strategy_type, **opts)
        raise ValueError(f"Invalid sampling strategy type: {type(strategy)}")

    def _coerce_feature_selection(self) -> FeatureSelectionStrategy:
        selection = self.feature_selection
        if selection is None:
            warnings.warn("No feature selection specified. Using automatic selection with mean squared z-score.")
            return AutoFeatureSelection(method='mean_sq_z', max_features=1)
        if isinstance(selection, (str, list)):
            return ManualFeatureSelection(selection)
        if isinstance(selection, FeatureSelectionStrategy):
            return selection
        if isinstance(selection, dict):
            return AutoFeatureSelection(
                method=selection.get('method', 'mean_sq_z'),
                threshold=selection.get('threshold', None),
                max_features=selection.get('max_features', 1),
                correlation_threshold=selection.get('correlation_threshold', 0.8),
            )
        raise ValueError(f"Invalid feature selection type: {type(selection)}")

    def _coerce_emulator_factory(self) -> EmulatorFactory:
        if self.emulator_factory is not None:
            return self.emulator_factory
        if self.emulator_type is not None:
            return EmulatorFactory(default_type=self.emulator_type)
        warnings.warn("No emulator type specified. Using Gaussian Process Regression (GPR).")
        return EmulatorFactory(default_type='gpr')

    def validate(self):
        """
        Validate the current configuration.

        Checks that the required components are present and that all numeric and
        enumerated options are within their valid ranges.  Called automatically by
        :meth:`build`; may also be called directly.

        Raises:
            ValueError: If any required component is missing or an option is invalid.
        """
        if self.parameter_space is None:
            raise ValueError("Parameter space is required. Use from_data(), from_dataframes(), or from_existing().")
        if self.observations is None:
            raise ValueError("Observations are required. Use from_data(), from_dataframes(), or from_existing().")

        if self.n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")
        if self.implausibility_threshold <= 0:
            raise ValueError("Implausibility threshold must be positive")

        if self.oversample_factor is not None and self.oversample_factor < 1.0:
            raise ValueError("Oversample factor must be >= 1.0")
        if self.max_batch_size is not None and self.max_batch_size < 100:
            raise ValueError("Max batch size must be >= 100")
        if self.convergence_threshold is not None and not (0.0 <= self.convergence_threshold <= 1.0):
            raise ValueError("Convergence threshold must be between 0.0 and 1.0")
        if self.nroy_method is not None and self.nroy_method not in _VALID_NROY_METHODS:
            raise ValueError(f"Unknown NROY method '{self.nroy_method}'. Valid: {_VALID_NROY_METHODS}")

    def _engine_settings(self) -> Dict[str, Any]:
        """Collect optional engine keyword arguments (skipping unset ``None`` knobs)."""
        extra = dict(self.settings)
        for key in ('auto_reduce_space', 'oversample_factor', 'max_batch_size',
                    'convergence_threshold', 'nroy_method', 'nroy_options'):
            value = getattr(self, key)
            if value is not None:
                extra[key] = value
        return extra

    def build(self) -> 'HistoryMatchingEngine':
        """
        Build the HistoryMatchingEngine with the current configuration.

        Validates the configuration, coerces friendly config values into their
        underlying objects (applying defaults with a warning where unset), and
        constructs the engine.

        Returns:
            Configured HistoryMatchingEngine ready for execution

        Raises:
            ValueError: If required components are missing or options are invalid
        """
        self.validate()

        engine = HistoryMatchingEngine(
            parameter_space=self.parameter_space,
            observations=self.observations,
            sampling_strategy=self._coerce_sampling(),
            feature_selection_strategy=self._coerce_feature_selection(),
            emulator_factory=self._coerce_emulator_factory(),
            emulator_bank=self.emulator_bank if self.emulator_bank is not None else EmulatorBank(),
            n_samples=self.n_samples,
            implausibility_threshold=self.implausibility_threshold,
            max_iterations=self.max_iterations,
            random_seed=self.random_seed,
            output_dir=self.output_dir,
            run_name=self.run_name,
            **self._engine_settings(),
        )
        return engine

    def preview_configuration(self) -> Dict[str, Any]:
        """
        Preview the current configuration without building.

        Returns:
            Dict with configuration summary
        """
        # Resolve via the real coercion helpers so the preview matches build(),
        # suppressing the "using default" warnings those emit for unset options.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            temp_sampling = self._coerce_sampling()
            temp_feature = self._coerce_feature_selection()
            emulator_type = self._coerce_emulator_factory().get_default_type()

        return {
            'parameter_space': {
                'parameters': self.parameter_space.get_parameter_names() if self.parameter_space else None,
                'n_parameters': len(self.parameter_space.get_parameter_names()) if self.parameter_space else 0
            },
            'observations': {
                'features': self.observations.get_feature_names() if self.observations else None,
                'n_features': len(self.observations.get_feature_names()) if self.observations else 0
            },
            'sampling_strategy': temp_sampling.get_strategy_name(),
            'feature_selection_strategy': temp_feature.get_strategy_name(),
            'emulator_type': emulator_type,
            'workflow_settings': {
                'n_samples': self.n_samples,
                'max_iterations': self.max_iterations,
                'implausibility_threshold': self.implausibility_threshold,
                'random_seed': self.random_seed
            },
            'additional_settings': self._engine_settings()
        }

    def __repr__(self) -> str:
        """String representation of builder state."""
        config = self.preview_configuration()
        return (f"HistoryMatchingBuilder(\n"
                f"  parameters={config['parameter_space']['n_parameters']}, "
                f"features={config['observations']['n_features']},\n"
                f"  sampling='{config['sampling_strategy']}',\n"
                f"  feature_selection='{config['feature_selection_strategy']}',\n"
                f"  emulator='{config['emulator_type']}',\n"
                f"  n_samples={config['workflow_settings']['n_samples']}\n"
                f")")
