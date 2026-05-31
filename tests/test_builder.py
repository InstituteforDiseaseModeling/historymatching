"""
Unit tests for HistoryMatchingBuilder.
"""


import pandas as pd
import pytest
from historymatching.builder import HistoryMatchingBuilder
from historymatching.engine import HistoryMatchingEngine
from historymatching.emulator_bank import EmulatorBank
from historymatching.observation_data import ObservationData
from historymatching.parameter_space import ParameterSpace
from historymatching.emulators.factory import EmulatorFactory
from historymatching.feature_selection import AutoFeatureSelection
from historymatching.feature_selection import ManualFeatureSelection
from historymatching.sampling import RandomSampling


class TestHistoryMatchingBuilder:
    """Test HistoryMatchingBuilder functionality."""

    @pytest.fixture
    def sample_parameter_bounds(self):
        """Sample parameter bounds for testing."""
        return {
            "param1": (0.0, 1.0),
            "param2": (-5.0, 5.0),
            "param3": (10.0, 100.0)
        }

    @pytest.fixture
    def sample_observations(self):
        """Sample observations for testing."""
        return {
            "output1": (25.0, 5.0),  # (target, std)
            "output2": (100.0, 10.0),
            "output3": (0.5, 0.1)
        }

    @pytest.fixture
    def sample_parameter_df(self):
        """Sample parameter space DataFrame."""
        return pd.DataFrame({
            "parameter": ["param1", "param2", "param3"],
            "minimum": [0.0, -5.0, 10.0],
            "maximum": [1.0, 5.0, 100.0]
        })

    @pytest.fixture
    def sample_observations_df(self):
        """Sample observations DataFrame."""
        return pd.DataFrame({
            "feature": ["output1", "output2", "output3"],
            "mean": [25.0, 100.0, 0.5],
            "std": [5.0, 10.0, 0.1]  # sqrt of previous variance values
        })

    def test_builder_initialization(self):
        """Test basic builder initialization."""
        builder = HistoryMatchingBuilder()

        # Should have None for required components
        assert builder.parameter_space is None
        assert builder.observations is None

        # Should have defaults for configuration
        assert builder.n_samples == 1000
        assert builder.implausibility_threshold == 3.0
        assert builder.max_iterations == 10
        assert builder.random_seed is None

    def test_from_data_constructor(self, sample_parameter_bounds, sample_observations):
        """Test from_data class method constructor."""
        builder = HistoryMatchingBuilder.from_data(
            parameter_bounds=sample_parameter_bounds,
            observations=sample_observations
        )

        # Check that domain objects were created
        assert isinstance(builder.parameter_space, ParameterSpace)
        assert isinstance(builder.observations, ObservationData)

        # Check parameter space
        param_names = builder.parameter_space.get_parameter_names()
        assert set(param_names) == set(sample_parameter_bounds.keys())

        # Check observations
        obs_features = builder.observations.get_feature_names()
        assert set(obs_features) == set(sample_observations.keys())

    def test_from_data_with_kwargs(self, sample_parameter_bounds, sample_observations):
        """Test from_data with additional configuration."""
        builder = HistoryMatchingBuilder.from_data(
            parameter_bounds=sample_parameter_bounds,
            observations=sample_observations,
            n_samples=500,
            max_iterations=5,
            sampling_strategy="grid",
            emulator_type="linear"
        )

        assert builder.n_samples == 500
        assert builder.max_iterations == 5

        # Preview should show configured values
        config = builder.preview_configuration()
        assert "Grid Sampling" in config["sampling_strategy"]
        assert config["emulator_type"] == "linear"

    def test_unknown_kwargs_go_to_settings(self, sample_parameter_bounds, sample_observations):
        """Unrecognised kwargs are stored in the settings dict."""
        builder = HistoryMatchingBuilder.from_data(
            parameter_bounds=sample_parameter_bounds,
            observations=sample_observations,
            max_candidate_factor=2000,
        )
        assert builder.settings["max_candidate_factor"] == 2000

    def test_from_dataframes_constructor(self, sample_parameter_df, sample_observations_df):
        """Test from_dataframes class method constructor."""
        builder = HistoryMatchingBuilder.from_dataframes(
            parameter_space_df=sample_parameter_df,
            observations_df=sample_observations_df
        )

        # Check that domain objects were created
        assert isinstance(builder.parameter_space, ParameterSpace)
        assert isinstance(builder.observations, ObservationData)

        # Check parameter space
        param_names = builder.parameter_space.get_parameter_names()
        assert set(param_names) == set(sample_parameter_df["parameter"])

        # Check observations
        obs_features = builder.observations.get_feature_names()
        assert set(obs_features) == set(sample_observations_df["feature"])

    def test_from_existing_constructor(self, sample_parameter_bounds, sample_observations):
        """Test from_existing class method constructor."""
        # Create domain objects
        parameter_space = ParameterSpace(sample_parameter_bounds)
        observations = ObservationData({
            name: (target, std**2) for name, (target, std) in sample_observations.items()
        })

        builder = HistoryMatchingBuilder.from_existing(
            parameter_space=parameter_space,
            observations=observations
        )

        assert builder.parameter_space is parameter_space
        assert builder.observations is observations

    def test_sampling_strategy_string(self, sample_parameter_bounds, sample_observations):
        """Test configuring sampling strategy by string."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        builder.sampling_strategy = "grid"

        config = builder.preview_configuration()
        assert "Grid Sampling" in config["sampling_strategy"]

    def test_sampling_strategy_object(self, sample_parameter_bounds, sample_observations):
        """Test configuring sampling strategy with object."""
        strategy = RandomSampling()

        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        builder.sampling_strategy = strategy

        assert builder.sampling_strategy is strategy

    def test_sampling_strategy_dict(self, sample_parameter_bounds, sample_observations):
        """Test configuring sampling strategy with dict."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        builder.sampling_strategy = {"type": "lhs", "criterion": "center"}

        config = builder.preview_configuration()
        assert "center" in config["sampling_strategy"]

    def test_feature_selection_list(self, sample_parameter_bounds, sample_observations):
        """Test configuring feature selection with list."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        builder.feature_selection = ["output1", "output2"]

        engine = builder.build()
        assert isinstance(engine.feature_selection_strategy, ManualFeatureSelection)

        config = builder.preview_configuration()
        assert "Manual Selection" in config["feature_selection_strategy"]

    def test_feature_selection_string(self, sample_parameter_bounds, sample_observations):
        """Test configuring feature selection with single string."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        builder.feature_selection = "output1"

        engine = builder.build()
        assert isinstance(engine.feature_selection_strategy, ManualFeatureSelection)

    def test_feature_selection_object(self, sample_parameter_bounds, sample_observations):
        """Test configuring feature selection with object."""
        strategy = AutoFeatureSelection(method="var", max_features=2)

        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        builder.feature_selection = strategy

        assert builder.feature_selection is strategy

    def test_feature_selection_dict(self, sample_parameter_bounds, sample_observations):
        """Test configuring feature selection with dict."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        builder.feature_selection = {
            "method": "var",
            "max_features": 2,
            "threshold": 0.5
        }

        engine = builder.build()
        strategy = engine.feature_selection_strategy
        assert isinstance(strategy, AutoFeatureSelection)
        assert strategy.method == "var"
        assert strategy.max_features == 2
        assert strategy.threshold == 0.5

    def test_emulator_type(self, sample_parameter_bounds, sample_observations):
        """Test configuring emulator type."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        builder.emulator_type = "linear"

        engine = builder.build()
        assert isinstance(engine.emulator_factory, EmulatorFactory)
        assert engine.emulator_factory.get_default_type() == "linear"

    def test_emulator_factory(self, sample_parameter_bounds, sample_observations):
        """Test configuring custom emulator factory (overrides emulator_type)."""
        factory = EmulatorFactory("gpr", kernel="rbf")

        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        builder.emulator_factory = factory

        assert builder.emulator_factory is factory
        assert builder.build().emulator_factory is factory

    def test_workflow_parameters(self, sample_parameter_bounds, sample_observations):
        """Test configuring workflow parameters."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        builder.n_samples = 2000
        builder.max_iterations = 20
        builder.implausibility_threshold = 2.5
        builder.random_seed = 42

        assert builder.n_samples == 2000
        assert builder.max_iterations == 20
        assert builder.implausibility_threshold == 2.5
        assert builder.random_seed == 42

    def test_space_reduction_options(self, sample_parameter_bounds, sample_observations):
        """Test space reduction configuration."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        builder.auto_reduce_space = True
        builder.oversample_factor = 5.0

        assert builder.auto_reduce_space is True
        assert builder.oversample_factor == 5.0

        # These knobs propagate to the engine.
        engine = builder.build()
        assert engine.auto_reduce_space is True
        assert engine.oversample_factor == 5.0

    def test_emulator_bank(self, sample_parameter_bounds, sample_observations):
        """Test configuring existing emulator bank."""
        bank = EmulatorBank()

        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        builder.emulator_bank = bank

        assert builder.emulator_bank is bank
        assert builder.build().emulator_bank is bank

    def test_custom_settings(self, sample_parameter_bounds, sample_observations):
        """Test adding custom settings via the settings dict."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        builder.settings["custom_param"] = "custom_value"
        builder.settings["debug"] = True

        assert builder.settings["custom_param"] == "custom_value"
        assert builder.settings["debug"] is True

    def test_validate_rejects_invalid_parameters(self, sample_parameter_bounds, sample_observations):
        """Test that validate() rejects out-of-range options."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)

        builder.n_samples = 0
        with pytest.raises(ValueError):
            builder.validate()

        builder.n_samples = 1000
        builder.max_iterations = -1
        with pytest.raises(ValueError):
            builder.validate()

        builder.max_iterations = 10
        builder.implausibility_threshold = 0
        with pytest.raises(ValueError):
            builder.validate()

        builder.implausibility_threshold = 3.0
        builder.oversample_factor = 0.5
        with pytest.raises(ValueError):
            builder.validate()

    def test_build_calls_validate(self, sample_parameter_bounds, sample_observations):
        """build() should run validation and reject invalid config."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        builder.n_samples = 0
        with pytest.raises(ValueError):
            builder.build()

    def test_build_success(self, sample_parameter_bounds, sample_observations):
        """Test successful engine building."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)

        engine = builder.build()

        assert isinstance(engine, HistoryMatchingEngine)
        assert engine.parameter_space is not None
        assert engine.observations is not None

    def test_build_missing_components(self):
        """Test building without required components."""
        builder = HistoryMatchingBuilder()

        # Missing parameter space
        with pytest.raises(ValueError, match="Parameter space is required"):
            builder.build()

    def test_build_with_defaults(self, sample_parameter_bounds, sample_observations):
        """Test that defaults are applied when building."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)

        # Don't configure strategies explicitly
        engine = builder.build()

        # Should have default strategies
        assert engine.sampling_strategy is not None
        assert engine.feature_selection_strategy is not None
        assert engine.emulator_factory is not None
        assert engine.emulator_bank is not None

    def test_preview_configuration(self, sample_parameter_bounds, sample_observations):
        """Test configuration preview."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        builder.sampling_strategy = "grid"
        builder.emulator_type = "linear"
        builder.n_samples = 500

        config = builder.preview_configuration()

        assert config["parameter_space"]["n_parameters"] == 3
        assert config["observations"]["n_features"] == 3
        assert "Grid Sampling" in config["sampling_strategy"]
        assert config["emulator_type"] == "linear"
        assert config["workflow_settings"]["n_samples"] == 500

    def test_builder_repr(self, sample_parameter_bounds, sample_observations):
        """Test string representation."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)

        repr_str = repr(builder)

        assert "HistoryMatchingBuilder" in repr_str
        assert "parameters=3" in repr_str
        assert "features=3" in repr_str

    def test_full_configuration(self, sample_parameter_bounds, sample_observations):
        """Test configuring every option via attribute assignment, then building."""
        builder = HistoryMatchingBuilder.from_data(sample_parameter_bounds, sample_observations)
        builder.sampling_strategy = "lhs"
        builder.feature_selection = ["output1"]
        builder.emulator_type = "gpr"
        builder.n_samples = 1500
        builder.max_iterations = 15
        builder.implausibility_threshold = 2.8
        builder.random_seed = 123
        builder.auto_reduce_space = True
        builder.oversample_factor = 3.0
        engine = builder.build()

        assert isinstance(engine, HistoryMatchingEngine)
        assert engine.n_samples == 1500
        assert engine.max_iterations == 15
        assert engine.implausibility_threshold == 2.8
        assert engine.random_seed == 123
        assert engine.auto_reduce_space is True
        assert engine.oversample_factor == 3.0


class TestBuilderFromData:
    """Test builder factory methods produce working engines."""

    def test_from_data_builds_engine(self):
        """Test building engine from dict inputs."""
        builder = HistoryMatchingBuilder.from_data(
            parameter_bounds={"param1": (0, 1), "param2": (-1, 1)},
            observations={"output1": (5.0, 1.0), "output2": (10.0, 2.0)},
        )
        builder.n_samples = 800
        engine = builder.build()

        assert isinstance(engine, HistoryMatchingEngine)
        assert engine.n_samples == 800
        assert len(engine.parameter_space.get_parameter_names()) == 2
        assert len(engine.observations.get_feature_names()) == 2

    def test_from_dataframes_builds_engine(self):
        """Test building engine from DataFrame inputs."""
        parameter_df = pd.DataFrame({
            "parameter": ["param1", "param2"],
            "minimum": [0, -1],
            "maximum": [1, 1]
        })

        observations_df = pd.DataFrame({
            "feature": ["output1", "output2"],
            "mean": [5.0, 10.0],
            "std": [1.0, 2.0]
        })

        builder = HistoryMatchingBuilder.from_dataframes(parameter_df, observations_df)
        builder.sampling_strategy = 'grid'
        builder.emulator_type = 'linear'
        builder.max_iterations = 8
        engine = builder.build()

        assert isinstance(engine, HistoryMatchingEngine)
        assert engine.max_iterations == 8
        assert "Grid" in engine.sampling_strategy.get_strategy_name()
        assert engine.emulator_factory.get_default_type() == "linear"


class TestBuilderEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_sampling_strategy_type(self):
        """Test invalid sampling strategy type is rejected at build()."""
        builder = HistoryMatchingBuilder.from_data({"param": (0, 1)}, {"output": (1, 0.1)})
        builder.sampling_strategy = 123  # Invalid type

        with pytest.raises(ValueError):
            builder.build()

    def test_invalid_feature_selection_type(self):
        """Test invalid feature selection type is rejected at build()."""
        builder = HistoryMatchingBuilder.from_data({"param": (0, 1)}, {"output": (1, 0.1)})
        builder.feature_selection = 123  # Invalid type

        with pytest.raises(ValueError):
            builder.build()

    def test_empty_parameter_bounds(self):
        """Test with empty parameter bounds."""
        with pytest.raises(ValueError):
            HistoryMatchingBuilder.from_data({}, {"output": (1, 0.1)})

    def test_empty_observations(self):
        """Test with empty observations."""
        with pytest.raises(ValueError):
            HistoryMatchingBuilder.from_data({"param": (0, 1)}, {})

    def test_inconsistent_data_formats(self):
        """Test with inconsistent data formats."""
        # Parameter bounds with wrong tuple size
        with pytest.raises((ValueError, TypeError)):
            HistoryMatchingBuilder.from_data(
                {"param": (0,)},  # Missing max value
                {"output": (1, 0.1)}
            )

    def test_build_configuration_conflicts(self):
        """Test building with conflicting configurations (last assignment wins)."""
        builder = HistoryMatchingBuilder.from_data(
            {"param": (0, 1)},
            {"output": (1, 0.1)}
        )

        # Last assignment wins: manual selection is overridden by auto.
        builder.feature_selection = ["output"]  # Manual selection
        builder.feature_selection = {"method": "fano"}  # Override with auto
        engine = builder.build()

        assert isinstance(engine.feature_selection_strategy, AutoFeatureSelection)
