import numpy as np
import pandas as pd
import logging
from typing import Callable, Optional

logger = logging.getLogger()




class Config:
    """
    Configuration for a history matching iteration.
    """
    # Attributes
    parameter_space : pd.DataFrame = None
    observations    : pd.DataFrame = None
    sample_points   : pd.DataFrame = None

    model             : Optional[Callable[[], any]] = None
    model_output      : str = None
    model_discrepancy : float = 1.0

    feature_selection_mode : str = 'manual'
    feature : str = None
    implausibility_threshold : float = 3.0

    draw_samples : str = 'lhs'
    n_candidates : int = 500
    
    
    def __init__( self, 

                  # Data
                  parameter_space : pd.DataFrame,
                  observations    : pd.DataFrame,
                  sample_points   : pd.DataFrame = None,

                  # Model
                  model             : Optional[Callable[[], any]] = None,
                  model_output      : str = None,
                  model_discrepancy : float = 1.0, 

                  # Features or emulator targets
                  feature_selection_mode : str  = 'manual',
                  feature : str = None,  # Emulator target
                  implausibility_threshold : float = 3.0,

                  # Sampling
                  draw_samples = 'lhs',
                  n_candidates = 500, 

                  # Others
                  **kwargs 
                 ):
        """
        Args:
            parameter_space   : a DataFrame containing the parameter space.
            observations      : a DataFrame containing the observations.
            sample_points     : (optional) a DataFrame containing the sample 
                                points.
            model             : function to run the model or simulator.
            model_output      : a DataFrame containing the model or simulator 
                                output.
            model_discrepancy : a measure of uncertainty of the model outputs.
            feature_selection_mode : method for the selection of features in an
                                iteration. It can be 'manual', in which the user
                                explicitly indicates the feature, or 'auto', in 
                                which a feature is automatically selected.
            feature           : feature to use as target for the emulator. This
                                argument is only considered if 
                                `feature_selection_mode` is 'manual'. If not 
                                provided, and `feature_selection_mode` is set
                                as 'manual', the user will be prompted to select
                                the desired feature.
            implausibility_threshold : threshold for implausibility.
            draw_samples      : method for proposing new samples. It can be:
                                'lhs'    : Latin Hypercube Sampling;
                                'grid'   : grid sampling;
                                'random' : random sampling.
            n_candidates      : number of candidate points to generate in an 
                                iteration.

        Keyword Args:
            user: dictionary of user-defined configuration parameters

        Returns:
            None
        """
        logger.info('Creating Config object')

        # Data parameters
        validate_parameter_space( parameter_space.reset_index() )
        validate_observations( observations )
        self.parameter_space = parameter_space.reset_index()
        self.observations = observations
        self.sample_points = sample_points
        
        # Model parameters
        if model is not None:    # The user provides a function to call the model
            self.model = model
            self.model_output = None
        else:    # The user runs the model externally
            logger.info('... model not defined, using model outputs provided by the user')
            self.model_output = model_output
        self.model_discrepancy = model_discrepancy
        
        # Emulator configuration parameters
        self.feature_selection_mode = feature_selection_mode.strip().lower()
        self.feature = feature
        self.implausibility_threshold = implausibility_threshold
        self.emulator_bank = dict()

        # Sampling parameters
        self.draw_samples = draw_samples
        self.n_candidates = n_candidates

        # Other parameters
        class User:
            pass
        self.user = User()
        self.user.__dict__.update(kwargs)

        return



def validate_parameter_space(parameter_space: pd.DataFrame) -> None:
    """
    Validate the parameter space.

    Args:
        parameter_space: the parameter space

    Raises:
        TypeError: if the parameter space is not a DataFrame
        ValueError: if the parameter space is empty
        ValueError: if the parameter space has duplicate parameter names
        ValueError: if the parameter space has a parameter with a single value
        ValueError: if the parameter space has a parameter with a negative value
        ValueError: if the parameter space has a parameter with a zero value
        ValueError: if the parameter space has a parameter with a non-numeric value
    """
    if not isinstance(parameter_space, pd.DataFrame):
        raise TypeError(f"Situation parameter space should be Pandas DataFrame, not '{type(parameter_space)}'")
    parameter_space_all_columns = ['parameter', 'minimum', 'maximum']
    if not all(column in parameter_space.columns for column in parameter_space_all_columns):
        raise RuntimeError(f"Situation parameter space must contain the columns {parameter_space_all_columns}. Found {parameter_space.columns}.")
    if len(parameter_space) == 0:
        raise RuntimeError("Situation parameter space must specify at least one parameter. Found none.")
    ordered = True
    msg = ""
    for row in parameter_space.itertuples():
        if row.minimum > row.maximum:
            msg += f"Parameter '{row.parameter}' minimum ({row.minimum}) > maximum ({row.maximum}).\n"
            ordered = False
    if not ordered:
        raise ValueError(msg)

    return



def validate_observations(observations: pd.DataFrame) -> None:
    """
    Validate the observations.

    Args:
        observations: the observations

    Raises:
        TypeError: if the observations is not a DataFrame
        ValueError: if the observations is empty
        ValueError: if the observations has duplicate feature names
        ValueError: if the observations has a feature with a single value
        ValueError: if the observations has a feature with a negative value
        ValueError: if the observations has a feature with a zero value
        ValueError: if the observations has a feature with a non-numeric value
    """
    if not isinstance(observations, pd.DataFrame):
        raise TypeError(f"Situation observations should be Pandas DataFrame, not '{type(observations)}'")

    observations_all_columns = ['feature', 'mean', 'variance']
    if set(observations.columns) != set(observations_all_columns):
        raise RuntimeError(f"Situation observations should have columns 'feature', 'mean', and 'variance'. Found {set(observations.columns)}")

    if len(observations) < 1:
        raise RuntimeError("Situation observations must have at least one feature.")

    return


        