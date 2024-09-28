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
        self.parameter_space = parameter_space
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
        