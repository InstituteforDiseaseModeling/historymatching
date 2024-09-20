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
    model_output      : pd.DataFrame = None
    model_discrepancy : float = 1.0



    
    def __init__( self, 

                  # Data
                  parameter_space : pd.DataFrame,
                  observations    : pd.DataFrame,
                  sample_points   : pd.DataFrame = None,

                  # Model
                  model             : Optional[Callable[[], any]] = None,
                  model_output      : pd.DataFrame = None,
                  model_discrepancy : float = 1.0, 

                  # Features or emulator targets
                  feature_selection_mode   = 'manual',
                  feature = None,  # Emulator target
                  implausibility_threshold = 3,

                  # Sampling
                  n_candidates = 500, 

                  # Others
                  non_implausible_target   = None, 
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
            n_candidates      : number of candidate points to generate in an 
                                iteration.
            non_implausible_target : target fraction of non-implausible 
                                points.

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
        """
        if sample_points is None:    # Could use Latin-hypercube sampling here
            logger.info('... generating sample_points using a uniform distribution')
            self.sample_points    \
                = pd.DataFrame( { parameter_space.loc[param,'parameter']: np.random.uniform( parameter_space.loc[param, 'minimum'], 
                                                                                             parameter_space.loc[param, 'maximum'],
                                                                                             candidates_per_iteration
                                                                          ) for param in parameter_space.index 
                                } 
                               )
        else:
            self.sample_points = sample_points
        """
        
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
        self.n_candidates = n_candidates

        # Other parameters
        self.non_implausible_target = non_implausible_target
        class User:
            pass
        self.user = User()
        self.user.__dict__.update(kwargs)

        return





class Config_orig:

    """
    Configuration for a history matching process.
    """

    def __init__( self, 
                  max_iterations,
                  feature_selection_mode,
                  candidates_per_iteration, 
                  implausibility_threshold, 
                  non_implausible_target, 
                  model_discrepancy, 
                  **kwargs 
                 ):
        """
        Args:
            max_iterations: maximum number of iterations to run.
            feature_selection_mode: method for the selection of 
                features at each iteration.
            candidates_per_iteration: number of candidate points to 
                generate per iteration.
            implausibility_threshold: threshold for implausibility.
            non_implausible_target: target fraction of non-implausible 
                points.

        Keyword Args:
            user: dictionary of user-defined configuration parameters

        Returns:
            None
        """

        logger.info("Creating Config object")
        self.max_iterations = max_iterations
        self.feature_selection_mode = feature_selection_mode.strip().lower()
        self.candidates_per_iteration = candidates_per_iteration
        self.implausibility_threshold = implausibility_threshold
        self.non_implausible_target = non_implausible_target
        self.model_discrepancy = model_discrepancy

        class User:
            pass

        self.user = User()
        self.user.__dict__.update(kwargs)

        return
