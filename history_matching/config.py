import logging
import numpy as np
import pandas as pd

logger = logging.getLogger()



class Config:

    """
    Configuration for a history matching iteration.
    """

    def __init__( self, 

                  # Data
                  parameter_space : pd.DataFrame,
                  observations    : pd.DataFrame,
                  sample_points   : pd.DataFrame = None,

                  # Model
                  model        = None,
                  model_output = None,
                  model_discrepancy = 2, 

                  # Features or emulator targets
                  feature_selection_mode   = 'manual',
                  feature = None,  # Emulator target
                  implausibility_threshold = 3,

                  # Sampling
                  candidates_per_iteration = 250, 

                  # Others
                  non_implausible_target   = None, 
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

        logger.info('Creating Config object')

        # Data parameters
        self.parameter_space = parameter_space
        self.observations = observations
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

        # Model parameters
        if model is None:
            logger.info('... model not defined, using model outputs provided by the user')
            self.model_outputs = model_outputs
        else:
            self.model = model
            self.model_output = None
        self.model_discrepancy = model_discrepancy

        # Emulator configuration parameters
        self.feature_selection_mode = feature_selection_mode.strip().lower()
        self.feature = feature
        self.implausibility_threshold = implausibility_threshold
        self.emulator_bank = dict()

        # Sampling parameters
        self.candidates_per_iteration = candidates_per_iteration

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
