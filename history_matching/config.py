import logging

logger = logging.getLogger()


class Config:

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
