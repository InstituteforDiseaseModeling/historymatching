#!/usr/bin/env python3
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .error import HistoryMatchingError


#TODO: Add predictor abstract class

def gfamily(family):
    gfamilies = {
        "poisson":  sm.families.Poisson,
        "binomial": sm.families.Binomial,
        "gamma":    sm.families.Gamma,
        "gaussian": sm.families.Gaussian
    }
    if not family in gfamilies:
        raise HistoryMatchingError(f"Invalid glm family '{family}'!")
    return gfamilies[family]()



class GLM:
    """Generalized Linear Modeling (GLM).

    This class implementes Generalized Linear Modeling using statsmodels as the
    engine.
    """
    def __init__(
            self,
            family,
    ):
        """Initialize the GLM class.

        Args:
            family: (str) The family of generalized linear model to use. 
                          Options include 'poisson', 'binomial', 'gamma', 
                          'negativebinomial', and 'gaussian'. 
        """
        # Make this call only to check that the family exists
        gfamily(family)

        self.family = family
        self.alpha  = None
        self.model  = None

    def fit(self, train_x, train_y, maxiter=1000):
        """Fit the GLM.

        Args:
            maxiter: (int)
                maxiter parameter passed to the statsmodels `fit` function.
        """
        logger = logging.getLogger("HistoryMatching")

        # We're using statsmodels GLM because sklearn GLM's doesn't have a
        # family option.
        self.model = sm.GLM(train_y, train_x, family=gfamily(self.family))

        self.glmfit = self.model.fit(maxiter=maxiter)

        logger.info(self.glmfit.summary())
        logger.info('GLM AIC:', self.glmfit.aic)
        logger.info('GLM BIC:', self.glmfit.bic)
        logger.info('GLM ITERATION:', self.glmfit.fit_history['iteration'])

    def predict(self, test_x):
        """Evaluate the GLM and return the mean prediction.

        Args:
            test_x: (Pandas DataFrame)
                Data frame of points similar to training_data.

        Returns:
            Predicted outputs at the inputs specified by data.
        """
        return self.model.predict(self.glmfit.params, test_x)
