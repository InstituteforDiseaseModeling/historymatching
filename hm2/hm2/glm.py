#!/usr/bin/env python3
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .basis import BasisBase
from .error import HistoryMatchingError



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
            basis,
            family,
    ):
        """Initialize the GLM class.

        Args:
            basis: Feature generator inheriting from BasisBase
            family: (str) The family of generalized linear model to use. 
                          Options include 'poisson', 'binomial', 'gamma', 
                          'negativebinomial', and 'gaussian'. 
        """
        if not isinstance(basis,BasisBase):
            raise HistoryMatchingError("`basis` must inherit from BasisBase!")

        # Make this call only to check that the family exists
        gfamily(family)

        self.basis  = basis
        self.family = family
        self.alpha  = None
        self.model  = None

    def fit(self, data, endog, maxiter=1000):
        """Fit the GLM.

        Args:
            maxiter: (int)
                maxiter parameter passed to the statsmodels `fit` function.
        """
        if not isinstance(data, pd.DataFrame):
          raise TypeError("data passed to GLM.fit must be a DataFrame!")

        logger = logging.getLogger("HistoryMatching")

        data = self.basis.fit_transform(data)
        self.model = sm.GLM(data, endog, family=gfamily(self.family))

        logger.info("Fitting GLM...")
        self.model = self.model.fit(maxiter=maxiter)

        logger.info(self.model.summary())
        logger.info('AIC:', self.model.aic)
        logger.info('BIC:', self.model.bic)
        logger.info('ITERATION:', self.model.fit_history['iteration'])

    def predict(self, data):
        """Evaluate the GLM and return the mean prediction.

        Args:
            data: (Pandas DataFrame)
                Data frame of points similar to training_data.

        Returns:
            Predicted outputs at the inputs specified by data.
        """
        if not isinstance(data, pd.DataFrame):
          raise TypeError("data passed to GLM.predict must be a DataFrame!")

        data = self.basis.fit_transform(data)
        return self.fitted_model.predict(data, transform=False)

    def residuals(self, data, endog):
        return endog-self.predict(data)