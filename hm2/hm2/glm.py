#!/usr/bin/env python3
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import statsmodels.api as sm



class GLM:
    """Generalized Linear Modeling (GLM).

    This class implementes Generalized Linear Modeling using statsmodels as the
    engine.
    """
    def __init__(
            self,
            polyorder,
            intercept,
            family = 'poisson',
    ):
        """Initialize the GLM class.

        Args:
            polyorder: Order of polynomial expansion of the data features
            intercept: Whether to add an intercept feature
            family: (str) The family of generalized linear model to use. 
                          Options include 'poisson', 'binomial', 'gamma', 
                          'negativebinomial', and 'gaussian'. 
        """
        self.polyorder = polyorder
        self.intercept = intercept
        self.family    = family
        self.alpha     = None
        self.model     = None

        self.gfamilies = {
            "poisson":  sm.families.Poisson(),
            "binomial": sm.families.Binomial(),
            "gamma":    sm.families.Gamma(),
            "gaussian": sm.families.Gaussian()
        }

        if not family in glms:
            raise HistoryMatchingError(f"Invalid glm family '{family}'!")

        self.polyfit = PolynomialFeatures(
          degree           = polyorder, 
          interaction_only = False, 
          include_bias     = intercept
        )

    def fit(self, data, y, maxiter=1000): #TODO: Rename y
        """Fit the GLM.

        Args:
            maxiter: (int)
                maxiter parameter passed to the statsmodels `fit` function.
        """
        logger = logging.getLogger("HistoryMatching")

        data = self.polyfit.fit_transform(data)
        self.model = sm.GLM(data, y, family=self.gfamilies[self.family])

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
        data = self.polyfit.fit_transform(data)
        return self.fitted_model.predict(data, transform=False)
