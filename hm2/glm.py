#!/usr/bin/env python3
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.graphics as smg
import statsmodels.api as sm

from .error import HistoryMatchingError
from .plotting import WrappedFigure


#TODO: Add predictor abstract class

def _gfamily(family):
    gfamilies = {
        "binomial": sm.families.Binomial,
        "gamma":    sm.families.Gamma,
        "gaussian": sm.families.Gaussian,
        "poisson":  sm.families.Poisson
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
        _gfamily(family)

        self.family  = family
        self.glm     = None
        self.glmfit  = None
        self._trainx = None
        self._trainy = None

    def fit(self, train_x, train_y, maxiter=1000):
        """Fit the GLM.

        Args:
            maxiter: (int)
                maxiter parameter passed to the statsmodels `fit` function.
        """
        logger = logging.getLogger("HistoryMatching")

        self._trainx = train_x
        self._trainy = train_y

        # We're using statsmodels GLM because sklearn GLM's doesn't have a
        # family option.
        self.glm = sm.GLM(train_y, train_x, family=_gfamily(self.family))

        self.glmfit = self.glm.fit(maxiter=maxiter)

        logger.info(self.glmfit.summary())
        #TODO
        # logger.info('GLM AIC:', self.glmfit.aic)
        # logger.info('GLM BIC:', self.glmfit.bic)
        # logger.info('GLM ITERATION:', self.glmfit.fit_history['iteration'])

    def predict(self, test_x):
        """Evaluate the GLM and return the mean prediction.

        Args:
            test_x (Pandas DataFrame):
                Data frame of points similar to training_data.

        Returns:
            Predicted outputs at the inputs specified by data.
        """
        if self.glm is None:
            raise HistoryMatchingError("GLM is untrained!")
        return self.glmfit.predict(test_x)


    def plot_fitted_vs_observed(self, figsize=None):
        """Generates a plot of the fitted values vs the observed values from the training data.

        Returns:
            A matplotlib figure handle.
        """
        if self.glm is None:
            raise HistoryMatchingError("GLM is untrained!")

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(self._trainy, self.glmfit.mu, marker='+')
        ax.set_title('Fitted versus Observed Values')
        ax.set_xlabel('Observed values')
        ax.set_ylabel('Fitted values')

        return WrappedFigure(fig)


    def plot_pearson_residuals(self, figsize=None):
        """Generates a plot of the peasron residuals.

        Returns:
            A matplotlib figure handle.
        """
        if self.glm is None:
            raise HistoryMatchingError("GLM is untrained!")

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(self.glmfit.mu, self.glmfit.resid_pearson, marker='+')
        ax.set_title('Residual Dependence Plot')
        ax.set_ylabel('Pearson Residuals')
        ax.set_xlabel('Fitted values')

        return WrappedFigure(fig)


    def plot_deviance_redisuals(self, figsize=None, bins=25):
        """Generates a plot of the deviance residuals.

        Returns:
            A matplotlib figure handle.
        """
        if self.glm is None:
            raise HistoryMatchingError("GLM is untrained!")

        fig, ax = plt.subplots(figsize=figsize)
        resid = self.glmfit.resid_deviance.copy()
        resid_std = sp.stats.zscore(resid)
        ax.hist(resid_std, bins=25)
        ax.set_title('Standardized deviance residuals')

        return WrappedFigure(fig)


    def plot_QQ(self, figsize=None):
        """Generates a QQ plot.

        Returns:
            A matplotlib figure handle.
        """
        if self.glm is None:
            raise HistoryMatchingError("GLM is untrained!")

        fig, ax = plt.subplots(figsize=figsize)
        smg.gofplots.qqplot(self.glmfit.resid_deviance, line='45', fit=True, ax=ax)

        return WrappedFigure(fig)
