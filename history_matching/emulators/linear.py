import logging
from typing import Optional

import numpy as np
import pandas as pd

import statsmodels.api as sm
import scipy
from sklearn import linear_model as sklm

from .base import BaseEmulator


class LinearModel(BaseEmulator):
    """
    Emulator based on an ordinary least squares linear regression.
    The emulator fits a linear regression model to minimize the residual sum of squares between observed targets in the training data and the targets predicted by the linear approximation.
    """

    def __init__(self, x: Optional[pd.DataFrame] = None, y: Optional[pd.DataFrame] = None, test_fraction: float = 0.25) -> None:
        """Initialize the emulator.

        Args:
            x : Input data. Pandas dataframe with columns representing parameter
                values.
            y : Output data. Pandas dataframe with columns representing
                observations and rows representing samples. Each row in this
                dataframe must match the corresponding row in `x`.
            test_fraction : Fraction of `x` and `y` samples to be used for
                testing. This is a scalar between 0 and 1.

        Returns:
            None
        """
        super().__init__(x, y, test_fraction)

        return

    
    def train(self):
        """
        Fits a linear regression model to minimize the residual sum of squares between observed targets in the training data and the targets predicted by the linear approximation.
        """
        logging.debug("... training emulator")

        x = sm.add_constant( self.X_train )
        self.model = sm.OLS( self.y_train, x )
        self.results = self.model.fit()
        
        self.training_complete = True
        logging.debug("     training complete")
        return

    
    def predict(self, x: pd.DataFrame()):
        """Predict an output using the trained emulator.

        Args:
            x : Input data. Pandas dataframe with columns representing parameter
                values.

        Returns:
            Pandas dataframe with predicted values and uncertainty intervals.
        """
        logging.debug("... predicting outputs using the trained emulator")

        # Compute the prediction
        x_pred = sm.add_constant( x )
        prediction_results = self.results.get_prediction( x_pred )
        predicted_mean = prediction_results.predicted_mean
        
        # Compute the confidence interval of the predicted mean
        low_mean = prediction_results.conf_int()[:,0]
        high_mean = prediction_results.conf_int()[:,1]

        # Compute the prediction intervals 
        low = prediction_results.summary_frame()['obs_ci_lower']
        high = prediction_results.summary_frame()['obs_ci_upper']
        
        # Prepare output and return
        out = pd.DataFrame(index=x.index)
        out['value'] = predicted_mean
        out['ci_obs_low'] = low    # CI of prediction
        out['ci_obs_high'] = high
        out['ci_pred_low'] = low_mean    # CI of predicted mean
        out['ci_pred_high'] = high_mean
        return out

    
    def print_emulator_description(self):
        """Display detailed specifications (for example, emulator coefficients)
        for the trained emulator.
        """
        print( self.results.summary() )
        return



class LinearModelScipy(BaseEmulator):
    """
    Emulator based on an ordinary least squares linear regression.
    The emulator fits a linear regression model to minimize the residual sum of squares between observed targets in the training data and the targets predicted by the linear approximation.
    """

    def __init__(self, x: Optional[pd.DataFrame] = None, y: Optional[pd.DataFrame] = None, test_fraction: float = 0.25) -> None:
        """Initialize the emulator.

        Args:
            x : Input data. Pandas dataframe with columns representing parameter
                values.
            y : Output data. Pandas dataframe with columns representing
                observations and rows representing samples. Each row in this
                dataframe must match the corresponding row in `x`.
            test_fraction : Fraction of `x` and `y` samples to be used for
                testing. This is a scalar between 0 and 1.

        Returns:
            None
        """
        super().__init__(x, y, test_fraction)

        return

    
    def train(self):
        """
        Fits a linear regression model to minimize the residual sum of squares between observed targets in the training data and the targets predicted by the linear approximation.
        """
        logging.debug("... training emulator")

        self.regression_model = sklm.LinearRegression()
        self.regression_model.fit(self.X_train, self.y_train)

        self.training_complete = True
        logging.debug("     training complete")
        return

    
    def predict(self, x: pd.DataFrame(), qlow=0.05, qhi=0.95):
        """Predict an output using the trained emulator.

        Args:
            x : Input data. Pandas dataframe with columns representing parameter
                values.
            qlow  : Lower quantile for the estimated uncertainty interval.
            qhigh : Upper quantile for the estimated uncertainty interval.

        Returns:
            Pandas dataframe with predicted values and uncertainty intervals.
        """
        logging.debug("... predicting outputs using the trained emulator")
        # Compute the prediction
        X_pred = x.to_numpy()
        if len(X_pred.shape) == 1:
            if X_pred.shape[0] > 1:
                X_pred = X_pred.reshape(-1, 1)
            else:
                X_pred = X_pred.reshape(1, -1)
        y_pred = self.regression_model.predict(X_pred)

        # Compute uncertainty bounds
        # NOTE: sklearn does not compute the variance of the predicted mean. 
        # We are using the variance of the samples here. However, this needs
        # to be updated to compute the variance of the predicted mean. 
        variance = np.var(self.y_train)
        sigma = variance**0.5
        low = scipy.stats.norm.ppf(q=qlow, scale=sigma)
        hi = scipy.stats.norm.ppf(q=qhi, scale=sigma)

        # Prepare output and return
        out = pd.DataFrame(index=x.index)
        out["value"] = y_pred
        out["low"] = out["value"] + low
        out["high"] = out["value"] + hi
        return out

    
    def print_emulator_description(self):
        """Display detailed specifications (for example, emulator coefficients)
        for the trained emulator.
        """
        print("      coefficients: ", self.regression_model.coef_)
        print("      intercept   : ", self.regression_model.intercept_)
        return

