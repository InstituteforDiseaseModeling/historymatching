import logging
from typing import Optional

import numpy as np
import pandas as pd
import scipy
from sklearn import linear_model as sklm

from .base import BaseEmulator


class LinearModel(BaseEmulator):

    """
    Emulator based on an ordinary least squares linear regression.
    The emulator fits a linear regression model to minimize the residual sum of squares between observed targets in the training data and the targets predicted by the linear approximation.
    """

    def __init__(self, x: Optional[pd.DataFrame] = None, y: Optional[pd.DataFrame] = None, test_fraction: float = 0.25) -> None:
        self.regression_model = None
        super().__init__(x, y, test_fraction)

        return

    def train(self):
        """
        Fits a linear regression model to minimize the residual sum of squares between observed targets in the training data and the targets predicted by the linear approximation.
        """
        logging.debug("... training emulator")

        self.regression_model = sklm.LinearRegression()
        self.regression_model.fit(self.X_train, self.y_train)

        # self.var = numpy.var(self.y_train)
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
