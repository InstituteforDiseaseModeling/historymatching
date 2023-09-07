import logging
from typing import Optional

import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel

from .base import BaseEmulator


class GaussianModel(BaseEmulator):
    def __init__(self, x: Optional[pd.DataFrame] = None, y: Optional[pd.DataFrame] = None, test_fraction: float = 0.25) -> None:
        self.regression_model = None
        super().__init__(x, y, test_fraction)

        return

    def train(self):
        """
        Fits a Gaussian Process model.
        """

        random_state = None
        stddev_y = self.y_train.std(axis=0)

        logging.debug("... training emulator")

        kernel = ConstantKernel(1.0) * RBF()
        self.regression_model = GaussianProcessRegressor(kernel=kernel, random_state=random_state, alpha=stddev_y, n_restarts_optimizer=10)
        self.regression_model.fit(self.X_train, self.y_train)

        self.training_complete = True
        logging.debug("     training complete")
        return

    def predict(self, x: pd.DataFrame(), qlow=0.05, qhigh=0.95):
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
        y_pred, y_std = self.regression_model.predict(X_pred, return_std=True)

        # Compute uncertainty bounds
        # import scipy.stats above if you uncomment lines below
        # variance = np.var(self.y_train)
        # sigma = variance**0.5
        # low = scipy.stats.norm.ppf(q=qlow, scale=sigma)
        # hi = scipy.stats.norm.ppf(q=qhigh, scale=sigma)

        # Prepare output and return
        out = pd.DataFrame(index=x.index)
        out["value"] = y_pred
        # TODO - remove? out["low"] = out["value"] + low
        # TODO - remove? out["high"] = out["value"] + hi
        out["variance"] = y_std**2

        return out
