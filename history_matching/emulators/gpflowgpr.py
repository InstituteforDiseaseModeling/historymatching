"""Gaussian Process Regression emulator implemented in GPFlow."""

import logging
import warnings
from typing import Optional

# gpflow import tensorflow-probability
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import gpflow as gpf


import pandas as pd

from .base import BaseEmulator


class GPFlowGPR(BaseEmulator):

    """Gaussian Process Regression emulator implemented in GPFlow."""

    def __init__(self, x: Optional[pd.DataFrame] = None, y: Optional[pd.DataFrame] = None, test_fraction=0.25):
        """Initialise the Gaussian Process Regression emulator implemented in GPFlow."""

        super().__init__(x, y, test_fraction)
        self.regression_model = None

        return

    def train(self):
        """Fits a Gaussian Process model."""

        logging.debug("... training emulator")

        kernel = gpf.kernels.Matern52()

        # TODO - do this elsewhere?
        if len(self.y_train.shape) == 1:
            self.y_train = self.y_train.reshape(-1, 1)

        self.regression_model = gpf.models.GPR(data=(self.X_train, self.y_train), kernel=kernel)
        gpf.utilities.print_summary(self.regression_model)
        opt = gpf.optimizers.Scipy()
        opt.minimize(self.regression_model.training_loss, self.regression_model.trainable_variables, options={"maxiter": 100})
        gpf.utilities.print_summary(self.regression_model)
        self.training_complete = True
        logging.debug("     training complete")

        return

    def predict(self, x: pd.DataFrame(), qlow=0.05, qhigh=0.95):
        """Predict an output using the trained emulator."""

        logging.debug("... predicting outputs using the trained emulator")
        # Compute the prediction
        X_pred = x.to_numpy()
        if len(X_pred.shape) == 1:
            if X_pred.shape[0] > 1:
                X_pred = X_pred.reshape(-1, 1)
            else:
                X_pred = X_pred.reshape(1, -1)
        y_pred, y_var = self.regression_model.predict_y(X_pred)
        out = pd.DataFrame({"value": y_pred.numpy().reshape(len(x)), "variance": y_var.numpy().reshape(len(x))}, index=x.index)

        return out
