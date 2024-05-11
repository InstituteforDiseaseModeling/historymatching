import logging
import warnings
from typing import Optional

import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow_probability as tfp
    logging.debug(f"Loaded tensorflow-probability version {tfp.__version__}.")

from .base import BaseEmulator


class TensorFlowGPR(BaseEmulator):
    """Gaussian Process Regression emulator implemented in TensorFlow."""

    def __init__(self, x: Optional[pd.DataFrame] = None, y: Optional[pd.DataFrame] = None, test_fraction=0.25):
        """Initialise the Gaussian Process Regression (GPR) emulator implemented in TensorFlow.

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
        """Fits a Gaussian Process model."""

        logging.debug("... training emulator")

        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic()
        self.regression_model = tfp.distributions.GaussianProcess(kernel=kernel, index_points=self.X_train)
        raise NotImplementedError("Haven't implemented training for TensorFlowGPR yet.")  # self.regression_model.fit(self.y_train)
        self.training_complete = True
        logging.debug("     training complete")

        return

    def predict(self, x: pd.DataFrame, qlow=0.05, qhigh=0.95):
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
        out = pd.DataFrame([y_pred, y_var], columns=["value", "variance"], index=x.index)

        return out
