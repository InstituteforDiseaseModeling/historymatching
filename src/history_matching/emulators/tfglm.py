import logging
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from .base import BaseEmulator


class TensorFlowGLM(BaseEmulator):

    """Generalised Linear Model emulator implemented in TensorFlow."""

    def __init__(self, x: Optional[pd.DataFrame] = None, y: Optional[pd.DataFrame] = None, test_fraction=0.25):
        """Initialise the Generalised Linear Model emulator implemented in TensorFlow."""
        super().__init__(x, y, test_fraction)
        self.regression_model = None

        return

    def train(self):
        """Fits a Generalised Linear Model."""

        logging.debug("... training emulator")

        # https://www.tensorflow.org/probability/api_docs/python/tfp/glm/fit
        # model_matrix (Batch of) float-like, matrix-shaped Tensor where each row represents a sample's features.
        # response     (Batch of) vector-shaped Tensor where each element represents a sample's observed response (to the corresponding row of features). Must have same dtype as model_matrix.
        # model        tfp.glm.ExponentialFamily-like instance which implicitly characterizes a negative log-likelihood loss by specifying the distribuion's mean, gradient_mean, and variance.

        model_coefficients, linear_response, is_converged, num_iter = tfp.glm.fit(model_matrix=self.X_train, response=self.y_train.reshape((len(self.y_train),)), model=tfp.glm.Normal())

        self.model_coefficients = model_coefficients
        self.linear_response = linear_response
        self.is_converged = is_converged
        self.num_iter = num_iter

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
        # y_pred, y_var = self.regression_model.predict_y(X_pred)

        # model_matrix is incoming parameter values
        # model_coefficients is the trained coefficients
        # offset is the [trained?] offset
        y_pred = tf.linalg.matvec(x.to_numpy(), self.model_coefficients)  # + offset
        # y_var = np.zeros(len(y_pred))

        out = pd.DataFrame()
        out["value"] = y_pred.numpy()
        out["variance"] = np.zeros(len(y_pred))

        return out
