#!/usr/bin/env python3
import logging

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
import numpy as np

from hm2.error import *



class SkGPR:
    """Gaussian Process Regression (GPR) using scikit-learn"""
    def __init__(self):
        self.model = None

    def fit(self, train_x, train_y, stdev_y=None, maxiter=1000):
        """Fit the GPR.

        Args:
            train_x: Training data
            train_y: Correct outputs
            stdev_y: Standard deviation of Y values (uncertainty)
            maxiter (int): Maximum number of training iterations

        Returns:
            None
        """
        # Initialize the model
        kernel = ConstantKernel(1.0) * RBF()
        self.model = GaussianProcessRegressor(kernel=kernel)

        logging.getLogger("HistoryMatching").warn("SkGPR ignores stdev_y")

        self.model.fit(train_x, train_y)

    @property
    def _trainx(self):
        assert self.model is not None
        return self.model.X_train_

    @property
    def _trainy(self):
        assert self.model is not None
        return self.model.y_train_

    def predict(self, test_x):
        """Evaluate the GLM and return the mean prediction.

        Args:
            test_x (Pandas DataFrame):
                Data frame of points similar to training_data.

        Returns:
            Predicted outputs at the inputs specified by data.
        """
        if self.model is None:
            raise HistoryMatchingError("SkGPR hasn't been trained yet!")

        return self.model.predict(test_x, return_std=True)
