import logging

import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from .base import BaseEmulator

logger = logging.getLogger()


class GprEmulator(BaseEmulator):
    
    def __init__(self, x:pd.DataFrame = None, y:pd.DataFrame = None, test_fraction: float = 0.25) -> None:

        super().__init__(x, y, test_fraction)
        self.gpr = None

        return

    def train(self) -> None:

        kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        self.gpr.fit(self.X_train, self.y_train)

        return

    def predict(self, X:pd.DataFrame, qlow: float = 0.05, ghi:float = 0.95) -> pd.DataFrame:

        if self.gpr is not None:

            # mean_prediction, std_prediction = self.gpr.predict(X, return_std=True)
            mean_prediction, stddev = self.gpr.predict(X.to_numpy(), return_std=True)

        else:
            raise RuntimeError("Emulator has not been trained.")

        return mean_prediction, stddev

