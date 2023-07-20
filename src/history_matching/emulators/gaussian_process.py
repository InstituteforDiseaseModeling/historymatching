import logging
from typing import ClassVar
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import scipy
from asdf.extension import Converter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel

from history_matching.utils import ndarray_to_dataframe

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
        variance = np.var(self.y_train)
        sigma = variance**0.5
        low = scipy.stats.norm.ppf(q=qlow, scale=sigma)
        hi = scipy.stats.norm.ppf(q=qhigh, scale=sigma)

        # Prepare output and return
        out = pd.DataFrame(index=x.index)
        out["value"] = y_pred
        # TODO - remove? out["low"] = out["value"] + low
        # TODO - remove? out["high"] = out["value"] + hi
        out["variance"] = y_std**2

        return out

    def to_yaml_tree(self, tag, ctx) -> Dict:
        dictionary = super().to_yaml_tree(tag, ctx)
        dictionary.update({"regression_model_params": self.regression_model.get_params(), "regression_model_state": self.regression_model.__getstate__()})

        return dictionary

    @staticmethod
    def from_yaml_tree(node, tag, ctx) -> "GaussianModel":
        # Would prefer something like the following:
        # emulator = super().from_yaml_tree(node, tag, ctx)
        # emulator.regression_model = node.regression_model
        # but BaseEmulator doesn't know to create a GaussianModel. :(

        emulator = GaussianModel()  # pass no initial values

        # BaseEmulator attributes
        emulator.X_df = ndarray_to_dataframe(node["X_df"])
        emulator.X_train = node["X_train"]
        emulator.X_test = node["X_test"]
        emulator.y_df = ndarray_to_dataframe(node["y_df"])
        emulator.y_train = node["y_train"]
        emulator.y_test = node["y_test"]
        emulator.y_pred = node["y_pred"]
        emulator.y_pred_test = node["y_pred_test"]
        emulator.y_test_pred_df = ndarray_to_dataframe(node["y_test_pred_df"])
        emulator.training_complete = node["training_complete"]
        emulator.testing_complete = node["testing_complete"]
        emulator.mse = node["mse"]
        emulator.r2score = node["r2score"]

        emulator.regression_model = GaussianProcessRegressor(**node["regression_model_params"])
        emulator.regression_model.__setstate__(node["regression_model_state"])

        return emulator


class GaussianModelConverter(Converter):
    tags: ClassVar[List[str]] = ["asdf://idmod.org/asdf/tags/emulators/gaussian_model-1.0.0"]
    types: ClassVar[List[type]] = ["history_matching.emulators.gaussian_process.GaussianModel"]

    def to_yaml_tree(self, obj, tag, ctx):
        return obj.to_yaml_tree(tag, ctx)
        # return {"pickle": pickle.dumps(obj)}

    def from_yaml_tree(self, node, tag, ctx):
        return GaussianModel.from_yaml_tree(node, tag, ctx)
        # return pickle.loads(node["pickle"])
