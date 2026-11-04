from typing import Optional
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy
from sklearn import linear_model as sklm

from .base import BaseEmulator
from .results import EmulationResults


class LinearModel(BaseEmulator):
    """
    Emulator based on an ordinary least squares linear regression.
    The emulator fits a linear regression model to minimize the residual sum of squares between observed targets in the training data and the targets predicted by the linear approximation.
    """

    def __init__(self, x: Optional[pd.DataFrame] = None, y: Optional[pd.DataFrame] = None, test_fraction: float = 0.25) -> None:
        """Initialize the emulator.

        Args:
            x: Input data. Pandas dataframe with columns representing parameter
                values.
            y: Output data. Pandas dataframe with columns representing
                observations and rows representing samples. Each row in this
                dataframe must match the corresponding row in `x`.
            test_fraction: Fraction of `x` and `y` samples to be used for
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

        x = sm.add_constant(self.X_train, has_constant='add')  # as in predict()
        self.model = sm.OLS( self.y_train, x )
        self.results = self.model.fit()
        
        self.training_complete = True
        logging.debug("     training complete")
        return

    
    def predict(self, x: pd.DataFrame) -> EmulationResults:
        """Predict an output using the trained emulator.

        Args:
            x: Input data. Pandas dataframe with columns representing parameter
                values.

        Returns:
            EmulationResults with predicted values and uncertainty intervals.
        """
        logging.debug("... predicting outputs using the trained emulator")

        # has_constant='add': a single row looks constant in every column, and the
        # default ('skip') would then leave out the intercept column.
        x_pred = sm.add_constant(x, has_constant='add')
        prediction_results = self.results.get_prediction(x_pred)
        frame = prediction_results.summary_frame()

        additional = pd.DataFrame({
            'ci_obs_low': frame['obs_ci_lower'].to_numpy(),    # interval for a new run
            'ci_obs_high': frame['obs_ci_upper'].to_numpy(),
            'ci_pred_low': frame['mean_ci_lower'].to_numpy(),  # interval for the mean
            'ci_pred_high': frame['mean_ci_upper'].to_numpy(),
        }, index=x.index)

        # The std of a new simulator run at x: uncertainty in the fitted mean plus
        # the residual variance, as BayesLinear's nugget term and GPR's predict_y
        # include. The mean's standard error alone understates it, which over-prunes.
        return EmulationResults(
            mean=np.asarray(prediction_results.predicted_mean).ravel(),
            std=np.asarray(prediction_results.se_obs).ravel(),
            additional_data=additional
        )

    
    def get_hyperparameters(self) -> dict:
        """Return linear model hyperparameters as a JSON-serializable dict."""
        if not self.training_complete:
            return {}

        param_names = list(self.X_df.columns) if self.X_df is not None else [f"x{i}" for i in range(self.X_train.shape[1])]
        coeffs = self.results.params
        return {
            'type': 'linear',
            'coefficients': {name: float(c) for name, c in zip(['intercept'] + param_names, coeffs)},
            'r_squared': float(self.results.rsquared),
            'r_squared_adj': float(self.results.rsquared_adj),
            'n_train': int(len(self.X_train)),
            'n_dims': int(self.X_train.shape[1]),
        }

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
            x: Input data. Pandas dataframe with columns representing parameter
                values.
            y: Output data. Pandas dataframe with columns representing
                observations and rows representing samples. Each row in this
                dataframe must match the corresponding row in `x`.
            test_fraction: Fraction of `x` and `y` samples to be used for
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

        # Pieces of the OLS predictive variance s^2 (1 + x0' (X'X)^-1 x0), which
        # sklearn does not provide.
        design = np.column_stack([np.ones(len(self.X_train)), self.X_train])
        dof = len(design) - design.shape[1]
        if dof <= 0:
            raise ValueError(f"LinearModelScipy needs more training points ({len(design)}) "
                             f"than coefficients ({design.shape[1]}) to estimate its residual variance")
        residuals = np.ravel(self.y_train) - np.ravel(self.regression_model.predict(self.X_train))
        self._residual_var = float(residuals @ residuals) / dof
        self._xtx_inv = np.linalg.pinv(design.T @ design)

        self.training_complete = True
        logging.debug("     training complete")
        return

    
    def predict(self, x: pd.DataFrame) -> EmulationResults:
        """Predict an output using the trained emulator.

        Args:
            x: Input data. Pandas dataframe with columns representing parameter
                values.

        Returns:
            EmulationResults with predicted values and uncertainty intervals.
        """
        logging.debug("... predicting outputs using the trained emulator")
        # Compute the prediction
        X_pred = x.to_numpy()
        if len(X_pred.shape) == 1:
            if X_pred.shape[0] > 1:
                X_pred = X_pred.reshape(-1, 1)
            else:
                X_pred = X_pred.reshape(1, -1)
        y_pred = np.ravel(self.regression_model.predict(X_pred))  # sklearn returns (n, 1)

        # The std of a new simulator run at x, the same quantity LinearModel reports:
        # residual variance times (1 + leverage). The spread of the training outputs,
        # used before, also counts the variation the regression explains.
        design = np.column_stack([np.ones(len(X_pred)), X_pred])
        leverage = np.einsum('ij,jk,ik->i', design, self._xtx_inv, design)
        sigma = np.sqrt(self._residual_var * (1.0 + leverage))

        # Central 90% interval for a new run
        z = scipy.stats.norm.ppf(0.95)
        additional = pd.DataFrame({
            "low": y_pred - z * sigma,
            "high": y_pred + z * sigma
        }, index=x.index)

        return EmulationResults(
            mean=y_pred,
            std=sigma,
            additional_data=additional
        )

    
    def get_hyperparameters(self) -> dict:
        """Return linear model (scipy) hyperparameters as a JSON-serializable dict."""
        if not self.training_complete:
            return {}

        param_names = list(self.X_df.columns) if self.X_df is not None else [f"x{i}" for i in range(self.X_train.shape[1])]
        coeffs = self.regression_model.coef_.flatten()
        return {
            'type': 'linear_scipy',
            'coefficients': {name: float(c) for name, c in zip(param_names, coeffs)},
            'intercept': float(self.regression_model.intercept_.item()),
            'n_train': int(len(self.X_train)),
            'n_dims': int(self.X_train.shape[1]),
        }

    def print_emulator_description(self):
        """Display detailed specifications (for example, emulator coefficients)
        for the trained emulator.
        """
        print("      coefficients: ", self.regression_model.coef_)
        print("      intercept   : ", self.regression_model.intercept_)
        return

