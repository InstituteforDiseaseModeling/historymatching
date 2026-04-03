import logging
from typing import Optional

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

        x = sm.add_constant( self.X_train )
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
        
        # Create additional data for emulator-specific outputs
        additional = pd.DataFrame({
            'ci_obs_low': low,    # CI of prediction
            'ci_obs_high': high,
            'ci_pred_low': low_mean,    # CI of predicted mean
            'ci_pred_high': high_mean
        }, index=x.index)
        
        return EmulationResults(
            mean=predicted_mean,
            std=prediction_results.summary_frame()['mean_se'],  # Standard error is already std
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
        y_pred = self.regression_model.predict(X_pred)

        # Compute uncertainty bounds
        # NOTE: sklearn does not compute the variance of the predicted mean. 
        # We are using the variance of the samples here. However, this needs
        # to be updated to compute the variance of the predicted mean. 
        variance = np.var(self.y_train)
        sigma = variance**0.5
        
        # Compute default confidence intervals for additional data
        low = scipy.stats.norm.ppf(q=0.05, scale=sigma)
        hi = scipy.stats.norm.ppf(q=0.95, scale=sigma)
        
        additional = pd.DataFrame({
            "low": y_pred.flatten() + low,
            "high": y_pred.flatten() + hi
        }, index=x.index)

        return EmulationResults(
            mean=y_pred.flatten(),  # sklearn returns 2D (n_samples, 1), flatten to 1D
            std=np.full(len(y_pred), sigma),  # Create array of constant std values
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

