import logging
from typing import Optional

import numpy as np
import pandas as pd

import statsmodels.api as sm
import scipy

from .base import BaseEmulator


class GLM(BaseEmulator):
    """ Generalized Linear Model (GLM) emulator.
    """

    def __init__(self, x: Optional[pd.DataFrame]=None, y: Optional[pd.DataFrame]=None, test_fraction: float=0.25, link='linear') -> None:
        """Initialize the emulator.

        Args:
            x : Input data. Pandas dataframe with columns representing parameter
                values.
            y : Output data. Pandas dataframe with columns representing
                observations and rows representing samples. Each row in this
                dataframe must match the corresponding row in `x`.
            test_fraction : Fraction of `x` and `y` samples to be used for
                testing. This is a scalar between 0 and 1.
            link : Link function for the GLM model. It can be either 'linear'
                or 'poisson'.
                
        Returns:
            None
        """
        super().__init__(x, y, test_fraction)
        self.family = sm.families.Gaussian()    if link=='linear'    else  \
                      sm.families.Poisson()   if link=='poisson'   else  \
                      None    # This last case should't happen; it should raise an error
        self.link = link

        return

    
    def train(self):
        """Fits a Generalised Linear Model."""
        logging.debug("... training emulator")

        self.model = sm.GLM( self.y_train, x, family=self.family )
        self.results = self.model.fit()
        
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
        prediction_results = self.results.get_prediction( x_pred )
        predicted_mean = prediction_results.predicted_mean
        predicted_var = prediction_results.var_pred_mean

        # Compute the uncertainty interval
        z = 1.96  # 95% confidence interval
        low  = predicted_mean - z * np.sqrt(predicted_var)
        high = predicted_mean + z * np.sqrt(predicted_var)

        # Prepare output and return
        out = pd.DataFrame(index=x.index)
        out["value"] = predicted_mean
        out["low"] = low
        out["high"] = high
        return out

    
    def print_emulator_description(self):
        """Display detailed specifications (for example, emulator coefficients)
        for the trained emulator.
        """
        print( self.results.summary() )
        return
