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

        x = sm.add_constant( self.X_train )
        self.model = sm.GLM( self.y_train, x, family=self.family )
        self.results = self.model.fit()
        
        self.training_complete = True
        logging.debug("     training complete")
        return

    
    def predict(self, x: pd.DataFrame()):
        """Predict an output using the trained emulator.

        Args:
            x : Input data. Pandas dataframe with columns representing parameter
                values.

        Returns:
            Pandas dataframe with predicted values and uncertainty intervals.
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
        # obs_ci not returned by summary_frame()
        #low = prediction_results.summary_frame()['obs_ci_lower']
        #high = prediction_results.summary_frame()['obs_ci_upper']
        low = prediction_results.summary_frame()['mean_ci_lower']
        high = prediction_results.summary_frame()['mean_ci_upper']
        
        # Prepare output and return
        out = pd.DataFrame(index=x.index)
        out['value'] = predicted_mean
        out['low'] = low
        out['high'] = high
        out['ci_value_low'] = low_mean
        out['ci_value_high'] = high_mean
        return out

    
    def print_emulator_description(self):
        """Display detailed specifications (for example, emulator coefficients)
        for the trained emulator.
        """
        print( self.results.summary() )
        return
