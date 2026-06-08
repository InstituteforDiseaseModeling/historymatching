import logging
from typing import Optional

import pandas as pd

import statsmodels.api as sm

from .base import BaseEmulator
from .results import EmulationResults


class GLM(BaseEmulator):
    """ Generalized Linear Model (GLM) emulator.
    """

    def __init__(self, x: Optional[pd.DataFrame]=None, y: Optional[pd.DataFrame]=None, test_fraction: float=0.25, link='linear') -> None:
        """Initialize the emulator.

        Args:
            x: Input data. Pandas dataframe with columns representing parameter
                values.
            y: Output data. Pandas dataframe with columns representing
                observations and rows representing samples. Each row in this
                dataframe must match the corresponding row in `x`.
            test_fraction: Fraction of `x` and `y` samples to be used for
                testing. This is a scalar between 0 and 1.
            link: Link function for the GLM model. It can be either 'linear'
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
        pred_ci = self.results.get_prediction( x_pred, linear=False)
        low = pred_ci.conf_int(obs=True)[:,0]
        high = pred_ci.conf_int(obs=True)[:,1]

        # Create additional data for emulator-specific outputs
        additional = pd.DataFrame({
            'ci_obs_low': low,
            'ci_obs_high': high,
            'ci_pred_low': low_mean,
            'ci_pred_high': high_mean
        }, index=x.index)
        
        return EmulationResults(
            mean=predicted_mean,
            std=pred_ci.se_mean,  # Standard error is already std
            additional_data=additional
        )

    
    def get_hyperparameters(self) -> dict:
        """Return GLM hyperparameters as a JSON-serializable dict."""
        if not self.training_complete:
            return {}

        param_names = list(self.X_df.columns) if self.X_df is not None else [f"x{i}" for i in range(self.X_train.shape[1])]
        coeffs = self.results.params
        return {
            'type': 'glm',
            'family': str(self.results.family),
            'coefficients': {name: float(c) for name, c in zip(['intercept'] + param_names, coeffs)},
            'deviance': float(self.results.deviance),
            'n_train': int(len(self.X_train)),
            'n_dims': int(self.X_train.shape[1]),
        }

    def print_emulator_description(self):
        """Display detailed specifications (for example, emulator coefficients)
        for the trained emulator.
        """
        print( self.results.summary() )
        return
