from typing import Optional
import logging

import numpy as np
import pandas as pd
import scipy.stats
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

        x = sm.add_constant(self.X_train, has_constant='add')  # as in predict()
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

        # has_constant='add': a single row looks constant in every column, and the
        # default ('skip') would then leave out the intercept column.
        x_pred = sm.add_constant(x, has_constant='add')
        prediction_results = self.results.get_prediction(x_pred)  # on the mean (response) scale
        mean = np.asarray(prediction_results.predicted_mean).ravel()
        se_mean = np.asarray(prediction_results.se_mean).ravel()
        mean_ci = np.asarray(prediction_results.conf_int())

        # The std of a new simulator run at x: uncertainty in the fitted mean plus
        # the family's variance at that mean (scale for Gaussian, the mean itself for
        # Poisson), as BayesLinear's nugget term and GPR's predict_y include. The
        # mean's standard error alone understates it, which over-prunes.
        obs_var = self.results.scale * self.results.family.variance(mean)
        std = np.sqrt(se_mean**2 + obs_var)

        # Normal-approximation 95% interval for a new run (statsmodels gives GLM
        # intervals for the mean only)
        z = scipy.stats.norm.ppf(0.975)
        additional = pd.DataFrame({
            'ci_obs_low': mean - z * std,
            'ci_obs_high': mean + z * std,
            'ci_pred_low': mean_ci[:, 0],
            'ci_pred_high': mean_ci[:, 1]
        }, index=x.index)

        return EmulationResults(
            mean=mean,
            std=std,
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
