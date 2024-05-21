import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import gpflow

from .base import BaseEmulator





class GPR(BaseEmulator):
    """Gaussian Process Regression emulator implemented in GPFlow."""

    def __init__(self, x: Optional[pd.DataFrame] = None, y: Optional[pd.DataFrame] = None, test_fraction=0.25):
        """Initialise the Gaussian Process Regression (GPR) emulator.

        Args:
            x : Input data. Pandas dataframe with columns representing parameter
                values.
            y : Output data. Pandas dataframe with columns representing
                observations and rows representing samples. Each row in this
                dataframe must match the corresponding row in `x`.
            test_fraction : Fraction of `x` and `y` samples to be used for
                testing. This is a scalar between 0 and 1.

        Returns:
            None
        """
        super().__init__(x, y, test_fraction)

        return

    
    def train(self):
        """Fits a GPR model."""

        logging.debug("... training emulator")

        x_gpf = np.hstack(  ( np.ones( (len(self.X_train), 1 ) ),  self.X_train ) )
        y_gpf = np.float64( self.y_train.reshape( (len(self.y_train),) ) ).reshape(-1,1)

        self.model = gpflow.models.GPR( (x_gpf, y_gpf), kernel=gpflow.kernels.SquaredExponential() )
        opt = gpflow.optimizers.Scipy()
        self.opt_logs = opt.minimize( self.model.training_loss, self.model.trainable_variables )

        self.training_complete = True
        logging.debug("     training complete")

        return

    
    def predict(self, x: pd.DataFrame, qlow=0.05, qhigh=0.95):
        """Predict an output using the trained emulator."""

        logging.debug("... predicting outputs using the trained emulator")

        # Make the prediction
        x_gpf = np.hstack( (np.ones( (len(x), 1 ) ),  x) )
        ymean, yvar = self.model.predict_y( x_gpf )

        # Compute the uncertainty interval
        z = 1.96  # 95% confidence interval
        low  = ymean - z * np.sqrt(yvar)
        high = ymean + z * np.sqrt(yvar)

        # Save outputs
        out = pd.DataFrame(index=x.index)
        out['value'] = ymean
        out['ci_obs_low' ] = low
        out['ci_obs_high'] = high
        out['ci_pred_low' ] = 0  # need to verify this
        out['ci_pred_high'] = 0
        return out


    def print_emulator_description(self):
        """Display detailed specifications (for example, emulator coefficients)
        for the trained emulator.
        """
        if self.training_complete:
            print('      model description:' )
            gpflow.utilities.print_summary( self.model )
            print('\n      optimization logs: \n', self.opt_logs )
        return