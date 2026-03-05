import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import gpflow

from .base import BaseEmulator
from .results import EmulationResults





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

        ls = np.ones( (x_gpf.shape[1],) )
        self.model = gpflow.models.GPR( (x_gpf, y_gpf), kernel=gpflow.kernels.SquaredExponential(lengthscales=ls))
        opt = gpflow.optimizers.Scipy()
        self.opt_logs = opt.minimize( self.model.training_loss, self.model.trainable_variables )

        # TODO: Validate that the optimization was successful (success = True) and respond accordingly or at least let the user know.
        if not self.opt_logs.success:
           logging.error("... training failed")
           return

        self.training_complete = True
        logging.debug("     training complete")

        return

    
    def predict(self, x: pd.DataFrame) -> EmulationResults:
        """Predict an output using the trained emulator."""

        logging.debug("... predicting outputs using the trained emulator")

        # Make the prediction
        x_gpf = np.hstack( (np.ones( (len(x), 1 ) ),  x) )
        f_mean, f_var = self.model.predict_f( x_gpf, full_cov=False )
        y_mean, y_var = self.model.predict_y( x_gpf )

        # Compute the uncertainty interval for additional data
        z = 1.96  # 95% confidence interval
        f_lower = f_mean - z * np.sqrt(f_var)
        f_upper = f_mean + z * np.sqrt(f_var)
        y_lower = y_mean - z * np.sqrt(y_var)
        y_upper = y_mean + z * np.sqrt(y_var)

        # Convert to numpy arrays and flatten for pandas compatibility
        def to_flat_array(tensor_or_array):
            if hasattr(tensor_or_array, 'numpy'):
                return tensor_or_array.numpy().flatten()
            else:
                return np.asarray(tensor_or_array).flatten()
        
        y_mean_flat = to_flat_array(y_mean)
        y_var_flat = to_flat_array(y_var)
        
        # Create additional data for emulator-specific outputs
        additional = pd.DataFrame({
            "f_var": to_flat_array(f_var),
            "ci_obs_low": to_flat_array(y_lower),
            "ci_obs_high": to_flat_array(y_upper),
            "ci_pred_low": to_flat_array(f_lower),
            "ci_pred_high": to_flat_array(f_upper)
        }, index=x.index)

        return EmulationResults(
            mean=y_mean_flat,
            std=np.sqrt(y_var_flat),
            additional_data=additional
        )


    def print_emulator_description(self):
        """Display detailed specifications (for example, emulator coefficients)
        for the trained emulator.
        """
        if self.training_complete:
            print('      model description:' )
            gpflow.utilities.print_summary( self.model )
            print('\n      optimization logs: \n', self.opt_logs )
        return