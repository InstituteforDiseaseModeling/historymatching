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

    
    def _normalize_x(self, x):
        """Normalize inputs to [0, 1] using training-set min/range."""
        return (np.float64(x) - self._x_min) / self._x_range

    def train(self):
        """Fits a GPR model.

        Inputs are normalized to [0, 1] and outputs are standardized
        (zero mean, unit variance) before training.  This ensures:
          - Kernel lengthscales are comparable across parameters with very
            different physical scales (e.g. 0.0005–0.006 vs 1.0–3.0).
          - The optimizer isn't confused by large output offsets
            (e.g. birth weight ~ 3000 g with a signal of a few grams).

        Predictions are un-standardized automatically in predict().
        """
        logging.debug("... training emulator")

        x_raw = np.float64(self.X_train)
        y_raw = np.float64(self.y_train.reshape(-1, 1))

        # Input normalization: min–max to unit box
        self._x_min = x_raw.min(axis=0)
        self._x_range = x_raw.max(axis=0) - self._x_min
        self._x_range = np.maximum(self._x_range, 1e-12)  # guard against zero-range columns

        # Output standardization: zero mean, unit variance
        self._y_mean = float(np.mean(y_raw))
        self._y_std = float(np.std(y_raw))
        if self._y_std < 1e-12:
            self._y_std = 1.0  # guard against constant output

        x_gpf = self._normalize_x(x_raw)
        y_gpf = (y_raw - self._y_mean) / self._y_std

        self.model = gpflow.models.GPR(
            (x_gpf, y_gpf),
            kernel=gpflow.kernels.SquaredExponential(),
            mean_function=gpflow.mean_functions.Constant(),
        )
        opt = gpflow.optimizers.Scipy()
        self.opt_logs = opt.minimize(self.model.training_loss, self.model.trainable_variables)

        if not self.opt_logs.success:
            logging.warning(
                "GPR optimization did not converge (scipy reported success=False). "
                "This is common when the noise variance hits its lower bound. "
                "The fitted model may still be usable — check emulator diagnostics."
            )

        self.training_complete = True
        logging.debug("     training complete")

        return

    
    def predict(self, x: pd.DataFrame) -> EmulationResults:
        """Predict an output using the trained emulator."""

        logging.debug("... predicting outputs using the trained emulator")

        # Normalize inputs using training-set min/range
        x_gpf = self._normalize_x(x)
        f_mean_z, f_var_z = self.model.predict_f(x_gpf, full_cov=False)
        y_mean_z, y_var_z = self.model.predict_y(x_gpf)

        # Convert to numpy arrays and flatten for pandas compatibility
        def to_flat(tensor_or_array):
            if hasattr(tensor_or_array, 'numpy'):
                return tensor_or_array.numpy().flatten()
            else:
                return np.asarray(tensor_or_array).flatten()

        # Un-standardize: mean → mean*std + mu, var → var*std²
        ys, ym = self._y_std, self._y_mean
        f_mean = to_flat(f_mean_z) * ys + ym
        f_var  = to_flat(f_var_z)  * ys ** 2
        y_mean = to_flat(y_mean_z) * ys + ym
        y_var  = to_flat(y_var_z)  * ys ** 2

        # Compute the uncertainty interval for additional data
        z = 1.96  # 95% confidence interval
        f_lower = f_mean - z * np.sqrt(f_var)
        f_upper = f_mean + z * np.sqrt(f_var)
        y_lower = y_mean - z * np.sqrt(y_var)
        y_upper = y_mean + z * np.sqrt(y_var)

        # Create additional data for emulator-specific outputs
        additional = pd.DataFrame({
            "f_var": f_var,
            "ci_obs_low": y_lower,
            "ci_obs_high": y_upper,
            "ci_pred_low": f_lower,
            "ci_pred_high": f_upper,
        }, index=x.index)

        return EmulationResults(
            mean=y_mean,
            std=np.sqrt(y_var),
            additional_data=additional,
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