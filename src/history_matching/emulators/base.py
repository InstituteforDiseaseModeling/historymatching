from abc import abstractmethod
import logging
from typing import Dict

from asdf.extension import Converter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection

from history_matching.utils import dataframe_to_ndarray, ndarray_to_dataframe


class BaseEmulator:
    """Base class for emulators."""

    def __init__(self,
                 x: pd.DataFrame = None,
                 y: pd.DataFrame = None,
                 test_fraction=0.25
                 ):
        """Initialize the emulator.

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

        # Data arrays
        self.X_df = None            # Input data (full set as a dataframe)
        self.X_train = None         # Input data/parameters for training emulators
        self.X_test = None          # Input data/parameters for testing emulators
        self.y_df = None            # Model ouput data (full set as a dataframe)
        self.y_train = None         # Model output data/observations for training emulators
        self.y_test = None          # Model output data/observations for testing emulators
        self.y_pred = None          # Array of data predicted by the emulator
        self.y_pred_test = None     # Array of testing data predicted by the emulator
        self.y_test_pred_df = None  # Testing data predicted by the emulator (dataframe)

        # Status flags
        self.training_complete = False
        self.testing_complete = False

        # Performance metrics
        self.mse = np.nan       # Mean Squared Error (MSE)
        self.r2score = np.nan   # R^2 regression score

        # Read arguments
        if (x is not None) and (y is not None):

            X = x.to_numpy()
            Y = y.to_numpy()

            # Split data into testing and training datasets
            self.X_train, self.X_test, self.y_train, self.y_test = \
                model_selection.train_test_split(X, Y, test_size=test_fraction)

            # Save some additional initialization data
            self.X_df = pd.DataFrame(x)
            self.y_df = pd.DataFrame(y)

        return

    @abstractmethod
    def train(self):
        """Trains the emulator.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def print_emulator_description(self):
        """Display detailed specifications (for example, emulator coefficients)
        for the trained emulator.
        """
        raise NotImplementedError

    def test(self):
        """Tests and runs diagnostics on the trained emulator."""
        logging.debug('... testing emulator')

        if not self.training_complete:
            logging.warning('this emulator has not been trained yet')
        else:
            X_test_df = pd.DataFrame(self.X_test, columns=self.X_df.columns)
            self.y_test_pred_df = self.predict(X_test_df)
            self.y_pred_test = self.y_test_pred_df['value'].to_numpy()

            self.mse = np.linalg.norm(self.y_test.flatten() - self.y_pred_test, ord=2)

        self.testing_complete = True
        logging.debug('     emulator testing completed')
        return

    def info(self):
        """Prints report about the emulator and its performance."""
        print('... General information:')
        print('      Number of parameters = ', len(self.X_df.columns))
        print('      Number of samples (total) = ', len(self.X_df))
        print('      Number of training samples = ', np.size(self.X_train))
        print('      Number of testing samples = ', np.size(self.X_test))
        print('')

        if not self.training_complete:
            print('      This emulator has not been trained yet')
        else:
            print('... Emulator configuration:')
            self.print_emulator_description()
            print('')

        if not self.testing_complete:
            print('      This emulator has not been tested yet')
        else:
            print('... Performance results:')
            print('      MSE = ', self.mse)
            print('      R2 = ', self.r2score)
        return

    def plot_diagnostics(self):
        """Diagnostics plots for the trained emulator."""

        self.plot_residuals()
        self.plot_predictions()

        return

    def plot_residuals(self):
        """Plot residuals of predicted vs. true testing values. Designed for
        LinearModel with only one parameter.
        """
        residuals = np.square(self.y_test.flatten() - self.y_pred_test)
        residuals_df = pd.DataFrame({'theta': self.X_test.flatten(),
                                     'residual': residuals.flatten()}
                                    )
        residuals_df.plot(x='theta',
                          y='residual',
                          title='Residuals',
                          legend=False,
                          style='.',
                          figsize=(8, 4)
                          )

        return

    def plot_predictions(self):
        """Plot the predicted and true testing values. Designed for LinearModel
        with only one parameter.
        """
        # Get data
        predictions_vs_true = self.y_test_pred_df.copy()
        predictions_vs_true['theta'] = self.X_test
        predictions_vs_true['true'] = self.y_test
        predictions_vs_true['error'] = predictions_vs_true['high'] - predictions_vs_true['low']

        # Classify data into success and failures
        test_success = predictions_vs_true[(predictions_vs_true['low'] <= predictions_vs_true['true']) & (predictions_vs_true['high'] >= predictions_vs_true['true'])].copy()
        test_success.rename(columns={'value': 'predicted (correct)', 'true': 'true value'}, inplace=True)

        test_failure = predictions_vs_true[(predictions_vs_true['low'] > predictions_vs_true['true']) | (predictions_vs_true['high'] < predictions_vs_true['true'])].copy()
        test_failure.rename(columns={'value': 'predicted (failed)', 'true': 'true value'}, inplace=True)

        # Draw plots
        fig_ts, ax_ts = plt.subplots(1, 1, figsize=(8, 4))
        test_success.plot(x='theta', y='predicted (correct)', style='o',
                          markersize=12, color='tab:green', alpha=0.7,
                          ax=ax_ts
                          )
        ax_ts.errorbar(test_success['theta'].to_numpy(),
                       (test_success['low'] + test_success['high']).to_numpy() / 2,
                       fmt='none',
                       yerr=test_success['error'].to_numpy() / 2,
                       ecolor='tab:green',
                       capsize=4
                       )

        test_failure.plot(x='theta', y='predicted (failed)', style='o',
                          markersize=12, color='tab:red', alpha=0.7,
                          ax=ax_ts
                          )
        ax_ts.errorbar(test_failure['theta'].to_numpy(),
                       (test_failure['low'] + test_failure['high']).to_numpy() / 2,
                       fmt='none',
                       yerr=test_failure['error'].to_numpy() / 2,
                       ecolor='tab:red',
                       capsize=4
                       )

        predictions_vs_true.plot(x='theta', y='true', style='x',
                                 color='k', markersize=14,
                                 ax=ax_ts
                                 )

        return

    def to_yaml_tree(self, tag, ctx) -> Dict:

        dictionary = dict(
            X_df=dataframe_to_ndarray(self.X_df),
            X_train=self.X_train,
            X_test=self.X_test,
            y_df=dataframe_to_ndarray(self.y_df),
            y_train=self.y_train,
            y_test=self.y_test,
            y_pred=self.y_pred,
            y_pred_test=self.y_pred_test,
            y_test_pred_df=dataframe_to_ndarray(self.y_test_pred_df),
            training_complete=self.training_complete,
            testing_complete=self.testing_complete,
            mse=self.mse,
            r2score=self.r2score
        )

        return dictionary

    @staticmethod
    def from_yaml_tree(node, tag, ctx) -> "BaseEmulator":

        emulator = BaseEmulator()   # pass no initial values

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

        return emulator


class BaseEmulatorConverter(Converter):

    tags = ["asdf://idmod.org/asdf/tags/emulators/baseemulator-1.0.0"]
    types = ["history_matching.emulators.base.BaseEmulator"]

    def to_yaml_tree(self, obj, tag, ctx):
        return obj.to_yaml_tree(tag, ctx)

    def from_yaml_tree(self, node, tag, ctx):
        return BaseEmulator.from_yaml_tree(node, tag, ctx)
