import logging
from abc import abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection

from history_matching.config import Config


class BaseEmulator:
    """Base class for emulators."""

    def __init__(self, x: Optional[pd.DataFrame] = None, y: Optional[pd.DataFrame] = None, test_fraction=0.25):
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
        self.X_df = None  # Input data (full set as a dataframe)
        self.X_train = None  # Input data/parameters for training emulators
        self.X_test = None  # Input data/parameters for testing emulators
        self.y_df = None  # Model ouput data (full set as a dataframe)
        self.y_train = None  # Model output data/observations for training emulators
        self.y_test = None  # Model output data/observations for testing emulators
        self.y_pred = None  # Array of data predicted by the emulator
        self.y_pred_test = None  # Array of testing data predicted by the emulator
        self.y_test_pred_df = None  # Testing data predicted by the emulator (dataframe)

        # Status flags
        self.training_complete = False
        self.testing_complete = False

        # Performance metrics
        self.mse = np.nan  # Mean Squared Error (MSE)
        self.r2score = np.nan  # R^2 regression score

        # Read arguments
        if (x is not None) and (y is not None):
            X = x.to_numpy()
            Y = y.to_numpy()

            # Split data into testing and training datasets
            self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, Y, test_size=test_fraction)
            self.X_train_df = pd.DataFrame( self.X_train, columns=x.columns )
            self.X_test_df  = pd.DataFrame( self.X_test , columns=x.columns )
            self.Y_train_df = pd.DataFrame( self.y_train, columns=y.columns )
            self.Y_test_df  = pd.DataFrame( self.y_test , columns=y.columns )
            
            # Save some additional initialization data
            self.X_df = pd.DataFrame(x)
            self.y_df = pd.DataFrame(y)

        return

    @abstractmethod
    def train(self):
        """Trains the emulator."""
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

    def get_implausibility(self, x: pd.DataFrame, target, target_var, model_var=0, qlow=0.05, qhigh=0.95):
        """Get implausibility for a given set of parameters.

        Args:
            x : Input data. Pandas dataframe with columns representing parameter
                values where the implausibility metric will be evaluated.
            target : Scalar indicating the value to use as reference for the 
                     implausiblity computation. This is typically extracted from
                     observed data.
            target_var : Variance of the target point.
            model_var : Model discrepancy or variance. This parameter quantifies
                        the discrepancy between the model output and real life
                        data.
            qlow: Lower quantile for the estimated uncertainty interval.
            qhigh: Upper quantile for the estimated uncertainty interval.
        Returns:
            Numpy array with implausibility values for each of the data points 
            in x.
        """
        if not self.training_complete:
            self.train()
        predictions = self.predict(x, qlow, qhigh)
        predictions_var = predictions['high'] - predictions['low']
        
        implausibility = ( predictions['value'] - target )**2 / np.sqrt( predictions_var + target_var + model_var )

        return implausibility
        
    
    def get_implausibility_old(self, x: pd.DataFrame, observations: pd.DataFrame, feature: str, config: Config, qlow=0.05, qhigh=0.95):
        """Get implausibility for a given set of parameters.

        Args:
            x : Input data. Pandas dataframe with columns representing parameter
                values.
            observations : 
            qlow  : Lower quantile for the estimated uncertainty interval.
            qhigh : Upper quantile for the estimated uncertainty interval.

        Returns:
            Numpy array with implausibility values.
        """
        predictions = self.predict(x, qlow, qhigh)

        emulator_mean = predictions["value"]
        emulator_variance = predictions["variance"]

        # implausibility = abs(mean - observation) / sqrt(variance + observation_variance + discrepancy_variance)
        observation = observations.at[feature, "mean"]
        observation_variance = observations.at[feature, "variance"]
        model_variance = config.model_variance  # clorton: model_variance = discrepancy_variance from OG code
        # implausibility = abs(mean - observation) / np.sqrt(variance + observation_variance + discrepancy_variance)
        implausibility = abs(emulator_mean - observation) / np.sqrt(emulator_variance + observation_variance + model_variance)

        return implausibility

    @abstractmethod
    def print_emulator_description(self):
        """Display detailed specifications (for example, emulator coefficients)
        for the trained emulator.
        """
        raise NotImplementedError

    def test(self):
        """Tests and runs diagnostics on the trained emulator."""
        logging.debug("... testing emulator")

        if not self.training_complete:
            logging.warning("this emulator has not been trained yet")
        else:
            X_test_df = pd.DataFrame(self.X_test, columns=self.X_df.columns)
            self.y_test_pred_df = self.predict(X_test_df)
            self.y_pred_test = self.y_test_pred_df["value"].to_numpy()

            self.mse = np.linalg.norm(self.y_test.flatten() - self.y_pred_test, ord=2)

        self.testing_complete = True
        logging.debug("     emulator testing completed")
        return

    def info(self):
        """Prints report about the emulator and its performance."""
        print("... General information:")
        print("      Number of parameters = ", len(self.X_df.columns))
        print("      Number of samples (total) = ", len(self.X_df))
        print("      Number of training samples = ", np.size(self.X_train))
        print("      Number of testing samples = ", np.size(self.X_test))
        print("")

        if not self.training_complete:
            print("      This emulator has not been trained yet")
        else:
            print("... Emulator configuration:")
            self.print_emulator_description()
            print("")

        if not self.testing_complete:
            print("      This emulator has not been tested yet")
        else:
            print("... Performance results:")
            print("      MSE = ", self.mse)
            print("      R2 = ", self.r2score)
        return

    def plot_diagnostics(self):
        """Diagnostics plots for the trained emulator."""

        self.plot_residuals()
        self.plot_predictions()

        return

    def plot_residuals(self):
        """Plot residuals of predicted vs. true testing values. 
        """
        residuals = np.square(self.y_test.flatten() - self.y_pred_test)

        params = self.X_df.columns
        n_params = len( params )

        residuals_df = pd.DataFrame( self.X_test_df )
        residuals_df['residual'] = residuals
        axs = residuals_df.plot.scatter( x = params,
                                         y = 'residual',
                                         title = 'Residuals',
                                         legend = False,
                                         subplots = True,
                                         figsize  = (4*n_params, 4),
                                         sharey   = True
                                        )
        fig = axs[0].get_figure()
        fig.tight_layout()
        
        return

    def plot_predictions(self):
        """Plot the predicted and true testing values. 
        """
        # Get data
        params = self.X_df.columns
        n_params = len( params )
        predictions_df = self.X_test_df
        predictions_df['true'] = self.y_test
        predictions_df['prediction'] = self.y_test_pred_df['value']
        predictions_df['prediction (low)'] = self.y_test_pred_df['low']
        predictions_df['prediction (high)'] = self.y_test_pred_df['high']

        # Classify as correct or incorrect; assume incorrect and then overwrite if needed
        predictions_correct = predictions_df[ (predictions_df['true']<=predictions_df['prediction (high)'])     
                                             &(predictions_df['true']>=predictions_df['prediction (low)' ])]    \
                              .rename( columns={'prediction':'prediction (correct)'} )
        predictions_failed  = predictions_df[ (predictions_df['true']  >predictions_df['prediction (high)'])     
                                             |(predictions_df['true'] <predictions_df['prediction (low)' ])]    \
                              .rename( columns={'prediction':'prediction (failed)'} )
        
        # Draw plot
        fig, axs = plt.subplots( 1, n_params, figsize=(4*n_params, 4), sharey=True )
        axs = np.atleast_1d(axs)
        for i, param in enumerate(params):
            predictions_correct.plot( x=param, y='prediction (correct)', style='s', color='tab:green', ax=axs[i] )
            predictions_failed .plot( x=param, y='prediction (failed)' , style='o', color='tab:red'  , ax=axs[i] )
            predictions_df.plot( x=param, y='true', style='x', color='black', ax=axs[i], title='Prediction Accuracy' )

            axs[i].vlines( predictions_correct[param].values,
                           predictions_correct['prediction (low)' ].values,
                           predictions_correct['prediction (high)'].values,
                           color = 'tab:green'
                          )
            axs[i].vlines( predictions_failed[param].values,
                           predictions_failed['prediction (low)' ].values,
                           predictions_failed['prediction (high)'].values,
                           color = 'tab:red'
                          )
        fig.tight_layout()
                
        return
