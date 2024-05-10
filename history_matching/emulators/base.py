import logging
from abc import abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
        # Get data
        params = self.X_df.columns
        n_params = len( params )

        residuals = np.square(self.y_test.flatten() - self.y_pred_test)
        residuals_df = pd.DataFrame( self.X_test_df )
        residuals_df['residual'] = residuals

        # Draw plot
        fig, axs = plt.subplots( 1, n_params, figsize=(4*n_params, 4), sharey=True )
        axs = np.atleast_1d(axs)
        for i, param in enumerate(params):
            residuals_df.plot.scatter( x = param,
                                       y = 'residual',
                                       title = 'Residuals',
                                       legend = False,
                                       ax = axs[i]
                                     )
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

    def plot_implausibility(self, x=None, target=0, target_var=0, model_var=0, threshold=3):
        """Get implausibility for a given set of parameters.

        Args:
            x : Input data. Pandas dataframe with columns representing parameter
                values where the implausibility metric will be evaluated.
                If not provided, both testing and training data will be used for
                generating the plots.
            target : Scalar indicating the value to use as reference for the 
                     implausiblity computation. This is typically extracted from
                     observed data.
            target_var : Variance of the target point.
            model_var : Model discrepancy or variance. This parameter quantifies
                        the discrepancy between the model output and real life
                        data.
            threshold: Implausibility threshold. Sets of parameters within this
                       threshold are deemed as non-implausible.
        """
        # Get data
        params = self.X_df.columns
        n_params = len( params )

        # Compute implausibility
        implausibility = x.copy()
        implausibility['implausibility'] = self.get_implausibility( x, target, target_var, model_var ).values
        implausibility['predicted'] = self.predict(x)['value'].values
        implausible = implausibility[ implausibility['implausibility'] > threshold ]
        non_implausible = implausibility[ implausibility['implausibility'] <= threshold ]

        # Plot implausibility (scatter)
        fig, axs = plt.subplots( 3, n_params, figsize=(3*n_params, 7.5), sharex=True, sharey=False )
        for i, param in enumerate(params):
            ax_predict = axs[0,i] if len(params)>1 else axs[0]
            ax_implausibility = axs[1,i] if len(params)>1 else axs[1]
            ax_logimplausibility = axs[2,i] if len(params)>1 else axs[2]
            
            implausibility.plot( x=param, y='predicted', style='.', legend=False, ax=ax_predict, logy=False )
            ax_predict.axhline( y=target, linestyle='--', color='tab:green' )
            
            implausibility.plot( x=param, y='implausibility', style='.', color='m', legend=False, ax=ax_implausibility, logy=False )
            ax_implausibility.axhline( y=threshold, linestyle='--', color='k' )

            implausibility.plot( x=param, y='implausibility', style='.', color='m', legend=False, ax=ax_logimplausibility, logy=True )
            ax_logimplausibility.axhline( y=threshold, linestyle='--', color='k' )
            
            if i==0:
                legend_target = [Line2D([0], [0], color='tab:green', linestyle='--', label='target')]
                ax_predict.legend(handles=legend_target)

                legend_threshold = [Line2D([0], [0], color='k', linestyle='--', label='threshold')]
                ax_implausibility.legend(handles=legend_threshold)
                ax_logimplausibility.legend(handles=legend_threshold)

        fig.tight_layout()

        # Plot implausibility (pairplot)
        if len(params)>1:
            implausibility['color'] = 'tab:cyan'
            implausibility.loc[ implausibility['implausibility']<=threshold ,'color'] = 'tab:orange' 
            axs_sm = pd.plotting.scatter_matrix( implausibility[params], 
                                                 alpha    = 0.9,
                                                 figsize  = (3*n_params, 3*n_params),
                                                 c        = implausibility['color'], 
                                                 marker   = 'o',
                                                 diagonal = 'kde',
                                                 density_kwds = {'color':'tab:blue'},
                                                )
            fig_sm = axs_sm[0][0].get_figure()
            fig_sm.suptitle( 'Scatter Matrix\n(orange: non-implausible;  cyan: implausible)' )
            fig_sm.tight_layout()

        # Plot histogram 
        fig_hist_nimp, axs_hist_nimp = plt.subplots( 1, n_params, figsize=(4*n_params, 4), sharey=True )
        fig_kde_nimp , axs_kde_nimp  = plt.subplots( 1, n_params, figsize=(4*n_params, 4), sharey=True )
        axs_hist_nimp = np.atleast_1d(axs_hist_nimp)
        axs_kde_nimp  = np.atleast_1d(axs_kde_nimp )
        n_bins = 12
        for i, param in enumerate(params):
            non_implausible[param].plot.hist( title='Non-implausible', 
                                              bins=n_bins, 
                                              color='tab:blue'  , alpha=0.5, ax=axs_hist_nimp[i] )
            non_implausible[param].plot.kde ( title='Non-implausible', 
                                              color='tab:blue'  , alpha=0.5, ax=axs_kde_nimp[i] )
        fig_hist_nimp.tight_layout()
        fig_kde_nimp .tight_layout()

        # Let's add histograms to the pairplot
        if len(params)>1:
            for i, param in enumerate(params):
                non_implausible[param].plot.hist( title = 'Non-implausible', 
                                                  bins  = n_bins, 
                                                  color = 'tab:orange'  , alpha=0.5, 
                                                  ax    = axs_sm[i,i]
                                                 )
        
            # Plot z-score

        
        
        return
