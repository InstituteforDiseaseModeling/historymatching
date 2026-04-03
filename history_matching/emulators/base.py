import logging
from abc import abstractmethod
from typing import Optional
import itertools

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import metrics

from .results import EmulationResults




class BaseEmulator:
    """Base class for emulators."""

    def __init__(self, x: Optional[pd.DataFrame] = None, y: Optional[pd.DataFrame] = None, test_fraction=0.25):
        """Initialize the emulator.

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
            self.emulator_metrics = {}

        return

    
    @abstractmethod
    def train(self):
        """Trains the emulator."""
        raise NotImplementedError

    
    @abstractmethod
    def predict(self, x: pd.DataFrame) -> EmulationResults:
        """Predict an output using the trained emulator.

        Args:
            x: Input data. Pandas dataframe with columns representing parameter
                values.

        Returns:
            EmulationResults with predicted values and uncertainty information.
        """
        raise NotImplementedError

    
    def get_implausibility(self, x: pd.DataFrame, target, target_var, model_discrepancy=0):
        """Get implausibility for a given set of parameters.

        Args:
            x: Input data. Pandas dataframe with columns representing parameter
                values where the implausibility metric will be evaluated.
            target: Scalar indicating the value to use as reference for the
                implausibility computation. This is typically extracted from
                observed data.
            target_var: Variance of the target point.
            model_discrepancy: Model discrepancy or variance. This parameter
                quantifies the discrepancy between the model output and real
                life data.
        Returns:
            Numpy array with implausibility values for each of the data points 
            in x.
        """
        if not self.training_complete:
            self.train()

        predictions = self.predict(x)
        predictions_var = predictions.get_variance()  # Get variance directly
        implausibility = abs( predictions.get_mean() - target ) / np.sqrt( predictions_var + target_var + model_discrepancy )
        
        return implausibility    
    

    @abstractmethod
    def print_emulator_description(self):
        """Display detailed specifications (for example, emulator coefficients)
        for the trained emulator.
        """
        raise NotImplementedError

    @abstractmethod
    def get_hyperparameters(self) -> dict:
        """Return emulator hyperparameters as a JSON-serializable dict.

        Subclasses should include all fitted hyperparameters relevant to
        understanding the emulator (e.g. lengthscales, noise variance,
        regression coefficients).  Parameter names in the input space should
        use the original column names where possible.
        """
        raise NotImplementedError

    
    def test(self):
        """Tests and runs diagnostics on the trained emulator."""
        logging.debug('... testing emulator')

        if not self.training_complete:
            logging.warning('this emulator has not been trained yet')
        else:
            
            self.y_test_pred_results = self.predict( self.X_test_df )
            self.y_test_pred = self.y_test_pred_results.get_mean().to_numpy()

            self.y_train_pred_results = self.predict( self.X_train_df )
            self.y_train_pred = self.y_train_pred_results.get_mean().to_numpy()

            y_test = self.y_test.flatten()
            y_pred = self.y_test_pred.flatten()
            n_test = len( self.y_test_pred )
            self.emulator_metrics['MSE'  ] = metrics.mean_squared_error ( y_test, y_pred )
            self.emulator_metrics['L-inf'] = metrics.max_error          ( y_test, y_pred )
            self.emulator_metrics['L-1'  ] = metrics.mean_absolute_error( y_test, y_pred )
            self.emulator_metrics['R2'   ] = metrics.r2_score           ( y_test, y_pred )
            self.emulator_metrics['AIC'  ] = n_test * np.log( self.emulator_metrics['MSE'] ) + 2*self.X_test.shape[1]
            self.emulator_metrics['BIC'  ] = n_test * np.log( self.emulator_metrics['MSE'] ) + 2*self.X_test.shape[1]*np.log(n_test)
        
        self.testing_complete = True
        logging.debug('     emulator testing completed')
        return

    
    def info(self):
        """Prints report about the emulator and its performance."""
        print("... General information:")
        print("      Number of parameters = ", len(self.X_df.columns))
        print("      Number of samples (total) = ", len(self.X_df))
        print("      Number of training samples = ", len(self.X_train))
        print("      Number of testing samples = ", len(self.X_test))
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
            for key, value in self.emulator_metrics.items():
                print( f'      {key:<6} = {value}' )
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

        residuals = np.square(self.y_test.flatten() - self.y_test_pred)
        residuals_df = pd.DataFrame( self.X_test_df )
        residuals_df['residual'] = residuals

        # Draw plot (wrap to max 5 columns)
        max_cols = 5
        n_cols = min(n_params, max_cols)
        n_rows = int(np.ceil(n_params / n_cols))
        fig, axs = plt.subplots( n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), sharey=True )
        axs = np.atleast_1d(axs).flatten()
        for i, param in enumerate(params):
            residuals_df.plot.scatter( x = param,
                                       y = 'residual',
                                       title = 'Residuals',
                                       legend = False,
                                       ax = axs[i]
                                     )
        for i in range(n_params, len(axs)):
            axs[i].set_visible(False)
        fig.tight_layout()
        
        return
    
    
    def plot_predictions(self):
        """Plot the predicted and true testing values. 
        """
        # Get data
        params = self.X_df.columns
        n_params = len( params )
        predictions_df = self.X_test_df.copy()
        predictions_df['true'] = self.y_test
        predictions_df['prediction'] = self.y_test_pred_results.get_mean()
        # Use observation CIs (includes noise variance) so that stochastic
        # test points are classified correctly.  Latent-function CIs (ci_pred)
        # exclude noise and will over-count "failed" predictions.
        additional_data = self.y_test_pred_results.get_additional_data()
        if additional_data is not None and 'ci_obs_low' in additional_data.columns:
            predictions_df['prediction (low)'] = additional_data['ci_obs_low']
            predictions_df['prediction (high)'] = additional_data['ci_obs_high']
        else:
            # Compute confidence intervals if not available
            ci_low, ci_high = self.y_test_pred_results.get_ci(0.95)
            predictions_df['prediction (low)'] = ci_low
            predictions_df['prediction (high)'] = ci_high
        predictions_df['error (normalized)'] = ( predictions_df['true'] - predictions_df['prediction'] )    \
                                               .div( predictions_df['true'] )

        # Classify as correct or incorrect; assume incorrect and then overwrite if needed
        predictions_correct = predictions_df[ (predictions_df['true']<=predictions_df['prediction (high)'])     
                                             &(predictions_df['true']>=predictions_df['prediction (low)' ])]    \
                              .rename( columns={'prediction':'prediction (correct)'} )
        predictions_failed  = predictions_df[ (predictions_df['true']  >predictions_df['prediction (high)'])     
                                             |(predictions_df['true'] <predictions_df['prediction (low)' ])]    \
                              .rename( columns={'prediction':'prediction (failed)'} )
        
        # Plot predictions (wrap to max 5 columns)
        max_cols = 5
        n_cols = min(n_params, max_cols)
        n_rows = int(np.ceil(n_params / n_cols))
        fig, axs = plt.subplots( n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), sharey=True )
        axs = np.atleast_1d(axs).flatten()
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
        for i in range(n_params, len(axs)):
            axs[i].set_visible(False)
        fig.tight_layout()


        # Plot predicted vs. observed — two panels: train (left) and test (right)
        fig_predobs, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(10, 5))

        # Common axis limits
        train_true = self.y_train.flatten()
        train_pred = self.y_train_pred_results.get_mean().values
        test_pred = predictions_df['prediction'].values
        test_true = predictions_df['true'].values
        lo = min(test_true.min(), test_pred.min(), train_true.min(), train_pred.min())
        hi = max(test_true.max(), test_pred.max(), train_true.max(), train_pred.max())
        margin = (hi - lo) * 0.05
        ref = [lo - margin, hi + margin]

        # ── Train panel ──
        train_additional = self.y_train_pred_results.get_additional_data()
        if train_additional is not None and 'ci_obs_low' in train_additional.columns:
            train_low = train_additional['ci_obs_low'].values
            train_high = train_additional['ci_obs_high'].values
            train_err = np.array([train_pred - train_low, train_high - train_pred])
            ax_train.errorbar(train_true, train_pred, yerr=train_err,
                              fmt='o', color='tab:blue', alpha=0.4, markersize=3,
                              elinewidth=0.5)
        else:
            ax_train.scatter(train_true, train_pred, color='tab:blue', alpha=0.4, s=10)
        ax_train.plot(ref, ref, '--', color='gray', alpha=0.5)
        ax_train.set_title(f'Train (n={len(train_true)})')
        ax_train.set_xlabel('True value')
        ax_train.set_ylabel('Predicted')
        ax_train.set_xlim(ref)
        ax_train.set_ylim(ref)

        # ── Test panel ──
        test_low  = predictions_df['prediction (low)'].values
        test_high = predictions_df['prediction (high)'].values
        test_err  = np.array([test_pred - test_low, test_high - test_pred])
        test_correct_mask = (test_true <= test_high) & (test_true >= test_low)
        n_correct = test_correct_mask.sum()
        n_total = len(test_true)

        ax_test.errorbar(test_true[test_correct_mask], test_pred[test_correct_mask],
                         yerr=test_err[:, test_correct_mask],
                         fmt='o', color='tab:green', alpha=0.6, markersize=4,
                         elinewidth=0.8, label=f'correct ({n_correct})')
        ax_test.errorbar(test_true[~test_correct_mask], test_pred[~test_correct_mask],
                         yerr=test_err[:, ~test_correct_mask],
                         fmt='o', color='tab:red', alpha=0.6, markersize=4,
                         elinewidth=0.8, label=f'failed ({n_total - n_correct})')
        ax_test.plot(ref, ref, '--', color='gray', alpha=0.5)
        ax_test.set_title(f'Test (n={n_total}, {100*n_correct/n_total:.0f}% correct)')
        ax_test.set_xlabel('True value')
        ax_test.set_ylabel('Predicted')
        ax_test.set_xlim(ref)
        ax_test.set_ylim(ref)
        ax_test.legend(fontsize=8)

        fig_predobs.tight_layout()
        
        
        # Compute normalized errors for train and test
        train_error_norm = ((self.y_train.flatten() - self.y_train_pred_results.get_mean().values)
                            / np.where(self.y_train.flatten() != 0, self.y_train.flatten(), 1.0))
        test_error_norm = predictions_df['error (normalized)'].values

        # plot predicted vs (normalized) error — train and test panels
        fig_prederr, (ax_prederr_tr, ax_prederr_te) = plt.subplots(1, 2, figsize=(10, 5))

        ax_prederr_tr.scatter(train_error_norm, train_pred, color='tab:blue', alpha=0.4, s=10)
        ax_prederr_tr.set_title(f'Train (n={len(train_true)})')
        ax_prederr_tr.set_xlabel('error (normalized)')
        ax_prederr_tr.set_ylabel('prediction')

        ax_prederr_te.scatter(test_error_norm, test_pred, color='tab:blue', alpha=0.4, s=10)
        ax_prederr_te.set_title(f'Test (n={len(test_true)})')
        ax_prederr_te.set_xlabel('error (normalized)')
        ax_prederr_te.set_ylabel('prediction')

        fig_prederr.tight_layout()


        # Plot histogram with normalized error — train and test panels
        n_bins = 20
        fig_errhist, (ax_hist_tr, ax_hist_te) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

        pd.Series(train_error_norm).replace([np.inf, -np.inf], np.nan).dropna().plot.hist(
            bins=n_bins, ax=ax_hist_tr, color='tab:blue', alpha=0.7)
        ax_hist_tr.set_title(f'Train (n={len(train_true)})')
        ax_hist_tr.set_xlabel('error (normalized)')
        ax_hist_tr.set_ylabel('Count')

        pd.Series(test_error_norm).replace([np.inf, -np.inf], np.nan).dropna().plot.hist(
            bins=n_bins, ax=ax_hist_te, color='tab:blue', alpha=0.7)
        ax_hist_te.set_title(f'Test (n={len(test_true)})')
        ax_hist_te.set_xlabel('error (normalized)')

        fig_errhist.tight_layout()
        
        return

    
    def plot_implausibility(self, x=None, target=0, target_var=0, model_discrepancy=0, threshold=3):
        """Get implausibility for a given set of parameters.

        Args:
            x: Input data. Pandas dataframe with columns representing parameter
                values where the implausibility metric will be evaluated.
                If not provided, both testing and training data will be used for
                generating the plots.
            target: Scalar indicating the value to use as reference for the 
                     implausiblity computation. This is typically extracted from
                     observed data.
            target_var: Variance of the target point.
            model_discrepancy: Model discrepancy or variance. This parameter quantifies
                        the discrepancy between the model output and real life
                        data.
            threshold: Implausibility threshold. Sets of parameters within this
                       threshold are deemed as non-implausible.
        """
        # Get data
        params = self.X_df.columns
        n_params = len( params )

        # Compute implausibility
        y_pred = self.predict(x)
        implausibility = x.copy()
        implausibility['implausibility'] = self.get_implausibility( x, target, target_var, model_discrepancy ).values
        implausibility['predicted'] = y_pred.get_mean().values
        implausible = implausibility[ implausibility['implausibility'] > threshold ]
        non_implausible = implausibility[ implausibility['implausibility'] <= threshold ]


        # Pair plots with model outputs (could be done easily with Seaborn, but trying to avoid adding more dependencies)
        # The colorbar (and marker color) is the model output; marker outline: red if implausible; black if non-implausible
        if n_params>1:  
            fig_pp, axs_pp = plt.subplots( n_params, n_params, figsize=(4*n_params,4*n_params), sharex=False, sharey=True )
            combinations = list( itertools.product(params, repeat=2) )
            for index, (param1, param2) in enumerate(combinations):
                i = index % n_params
                j = index // n_params
                edge_colors = implausibility.apply(lambda row: 'none' if row['implausibility'] > threshold else 'tab:red', axis=1)
                
                if i != j:
                    x.plot.scatter( x = param1, 
                                    y = param2, 
                                    c = y_pred.get_mean().values,
                                    cmap = 'viridis',
                                    colorbar = True, 
                                    edgecolors = edge_colors,
                                    ax = axs_pp[i,j] 
                                   )
                else:
                    axs_pp[i,j].spines['top'   ].set_visible(False)
                    axs_pp[i,j].spines['bottom'].set_visible(False)
                    axs_pp[i,j].spines['left'  ].set_visible(False)
                    axs_pp[i,j].spines['right' ].set_visible(False)
            fig_pp.suptitle( 'Predicted Output Pair Plots\n(red marker: non-implausible)' )
            fig_pp.tight_layout()
            
        
        # Plot implausibility (scatter)
        fig_imp, axs_imp = plt.subplots( 3, n_params, figsize=(4*n_params, 12), sharex=True, sharey=False )
        for i, param in enumerate(params):
            ax_predict = axs_imp[0,i] if len(params)>1 else axs_imp[0]
            ax_implausibility = axs_imp[1,i] if len(params)>1 else axs_imp[1]
            ax_logimplausibility = axs_imp[2,i] if len(params)>1 else axs_imp[2]
            
            implausibility.plot( x=param, y='predicted', style='.', legend=False, ax=ax_predict, logy=False )
            ax_predict.axhline( y=target, linestyle='--', color='tab:green' )
            ax_predict.set_ylabel( 'Predicted' )
            
            implausibility.plot( x=param, y='implausibility', style='.', color='m', legend=False, ax=ax_implausibility, logy=False )
            ax_implausibility.axhline( y=threshold, linestyle='--', color='k' )
            ax_implausibility.set_ylabel( 'Implausibility' )
            
            implausibility.plot( x=param, y='implausibility', style='.', color='m', legend=False, ax=ax_logimplausibility, logy=True )
            ax_logimplausibility.axhline( y=threshold, linestyle='--', color='k' )
            ax_logimplausibility.set_ylabel( 'Implausibility' )
            
            if i==0:
                legend_target = [Line2D([0], [0], color='tab:green', linestyle='--', label='target')]
                ax_predict.legend(handles=legend_target)

                legend_threshold = [Line2D([0], [0], color='k', linestyle='--', label='threshold')]
                ax_implausibility.legend(handles=legend_threshold)
                ax_logimplausibility.legend(handles=legend_threshold)
        fig_imp.tight_layout()

        
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

            # Let's add histograms to the pairplot
            if ( len(params)>1 ) and ( len(non_implausible)>1 ):
                for i, param in enumerate(params):
                    non_implausible[param].plot.kde( #title = 'Non-implausible', 
                                                     color = 'tab:orange'  , alpha=0.5, 
                                                     ax    = axs_sm[i,i], 
                                                     secondary_y = True
                                                    )

            # Tidy-up figure   
            fig_sm = axs_sm[0][0].get_figure()
            fig_sm.suptitle( 'Scatter Matrix\n(orange: non-implausible;  cyan: implausible)' )
            fig_sm.tight_layout()

        
        # Plot histogram 
        fig_hist_nimp, axs_hist_nimp = plt.subplots( 1, n_params, figsize=(4*n_params, 4), sharey=True )
        fig_kde_nimp , axs_kde_nimp  = plt.subplots( 1, n_params, figsize=(4*n_params, 4), sharey=True )
        axs_hist_nimp = np.atleast_1d( axs_hist_nimp )
        axs_kde_nimp  = np.atleast_1d( axs_kde_nimp  )
        n_bins = 12
        for i, param in enumerate(params):
            non_implausible[param].plot.hist( title = 'Non-implausible Points (histogram)', 
                                              bins  = n_bins, 
                                              color = 'tab:blue', alpha=0.5, ax=axs_hist_nimp[i] )
            axs_hist_nimp[i].set_xlabel( param )
            if len(non_implausible)>1:
                non_implausible[param].plot.kde( title = 'Non-implausible Points (KDE)', 
                                                 color = 'tab:blue', 
                                                 ax    = axs_kde_nimp[i] )
                axs_kde_nimp[i].set_xlabel( param )
        fig_hist_nimp.tight_layout()
        fig_kde_nimp .tight_layout()

        return


    def plot_zscore( self, target=0, target_var=0, model_var=0, threshold=3 ):
        """Plot a Z-score diagnoses for testing and training data.

        Args:
            target: Scalar indicating the value to use as reference for the 
                     implausiblity computation. This is typically extracted from
                     observed data.
            target_var: Variance of the target point.
            model_var: Model discrepancy or variance. This parameter quantifies
                        the discrepancy between the model output and real life
                        data.
            threshold: Implausibility threshold. Sets of parameters within this
                       threshold are deemed as non-implausible.
        """
       
        # Compute Z-values
        data_train = self.X_train_df.copy()
        data_train['output (true)'] = self.y_train.flatten()
        data_train['implausibility'] = self.get_implausibility( self.X_train_df, target, target_var, model_var ).values
        data_train['predicted'] = self.y_train_pred_results.get_mean().values
        # Try to get confidence intervals from additional data, fallback to computed ones
        train_additional = self.y_train_pred_results.get_additional_data()
        if train_additional is not None and 'ci_obs_high' in train_additional.columns:
            data_train['predicted_obs_var' ] = ( ( train_additional['ci_obs_high'] - train_additional['ci_obs_low'] )/3 )**2
            data_train['predicted_pred_var'] = ( ( train_additional['ci_pred_high'] - train_additional['ci_pred_low'] )/3 )**2
        else:
            # Use standard deviation as fallback
            train_std = self.y_train_pred_results.get_std()
            data_train['predicted_obs_var'] = train_std**2
            data_train['predicted_pred_var'] = train_std**2
        data_train['error'] = self.y_train.flatten() - self.y_train_pred_results.get_mean().values
        data_train['Z_noisy'] = data_train['error'].div( np.sqrt( data_train['predicted_obs_var' ] ) )
        data_train['Z_noiseless'] = data_train['error'].div( np.sqrt( data_train['predicted_pred_var' ] ) )
        data_train_implausible    = data_train[ data_train['implausibility']> threshold ]
        data_train_nonimplausible = data_train[ data_train['implausibility']<=threshold ]

        data_test = self.X_test_df.copy()
        data_test['output (true)'] = self.y_test.flatten()
        data_test['implausibility'] = self.get_implausibility( self.X_test_df, target, target_var, model_var ).values
        data_test['predicted'] = self.y_test_pred_results.get_mean().values
        # Try to get confidence intervals from additional data, fallback to computed ones
        test_additional = self.y_test_pred_results.get_additional_data()
        if test_additional is not None and 'ci_obs_high' in test_additional.columns:
            data_test['predicted_obs_var' ] = ( ( test_additional['ci_obs_high'] - test_additional['ci_obs_low'] )/3 )**2
            data_test['predicted_pred_var'] = ( ( test_additional['ci_pred_high'] - test_additional['ci_pred_low'] )/3 )**2
        else:
            # Use standard deviation as fallback
            test_std = self.y_test_pred_results.get_std()
            data_test['predicted_obs_var'] = test_std**2
            data_test['predicted_pred_var'] = test_std**2
        data_test['error'] = self.y_test.flatten() - self.y_test_pred_results.get_mean().values
        data_test['Z_noisy'] = data_test['error'].div( np.sqrt( data_test['predicted_obs_var' ] ) )
        data_test['Z_noiseless'] = data_test['error'].div( np.sqrt( data_test['predicted_pred_var' ] ) )
        data_test_implausible    = data_test[ data_test['implausibility']> threshold ]
        data_test_nonimplausible = data_test[ data_test['implausibility']<=threshold ]

        # Draw plots
        fig_z, axs_z = plt.subplots( 2, 1, figsize=(10,8), sharex=True )
        
        data_train_implausible   .plot.scatter( x='output (true)', y='Z_noisy', marker='x', c='tab:gray', label='implausible (noisy)'    , ax=axs_z[0] )
        data_train_nonimplausible.plot.scatter( x='output (true)', y='Z_noisy', marker='o', c='tab:blue', label='non-implausible (noisy)', ax=axs_z[0] )
        data_train_implausible   .plot.scatter( x='output (true)', y='Z_noiseless', marker='x', c='black' , label='implausible'    , ax=axs_z[0] )
        data_train_nonimplausible.plot.scatter( x='output (true)', y='Z_noiseless', marker='o', c='tab:green', label='non-implausible', ax=axs_z[0] )
        axs_z[0].set_title('Training data\n(red/dashed line: target observation)')
        
        data_test_implausible   .plot.scatter( x='output (true)', y='Z_noisy', marker='x', c='tab:gray', label='implausible (noisy)'    , ax=axs_z[1] )
        data_test_nonimplausible.plot.scatter( x='output (true)', y='Z_noisy', marker='o', c='tab:blue', label='non-implausible (noisy)', ax=axs_z[1] )
        data_test_implausible   .plot.scatter( x='output (true)', y='Z_noiseless', marker='x', c='black' , label='implausible'    , ax=axs_z[1] )
        data_test_nonimplausible.plot.scatter( x='output (true)', y='Z_noiseless', marker='o', c='tab:green', label='non-implausible', ax=axs_z[1] )
        axs_z[1].set_title('Testing data')

        xlim = axs_z[0].get_xlim()
        for i in [0, 1]:
            axs_z[i].set_ylabel('Z')
            axs_z[i].set_xlabel('y (observed data)')
            axs_z[i].axvline( target, color='tab:red', linestyle='--' )
            axs_z[i].fill_between( xlim, -2  ,  2  , color='tab:green' , alpha=0.2 )
            axs_z[i].fill_between( xlim, -2.7, -2  , color='yellow', alpha=0.2 )
            axs_z[i].fill_between( xlim,  2  ,  2.7, color='yellow', alpha=0.2 )
            axs_z[i].fill_between( xlim, -3  , -2.7, color='tab:red'   , alpha=0.2 )
            axs_z[i].fill_between( xlim,  2.7,  3  , color='tab:red'   , alpha=0.2 )
        
        fig_z.tight_layout()

        return
