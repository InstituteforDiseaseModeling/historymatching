import logging
import numpy as np
import pandas as pd
import scipy
from sklearn import linear_model as sklm

from .base import BaseEmulator




class LinearModel(BaseEmulator):
    """Emulator based on an ordinary least squares linear regression. The 
    emulator fits a linear regression model to minimize
    """    
    # Emulator data
    regression_model = None
    
    
    def train(self):
        """Fits a linear regression model to minimize the residual sum of 
        squares between observed targets in the training data and the targets
        predicted by the linear approximation.
        """
        logging.debug('... training emulator')
     
        self.regression_model = sklm.LinearRegression()
        self.regression_model.fit( self.X_train, self.y_train )
        
        #self.var = numpy.var( self.y_train )
        self.training_complete = True
        logging.debug('     training complete')
        return
    
    
    def predict(self, x:pd.DataFrame(), qlow=0.05, qhi=0.95 ):    
        """Predict an output using the trained emulator.
        
        Args:
            x : Input data. Pandas dataframe with columns representing parameter
                values.
            qlow  : Lower quantile for the estimated uncertainty interval.
            qhigh : Upper quantile for the estimated uncertainty interval.
            
        Returns:
            Pandas dataframe with predicted values and uncertainty intervals.
        """
        logging.debug('... predicting outputs using the trained emulator')
        # Compute the prediction
        X_pred = x.to_numpy()
        y_pred = self.regression_model.predict( X_pred )
        
        # Compute uncertainty bounds
        variance = np.var( self.y_train )
        sigma = variance**0.5
        low = scipy.stats.norm.ppf( q=qlow, scale=sigma )
        hi  = scipy.stats.norm.ppf( q=qhi , scale=sigma )
        
        # Prepare output and return
        out = pd.DataFrame( index=x.index )
        out['value'] = y_pred
        out['low' ] = out['value'] + low
        out['high'] = out['value'] + hi
        return out
    
    
    def print_emulator_description(self):
        """Display detailed specifications (for example, emulator coefficients)
        for the trained emulator.
        """
        print('      coefficients: ', self.regression_model.coef_ )
        print('      intercept   : ', self.regression_model.intercept_ )
        return

    """
    def diagnostics(self, qlow=0.05, qhi=0.95 ):
        
        # Compute uncertainty bounds
        sigma = self.var**0.5
        low = scipy.stats.norm.ppf( q=qlow, scale=sigma )
        hi  = scipy.stats.norm.ppf( q=qhi , scale=sigma )
        
        # Prepare data
        test_data = pandas.DataFrame(index=range(len(self.X_test)))
        test_data['x'] = self.X_test
        test_data['y_pred'] = self.regr.predict( self.X_test )
        test_data['y_low' ] = test_data['y_pred'] + low
        test_data['y_hi'  ] = test_data['y_pred'] + hi
        test_data['y_err' ] = test_data['y_hi'] - test_data['y_low']
        test_data['data'  ] = self.y_test
        
        test_success = test_data[ ( test_data['y_low'] <= test_data['data'] ) \
                                 &( test_data['y_hi' ] >= test_data['data'] ) ]
        test_success.rename( columns={'y_pred': 'predicted (correct)'}, inplace=True )
        test_failure = test_data[ ( test_data['y_low'] > test_data['data'] ) \
                                 |( test_data['y_hi' ] < test_data['data'] ) ]
        test_failure.rename( columns={'y_pred': 'predicted (error)'}, inplace=True )
        
        # Draw plots
        fig_ts, ax_ts = plt.subplots(1, 1, figsize=(7,5))
        test_success.plot( x='x', y='predicted (correct)', 
                           style='o', markersize=12, color='tab:green', alpha=0.7, ax=ax_ts )
        ax_ts.errorbar( test_success['x'].to_numpy(),
                        ( test_success['y_low'] + test_success['y_hi'] ).to_numpy()/2,
                        fmt = 'none',
                        yerr = test_success['y_err'].to_numpy()/2,
                        ecolor = 'tab:green',
                        capsize = 4
                       )

        test_failure.plot( x='x', y='predicted (error)', 
                           style='o', markersize=12, color='tab:red', alpha=0.7, ax=ax_ts )
        ax_ts.errorbar( test_failure['x'].to_numpy(),
                        ( test_failure['y_low'] + test_failure['y_hi'] ).to_numpy()/2,
                        fmt = 'none',
                        yerr = test_failure['y_err'].to_numpy()/2,
                        ecolor = 'tab:red',
                        capsize = 4
                       )
                
        test_data.plot( x='x', y='data', style='x', color='k', markersize=14, ax=ax_ts )
        
        
        return
    """