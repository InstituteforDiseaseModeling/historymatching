import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow_probability as tfp
    logging.debug(f"Loaded tensorflow-probability version {tfp.__version__}.")

from .base import BaseEmulator



class TensorFlowGLM(BaseEmulator):
    """Generalised Linear Model emulator implemented in TensorFlow."""

    def __init__(self, x: Optional[pd.DataFrame] = None, y: Optional[pd.DataFrame] = None, test_fraction=0.25, link='linear'):
        """Initialise the Generalised Linear Model (GLM) emulator implemented in TensorFlow.

        Args:
            x : Input data. Pandas dataframe with columns representing parameter
                values.
            y : Output data. Pandas dataframe with columns representing
                observations and rows representing samples. Each row in this
                dataframe must match the corresponding row in `x`.
            test_fraction : Fraction of `x` and `y` samples to be used for
                testing. This is a scalar between 0 and 1.
            link : Link function for the GLM model. It can be either 'linear'
                or 'poisson'.

        Returns:
            None
        """
        super().__init__(x, y, test_fraction)
        self.model = tfp.glm.Normal()    if link=='linear'    else  \
                     tfp.glm.Poisson()   if link=='poisson'   else  \
                     None    # This last case should't happen; it should raise an error
        self.link = link

        return

    
    def train(self):
        """Fits a Generalised Linear Model."""

        logging.debug("... training emulator")

        # https://www.tensorflow.org/probability/api_docs/python/tfp/glm/fit
        # model_matrix (Batch of) float-like, matrix-shaped Tensor where each row represents a sample's features.
        # response     (Batch of) vector-shaped Tensor where each element represents a sample's observed response (to the corresponding row of features). Must have same dtype as model_matrix.
        # model        tfp.glm.ExponentialFamily-like instance which implicitly characterizes a negative log-likelihood loss by specifying the distribuion's mean, gradient_mean, and variance.

        # tensorflow-probability uses deprecated portions of setuptools which don't warn
        # until tensorflow-probability is lazily loaded here.
        x_tf = np.hstack(  ( np.ones( (len(self.X_train), 1 ) ), 
                             self.X_train.reshape(-1,1) )  )
        y_tf = np.float64( self.y_train.reshape( (len(self.y_train),) ) )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_coefficients,   \
                linear_response,  \
                is_converged,     \
                num_iter = tfp.glm.fit( model_matrix = x_tf,
                                        response     = y_tf,
                                        model        = self.model    #tfp.glm.Normal()
                                       )

        self.model_coefficients = model_coefficients
        self.linear_response = linear_response
        self.is_converged = is_converged
        self.num_iter = num_iter

        self.training_complete = True
        logging.debug("     training complete")
        return

    
    def predict(self, x: pd.DataFrame, qlow=0.05, qhigh=0.95):
        """Predict an output using the trained emulator."""

        logging.debug("... predicting outputs using the trained emulator")
        # Compute the prediction
        x_tf = np.hstack(  ( np.ones( (len(x), 1 ) ), 
                             x.to_numpy().reshape(-1,1) )  )

        # model_matrix is incoming parameter values
        out = pd.DataFrame(index=x.index)
        linear_response_pred = tf.linalg.matvec( x_tf, self.model_coefficients ) 
        if self.link=='linear':        
            out['value'] = linear_response_pred.numpy()
        elif self.link=='poisson':
            out['value'] = tf.math.exp( linear_response_pred ).numpy()
        else:
            out['value'] = np.nan  # this shouldn't happen. Need to raise an error.

        # Calculate uncertainty intervals for the predicted outputs
        predicted_dist = tfp.distributions.Normal( loc = linear_response_pred,
                                                   scale = 7.5,  # Assuming constant; this should be sigma
                                                  )
        num_samples_predictive = 1000
        predicted_samples = predicted_dist.sample( num_samples_predictive )
        lower_bounds = np.percentile(predicted_samples, 2.5, axis=0)
        upper_bounds = np.percentile(predicted_samples, 97.5, axis=0)
        out['low']  = lower_bounds
        out['high'] = upper_bounds
        
        return out

    
    def print_emulator_description(self):
        """Display detailed specifications (for example, emulator coefficients)
        for the trained emulator.
        """
        if self.training_complete:
            print('      coefficients: '        , self.model_coefficients.numpy() )
            print('      convergence : '        , self.is_converged      .numpy() )
            print('      number of iterations: ', self.num_iter          .numpy() )
        return