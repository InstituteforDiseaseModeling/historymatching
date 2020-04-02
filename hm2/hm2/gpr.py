#!/usr/bin/env python3
import logging

from sklearn.preprocessing import PolynomialFeatures
import gpytorch as gpt
import torch as T
import numpy as np
import pandas as pd

from .basis import BasisBase
from .error import HistoryMatchingError


#TODO: Add predictor abstract class

class _ExactGPModel(gpt.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(_ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpt.means.ConstantMean()
        self.covar_module = gpt.kernels.ScaleKernel(gpt.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpt.distributions.MultivariateNormal(mean_x, covar_x)



#TODO(r-barnes): Consider saving only the state dict and not the intermediate variables
class GPR:
    """Gaussian Process Regression (GPR)

    This class implementes Generalized Linear Modeling using statsmodels as the
    engine.
    """
    def __init__(
            self,
            basis
    ):
        """Initialize the GLM class.

        Args:
            basis: Feature generator inheriting from BasisBase
        """
        if not isinstance(basis,BasisBase):
            raise HistoryMatchingError("`basis` must inherit from BasisBase!")

        self.basis = basis
        self.model = None
        self.likelihood = None

    def fit(self, train_x, train_y, stdev_y, maxiter=1000):
        """Fit the GLM.

        Args:
            train_x: Training data
            train_y: Correct outputs
            stdev_y: Standard deviation of Y values (uncertainty)
            maxiter: Maximum number of training iterations

        Returns: None
        """
        print(train_y)
        if not isinstance(train_x, pd.DataFrame):
          raise TypeError("train_x passed to GPR.fit must be a DataFrame!")
        if not isinstance(train_y, pd.DataFrame):
          raise TypeError("train_y passed to GPR.fit must be a DataFrame!")
        if not isinstance(stdev_y, pd.DataFrame):
          raise TypeError("stdev_y passed to GPR.fit must be a DataFrame!")
        return self._fit_new(train_x, train_y, stdev_y, maxiter)

    @staticmethod
    def _convert_xdata(data):
        ret = T.from_numpy(data.to_numpy()[:,0])
        return ret #TODO: Remove temp var

    @staticmethod
    def _convert_ydata(data):
        ret = T.from_numpy(data.to_numpy()[:,0])
        return ret #TODO: Remove temp var

    def _fit_new(self, train_x, train_y, stdev_y, maxiter):
        """Fit the GLM.

        Args:
            train_x: Training data
            train_y: Correct outputs
            stdev_y: Standard deviation of Y values (uncertainty)
            maxiter: Maximum number of training iterations

        Returns: None
        """
        logger = logging.getLogger("HistoryMatching")

        # Initialize likelihood and model
        train_x = self._convert_xdata(self.basis.fit_transform(train_x))
        train_y = self._convert_ydata(train_y)

        if (stdev_y['stdev']==0).all():
            self.likelihood = gpt.likelihoods.GaussianLikelihood()
        else:
            stdev = self._convert_ydata(stdev_y)
            self.likelihood = gpt.likelihoods.FixedNoiseGaussianLikelihood(noise=stdev**2, learn_additional_noise=True)

        self.model = _ExactGPModel(train_x, train_y, self.likelihood)

        self.likelihood.train()
        self.model.train()

        # Use the adam optimizer. `parameters` includes GaussianLikelihood
        # parameters
        optimizer = T.optim.Adam([{'params': self.model.parameters()}], lr=0.1) 

        # "Loss" for GPs - the marginal log likelihood
        mll = gpt.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(maxiter):
            optimizer.zero_grad()         # Zero gradients from previous iteration
            output = self.model(train_x)  # Output from model
            loss = -mll(output, train_y)  # Calc loss and backprop gradients
            loss.backward()
            #TODO: Use logger
            print('Iter {i}/{maxiter} - Loss: {loss:.3f} lengthscale: {lenscale:.3f} noise: {noise:.3f}'.format(
                i        = i,
                maxiter  = maxiter,
                loss     = loss.item(),
                lenscale = self.model.covar_module.base_kernel.lengthscale.item(),
                noise    = self.model.likelihood.noise.item()
            ))
            optimizer.step()

    def predict(self, test_x):
        """Evaluate the GLM and return the mean prediction.

        Args:
            test_x: (Pandas DataFrame)
                Data frame of points similar to training_data.

        Returns:
            Predicted outputs at the inputs specified by data.
        """
        if not isinstance(test_x, pd.DataFrame):
          raise TypeError("data passed to GPR.predict must be a DataFrame!")

        test_x = self._convert_xdata(self.basis.fit_transform(test_x))

        # Put likelihood and model into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()

        with T.no_grad(), gpt.settings.fast_pred_var():
            y_pred        = self.model(test_x)
            observed_pred = self.likelihood(y_pred)
            mean          = observed_pred.mean
            #TODO: What is the difference between these?
            stdy          = T.sqrt(y_pred.variance)
            stdf          = T.sqrt(observed_pred.variance)

        return mean.numpy(), stdy.numpy()

    def residuals(self, data, endog):
        return endog-self.predict(data)

#TODO: Retrain with https://gpt.readthedocs.io/en/latest/models.html#gpt.models.ExactGP.get_fantasy_model
