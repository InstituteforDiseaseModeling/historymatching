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
        # self.covar_module = gpt.kernels.ScaleKernel(gpt.kernels.RBFKernel()) #TODO
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
        self.likelihood = gpt.likelihoods.GaussianLikelihood()
        self.model = None

    def fit(self, data, endog, maxiter=1000):
        """Fit the GLM.

        Args:
            data: Training data
            endog: Correct outputs
            maxiter: Maximum number of training iterations

        Returns: None
        """
        if not isinstance(data, pd.DataFrame):
          raise TypeError("data passed to GPR.fit must be a DataFrame!")
        return self._fit_new(data, endog, maxiter)

    @staticmethod
    def _convert_data(data):
        return T.squeeze(T.from_numpy(data.to_numpy()))

    def _fit_new(self, train_x, train_y, maxiter):
        logger = logging.getLogger("HistoryMatching")

        # Initialize likelihood and model
        train_x    = self._convert_data(self.basis.fit_transform(train_x))
        train_y = self._convert_data(train_y)
 
        self.model = _ExactGPModel(train_x, train_y, self.likelihood)

        self.likelihood.train()
        self.model.train()

        # Use the adam optimizer. `parameters` includes GaussianLikelihood
        # parameters
        optimizer = T.optim.Adam([{'params': self.model.parameters()}], lr=0.1) 

        # "Loss" for GPs - the marginal log likelihood
        mll = gpt.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(maxiter):
            optimizer.zero_grad()        # Zero gradients from previous iteration
            output = self.model(train_x) # Output from model
            loss = -mll(output, train_y) # Calc loss and backprop gradients
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

        test_x = self._convert_data(self.basis.fit_transform(test_x))

        # Put likelihood and model into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()

        with T.no_grad(), gpt.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))
            mean = observed_pred.mean
            lower, upper = observed_pred.confidence_region()

        return mean.numpy(), lower.numpy(), upper.numpy()

    def residuals(self, data, endog):
        return endog-self.predict(data)

#TODO: Retrain with https://gpt.readthedocs.io/en/latest/models.html#gpt.models.ExactGP.get_fantasy_model
