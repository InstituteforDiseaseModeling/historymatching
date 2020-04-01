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



#TODO(r-barnes): Consider saving only the state dict and not the interediate variables
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

    def predict(self, data):
        """Evaluate the GLM and return the mean prediction.

        Args:
            data: (Pandas DataFrame)
                Data frame of points similar to training_data.

        Returns:
            Predicted outputs at the inputs specified by data.
        """
        if not isinstance(data, pd.DataFrame):
          raise TypeError("data passed to GPR.predict must be a DataFrame!")

        data = self._convert_data(self.basis.fit_transform(data))

        # Put likelihood and model into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()

        print("data",data)
        f_preds = self.model(data)
        y_preds = self.likelihood(self.model(data))

        #TODO
        f_mean = f_preds.mean.detach().numpy()
        f_var = f_preds.variance.detach().numpy()
        # f_covar = f_preds.covariance_matrix
        # f_samples = f_preds.sample(sample_shape=T.Size(1000,))

        return f_mean, f_var

        #TODO
        # # Test points are regularly spaced along [0,1]
        # # Make predictions by feeding model through likelihood
        # with torch.no_grad(), gpt.settings.fast_pred_var():
        #     test_x = torch.linspace(0, 1, 51)
        #     observed_pred = likelihood(model(test_x))

    def residuals(self, data, endog):
        return endog-self.predict(data)

#TODO: Retrain with https://gpt.readthedocs.io/en/latest/models.html#gpt.models.ExactGP.get_fantasy_model


#TODO
#In the next cell, we plot the mean and confidence region of the Gaussian
#process model. The confidence_region method is a helper method that returns 2
#standard deviations above and below the mean.

#TODO
# with torch.no_grad():
#     # Initialize plot
#     f, ax = plt.subplots(1, 1, figsize=(4, 3))

#     # Get upper and lower confidence bounds
#     lower, upper = observed_pred.confidence_region()
#     # Plot training data as black stars
#     ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
#     # Plot predictive means as blue line
#     ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
#     # Shade between the lower and upper confidence bounds
#     ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
#     ax.set_ylim([-3, 3])
#     ax.legend(['Observed Data', 'Mean', 'Confidence'])
