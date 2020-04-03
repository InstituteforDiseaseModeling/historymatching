#!/usr/bin/env python3
import logging

import gpytorch as gpt
import torch as T
import numpy as np



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
class TorchGPR:
    """Gaussian Process Regression (GPR) using GPyTorch"""
    def __init__(self):
        self.model = None
        self.likelihood = None

    def fit(self, train_x, train_y, stdev_y, maxiter=1000):
        """Fit the GPR.

        Args:
            train_x: Training data
            train_y: Correct outputs
            stdev_y: Standard deviation of Y values (uncertainty)
            maxiter: Maximum number of training iterations

        Returns: None
        """
        logger = logging.getLogger("HistoryMatching")

        train_x = T.from_numpy(train_x)
        train_y = T.from_numpy(train_y)

        # Initialize likelihood and model
        if np.all(stdev_y==0):
            self.likelihood = gpt.likelihoods.GaussianLikelihood()
        else:
            stdev_y = T.from_numpy(stdev_y)
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
        # Put likelihood and model into evaluation (predictive posterior) mode
        if self.model is None:
            raise HistoryMatchingError("TorchGPR hasn't been trained yet!")

        test_x = T.from_numpy(test_x)

        self.model.eval()
        self.likelihood.eval()

        print(test_x)

        with T.no_grad(), gpt.settings.fast_pred_var():
            y_pred        = self.model(test_x)
            observed_pred = self.likelihood(y_pred)
            mean          = observed_pred.mean
            #TODO: What is the difference between these?
            stdy          = T.sqrt(y_pred.variance)
            stdf          = T.sqrt(observed_pred.variance)

        return mean.numpy(), stdy.numpy()

#TODO: Retrain with https://gpt.readthedocs.io/en/latest/models.html#gpt.models.ExactGP.get_fantasy_model
