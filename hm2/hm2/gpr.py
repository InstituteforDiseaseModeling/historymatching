#!/usr/bin/env python3
from sklearn.preprocessing import PolynomialFeatures
import gpytorch
import numpy as np
import pandas as pd

from .basis import BasisBase
from .error import HistoryMatchingError



class _ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) #TODO
        self.covar_module = gpytorch.kernels.RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



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
        self.model = None

    def fit(self, data, endog, maxiter=1000):
        """Fit the GLM.

        Args:
            maxiter: (int)
                maxiter parameter passed to the statsmodels `fit` function.
        """
        if not isinstance(data, pd.DataFrame):
          raise TypeError("data passed to GPR.fit must be a DataFrame!")

        logger = logging.getLogger("HistoryMatching")

        data = self.basis.fit_transform(data)

        # Initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = _ExactGPModel(data, endog, likelihood)

        # Put likelihood and model into training mode
        likelihood.train()
        self.model.train()

        # Use the adam optimizer. `parameters` includes GaussianLikelihood
        # parameters
        optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=0.1) 

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self.model)

        for i in range(training_iter):
            optimizer.zero_grad()        # Zero gradients from previous iteration
            output = self.model(data)    # Output from model
            loss = -mll(output, train_y) # Calc loss and backprop gradients
            loss.backward()
            logger.info(f'Iter {i}/{maxiter} - Loss: {loss:.3f} lengthscale: {lenscale:.3f} noise: {noise:.3f}'.format(
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

        data = self.polyfit.fit_transform(data)


        # Put likelihood and model into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        f_preds = model(data)
        y_preds = likelihood(model(data))

        #TODO
        f_mean = f_preds.mean
        f_var = f_preds.variance
        f_covar = f_preds.covariance_matrix
        f_samples = f_preds.sample(sample_shape=torch.Size(1000,))

        #TODO
        # # Test points are regularly spaced along [0,1]
        # # Make predictions by feeding model through likelihood
        # with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #     test_x = torch.linspace(0, 1, 51)
        #     observed_pred = likelihood(model(test_x))

    def residuals(self, data, endog):
        return endog-self.predict(data)



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
