#!/usr/bin/env python3

#Adapted from: https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html

import matplotlib.pyplot as plt
import numpy as np

from hm2.gpr import SkGPR

def f(x):
    """The function to predict."""
    return x * np.sin(x)



# ----------------------------------------------------------------------
#  First the noiseless case
real_observations = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T

# Observations
y = f(real_observations).ravel()

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
xmesh = np.atleast_2d(np.linspace(0, 10, 1000)).T

# Instantiate a Gaussian Process model
gpr = SkGPR()

# kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
# gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gpr.fit(real_observations, y, random_state=123456)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gpr.predict(xmesh)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(xmesh, f(xmesh), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.plot(real_observations, y, 'r.', markersize=10, label='Observations')
plt.plot(xmesh, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([xmesh, xmesh[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')



# ----------------------------------------------------------------------
# now the noisy case
real_obs_with_noise = np.linspace(0.1, 9.9, 20)
real_obs_with_noise = np.atleast_2d(real_obs_with_noise).T

# Observations and noise
y = f(real_obs_with_noise).ravel()
dy = 0.5 + 1.0 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise

# Instantiate and fit a Gaussian Process model
gpr = SkGPR()
gpr.fit(real_obs_with_noise, y, stdev_y=dy, random_state=123456)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gpr.predict(xmesh)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(xmesh, f(xmesh), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.errorbar(real_obs_with_noise.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')
plt.plot(xmesh, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([xmesh, xmesh[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(xmesh)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')

plt.show()