import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hm2.glm import GLM
from hm2.basis import PolynomialBasis



#Training data is 100 points in [0,1] inclusive regularly spaced
train_x = np.linspace(0, 1, 100)

# True function is sin(2*pi*x)
true_y = np.sin(train_x * (2 * np.pi))

# Corrupt the true function with Gaussian noise
train_y = true_y + np.random.normal(size=len(train_x)) * np.sqrt(0.04)

# Emulator requires a DataFrame
train_x = pd.DataFrame({"x":train_x})
train_y = pd.DataFrame({"y":train_y})

basis = PolynomialBasis(degree=4, intercept=True)
glm   = GLM(family="gaussian")
glm.fit(basis(train_x), train_y)

glm.plot_training_vs_trained(colname='x')
glm.plot_fitted_vs_observed()
glm.plot_pearson_residuals()
glm.plot_deviance_redisuals()
glm.plot_QQ()
