import code
import logging
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hm2.emulator import *
from hm2.basis import IdentityBasis

import sys

# logging.getLogger("HistoryMatching").setLevel(logging.DEBUG)

#Training data is 100 points in [0,1] inclusive regularly spaced
train_x = np.linspace(0, 1, 100)

# True function is sin(2*pi*x)
true_y = np.sin(train_x * (2 * np.pi))
# Corrupt the true function with Gaussian noise
train_y = true_y + np.random.normal(size=len(train_x)) * np.sqrt(0.04)

# Emulator requires a DataFrame 
train_x = pd.DataFrame({"x":train_x})
train_y = pd.DataFrame({"y":train_y})

# Build the emulator
gpr = GLM_GPR_Emulator(glm_basis=IdentityBasis(intercept=False), gpr_basis=IdentityBasis(intercept=False))
# Fit the emulator for 20 iterations
gpr.fit(train_x, train_y, maxiter=50)

# Get predictions from the emulator
predict_x = pd.DataFrame({"x":np.linspace(0,2,23)})
mean, std = gpr.predict(predict_x)

#Plot the function
plt.plot(train_x, true_y)
plt.plot(train_x,train_y,'k*')
plt.plot(predict_x, mean, 'b')
plt.fill_between(predict_x['x'], mean-1.96*std, mean+1.96*std, alpha=0.5)
plt.legend(['True function', 'Observed Data', 'Emulated Mean', 'Emulated Confidence'])
plt.show()
