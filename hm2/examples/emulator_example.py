import code
import logging
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hm2.emulator import GPR_Emulator
from hm2.basis import IdentityBasis

import sys

# logging.getLogger("HistoryMatching").setLevel(logging.DEBUG)

#Training data is 100 points in [0,1] inclusive regularly spaced
train_x = np.linspace(0, 1, 100)

# True function is sin(2*pi*x) with Gaussian noise
train_y = np.sin(train_x * (2 * np.pi)) + np.random.normal(size=len(train_x)) * np.sqrt(0.04)

train_x = pd.DataFrame({"x":train_x})
train_y = pd.DataFrame({"y":train_y})

gpr = GPR_Emulator(basis=IdentityBasis(intercept=True))
gpr.fit(train_x, train_y, maxiter=20)

plt.plot(train_x,train_y)
predicted = gpr.predict(train_x)
plt.errorbar(train_x, predicted[0], yerr=predicted[1], fmt='.k')
plt.show()
