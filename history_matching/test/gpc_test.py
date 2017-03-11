import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from history_matching.gpc import GPC

# WARNING: FIXING RANDOM SEED!
np.random.seed(0)

N = 250
x = np.linspace(0,2*np.pi,N)
f = (np.sin(x) + 1)/2.
y = 2 * (np.random.rand(N) < f) - 1

data = pd.DataFrame({
    'x': x,
    'y': y
})
data.index.name = 'Sample'

param_info = pd.DataFrame({
    'Name': ['x'],
    'Min': [0],
    'Max': [2*np.pi]
}).set_index('Name')

g = GPC(['x'], 'y', data, param_info,
            kernel_mode = 'RBF',
            kernel_params = [100, 0.03],
            verbose = False,
            debug = False
        )

if True:
    g.optimize_hyperparameters(
        x0 = [10, 0.03],
        bounds = ((0.01,20),(0.01,0.1)),
        K=5,
        eps=1e-2,
        disp=True,
        maxiter=15000
    )

p = pd.DataFrame({'x':np.linspace(-2*np.pi, 2*np.pi + 2*np.pi, 100)})
prediction = g.evaluate(p)

# PLOT
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(x,f, 'r-')
ax1.scatter(x,(y+1)/2.,s=25)
ax1.plot(p['x'], prediction['Trapz'], 'b.:')

ax2.errorbar(x=p['x'], y=prediction['Mean'], yerr=np.sqrt(prediction['Var']))
plt.show()



