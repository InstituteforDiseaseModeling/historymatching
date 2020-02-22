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
            #kernel_params = [0.001, 0.04],
            kernel_params = [4, 1.4],
            verbose = False,
            debug = False
        )

kxx = g.kxx_gpu_wrapper(x, np.array([5, 0.7]), deriv=-1)
dkxx_dtheta0 = g.kxx_gpu_wrapper(x, np.array([5, 0.7]), deriv=0)
dkxx_dtheta1 = g.kxx_gpu_wrapper(x, np.array([5, 0.7]), deriv=1+1)

g = GPC(['x'], 'y', data, param_info,
            kernel_mode = 'RBF',
            #kernel_params = [0.001, 0.04],
            kernel_params = [5, 1.4],
            verbose = False,
            debug = False
        )

fig, ax = plt.subplots(1,1)
ax.plot(x, kxx[N/2,:], 'k', lw=2)
ax.plot(x, dkxx_dtheta0[N/2,:], 'r')
ax.plot(x, dkxx_dtheta1[N/2,:], 'b')
plt.show()

