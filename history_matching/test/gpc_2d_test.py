import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from history_matching.gpc import GPC

# WARNING: FIXING RANDOM SEED!
np.random.seed(10)

f = lambda x,y: 1/4 * (np.tanh(10*x-5)+1) * (np.tanh(3*y-0.9)+1)
target = 0.7
implausibility_threshold = 3

N = 100

pts = np.random.rand(N,2)
data = pd.DataFrame({
    'x': pts[:,0],
    'y': pts[:,1],
    })
data['f'] = data.apply( lambda d: f(d['x'], d['y']), axis=1)
data['z'] = 2 * (np.random.rand(N) < data['f']) - 1
data.index.name = 'Sample'

param_info = pd.DataFrame({
    'Name': ['x', 'y'],
    'Min': [0, 0],
    'Max': [1, 1],
}).set_index('Name')

g = GPC(['x', 'y'], 'z', data, param_info,
            kernel_mode = 'RBF',
            kernel_params = [20.57666683, 0.20004966, 1.96484556], # Sigma_f^2 and lengthscale_x^2 lengthscale_y^2
            verbose = False,
            debug = False
        )

### Test find posterior mode against eq 3.17
theta = [2, 0.1, 0.1]
ret = g.find_posterior_mode(theta, f_guess=None, tol_grad=1e-6, maxiter=10000)
assert( np.allclose(ret['f_hat'], np.dot(ret['K'], ret['d_df_log_p_y_given_f']), atol=1e-5) )


##########################################


s2_range = (3, 100)
lx2_range = (0.1, 1)
ly2_range = (0.1, 5)

optim = None
if True:
    print('BEGIN: optimize_hyperparameters')
    optim = g.optimize_hyperparameters(
        x0 = [30, 0.2, 1],
        bounds = (s2_range, lx2_range, ly2_range),
        eps=1e-3,
        disp=True,
        maxiter=15000
    )
    print('DONE optimize_hyperparameters')

# Prediction grid
Px = Py = 25
px = np.linspace(0,1,Px)
py = np.linspace(0,1,Py)
Px,Py = np.meshgrid(px, py)
p = pd.DataFrame({'x':Px.flatten(), 'y':Py.flatten()})

prediction = g.evaluate(p)


# Proposal
P = 100
pts = np.random.rand(P,2)
proposal = pd.DataFrame({
    'x': pts[:,0],
    'y': pts[:,1],
    })
#proposal['f'] = proposal.apply( lambda d: f(d['x'], d['y']), axis=1)
'''
ret = g.evaluate(proposal).set_index('Sample') # Should merge
proposal['Mean'] = ret['Mean']
proposal['Var'] = ret['Var']
proposal['Logit-Mean'] = ret['Logit-Mean']
proposal['Logit-Var'] = ret['Logit-Var']
'''

# ret = g.evaluate(proposal).set_index('Sample')
ret = g.evaluate(proposal)
ret.set_index('Sample')
proposal = proposal.merge(ret, left_index=True, right_index=True)

logit_target = np.log(target / (1-target))
proposal['Implausibility'] = (proposal['Logit-Mean'] - logit_target)**2 / proposal['Logit-Var']

# PLOT
############### 2D

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
fig = plt.figure(figsize=(16,12), dpi=300)
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax2 = fig.add_subplot(1,2,2, projection='3d')


# Plot the surface.
Xf, Yf = np.meshgrid(np.linspace(0,1,25), np.linspace(0,1,25))
truth = pd.DataFrame({'x': Xf.flatten(), 'y': Yf.flatten()})
truth['f'] = truth.apply( lambda d: f(d['x'], d['y']), axis=1)
Ff = truth['f'].values.reshape(Xf.shape)

surf1 = ax1.plot_surface(Xf, Yf, Ff, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)
ax1.scatter(data['x'], data['y'], 0.5*(data['z']+1), c='k', marker='*')#, 25, marker='*', color='k')
ax1.scatter(p['x'], p['y'], prediction['Mean'], 'b.')
ax1.scatter(p['x'], p['y'], prediction['Mean'] + 2*np.sqrt(prediction['Var']), 'bo')
ax1.scatter(p['x'], p['y'], prediction['Mean'] - 2*np.sqrt(prediction['Var']), 'bo')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('GPC Metamodel')

proposal['Keep'] = proposal['Implausibility'] < implausibility_threshold
proposal['Color'] = 'r'
proposal.loc[proposal['Keep'], 'Color'] = 'b'

surf2 = ax2.scatter(proposal['x'], proposal['y'], proposal['Implausibility'], c=proposal['Color'], s=25, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=1)
ax2.plot_surface(Xf, Yf, implausibility_threshold * np.ones_like(Xf), color='k', linewidth=0, antialiased=False, alpha=0.5)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Implausibility')

plt.show()

