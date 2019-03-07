import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from history_matching.gpc import GPC

# WARNING: FIXING RANDOM SEED!
np.random.seed(0)

N = 25
x = np.linspace(0,2*np.pi,N)
f = (np.sin(x) + 1)/2.
#f = 1 / (1 + np.exp(-x))
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

if True:
    g.optimize_hyperparameters(
        x0 = [4, 0.1],
        bounds = ((0.01,20),(0.01,0.1)),
        eps=1e-2,
        disp=True,
        maxiter=15000
    )
    #MIN: 11.3391800355
    #S2: 3.9387755102
    #L2: 0.0559183673469

p = pd.DataFrame({'x':np.linspace(-2*np.pi, 4*np.pi, 100)})
prediction = g.evaluate(p)

# PLOT
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(x,f, 'r-')
ax1.scatter(x,(y+1)/2.,s=25)
ax1.plot(p['x'], prediction['Trapz'], 'b.:')

ax2.errorbar(x=p['x'], y=prediction['Mean'], yerr=np.sqrt(prediction['Var']))

#plt.show(); exit()

# THETA PLOT
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax2 = fig.add_subplot(1,2,2) #, projection='3d')

# Make data.
sigma2_vec = np.linspace(1, 15, 25)
l2_vec = np.linspace(0.01, 0.1, 25)
S2 = np.zeros( [len(sigma2_vec), len(l2_vec)] )
L2 = np.zeros_like(S2)
NLML = np.zeros_like(S2)
NormDF = np.zeros_like(S2)
dS2 = np.zeros_like(S2)
dL2 = np.zeros_like(S2)

f_hat = None
for i, s2 in enumerate(sigma2_vec):
    for j, l2 in enumerate(l2_vec):
        S2[i,j] = s2
        L2[i,j] = l2
        f, df, f_hat = g.negative_log_marginal_likelihood_and_gradient(np.array([s2, l2]), f_hat)
        NLML[i,j] = f
        dS2[i,j] = df[0] * (np.max(sigma2_vec)-np.min(sigma2_vec))**2 #* ((np.max(sigma2_vec) - np.min(sigma2_vec))/(np.max(l2_vec)-np.min(l2_vec)))**2
        dL2[i,j] = df[1] * (np.max(l2_vec)-np.min(l2_vec))**2
        NormDF[i,j] = np.linalg.norm(df)

amin = NLML.argmin()
i,j = np.unravel_index(amin, NLML.shape)
print('MIN:', NLML[i,j])
print('S2:', S2[i,j])
print('L2:', L2[i,j])


# Plot the surface.
surf1 = ax1.plot_surface(S2, L2, NLML, cmap=cm.coolwarm, linewidth=0, antialiased=False)
q1 = ax2.quiver(S2, L2, dS2, dL2, angles='xy', scale_units='xy', units='xy') #, scale=1, units='xy', scale_units='xy', angles='xy')
ax1.scatter(S2[i,j], L2[i,j], NLML[i,j]*1.1, c='r', s=500, marker='*')
ax2.scatter(S2[i,j], L2[i,j], c='r', marker='*', s=50)
#surf2 = ax2.plot_surface(S2, L2, NormDF, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax1.set_xlabel('sigma2_n')
ax1.set_ylabel('l^2')
ax1.set_xlim([np.min(sigma2_vec), np.max(sigma2_vec)])
ax1.set_ylim([np.min(l2_vec), np.max(l2_vec)])
ax2.set_xlabel('sigma2_n')
ax2.set_ylabel('l^2')
ax2.set_xlim([np.min(sigma2_vec), np.max(sigma2_vec)])
ax2.set_ylim([np.min(l2_vec), np.max(l2_vec)])
#plt.axis('equal')

#ax1.view_init(azim=0, elev=90)
#plt.zlabel('Negative Log Marginal Likelihood')
# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#plt.add_colorbar(surf1, shrink=0.5, aspect=5)

plt.show()

