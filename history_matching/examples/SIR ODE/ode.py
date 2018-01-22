from scipy.integrate import ode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

import matplotlib as mpl
mpl.rcParams.update({'font.size': 22})

import seaborn as sns
sns.set_palette(sns.color_palette("hls", 25))


from history_matching.gpr import GPR
from history_matching.basis import Basis

start = 0.0
stop = 1.0
number_of_lines= 25
cm_subsection = np.linspace(start, stop, number_of_lines) 
colors = [ cm.rainbow(x) for x in cm_subsection ]

np.random.seed(1)

f1, ax1 = plt.subplots(1,1,figsize=(15,7))
f2, ax2 = plt.subplots(1,1,figsize=(15,7))

beta_vec = np.linspace(0.025, 0.3, 25)
#beta_vec = np.array([0.025, 0.05, 0.075, 0.20, 0.25, 0.30])
D = len(beta_vec)
t_star = 25
y_star = []

implausibility_thresh = 3
idx = 8 # Where meas is shown
##############################################
sigma_y = 0.111
##############################################
sigma_n = 0.025 # Meas noise
#obs = [4,5,6,-5,-4,-3] # Observed inds i1
obs = [0, 1, 8, 10, 13, 15] # Observed inds

fmt='png'
#res=1200
figdir = 'fig sigma_y=%f'%sigma_y
try:
    os.stat(figdir)
except:
    os.mkdir(figdir)


for beta_idx, beta in enumerate(beta_vec):

    y = [[0.95, 0.05]]
    t = [0]

    def f(t, y, beta):
        S = y[0]
        I = y[1]

        return [
            -beta*S*I,
            beta*S*I
        ]

    r = ode(f).set_integrator('dopri5')
    r.set_initial_value(y[0], t[0]).set_f_params(beta)

    T = 50
    dt = 0.1 # <-- Doesn't affect accuracy, just plotting
    found = False
    while r.successful() and r.t < T:
        r.integrate(r.t+dt)
        t.append(r.t)
        #np.vstack([y, r.y])
        y.append(r.y)

        if (not found) and (r.t >= t_star):
            y_star.append(r.y)
            found = True

    Y = np.vstack(y)

    #ax1.plot(t, Y[:,0], label='Susceptible')
    if beta_idx % 5 == 0:
        ax1.plot(t, Y[:,1], label='Beta=%.2f'%beta)#, color=colors[beta_idx])
    else:
        ax1.plot(t, Y[:,1], label=None)#, color=colors[beta_idx])
ax1.legend()
ax1.set_xlim([t[0], t[-1]])
ax1.set_ylim([0,1])
ax1.set_ylabel('Infected (%)')
ax1.set_xlabel('Time')

f1.savefig(os.path.join(figdir, 'Infected_beta_sweep_1.%s'%fmt))

ax1.plot([t_star, t_star], [0,1], 'k', lw=2)
Y_star = np.stack(y_star)
ax1.plot(t_star*np.ones([1, D]), np.reshape(Y_star[:,1], (1, D)), '.', ms=15)
f1.savefig(os.path.join(figdir, 'Infected_beta_sweep_2.%s'%fmt))

ax2.plot(beta_vec, Y_star[:,1], '-', marker='.', ms=15)
ax2.set_xlabel('Beta')
ax2.set_ylabel('Infected at t=%.0f (%%)'%t_star)
ax2.set_xlim(beta_vec[[0,-1]])
ax2.set_ylim([0,1])
f2.savefig(os.path.join(figdir, 'Infected_beta_sweep_3.%s'%fmt))

target = Y_star[idx,1]

ax2.plot(beta_vec[[0,-1]], Y_star[[idx,idx],[1,1]], 'k', lw=2)
ax2.plot(beta_vec[[idx,idx]], [0, target], 'k:', lw=2)

ax2.fill_between(
    [beta_vec[0], beta_vec[-1]], 
    [ Y_star[idx,1] + 2*sigma_y, Y_star[idx,1] + 2*sigma_y],
    [ Y_star[idx,1] - 2*sigma_y, Y_star[idx,1] - 2*sigma_y],
    facecolor='k', alpha=0.25, zorder=0)

ax2.plot(beta_vec[idx], [0], 'kx', ms=15)
f2.savefig(os.path.join(figdir, 'Infected_beta_sweep_4.%s'%fmt))

### GPR
params = ['Beta']
param_info = pd.DataFrame({'Name':['Beta'], 'Min':[0], 'Max':[1]}).set_index('Name')
basis = Basis.polynomial_basis(
            params,
            intercept = False,
            first_order = True,
            second_order = False,
            third_order = False,
            fourth_order = False,
            fifth_order = False,
            higher_order = False,
            param_info = param_info,
            verbose = True
    )

Y_noisy = Y_star[obs,1] + sigma_n * np.random.randn(len(obs))
Y_mean = np.mean(Y_star[obs,1])
ax2.plot(beta_vec[obs], Y_noisy, 'c+', ms=15, mew=3)
Y_noisy -= Y_mean

f2.savefig(os.path.join(figdir, 'Infected_beta_sweep_5.%s'%fmt))

training_data = pd.DataFrame({
    'Beta': beta_vec[obs],
    'Sim_Result': Y_noisy
})

training_data.index.name = 'Sample_Id'
#training_data.reset_index(inplace=True)

Ycol = 'Sim_Result'
g = GPR(basis, Ycol, training_data, param_info,
        kernel_mode = 'RBF',
        theta = None,   # kernel_params
        #is_poisson = False,
        normalize_y = True,
        verbose = True,
        debug = False
    )
sigma2_f_guess = 1
sigma2_n_guess = 1
lengthscale_guess = basis.D * [0.2]
x0 =np.array([sigma2_f_guess, sigma2_n_guess] +  lengthscale_guess)

sigma2_f = 5*np.var(Y_noisy)
sigma2_f_bounds = (0.99*sigma2_f, 1.01*sigma2_f) #(0.5, 100)
sigma2_n_bounds = ((0.99*sigma_n)**2, (1.01**sigma_n)**2) #(0.001, 10)
lengthscale_bounds = (0.0001, 0.015) # 0.0025

bounds = (sigma2_f_bounds,)+(sigma2_n_bounds,) + basis.D*(lengthscale_bounds,)
g.optimize_hyperparameters(
    x0,
    bounds,
    optimizer_options={}
)

test = pd.DataFrame({
    'Beta': np.linspace(beta_vec[0],beta_vec[-1],1000),
})

ret = g.evaluate(test)
ret['Mean'] += Y_mean

ax2.plot(test['Beta'], ret['Mean'], 'b-', lw=2)
ax2.fill_between(
    test['Beta'], 
    ret['Mean'] + 2*np.sqrt(ret['Var_Predictive']),
    ret['Mean'] - 2*np.sqrt(ret['Var_Predictive']),
    facecolor='b', color='b', edgecolor='b', alpha=0.15
)
f2.savefig(os.path.join(figdir, 'Infected_beta_sweep_6.%s'%fmt))

implausibility = np.sqrt( (ret['Mean'] - target)**2 / (ret['Var_Predictive'] + sigma_y**2) )

ax2_1 = ax2.twinx()
ax2_1.plot(test['Beta'], implausibility, 'g')
#ax2_1.set_ylim([0,1.05*max(implausibility)])
ax2_1.set_ylim([0,11])
ax2_1.set_ylabel('Implausibility')
ax2_1.spines['right'].set_color('g')
#ax.spines['top'].set_color('red')
ax2_1.yaxis.label.set_color('g')
ax2_1.tick_params(axis='y', colors='g')

cm = plt.get_cmap('seismic')

dBeta = test.loc[1,'Beta'] - test.loc[0,'Beta']
for i,b in test['Beta'].iteritems():
    ax2.fill_between([b, b+dBeta], [0.05, 0.05], facecolor=cm( 1-min(implausibility[i]/implausibility_thresh,1) ))

ax2.set_xlim(beta_vec[[0,-1]])
ax2_1.set_xlim(ax2.get_xlim())
f2.savefig(os.path.join(figdir, 'Infected_beta_sweep_7.%s'%fmt))

ax2_1.plot(test['Beta'], 3*np.ones(test['Beta'].shape[0]), 'g:')
for i,b in test['Beta'].iteritems():
    '''
    print(b, implausibility[i] >= implausibility_thresh)

    # From previous iter
    if (b >= 0.0558308308308 and b <= 0.096021021021) or (b >= 0.213288288288):
        ax2.fill_between([b, b+dBeta], [1,1], hatch = '///', facecolor='k', alpha=0.5)
    '''

    if implausibility[i] >= implausibility_thresh:
        ax2.fill_between([b, b+dBeta], [1,1], hatch = '\\\\\\', facecolor='k', alpha=0.5)
f2.savefig(os.path.join(figdir, 'Infected_beta_sweep_8.%s'%fmt))
