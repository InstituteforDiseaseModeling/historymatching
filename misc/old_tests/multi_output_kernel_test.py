import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from history_matching.gpr import GPR
from history_matching.gpr_mo import GPR_MO
from history_matching.basis import Basis

from pyDOE import lhs

from sir import SIR
OPTIMIZE =[True, True]
N_SAMPLES = 200
OBSERVATION_TIMES = [60, 80] # [15, 20] or [20, 40]
N_TEST = 5

# Here we define the parameter names and ranges
param_info = pd.DataFrame({
    'Name':['Beta', 'Gamma'],
    'Min':[0.0001, -2],
    'Max':[0.00025, -0.5]
}).set_index('Name')
params = param_info.index.values
n_params = param_info.shape[0] # We'll use this one place later
print(param_info)

samples = pd.DataFrame(lhs(n_params, N_SAMPLES), columns=param_info.index.values)
samples.index.name = 'Sample_Id'

# The LHS samples are on range 0-1, let's stretch to the real parameter ranges
for param_name, values in samples.iteritems():
    samples[param_name] = \
        param_info.loc[param_name,'Min'] + \
        values*(param_info.loc[param_name,'Max']-param_info.loc[param_name,'Min'])

f, ax = plt.subplots(1,3,figsize=(16,10))

def sim(samples, ax, reps=1):
    sim_results = []
    for idx, sample in samples.iterrows():
        for rep in range(reps):
            z = SIR(beta=sample['Beta'], gamma=np.power(10, sample['Gamma']), x0=[1990,10,0], )
            T,_,P = z.sim()
            prevalence = [p[1] for p in P] # Analyze the simulation to get the prevalence
            for i, t_obs in enumerate(OBSERVATION_TIMES):
                value = next((p[1] for t,p in zip(T,P) if t>t_obs), None)
                if not value:
                    value = P[-1][1]
                sim_results.append([idx, rep, i, t_obs, value])

            # Plot
            ax.plot(T,prevalence)
    return sim_results

sim_results = sim(samples, ax[0])

# Convert sim_results into a pandas DataFrame
sim_results = pd.DataFrame(sim_results, columns=['Sample_Id', 'Rep', 'ObsIdx', 'ObsTime', 'Prevalence'])
data = sim_results.set_index(['Sample_Id', 'Rep', 'ObsIdx'])['Prevalence'].unstack('ObsIdx')
data.rename(columns={0:'Y1', 1:'Y2'}, inplace=True)
ax[1].scatter(data['Y1'], data['Y2'])

data = pd.merge(samples, data, left_index=True, right_index=True)

#np.random.seed(0) # WARNING: FIXING RANDOM SEED!

data.index.name = 'Sample_Id'

b = Basis.identity_basis(['Beta', 'Gamma'], param_info)

g1 = GPR(b, 'Y1', data, param_info,
            kernel_mode = 'RBF',
            theta = np.array([39.80141277,  0.14869185,  1.71068961,  2.        ]), #np.array([4, 1, 1.4, 1.4]),
            verbose = True,
            debug = False
        )
sigma2_f_bounds = (0.005, 30)
sigma2_n_bounds = (0.005, 30)
lengthscale_bounds = (0.01, 2)
if OPTIMIZE[0]:
    g1.optimize_hyperparameters(
        x0 = [2, 1, 1.4, 1.4],
        bounds = (sigma2_f_bounds,) + (sigma2_n_bounds,) + b.D*(lengthscale_bounds,)
    )


g2 = GPR(b, 'Y2', data, param_info,
            kernel_mode = 'RBF',
            theta = g1.theta, #np.array([1, 1, 1.4, 1.4]),
            verbose = True,
            debug = False
        )
if OPTIMIZE[1]:
    g2.optimize_hyperparameters(
        x0 = g1.theta,
        bounds = (sigma2_f_bounds,) + (sigma2_n_bounds,) + b.D*(lengthscale_bounds,)
    )


g = GPR_MO(b, ['Y1', 'Y2'], data, param_info,
            kernel_mode = 'RBF',
            #theta = 0.5*(g1.theta + g2.theta),
            verbose = True,
            debug = False
        )

g.optimize_hyperparameters(
    x0 = [15, 15, 0.2, 0.2, 1, 0],
    bounds = (sigma2_f_bounds,) + (sigma2_n_bounds,) + b.D*(lengthscale_bounds,) + ((0.01,1),) + ((-0.95,0.95),)
)

def confidence_ellipse(mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[:,0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[:,1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


for trial in range(N_TEST):
    test = pd.DataFrame(lhs(n_params, 1), columns=param_info.index.values)
    test.index.name = 'Sample_Id'

# The LHS samples are on range 0-1, let's stretch to the real parameter ranges
    for param_name, values in test.iteritems():
        test[param_name] = \
            param_info.loc[param_name,'Min'] + \
            values*(param_info.loc[param_name,'Max']-param_info.loc[param_name,'Min'])

    fig_test, ax_test = plt.subplots(1, 3, figsize=(16,10))
    sim_results_test = sim(test, ax_test[0], reps=100)
    sim_results_test = pd.DataFrame(sim_results_test, columns=['Sample_Id', 'Rep', 'ObsIdx', 'ObsTime', 'Prevalence'])

    data_test = sim_results_test.set_index(['Sample_Id', 'Rep', 'ObsIdx'])['Prevalence'].unstack('ObsIdx')
    data_test.rename(columns={0:'Y1', 1:'Y2'}, inplace=True)
    ax_test[1].scatter(data_test['Y1'], data_test['Y2'])
    data_test = pd.merge(test, data_test, left_index=True, right_index=True)


    ret1 = g1.evaluate(test)
    mean1 = ret1['Mean']

    ret2 = g2.evaluate(test)
    mean2 = ret2['Mean']

    ret = g.evaluate(test)
    print('Cov_Latent:', ret['Cov_Latent'])
    print('Var_Latent:', ret['Var_Latent'])
    mean = ret['Mean']

    ax_test[2].scatter(mean1, mean2, c='r', marker='*', label='Independent')
    ax_test[2].plot([mean1-2*np.sqrt(ret1['Var_Latent']), mean1+2*np.sqrt(ret1['Var_Latent'])], [mean2, mean2], c='r')
    ax_test[2].plot([mean1, mean1], [mean2-2*np.sqrt(ret2['Var_Latent']), mean2+2*np.sqrt(ret2['Var_Latent'])], c='r')

    ax_test[2].scatter(mean[:,0], mean[:,1], c='b', marker='x', label='Correlated')

    for cov, color in zip([ret['Cov_Latent'], ret['Cov_Predictive']], ['b', 'c']):
        '''
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        for j in range(1, 4):
            ell = Ellipse(xy=(mean[:,0], mean[:,1]),
                          width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                          angle=np.rad2deg(np.arccos(v[0, 0])),
                          color=color)
            ell.set_facecolor('none')
            ax_test[2].add_artist(ell)
        '''
        confidence_ellipse(mean, cov, ax_test[2], n_std=2, edgecolor=color)

    ax_test[2].scatter(data_test['Y1'], data_test['Y2'], c='k', marker='o', label='Simulation')
    ax_test[2].legend()

    #for idx, result in data_test.iterrows():
    #    plt.plot( [result['Y1'], mean1[idx]], [result['Y2'], mean2[idx]], 'r-', lw=0.5, alpha=0.25)
    #    plt.plot( [result['Y1'], mean[idx,0]], [result['Y2'], mean[idx,1]], 'b-', lw=0.5, alpha=0.25)

    '''
    fig, ax = plt.subplots(1,2, figsize=(16,10))
    ax[0].scatter(data['Beta'], data['Gamma'], data['Y1'], c='b', label='Y1')
    ax[0].scatter(data_test['Beta'], data_test['Gamma'], data_test['Y1'], c='r', label='Y1')
    ax[0].set_xlim(param_info.loc['Beta'][['Min', 'Max']])
    ax[0].set_ylim(param_info.loc['Gamma'][['Min', 'Max']])
    ax[1].scatter(data['Beta'], data['Gamma'], data['Y2'], c='b', label='Y2')
    ax[1].scatter(data_test['Beta'], data_test['Gamma'], data_test['Y2'], c='r', label='Y1')
    ax[1].set_xlim(param_info.loc['Beta'][['Min', 'Max']])
    ax[1].set_ylim(param_info.loc['Gamma'][['Min', 'Max']])
    '''


plt.show()
