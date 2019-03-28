import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from history_matching.gpc import GPC
from scipy.stats import norm

#np.random.seed(0) # Fix random seed for reproducibility

func = lambda x,y: 1/4 * (np.tanh(10*x-5)+1) * (np.tanh(3*y-0.9)+1)
target = 0.7
logit_target = np.log(target / (1-target))
probit_target = norm.ppf(target)
implausibility_threshold = 5
training_frac = 0.75

mean_var = 'Mean-Transformed'
var_var = 'Var-Transformed'

num_points_per_iteration = 40 # Number of points per iteration
num_iterations = 10 # Number of iterations
num_past_iterations_to_include_in_metamodel = 2 # Number of previous iterations to include in metamodel

# Parameter information
param_info = pd.DataFrame({
    'Name': ['x', 'y'],
    'Min': [0, 0],
    'Max': [1, 1],
}).set_index('Name')

# Select initial samples
samples = pd.DataFrame(np.random.rand(num_points_per_iteration, 2), columns=['x', 'y'])
samples.index.name = 'Sample'
samples.reset_index(inplace=True)
samples['Train' ] = False
samples.loc[ np.random.binomial(n=1, p=training_frac, size=samples.shape[0])==1, 'Train' ] = True
samples['Iteration'] = 0
samples['Iter-Sample'] = samples['Sample']

# GPC hyperparameter guess and range bounds
theta_guess = [25, 0.2, 0.5] # S2, lx2, ly2
s2_range = (1, 100)
lx2_range = (0.05, 10)
ly2_range = (0.05, 10)

# Prediction grid - for plotting
Px = Py = 25
px = np.linspace(0,1,Px)
py = np.linspace(0,1,Py)
Px,Py = np.meshgrid(px, py)
p = pd.DataFrame({'x':Px.flatten(), 'y':Py.flatten()})
pf = p.apply( lambda d: func(d['x'], d['y']), axis=1)
Pf = pf.values.reshape(Px.shape)

g = [] # Array to hold the GPC for each iteration
for iteration in range(num_iterations):
    new_samples_this_iter = samples.loc[samples['Iteration']==iteration]

    # Simulate
    f = new_samples_this_iter.apply( lambda d: func(d['x'], d['y']), axis=1)
    new_samples_this_iter['z'] = 2 * (np.random.rand(num_points_per_iteration) < f) - 1
    samples.loc[samples['Iteration']==iteration, 'z'] = new_samples_this_iter['z']

    # Super inefficient train/test split:
    samples_to_use_this_iter = samples.loc[samples['Iteration'] >= iteration - num_past_iterations_to_include_in_metamodel]# \
    #samples_to_use_this_iter.reset_index(drop=True, inplace=True)
    #samples_to_use_this_iter.index.name = 'Sample'
    #print(samples_to_use_this_iter)
    #exit()

#        .reset_index(drop=True)
    #samples_to_use_this_iter.index.name = 'Sample'
    #samples_to_use_this_iter.reset_index(inplace=True)
    #samples_to_use_this_iter.set_index('Train', inplace=True)
    #train = samples_to_use_this_iter.loc[True].reset_index().set_index('Sample')
    #test = samples_to_use_this_iter.loc[False].reset_index().set_index('Sample')
    train = samples_to_use_this_iter.loc[samples_to_use_this_iter['Train'] == True] #samples_to_use_this_iter.where('Train' == True)
    test = samples_to_use_this_iter.loc[samples_to_use_this_iter['Train'] == False] #samples_to_use_this_iter.where('Train' == False)

    print('TEST 1:')
    print(test)


    # 1. FIT GPC
    g.append( GPC(['x', 'y'], 'z', train, param_info,
                kernel_mode = 'RBF',
                kernel_params = theta_guess, # Sigma_f^2 and lengthscale_x^2 lengthscale_y^2
                verbose = False,
                debug = False
            )
        )

    theta_guess = g[iteration].theta

    # Optimize hyperparameters
    optim = g[iteration].optimize_hyperparameters(
        x0 = [20, 0.2, 2],
        bounds = (s2_range, lx2_range, ly2_range),
        eps = 1e-3,
        disp = True,
        maxiter = 15000
    )

    # Refocusing
    # TODO Consider: Keep points from prev iter that are not implausible on this iter?
    next_samples = pd.DataFrame(columns=['x', 'y', 'Implausible'])
    accepted = 0
    tried = 0
    for_plotting = pd.DataFrame(columns=['x', 'y'])
    while next_samples.shape[0] < num_points_per_iteration:
        n = num_points_per_iteration - next_samples.shape[0]
        if tried > 0 and accepted > 0:
            n = int(np.ceil(n / (accepted / tried)))
        proposal = pd.DataFrame(np.random.rand(n,2), columns=['x', 'y'])
        proposal['Implausible'] = False
        proposal['Max_Implausibility'] = -1 # For plotting
        for it in reversed(range(iteration+1)):
            # TODO: Only evaluate non-implausible points to save time, although will degrate plotting
            ret = g[it].evaluate(proposal).set_index('Sample')
            proposal['Implausibility_%d'%it] = (ret[mean_var] - logit_target)**2 / ret[var_var]
            proposal['Implausibile_%d'%it] = proposal['Implausibility_%d'%it] > implausibility_threshold
            proposal['Implausible'] = proposal['Implausible'] | proposal['Implausibile_%d'%it]
            proposal['Max_Implausibility'] = pd.concat([proposal['Max_Implausibility'], proposal['Implausibility_%d'%it]], axis=1).max(axis=1) # Better way?

        for_plotting = for_plotting.append(proposal[['x', 'y', 'Max_Implausibility']], ignore_index=True)
        new_samples = proposal.loc[~proposal['Implausible']]
        next_samples = next_samples.append(proposal.loc[~proposal['Implausible']])
        tried = tried + n
        accepted = accepted + new_samples.shape[0]
        print('Found %d new samples, now have %d of %d. Acceptance rate is %.0f%%'%(new_samples.shape[0], next_samples.shape[0], num_points_per_iteration, 100*accepted/tried))

    next_samples = next_samples.iloc[:num_points_per_iteration] # Trim if needed
    next_samples['Iteration'] = iteration+1
    print('NS\n', next_samples)
    exit()
    next_samples.index.name = 'Iter-Sample'
    next_samples.reset_index(inplace=True)
    next_samples['Train' ] = False
    next_samples.loc[ np.random.binomial(n=1, p=training_frac, size=next_samples.shape[0])==1, 'Train' ] = True

    # PLOTS ###########################################################################################3

    # Evaluate GPC on prediction grid for plotting
    prediction = g[iteration].evaluate(p).set_index('Sample')

    train = train \
        .merge( g[iteration].evaluate(train).set_index('Sample'), left_index=True, right_index=True) \
        .reset_index()
    train['Truth'] = train.apply( lambda d: func(d['x'], d['y']), axis=1)
    test = test \
        .merge( g[iteration].evaluate(test).set_index('Sample'), left_index=True, right_index=True) \
        .reset_index()
    test['Truth'] = test.apply( lambda d: func(d['x'], d['y']), axis=1)
    print('TEST:\n', test)
    fig = g[iteration].plot_errors(train, test, 'Mean', 'Var', truth_col='Truth')
    plt.savefig('Separatrix_it%d_Errors.png'%iteration)

    ###
    fig = plt.figure(figsize=(16,10))
    fig.suptitle('Iteration %d'%iteration, fontsize=12)
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2)

    Xf, Yf = np.meshgrid(np.linspace(0,1,25), np.linspace(0,1,25))
    truth = pd.DataFrame({'x': Xf.flatten(), 'y': Yf.flatten()})
    truth['f'] = truth.apply( lambda d: func(d['x'], d['y']), axis=1)
    Ff = truth['f'].values.reshape(Xf.shape)

    surf1 = ax1.plot_surface(Xf, Yf, Ff, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)
    ax1.scatter(samples['x'], samples['y'], 0.5*(samples['z']+1), c='k', marker='*')#, 25, marker='*', color='k')
    ax1.scatter(p['x'], p['y'], prediction['Mean'], 'b.')
    ax1.scatter(p['x'], p['y'], prediction['Mean'] + 2*np.sqrt(prediction['Var']), 'bo')
    ax1.scatter(p['x'], p['y'], prediction['Mean'] - 2*np.sqrt(prediction['Var']), 'bo')
    ax1.set_xlabel('X')
    ax1.set_xlim([param_info.loc['x','Min'], param_info.loc['x','Max']])
    ax1.set_ylabel('Y')
    ax1.set_ylim([param_info.loc['y','Min'], param_info.loc['y','Max']])
    ax1.set_zlabel('Z')
    ax1.set_zlim([-1.2, 1.2])
    ax1.set_title('GPC Metamodel')

    # Add in points from p to increase plotting resolution
    proposal = p.copy()
    proposal['Implausible'] = False
    proposal['Max_Implausibility'] = -1 # For plotting
    for it in reversed(range(iteration+1)):
        # TODO: Only evaluate non-implausible points to save time, although will degrate plotting
        ret = g[it].evaluate(proposal).set_index('Sample')
        proposal['Implausibility_%d'%it] = (ret[mean_var] - logit_target)**2 / ret[var_var]
        proposal['Implausibile_%d'%it] = proposal['Implausibility_%d'%it] > implausibility_threshold
        proposal['Implausible'] = proposal['Implausible'] | proposal['Implausibile_%d'%it]
        proposal['Max_Implausibility'] = pd.concat([proposal['Max_Implausibility'], proposal['Implausibility_%d'%it]], axis=1).max(axis=1) # Better way?

    for_plotting = for_plotting.append(proposal[['x', 'y', 'Max_Implausibility']], ignore_index=True)

    m = np.amax(for_plotting['Max_Implausibility'])
    ax2.tricontour(for_plotting['x'], for_plotting['y'], for_plotting['Max_Implausibility'], levels=[implausibility_threshold], linewidths=2, colors='k')
    levels = list(range(implausibility_threshold+1))
    if m > implausibility_threshold:
        levels = levels + [m]
    cntr2 = ax2.tricontourf(for_plotting['x'], for_plotting['y'], for_plotting['Max_Implausibility'], levels=levels, cmap="RdBu_r")

    ax2.plot(next_samples['x'], next_samples['y'], 'ro')
    success = samples['z'] == 1
    ax2.plot(samples.loc[success, 'x'], samples.loc[success, 'y'], 'wo')
    ax2.plot(samples.loc[~success, 'x'], samples.loc[~success, 'y'], 'ko')
    ax2.plot(train['x'], train['y'], 'cx')
    ax2.plot(test['x'], test['y'], 'mx')
    for idx, row in test.iterrows():
        ax2.annotate(xy=(row['x'], row['y']), s=str(row['Sample']))
    ax2.contour(Px, Py, Pf, levels = [target], colors='k', linestyles='dashed', linewidths=2)
    ax2.set_xlabel('X')
    ax2.set_xlim([param_info.loc['x','Min'], param_info.loc['x','Max']])
    ax2.set_ylim([param_info.loc['y','Min'], param_info.loc['y','Max']])
    ax2.set_ylabel('Y')
    ax2.set_title('Implausibility & Next Samples')

    plt.savefig('Separatrix_it%d.png'%iteration)

    ##### APPEND NEXT_SAMPLES TO SAMPLES FOR NEXT ITERATION ##############
    print('BEFORE:', samples)
    n = samples.iloc[-1]['Sample']
    next_samples['Sample'] = list(range(n, n+next_samples.shape[0]))
    samples = samples.append(next_samples[['Iteration', 'Sample', 'Iter-Sample', 'x','y', 'Train']])
    print('AFTER:', samples)
    print('AFTER ... 1:', samples.loc[samples['Iteration']==1])
    exit()

