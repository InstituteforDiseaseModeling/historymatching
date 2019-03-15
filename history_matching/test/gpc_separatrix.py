import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from history_matching.gpc import GPC

#np.random.seed(0) # Fix random seed for reproducibility

func = lambda x,y: 1/4 * (np.tanh(10*x-5)+1) * (np.tanh(3*y-0.9)+1)
target = 0.7
logit_target = np.log(target / (1-target))
implausibility_threshold = 4
training_frac = 0.8

num_points_per_iteration = 25 # Number of points per iteration
num_iterations = 10 # Number of iterations
num_past_iterations_to_include_in_metamodel = 3 # Number of previous iterations to include in metamodel

# Parameter information
param_info = pd.DataFrame({
    'Name': ['x', 'y'],
    'Min': [0, 0],
    'Max': [1, 1],
}).set_index('Name')

# Select initial samples
samples = pd.DataFrame(np.random.rand(num_points_per_iteration, 2), columns=['x', 'y'])
samples['Iteration'] = 0

# GPC hyperparameter guess and range bounds
theta_guess = [20, 0.2, 2] # S2, lx2, ly2
s2_range = (0.1, 100)
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
    samples_this_iter = samples.loc[samples['Iteration']==iteration]

    # Simulate
    f = samples_this_iter.apply( lambda d: func(d['x'], d['y']), axis=1)
    samples_this_iter['z'] = 2 * (np.random.rand(num_points_per_iteration) < f) - 1
    samples.loc[samples['Iteration']==iteration, 'z'] = samples_this_iter['z']

    # 1. FIT GPC
    all_samples = samples.loc[samples['Iteration'] >= iteration - num_past_iterations_to_include_in_metamodel]
    all_samples['Train'] = False
    print(all_samples.shape[0])
    print(training_frac)
    print(training_frac*np.ones((all_samples.shape[0],1)))
    print(np.random.choice(1, size=all_samples.shape[0], p=training_frac*np.ones((all_samples.shape[0],1))))
    exit()
    all_samples.loc[ np.random.choice(1, size=all_samples.shape[0], p=training_frac), 'Train' ] = True
    print(all_samples)
    exit()
    train = all_samples.sample( n=int(np.round(training_frac * all_samples.shape[0])), replace=False )
    print(train)
    exit()
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
            proposal['Implausibility_%d'%it] = (ret['Logit-Mean'] - logit_target)**2 / ret['Logit-Var']
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
    samples = samples.append(next_samples[['Iteration', 'x','y']])

    # PLOT ###########################################################################################3
    fig = plt.figure(figsize=(16,10))
    fig.suptitle('Iteration %d'%iteration, fontsize=12)
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2)

    # Evaluate GPC on prediction grid for plotting
    prediction = g[iteration].evaluate(p).set_index('Sample')

    '''
    ddd = samples.loc[samples['Iteration'] >= iteration - num_past_iterations_to_include_in_metamodel]
    ddd = ddd.merge(g[iteration].evaluate(ddd).set_index('Sample'), left_index=True, right_index=True)
    ddd.index.name='Sample'
    ddd.reset_index(inplace=True)
    print(ddd.head())
    fig = g[iteration].plot_errors(ddd, ddd, 'Mean', 'Var', 'Var')
    plt.show()
    exit()
    '''

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

    m = np.amax(for_plotting['Max_Implausibility'])
    ax2.tricontour(for_plotting['x'], for_plotting['y'], for_plotting['Max_Implausibility'], levels=[implausibility_threshold], linewidths=2, colors='k')
    cntr2 = ax2.tricontourf(for_plotting['x'], for_plotting['y'], for_plotting['Max_Implausibility'], levels=list(range(implausibility_threshold)+1)+[m], cmap="RdBu_r")

    ax2.plot(next_samples['x'], next_samples['y'], 'ro')
    ax2.plot(samples['x'], samples['y'], 'k.')
    ax2.contour(Px, Py, Pf, levels = [target], colors='k', linestyles='dashed', linewidths=2)
    ax2.set_xlabel('X')
    ax2.set_xlim([param_info.loc['x','Min'], param_info.loc['x','Max']])
    ax2.set_ylim([param_info.loc['y','Min'], param_info.loc['y','Max']])
    ax2.set_ylabel('Y')
    ax2.set_title('Implausibility & Next Samples')

    plt.savefig('Separatrix_it%d.png'%iteration)

