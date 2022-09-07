#! /usr/bin/env python3

#%% 1

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyDOE import lhs

from wand.image import Image as WImage
from IPython.display import display

from history_matching import HistoryMatching, HistoryMatchingCut, quick_read, Basis
from history_matching.examples.sir import SIR

#%% 2

# WORK_DIR = Path.cwd().absolute()                # notebook version
WORK_DIR = Path(__file__).parent.absolute()     # script version
iteration = int(re.search(r'iter(\d+)', WORK_DIR.parts[-1]).group(1)) # Index of the current iteration

print(f"__file__ is '{__file__}'")
print(f"WORK_DIR is '{WORK_DIR}'")
print(f"Current iteration = {iteration}")

#%% 3

# The implausibility threshold determines how willing we are to retain regions
# of parameter space that are inconsistent with the underlying data. A higher
# threshold is more risk averse in that potentially good regions are less likely
# to be rejected, however it will take more iterations/simulations to achieve results.
implausibility_threshold = 3

n_samples_this_iter = 100 # Number of simulations to conduct on this iteration
n_samples_to_generate_for_next_iter = 100 # Number of simulations to conduct on this iteration

training_fraction = 0.75 # Fraction of simulations to use as training
discrepancy_std = 3 # Accounts for uncertainty w.r.t model structure

#%% 4

# Observed data
observations = pd.DataFrame({
    'Times': [3, 15],
    'Prevalence': [15, 40],
    'Stdev': [4, 2.3],
})
display(observations)

# For this first iteration, we're going to make one "cut" using the first observation, but 
# you can separately do multiple "cuts" per iteration using several obervations (separately)
# We'll need a name for this cut and the desired results

desired_result_idx = 0 # Pick observation 0 or 1 [integers]

cut_name = 'Prevalence_Meas_%d'%desired_result_idx # No spaces or strange characters!

desired_result = observations.iloc[desired_result_idx]['Prevalence']
desired_result_var = observations.iloc[desired_result_idx]['Stdev']**2

#%% 5

# Here we define the parameter names and ranges
param_info = pd.DataFrame({
    'Name':['Beta', 'Gamma'],
    'Min':[1e-6, 1e-6],
    'Max':[0.01, 0.5]
}).set_index('Name')
params = param_info.index.values
n_params = param_info.shape[0] # We'll use this one place later
display(param_info)

#%% 6

# Just for fun, let's visualize some SIR trajectories and observations / target data
z = SIR(beta=0.003, gamma=0.1)
f, ax = z.plot()
for i,obs in observations.iterrows():
    ax.plot(obs['Times'], obs['Prevalence'], 'ko')
    ax.plot(
        [obs['Times'],obs['Times']], 
        [obs['Prevalence']-2*obs['Stdev'],obs['Prevalence']+2*obs['Stdev']],
        'k-')

#%% 7

# For the first iteration, the samples are random.  We'll use Latin Hypercube Sampling
# to make the samples more uniformly random.
# samples should be a pandas data frame, and must have an index named 'Sample_Id'
samples = pd.DataFrame(lhs(n_params, n_samples_this_iter), columns=param_info.index.values)
samples.index.name = 'Sample_Id'

# The LHS samples are on range 0-1, let's stretch to the real parameter ranges
for param_name, values in samples.iteritems():
    samples[param_name] = \
        param_info.loc[param_name,'Min'] + \
        values*(param_info.loc[param_name,'Max']-param_info.loc[param_name,'Min'])

# Plot the samples
f, ax = plt.subplots(figsize=(6,6));
ax.scatter(x=samples[param_info.index.values[0]], y=samples[param_info.index.values[1]]);
ax.set_xlim([param_info['Min'][0], param_info['Max'][0]]);
ax.set_ylim([param_info['Min'][1], param_info['Max'][1]]);
ax.set_xlabel(param_info.index.values[0])
ax.set_ylabel(param_info.index.values[1])

#%% 8

# Run the simulations specified by the samples and plot the results

f = plt.figure(figsize=(16,10))
sim_results = []
for idx, sample in samples.iterrows():
    z = SIR(beta=sample['Beta'], gamma=sample['Gamma'])
    T,_,P = z.sim() # Run the simulation
    prevalence = [p[1] for p in P] # Analyze the simulation to get the prevalence
    
    # Because we used SSA, the time vector T does not contain the prevalence at the 
    # exact observation time, t_obs.  Let's find the first measurement after t_obs
    # Here I'm getting both observations, but we really only need one
    for i, t_obs in enumerate(observations['Times']):
        value = next((p[1] for t,p in zip(T,P) if t>t_obs), None)
        if not value:
            value = P[-1][1]
        sim_results.append([idx, i, t_obs, value])
    
    # Plot
    plt.plot(T,prevalence)

# Convert sim_results into a pandas DataFrame
sim_results = pd.DataFrame(sim_results, columns=['Sample_Id', 'ObsIdx', 'ObsTime', 'Prevalence'])

# Simulation results will ultimately need a 'Sample_Id' and 'Sim_Id'
# The Sample_Id corresponds to the index in the samples dataframe above
# You can do more than one replicate of each sample, in which case you'd
# end up wiht more than one Sim_Id per Sample_Id.  We're not doing that here,
# so we can basically set Sim_Id to anything.
sim_results['Sim_Id'] = sim_results['Sample_Id']
    
# Plot the observations over the realized trajectories
for i,obs in observations.iterrows():
    plt.plot(obs['Times'], obs['Prevalence'], 'ko')
    plt.plot(
        [obs['Times'],obs['Times']], 
        [obs['Prevalence']-2*obs['Stdev'],obs['Prevalence']+2*obs['Stdev']],
        'k-')

#%% 9

# sim_results contains the simulated values at both observation times, but for this iteration
# we only want to use the first one (ObsIdx == 0).
# Also note that results must be a Series with index containing 'Sample_Id' and 'Sim_Id'
results = sim_results \
    .query('ObsIdx==%d'%desired_result_idx)[['Sample_Id', 'Sim_Id', 'Prevalence']] \
    .set_index(['Sample_Id', 'Sim_Id'])['Prevalence']
display(results.tail())

#%% 10

# Finally we get to do some History Matching!

# Begin by creating an instance of the HistoryMatching class
hm = HistoryMatching(
    cut_name = cut_name,
    param_info = param_info,
    inputs = samples,
    results = results,
    desired_result = desired_result,
    desired_result_var = desired_result_var,
    iteration = iteration,
    implausibility_threshold = implausibility_threshold,
    discrepancy_var = discrepancy_std**2,
    training_fraction = training_fraction,
    iterdir = WORK_DIR
)
hm.save() # Save to disk

#%% 11

# Now we begin the process of emulating the simulation output
# This process contains two steps.  The first step is to fit a deterministic model, here
# we use a generalized linear model (GLM).  The glm will attempt to model the output (prevalence 
# at the first observation) as a function of some inputs.  Those inputs need not be the model
# parameters directly!  The inputs could be anything from a constant intercept up to third or higher
# order interaction terms between parameters.  The following Basis instance builds out the GLM input
# parameters from the overall simulation input parameters.
#
# Some strategy is required when choosing these.  If you know which parameters matter, there's a way
# to directly specify those parameters.  Alternatively, if you have no idea, you can initially include 
# an intercept, first, second, and maybe also third order interaction terms.  The Basis class has a 
# built-in penalized regression that throws away unneeded terms (basis vectors).
# The second step of emulation, as demonstrated here, fits a GPR to the redisual error between the 
# simulated outputs and the GLM estimates.  If the GLM fits really well, the residual is mostly noise and
# the GPR has a hard time fitting / isn't very informative.  I actually prefer to weaken the GLM enough
# to leave plenty of residual signal for the GPR.  Here, I use only first-order (beta and gamma) terms.
basis_glm = Basis.polynomial_basis(
    params = param_info.index.values,
    intercept = True,
    first_order = True,
    second_order = False,
    third_order = False,
    param_info = param_info)

#%% 12

# Now fit the glm and plot

### GLM ###############################################################
print("="*80, "\nGeneralized Linear Modeling\n", "="*80)
#######################################################################
f = hm.glm(
    basis = basis_glm,
    family = 'Gaussian',
    force_optimize_glm = True,
    glm_fit_maxiter = 100000,
    plot = True, #force_optimize_glm,
    plot_data = True
)

#%% 13

for file in (Path(hm.glmdir) / "PairwiseResults").glob("*.pdf"):
    img = WImage(filename=file)
    print(file)
    display(img)

#%% 14

# Results get saved to disk, so load and display:
filename = Path(hm.glmdir) / "GLM Predicted vs Actual.pdf"
img = WImage(filename=filename)
print(filename)
display(img)

#%% 15

basis_gpr = Basis.polynomial_basis(
    params=param_info.index.values, 
    intercept = False, 
    first_order=True, 
    param_info=param_info)

#%% 16

### GPR ###############################################################
print("="*80, "\nGaussian Process Regression\n", "="*80)
#######################################################################
hm.gpr(
    basis = basis_gpr,
    force_optimize_gpr = True,

    sigma2_f_guess = 0.6,
    sigma2_f_bounds = (0.1, 1000),
    sigma2_n_guess =  2.0,
    sigma2_n_bounds = (0.01, 100),

    #lengthscale_guess = [0.09844299, 0.1256657, 0.0976875, 0.09889085, 0.1051974, 0.0950809, 0.10032171, 0.10599185, 0.10627393, 0.09950996, 0.09445544, 0.10285915, 0.10007409, 0.09847433, 0.08963389, 0.10205652, 0.09360044, 0.1024141, 0.09786228, 0.10247492, 0.09852253, 0.09632744, 0.09997534, 0.10767302, 0.10095249, 0.09941825, 0.10214923, 0.10221497, 0.09734157, 0.09093285, 0.10780673, 0.09881377, 0.10597152],
    lengthscale_guess = 0.25,
    lengthscale_bounds = (0.01, 100),

    optimize_sigma2_n = True,
    log_transform = False,

    verbose = True,
    optimizer_options = {
        'eps': 5e-3,
        'disp': True,
        'maxiter': 15000,
        'ftol': 2 * np.finfo(float).eps,
        'gtol': 2 * np.finfo(float).eps,
    },
    plot = True, #force_optimize_gpr,
    plot_data = True
)

#%% 17

# Results get saved to disk, so load and display:
for file in (Path(hm.gprdir) / "PairwiseResults").glob("*.pdf"):
    img = WImage(filename=file)
    print(file)
    display(img)

#%% 18

# Results get saved to disk, so load and display:
filename = Path(hm.gprdir) / "gpr.pdf"
img = WImage(filename=filename)
print(file)
display(img)

#%% 19

hm.plot()
# Results get saved to disk, so load and display:
filename = Path(hm.cutdir) / "emulation.pdf"
img = WImage(filename=filename)
print(filename)
display(img)

#%% 20

### Implausibility ############################################################
print("="*80, "\nImplausibility\n", "="*80)
###############################################################################
hm.calc_and_plot_implausibility(
    plot = True,
    do_plot_data = True,
    plot_data_highlight = pd.DataFrame() #hm.test_data.loc['prime.000049']
) 
    #plot_data_highlight=pd.DataFrame() # plot_data_highlight=hm.training_data.loc['prime.000049']

hm.training_data.to_excel(WORK_DIR / "Cuts" / cut_name / "train_data.xlsx")
hm.test_data.to_excel(WORK_DIR / "Cuts" / cut_name / "test_data.xlsx")

print('Good')

#%% 21

# Results get saved to disk, so load and display:
# for file in glob.glob(os.path.join(hm.combineddir, "PairwiseResults", "*", "*.pdf")):
for file in (Path(hm.combineddir) / "PairwiseResults").glob("*/*.pdf"):
    img = WImage(filename=file)
    print(file)
    display(img)

#%% 22

# Results get saved to disk, so load and display:
# from wand.image import Image as WImage
# import glob, os
# from IPython.display import display
for file in Path(hm.combineddir).glob("*.pdf"):
    img = WImage(filename=file)
    print(file)
    display(img)

#%% 23

### Cut #######################################################################
print("="*80, "\nCut\n", "="*80)
###############################################################################
# History Matching!
_cut_folder = "Cuts"
_iteration = int(re.search(r"iter(\d+)", WORK_DIR.parts[-1]).group(1))
_iterdir_parent = WORK_DIR.parent
print(_cut_folder)
print(_iteration)
print(_iterdir_parent)
hmc = HistoryMatchingCut(
    cut_folder = 'Cuts',
    iteration = int(re.search(r'iter(\d+)', WORK_DIR.parts[-1]).group(1)),
    iterdir_parent = WORK_DIR.parent
)

(_, rejected_percent) = hmc.cut(num_desired_candidates=n_samples_to_generate_for_next_iter, constraint = None)

#%% 24

# Samples for the next iteration are saved to file, in this case it's "Candidates_for_iter1.csv"
# Just to see what they look like, we'll read that file and plot the samples
# Notice how the entire bottom right of the paramter space is empty?  Those points were rejected!

samples = pd.read_csv('Candidates_for_iter%d.csv'%(iteration+1))

f, ax = plt.subplots(figsize=(6,6))
ax.scatter(x=samples[param_info.index.values[0]], y=samples[param_info.index.values[1]])
ax.set_xlim([param_info['Min'][0], param_info['Max'][0]])
ax.set_ylim([param_info['Min'][1], param_info['Max'][1]])
ax.set_xlabel(param_info.index.values[0])
ax.set_ylabel(param_info.index.values[1])
