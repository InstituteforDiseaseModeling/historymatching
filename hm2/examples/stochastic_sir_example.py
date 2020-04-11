import code
import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from hm2.examples.sir import SIR
import hm2.sampling
from hm2.boilerplate import *
import hm2.basis
from hm2.emulators import *
from hm2.wrapped_model import *



################################################
#Observational Data
################################################

time_observations = pd.DataFrame({
    'observation_id': [           0,            1],
    'time':           [         3.0,         15.0],
    'observation':    ['prevalence', 'prevalence'],
    'value':          [          15,           40],
    'stdev':          [           4,          2.3]
})

summary_observations = None



################################################
#Parameters
################################################

# Here we define the parameter names and ranges for our model
param_info = pd.DataFrame({
    'name': ['beta', 'gamma'],
    'min':  [  1e-6,    1e-6],
    'max':  [  0.01,     0.5]
})



################################################
#Start sampling
################################################

# For the first iteration, the samples are random.  We'll use Latin Hypercube
# Sampling to make the samples more uniformly spaced.
parameter_samples = hm2.sampling.latin_hypercube(param_info, samples=50)



################################################
#Start sampling
################################################

# HistoryMatching requires that the model be wrapped in the special ModelWrapper
# class. This standardize HistoryMatching's interaction with models 

def wrapped_model(**kwargs):
    model = SIR(**kwargs)

    results = model.sim()
    results['prevalence'] = results['per_infected']
    #Reshape DataFrame into the tidy form expected by HistoryMatching
    results = pd.melt(results, id_vars='time', var_name='observation')
    #We have no uncertainty about our results
    results['stdev'] = 0
    #Add observation ids
    results['observation_id'] = list(range(len(results)))
    #Sort by time
    results.sort_values(by='time', inplace=True)
    return results, None
    # return ValidateObservationFrames((results, None), copy=False)



#Run the model a number of times
sim_replicates = run_replicates(
  wrapped_model = wrapped_model,
  param_sets    = [x.to_dict() for _, x in parameter_samples.iterrows()],
  replicates    = 2,
  processes     = None,
)


################################################
#Plot a few runs
################################################

#print(plot_runs_time_series(sim_replicates))


################################################
#Match to observations
################################################

matched = match_sim_outputs_to_observations(
  sim_replicates,
  time_observations,
  summary_observations,
  processes=12
)

################################################
#Fit emulators
################################################

psamples_within = hm2.sampling.latin_hypercube_within(parameter_samples, 1000)

#Prepare data for emulator (make a train_x, train_y, stdev_y tuple)
time_emulators = dict()
for obs, params, y, stdev in get_data_for_emulators(parameter_samples, matched[0]):
  #Train emulator
  time_emulators[obs] = hm2.emulators.GLM_GPR_Emulator(
    glm_basis=hm2.basis.IdentityBasis(intercept=False),
    gpr_basis=hm2.basis.IdentityBasis(intercept=False)
  ).fit(params, y, stdev, gpr_maxiter=10000)
  # figs = gpremu.plot_data() #TODO

#Validate emulators here


implausibilities = get_implausibility(time_emulators, psamples_within, time_observations)
implausibilities = max_implausibility_per_param(implausibilities)
implausibilities = filter_implausibilities(implausibilities, threshold=0.2)


plausible_params = get_plausible_parameters(implausibilities, psamples_within)




################################
# WAVE 2

parameter_samples = hm2.sampling.latin_hypercube_within(plausible_params, 100)

sys.exit(0)



#TODO: plot_errors test_mean train_mean

# gpremu.fit(*prepped)






# Reduce replicates taking the mean of what the model produced for each time
# point
# runs = (
#   hm2.boilerplate.replicate_reducer(runs[0], "mean"),
#   hm2.boilerplate.replicate_reducer(runs[1], "mean")
# )

# a = hm2.boilerplate.extract_training_set_from_keyed_frame(
#   observation_key = 0,
#   parameter_samples = parameter_samples,
#   observations = time_observations,
#   keyed_frame = mr['time'],
#   frame_type = 'time',
# )


sys.exit(-1)

fig, ax=plt.subplots(2,1)

ax[0].scatter(
  parameter_samples.iloc[this_obs['param_id']]['beta'],
  parameter_samples.iloc[this_obs['param_id']]['gamma'],
  c=this_obs['value']
)

ax[1].scatter(
  parameter_samples.iloc[this_obs['param_id']]['beta'],
  parameter_samples.iloc[this_obs['param_id']]['gamma'],
  c=gpremu.predict(parameter_samples.iloc[this_obs['param_id']][['beta','gamma']])[0]
)

plt.show()

sys.exit(-1)




################################################
################################################
################################################
################################################
#Example ends here
################################################
################################################
################################################
################################################






# Plot the samples
# f, ax = plt.subplots(figsize=(6,6));
# ax.scatter(x=samples[param_info.index.values[0]], y=samples[param_info.index.values[1]]);
# ax.set_xlim([param_info['Min'][0], param_info['Max'][0]]);
# ax.set_ylim([param_info['Min'][1], param_info['Max'][1]]);
# ax.set_xlabel(param_info.index.values[0]);
# ax.set_ylabel(param_info.index.values[1]);



################################################
# Make a cut
################################################



# The implausibility threshold determines how willing we are to retain regions
# of parameter space that are inconsistent with the underlying data. A higher
# threshold is more risk averse in that potentially good regions are less likely
# to be rejected, however it will take more iterations/simulations to achieve results.
# implausibility_threshold = 3

# n_samples_this_iter = 100 # Number of simulations to conduct on this iteration
# n_samples_to_generate_for_next_iter = 100 # Number of simulations to conduct on this iteration

# training_fraction = 0.75 # Fraction of simulations to use as training
# discrepancy_std = 3 # Accounts for uncertainty w.r.t model structure


# For this first iteration, we're going to make one "cut" using the first observation, but 
# you can separately do multiple "cuts" per iteration using several observations (separately)
# We'll need a name for this cut and the desired results

# desired_result_idx = 0 # Pick observation 0 or 1 [integers]

# cut_name = 'Prevalence_Meas_%d'%desired_result_idx # No spaces or strange characters!

# desired_result = observations.iloc[desired_result_idx]['Prevalence']
# desired_result_var = observations.iloc[desired_result_idx]['Stdev']**2




# Finally we get to do some History Matching!

# Begin by creating an instance of the HistoryMatching class
# hm = HistoryMatching(
#     cut_name = cut_name,
#     param_info = param_info,
#     inputs = samples,
#     results = results,
#     desired_result = desired_result,
#     desired_result_var = desired_result_var,
#     iteration = iteration,
#     implausibility_threshold = implausibility_threshold,
#     discrepancy_var = discrepancy_std**2,
#     training_fraction = training_fraction
# )
# hm.save() # Save to disk



code.interact(local=locals())


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
basis_glm = hm2.basis.PolynomialBasis(
    degree = 1,
    intercept = True
)


# Now fit the glm and plot

### GLM ###############################################################
print("="*80, "\nGeneralized Linear Modeling\n", "="*80)
#######################################################################
glm = hm2.glm.GLM(basis=basis_glm, family='Gaussian').fit(runs)



for file in glob.glob(os.path.join(hm.glmdir, "PairwiseResults", "*.pdf")):
    img = WImage(filename=file)
    print(file)
    display(img)

# Results get saved to disk, so load and display:
for file in glob.glob(os.path.join(hm.glmdir, "GLM Predicted vs Actual.pdf")):
    img = WImage(filename=file)
    print(file)
    display(img)

code.interact(local=locals())
sys.exit(-1)













# Just for fun, let's visualize some SIR trajectories and observations / target data
# z = SIR(beta=0.003, gamma=0.1)
# f, ax = z.plot()
# for i,obs in observations.iterrows():
#     ax.plot(obs['Times'], obs['Prevalence'], 'ko')
#     ax.plot(
#         [obs['Times'],obs['Times']], 
#         [obs['Prevalence']-2*obs['Stdev'],obs['Prevalence']+2*obs['Stdev']],
#         'k-')




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











# sim_results contains the simulated values at both observation times, but for this iteration
# we only want to use the first one (ObsIdx == 0).
# Also note that results must be a Series with index containing 'Sample_Id' and 'Sim_Id'
results = sim_results \
    .query('ObsIdx==%d'%desired_result_idx)[['Sample_Id', 'Sim_Id', 'Prevalence']] \
    .set_index(['Sample_Id', 'Sim_Id'])['Prevalence']
display(results.tail())
