import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hm2.boilerplate import *
from hm2.emulators import *
from hm2.models import SIR
from hm2.plotting import *
from hm2.wrapped_model import *
import hm2.basis
import hm2.sampling



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

#TODO
#p=plot_pairwise(parameter_samples.drop(columns='param_id'))
#print(p['all'])


################################################
#Start sampling
################################################

# HistoryMatching requires that the model be wrapped in a special function
# which standardizes HistoryMatching's interaction with models, like so:

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


########################################
#WAVE 1
########################################

################################################
#Run the model a number of times
sim_replicates = run_replicates(
  wrapped_model = wrapped_model,
  param_sets    = [x.to_dict() for _, x in parameter_samples.iterrows()],
  replicates    = 2,
  processes     = None,
)

#TODO
#Just for fun, let's visualize some SIR trajectories and observations / target data
#p = plot_runs_time_series(sim_replicates, samples=20, time_observations=time_observations)
#Print plot every time you want to show it
#print(p)


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

psamples_within = hm2.sampling.latin_hypercube_within(parameter_samples, 1000)

#Prepare data for emulator (make a train_x, train_y, stdev_y tuple)
time_emulators = dict()
for obs, params, y, stdev in get_data_for_emulators(parameter_samples, matched[0]):
  #Train emulator
  time_emulators[obs] = hm2.emulators.GLM_GPR_Emulator(
    glm_basis=hm2.basis.IdentityBasis(intercept=False),
    gpr_basis=hm2.basis.IdentityBasis(intercept=False)
  ).fit(params, y, stdev, gpr_maxiter=10000)
  # time_emulators[obs].glm.plot_fitted_vs_observed()
  # time_emulators[obs].glm.plot_pearson_residuals()
  # time_emulators[obs].glm.plot_deviance_redisuals()
  # time_emulators[obs].glm.plot_QQ()
  # figs = time_emulators[obs].plot_data() #TODO




#TODO: Validate emulators here


implausibilities = get_implausibility(time_emulators, psamples_within, time_observations)
implausibilities = max_implausibility_per_param(implausibilities)

# The implausibility threshold determines how willing we are to retain regions
# of parameter space that are inconsistent with the underlying data. A higher
# threshold is more risk averse in that potentially good regions are less likely
# to be rejected, however it will take more iterations/simulations to achieve results.
implausibilities = filter_implausibilities(implausibilities, threshold=0.2)
plausible_params = get_plausible_parameters(implausibilities, psamples_within)

if len(plausible_params)==0:
  print("No non-implausible parameter samples!")


################################
# WAVE 2

parameter_samples = hm2.sampling.latin_hypercube_within(plausible_params, 100)



#TODO
# sys.exit(0)
# fig, ax=plt.subplots(2,1)

# ax[0].scatter(
#   parameter_samples.iloc[this_obs['param_id']]['beta'],
#   parameter_samples.iloc[this_obs['param_id']]['gamma'],
#   c=this_obs['value']
# )

# ax[1].scatter(
#   parameter_samples.iloc[this_obs['param_id']]['beta'],
#   parameter_samples.iloc[this_obs['param_id']]['gamma'],
#   c=gpremu.predict(parameter_samples.iloc[this_obs['param_id']][['beta','gamma']])[0]
# )

# plt.show()
