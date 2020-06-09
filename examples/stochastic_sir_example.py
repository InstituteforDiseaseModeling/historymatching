#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hm2.boilerplate import *
from hm2.emulators import *
from hm2.models import SIR
from hm2.plotting import *
from hm2.sampling import *
import hm2.basis

# Set this to True to show plots. Plots are disabled by default to ensure this
# script can be run by our automated testing framework.
SHOW_PLOTS = False
RUNS_PER_WAVE = 50



################################################
#Observational Data
################################################

#Note that there are no summary observations. If there were, they would have
#time values of `NaN`.
real_observations = pd.DataFrame({
    'observation_id': [           0,            1],
    'time':           [         3.0,         15.0],
    'observation':    ['prevalence', 'prevalence'],
    'value':          [          15,           40],
    'stdev':          [           4,          2.3]
})

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
parameter_samples = hm2.sampling.latin_hypercube(param_info, samples=RUNS_PER_WAVE, random_state=123456)

# Save the plot in p so we can easily plot it again and again
p=plot_pairwise(parameter_samples)
if SHOW_PLOTS:
  # `plot_pairwise` returns many plots. We view all of them at once this way.
  print(p['all'])

################################################
#Start sampling
################################################

# HistoryMatching requires that the model be wrapped in a special function
# which standardizes HistoryMatching's interaction with models, like so:

def wrapped_model(**kwargs):
    model = SIR(**kwargs)
    results = model.run()
    #Rename model result so it matches the name of an Observation
    results['prevalence'] = results['per_infected']
    #Reshape DataFrame into the tidy form expected by HistoryMatching
    results = pd.melt(results, id_vars='time', var_name='observation')
    #We have no uncertainty about our results
    results['stdev'] = 0
    #Add observation ids
    results['observation_id'] = list(range(len(results)))
    #Sort by time
    results.sort_values(by='time', inplace=True)
    return results



########################################
#WAVE 1
########################################

#Run the model a number of times
sim_replicates = run_replicates(
  wrapped_model = wrapped_model,
  param_sets    = parameter_samples,
  replicates    = 4,
  processes     = None  # None implies that all cores are used to make multiple runs in parallel
)

#Just for fun, let's visualize some SIR trajectories and observations / target data
p = plot_runs_time_series(sim_replicates, samples=20, real_observations=real_observations)
if SHOW_PLOTS:
  print(p)



################################################
#Match to observations
################################################

matched = match_sim_outputs_to_observations(
  sim_replicates,
  real_observations,
  processes=None
)



################################################
#Fit emulators
################################################

# TODO: Revise this text to match the new process
# Now we begin the process of emulating the simulation output This process
# contains two steps. The first step is to fit a deterministic model, here we
# use a generalized linear model (GLM). The GLM will attempt to model the output
# (prevalence at the first observation) as a function of some inputs. Those
# inputs need not be the model parameters directly! The inputs could be anything
# from a constant intercept up to third or higher order interaction terms
# between parameters. The Basis module provides several options. Here we build
# out the GLM input parameters from the overall simulation input parameters.
#
# Some strategy is required when choosing these.  If you know which parameters
# matter, there's a way to directly specify those parameters.  Alternatively, if
# you have no idea, you can initially include an intercept, first, second, and
# maybe also third order interaction terms.  The Basis class has a built-in
# penalized regression that throws away unneeded terms (basis vectors). The
# second step of emulation, as demonstrated here, fits a GPR to the redisual
# error between the simulated outputs and the GLM estimates.  If the GLM fits
# really well, the residual is mostly noise and the GPR has a hard time fitting
# / isn't very informative. We actually prefer to weaken the GLM enough to leave
# plenty of residual signal for the GPR.  Here, I use only first-order (beta and
# gamma) terms.

#Prepare data for emulator (make a train_x, train_y, stdev_y tuple)
params, y, stdev = get_single_obs_data_for_emulators(parameter_samples, matched, observation_id=0)

#Train emulator
emulator = hm2.emulators.GLM_GPR_Emulator(
  glm_basis=hm2.basis.PolynomialBasis(intercept=True, degree=1),
  gpr_basis=hm2.basis.IdentityBasis(intercept=False)
).fit(params, y, stdev, gpr_maxiter=10000, glm_seed=123456)

#Here, we validate the emulator by looking at various plots and trying to
#convince ourselves it's doing a good job
p1 = emulator.glm.plot_fitted_vs_observed()
p2 = emulator.glm.plot_pearson_residuals()
p3 = emulator.glm.plot_deviance_redisuals()
p4 = emulator.glm.plot_QQ()
p5 = emulator.plot_data()['all']
p6 = emulator.plot_emulated_vs_predicted()
if SHOW_PLOTS:
  print(p1,p2,p3,p4,p5,p6)


# Now that we've fit the emulator, we use it to find new non-implausible
# parameters which, hopefully, are in a smaller space.
new_parameter_samples = generate_n_new_plausible_parameters(
  count=RUNS_PER_WAVE,
  emulators={0:emulator},
  parameter_samples=parameter_samples,
  real_observations=real_observations,
  threshold=2,
  generation_count=1_000
)

vol_old = get_size_of_parameter_space(parameter_samples)
vol_new = get_size_of_parameter_space(new_parameter_samples)
print(f"Volume of old space: {vol_old}")
print(f"Volume of new space: {vol_new}")
print(f"Percent change: {percent_change_vol(vol_old, vol_new):.2}%")

# If there are no non-implausible parameters, we should stop
if len(new_parameter_samples)==0:
  print("No non-implausible parameter samples!")



################################
# WAVE 2

# Now, we repeat the above analysis in the reduced parameter space.
parameter_samples = new_parameter_samples
