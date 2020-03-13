from history_matching.HistoryMatching import HistoryMatching
from history_matching.quick_read import quick_read
from history_matching.basis import Basis

import pandas as pd
import os
import re
import pickle
import numpy as np
import glob

force_optimize_glm = False
force_optimize_gpr = True

iteration = int(re.search(r'[+-]?\d+', os.getcwd()).group())
exp_ids = glob.glob('Data_*')
training_fraction = 0.75
implausibility_threshold = 3

cut_name = 'RadiusShouldBe15'
desired_result = 15
discrepancy_std = 0.1 * desired_result
print('Desired result is', desired_result)

# Data
params_file = os.path.join('..', 'Params.xlsx')
samples_fn = 'Samples.xlsx'
results_fn = 'Results.xlsx'

sim_inputs = []
sim_results = []
for idx, exp_id in enumerate(exp_ids):
    read = quick_read(os.path.join(exp_id, samples_fn), 'Samples')
    read['Exp_Id'] = exp_id
    read['Sample_Id'] = read['Sample'].apply(lambda x: '%s.%06d'%(exp_id,x))
    read = read.set_index('Sample_Id').sort_index()

    # Train/test split
    if idx == 0:
        read['Train'] = False
        nSamp = len(read.index.get_level_values('Sample_Id'))
        nTrain = int(round(training_fraction * nSamp))
        read.iloc[:nTrain-1]['Train'] = True
    else:
        read['Train'] = True

    sim_inputs.append(read)

    read = quick_read(os.path.join(exp_id, results_fn), 'Sheet1')
    read['Exp_Id'] = exp_id
    read['Sample_Id'] = read['Sample'].apply(lambda x: '%s.%06d'%(exp_id,x))
    sim_results.append(read.set_index('Sample_Id').sort_index())

inputs = pd.concat(sim_inputs)
sim_results_all = pd.concat(sim_results)

sim_results_all.set_index(['Sim_Id'], append=True, inplace=True)
results = sim_results_all['Sim_Result']

if not os.path.exists(os.path.join('Cuts', cut_name)):
    os.makedirs(os.path.join('Cuts', cut_name))

param_info = quick_read(params_file, 'Params').set_index('Name')
param_names = param_info.index.tolist()
print('All available parameters:')
print(' *','\n * '.join(param_names))

# Choose GLM inputs
try:
    with open(os.path.join('Cuts', cut_name, 'basis_glm.pickle'), 'rb') as data_file:
        config = pickle.load( data_file )
        basis_glm = config['Basis']
        fitted_values = config['Fitted_Values']
except:
    basis_glm = Basis.make_polynomial_basis(params=param_names, intercept = True, first_order=True, second_order=True, third_order=False, param_info=param_info)

    basis_glm.plot_regularize(inputs, results, alpha = np.logspace(-3,1, 25), scaleX=True)
    alpha_glm = float(input('What would you like to use for the GLM regularization parameter, alpha_glm = '))
    #alpha_glm = 1e-3

    fitted_values = basis_glm.regularize(inputs, results, alpha = alpha_glm, scaleX=True) # 100 for thrid_order

    print(type(basis_glm.get_terms()))
    print('Regularization for GLM selected:\n', ' *','\n * '.join(basis_glm.get_terms()))
    with open(os.path.join('Cuts', cut_name, 'basis_glm.pickle'), 'wb') as fout:
        pickle.dump( {
            'Basis': basis_glm,
            'Fitted_Values': fitted_values
        }, fout, indent=4)

# Choose GPR inputs
try:
    with open(os.path.join('Cuts', cut_name, 'basis_gpr.pickle'), 'rb') as data_file:
        config = pickle.load( data_file )
        basis_gpr = config['Basis']
except:
    basis_gpr = Basis.make_polynomial_basis(params=param_names, intercept = False, first_order=True, param_info=param_info)
    results_err = results - fitted_values

    basis_gpr.plot_regularize(inputs, results_err, alpha = np.logspace(-3, 1, 25), scaleX=True)
    alpha_gpr = float(input('What would you like to use for the GPR regularization parameter, alpha_gpr = '))

    basis_gpr.regularize(inputs, results_err, alpha = alpha_gpr, scaleX=True)
    print('Regularization for GPR selected:\n', ' *','\n * '.join(basis_gpr.get_terms()))
    with open(os.path.join('Cuts', cut_name, 'basis_gpr.pickle'), 'wb') as fout:
        pickle.dump({'Basis': basis_gpr}, fout)


#basis_gpr = Basis.make_identity_basis(params=['Protection per Infection', 'Symptomatic Fraction', 'LOG Contact Exposure Period', 'LOG Environmental Exposure Period', 'LOG Acute Infectiousness'], param_info=param_info)

# History Matching!
hm = HistoryMatching(
    cut_name = cut_name,
    param_info = param_info,
    inputs = inputs,
    results = results,
    desired_result = desired_result,
    iteration = iteration,
    implausibility_threshold = implausibility_threshold,
    discrepancy_var = discrepancy_std**2,
    training_fraction = training_fraction
)
hm.save()


# If desired, you can filter train/test/both data with lower and upper bounds on the result
#hm.filter_data(source='Both', lower=0)

### GLM ###############################################################
print("="*80, "\nGeneralized Linear Modeling\n", "="*80)
#######################################################################
hm.glm(
    basis = basis_glm,
    family = 'Gaussian',
    force_optimize_glm = force_optimize_glm,
    glm_fit_maxiter = 100000,
    plot = force_optimize_glm,
    plot_data = False
)


### GPR ###############################################################
print("="*80, "\nGaussian Process Regression\n", "="*80)
#######################################################################
hm.gpr(
    basis = basis_gpr,
    force_optimize_gpr = force_optimize_gpr,
    sigma2_f_guess = 4,
    sigma2_f_bounds = (0.1, 1000),
    sigma2_n_guess = 0.1,
    sigma2_n_bounds = (0.001, 100),
    #lengthscale_guess = [0.04313128, 0.2, 0.14240553, 0.01418867, 0.2, 0.17683428],
    lengthscale_guess = 0.1,
    lengthscale_bounds = (0.001, 0.5),
    optimizer_options = {
        'eps': 5e-3,
        'disp': True,
        'maxiter': 15000,
        #'ftol': 1e-1,
        #'gtol': 1e-1,
        #'factr': 1e12 # <-- Not working?
    },
    optimize_sigma2_n = True,
    log_transform = False,
    plot = True, #force_optimize_gpr,
    plot_data = False
)

### Implausibility ############################################################
print("="*80, "\nImplausibility\n", "="*80)
###############################################################################
hm.calc_and_plot_implausibility(plot=True, do_plot_data=True, plot_data_highlight=pd.DataFrame()) # plot_data_highlight=hm.training_data.loc['8c7e4af7-1120-e711-9400-f0921c16849c.003328']

hm.training_data.to_excel(os.path.join('Cuts', cut_name, 'train_data.xlsx'))
hm.test_data.to_excel(os.path.join('Cuts', cut_name, 'test_data.xlsx'))

print('Good')

