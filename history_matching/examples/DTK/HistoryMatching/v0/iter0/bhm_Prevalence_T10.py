from history_matching import HistoryMatching, quick_read, Basis
import pandas as pd
import os
import re
import pickle
import numpy as np

force_optimize_glm = False
force_optimize_gpr = False

iteration = int(re.search(r'iter(\d+)', os.getcwd()).group(1))
TIMESTEP = 10

exp_ids = ['prime']#, 'iter2prime', 'iter1prime']
training_fraction = 0.75
implausibility_threshold = 3

# Data
params_file = os.path.join('..', 'Params.xlsx')
samples_fn = 'Samples.xlsx'
results_fn = 'Results_PrevalenceAnalyzer.csv'

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

    read = pd.read_csv(os.path.join(exp_id, results_fn), skipinitialspace=True) \
        .groupby(['Timestep', 'Sample', 'Sim_Id']) \
        .sum() \
        [['Prevalence']] \
        .reset_index()
    read['Exp_Id'] = exp_id
    read['Sample_Id'] = read['Sample'].apply(lambda x: '%s.%06d'%(exp_id,x))
    sim_results.append(read.set_index('Sample_Id').sort_index())

inputs = pd.concat(sim_inputs)
sim_results_all = pd.concat(sim_results)

ref_data_all = quick_read( os.path.join('..', '..', '..', 'Data', 'CountryData.xlsx'), 'Prevalence').set_index(['Timestep'])
desired_result = ref_data_all.loc[TIMESTEP]['Prevalence']
desired_result_std = ref_data_all.loc[TIMESTEP]['Stdev']

sim_results_all = sim_results_all.reset_index().set_index(['Timestep', 'Sample_Id', 'Sim_Id']).sort_index()
print(sim_results_all.head())
results = sim_results_all.loc[TIMESTEP, 'Prevalence']

# LOGIT
results = results.loc[results > 0] # Remove zero prevalence numbers
results = np.log( results / (1-results) )
lower = desired_result-2*desired_result_std
lower_l = np.log(lower / (1-lower))
upper = desired_result+2*desired_result_std
upper_l = np.log(upper/(1-upper))
desired_result_std = (upper_l - lower_l) / (2*1.96)
desired_result = np.log( desired_result / (1-desired_result) )

print('Desired result is {0:.3f} [{1:.3f}, {2:.3f}]: '.format(desired_result, desired_result-2*desired_result_std, desired_result+2*desired_result_std))
discrepancy_std = 0.25 #* desired_result


cut_name = 'Prevalence LOGIT (%d) v0' % (TIMESTEP)
if not os.path.exists(os.path.join('Cuts', cut_name)):
    os.makedirs(os.path.join('Cuts', cut_name))

param_info = quick_read(params_file, 'Params').set_index('Name')
param_names = param_info.index.tolist()
print('All available parameters:')
print(' *','\n * '.join(param_names))

# Choose GLM inputs
try:
    with open(os.path.join('Cuts', cut_name, 'basis_glm.pickle'), 'rb') as data_file:
        config = pickle.load(data_file)
        basis_glm = config['Basis']
        fitted_values = config['Fitted_Values']
except:
    basis_glm = Basis.make_polynomial_basis(params=param_names, intercept = True, first_order=True, second_order=True, third_order=False, param_info=param_info)

    basis_glm.plot_regularize(inputs, results, alpha = np.logspace(-4,0, 25), scaleX=True)
    alpha_glm = float(input('What would you like to use for the GLM regularization parameter, alpha_glm = '))
    #alpha_glm = 1e-3 #5e-3

    fitted_values = basis_glm.regularize(inputs, results, alpha = alpha_glm, scaleX=True)
    print('Regularization for GLM selected:\n', ' *','\n * '.join(basis_glm.get_terms()))
    with open(os.path.join('Cuts', cut_name, 'basis_glm.pickle'), 'wb') as fout:
        pickle.dump({
            'Basis': basis_glm,
            'Fitted_Values': fitted_values
        }, fout)


# History Matching!
hm = HistoryMatching(
    cut_name = cut_name,
    param_info = param_info,
    inputs = inputs,
    results = results,
    desired_result = desired_result,
    desired_result_var = desired_result_std**2,
    iteration = iteration,
    implausibility_threshold = implausibility_threshold,
    discrepancy_var = discrepancy_std**2,
    training_fraction = training_fraction
)
hm.save()

# If desired, you can filter train/test/both data
def at_least(x, lower): return x[ x['Sim_Result'] >= lower ]
def at_most(x, upper): return x[ x['Sim_Result'] <= upper ]
#hm.filter(train=True, func=partial(at_least, lower=1))

### GLM ###############################################################
print("="*80, "\nGeneralized Linear Modeling\n", "="*80)
#######################################################################
hm.glm(
    basis = basis_glm,
    family = 'Gaussian',
    force_optimize_glm = force_optimize_glm,
    glm_fit_maxiter = 100000,
    plot = True, #force_optimize_glm,
    plot_data = True
)

# Choose GPR inputs
#basis_gpr = Basis.make_identity_basis(params=['LOG Environmental Exposure Period', 'LOG Acute Infectiousness'], param_info=param_info)
try:
    with open(os.path.join('Cuts', cut_name, 'basis_gpr.pickle'), 'rb') as data_file:
        basis_gpr = pickle.load(data_file)
except:
    basis_gpr = Basis.make_polynomial_basis(params=param_names, intercept = False, first_order=True, param_info=param_info)

    results_err = results - fitted_values

    basis_gpr.plot_regularize(inputs, results_err, alpha = np.logspace(-5, -2, 25), scaleX=True)
    alpha_gpr = float(input('What would you like to use for the GPR regularization parameter, alpha_gpr = '))
    #alpha_gpr = 1e-3 #1e-3 # 5e-4

    basis_gpr.regularize(inputs, results_err, alpha = alpha_gpr, scaleX=True)
    print('Regularization for GPR selected:\n', ' *','\n * '.join(basis_gpr.get_terms()))
    with open(os.path.join('Cuts', cut_name, 'basis_gpr.pickle'), 'wb') as fout:
        pickle.dump(basis_gpr, fout)

### GPR ###############################################################
print("="*80, "\nGaussian Process Regression\n", "="*80)
#######################################################################
hm.gpr(
    basis = basis_gpr,
    force_optimize_gpr = force_optimize_gpr,

    sigma2_f_guess = 0.6,
    sigma2_f_bounds = (0.001, 100), #!!(0.1, 100),
    sigma2_n_guess =  2.0,
    sigma2_n_bounds = (0.01, 100),

    #lengthscale_guess = [0.09844299, 0.1256657, 0.0976875, 0.09889085, 0.1051974, 0.0950809, 0.10032171, 0.10599185, 0.10627393, 0.09950996, 0.09445544, 0.10285915, 0.10007409, 0.09847433, 0.08963389, 0.10205652, 0.09360044, 0.1024141, 0.09786228, 0.10247492, 0.09852253, 0.09632744, 0.09997534, 0.10767302, 0.10095249, 0.09941825, 0.10214923, 0.10221497, 0.09734157, 0.09093285, 0.10780673, 0.09881377, 0.10597152],
    lengthscale_guess = 0.25,
    lengthscale_bounds = (0.0001, 10), #!! (0.001, 0.5),

    optimize_sigma2_n = True,
    log_transform = True,

    verbose = True,
    optimizer_options = {
        'eps': 5e-3,
        'disp': True,
        'maxiter': 15000,
        'ftol': 2 * np.finfo(float).eps,
        'gtol': 2 * np.finfo(float).eps,
    },
    plot = True, #force_optimize_gpr,
    plot_data = False
)

### Implausibility ############################################################
print("="*80, "\nImplausibility\n", "="*80)
###############################################################################
hm.calc_and_plot_implausibility(
    plot=True,
    do_plot_data = True,
    plot_data_highlight = hm.test_data.loc['prime.000049']
) 
    #plot_data_highlight=pd.DataFrame() # plot_data_highlight=hm.training_data.loc['prime.000049']

hm.training_data.to_excel(os.path.join('Cuts', cut_name, 'train_data.xlsx'))
hm.test_data.to_excel(os.path.join('Cuts', cut_name, 'test_data.xlsx'))

print('Good')

