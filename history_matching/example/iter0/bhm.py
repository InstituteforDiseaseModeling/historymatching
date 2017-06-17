from history_matching import HistoryMatching, quick_read, Basis
import pandas as pd
import os
import re, time
import json
import numpy as np
import glob

force_optimize_glm = True
force_optimize_gpr = True

iteration = int(re.search(r'[+-]?\d+', os.getcwd()).group())
exp_ids = glob.glob('Data_*')
training_fraction = 0.75
implausibility_threshold = 3

cut_name = 'RadiusShouldBe15'
desired_result = 15
discrepancy_std = 0.1 * desired_result
print 'Desired result is: ', desired_result

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


sim_results_all.set_index(['Exp_Id', 'Sample', 'Sim_Id'], append=True, inplace=True)
results = sim_results_all['Sim_Result']

if not os.path.exists(os.path.join('Cuts', cut_name)):
    os.makedirs(os.path.join('Cuts', cut_name))

param_info = quick_read(params_file, 'Params').set_index('Name')
param_names = param_info.index.tolist()
print 'All available parameters:'
print ' *','\n * '.join(param_names)

# Choose GLM inputs
try:
    with open(os.path.join('Cuts', cut_name, 'basis_glm.json')) as data_file:
        config = json.load( data_file )
        basis_glm = Basis.deserialize(config['Basis'])
        fitted_values = pd.read_json(config['Fitted_Values'], orient='split').set_index(['Sample_Id', 'Exp_Id', 'Sample', 'Sim_Id']).squeeze()
except:
    basis_glm = Basis.polynomial_basis(params=param_names, intercept = True, first_order=True, second_order=True, third_order=False, param_info=param_info)

    basis_glm.plot_regularize(inputs, results, alpha = np.logspace(-3,0, 25), scaleX=True)

    alpha_glm = float(raw_input('What would you like to use for the GLM regularization parameter, alpha_glm = '))

    fitted_values = basis_glm.regularize(inputs, results, alpha = alpha_glm, scaleX=True) # 100 for thrid_order
    print 'Regularization for GLM selected:\n', ' *','\n * '.join(basis_glm.get_terms())
    with open(os.path.join('Cuts', cut_name, 'basis_glm.json'), 'w') as fout:
        json.dump( {
            'Basis': basis_glm.serialize(),
            'Fitted_Values': fitted_values.reset_index().to_json(orient='split')
        }, fout, indent=4)

# Choose GLM inputs
try:
    with open(os.path.join('Cuts', cut_name, 'basis_gpr.json')) as data_file:
        config = json.load( data_file )
        basis_gpr = Basis.deserialize(config['Basis'])
except:
    basis_gpr = Basis.polynomial_basis(params=param_names, intercept = False, first_order=True, param_info=param_info)
    results_err = results - fitted_values

    basis_gpr.plot_regularize(inputs, results_err, alpha = np.logspace(-6, 0, 25), scaleX=True)
    alpha_gpr = float(raw_input('What would you like to use for the GPR regularization parameter, alpha_gpr = '))

    basis_gpr.regularize(inputs, results_err, alpha = alpha_gpr, scaleX=True)
    print 'Regularization for GPR selected:\n', ' *','\n * '.join(basis_gpr.get_terms())
    with open(os.path.join('Cuts', cut_name, 'basis_gpr.json'), 'w') as fout:
            json.dump( { 'Basis': basis_gpr.serialize(), }, fout, indent=4)


#basis_gpr = Basis.identity_basis(params=['Protection per Infection', 'Symptomatic Fraction', 'LOG Contact Exposure Period', 'LOG Environmental Exposure Period', 'LOG Acute Infectiousness'], param_info=param_info)

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
print "="*80, "\nGeneralized Linear Modeling\n", "="*80
#######################################################################
hm.glm(
    basis = basis_glm,
    family = 'Gaussian',
    force_optimize_glm = force_optimize_glm,
    glm_fit_maxiter = 100000,
    plot = force_optimize_glm,
    plot_data = False
)


### GLM ###############################################################
print "="*80, "\nGaussian Process Regression\n", "="*80
#######################################################################
hm.gpr(
    basis = basis_gpr,
    force_optimize_gpr = force_optimize_gpr,
    K_folds = 10,
    sigma2_f_guess = 1,
    sigma2_f_bounds = (0.1, 100),
    sigma2_n_guess = 1,
    sigma2_n_bounds = (0.001, 100),
    #lengthscale_guess = [0.04313128, 0.2, 0.14240553, 0.01418867, 0.2, 0.17683428],
    lengthscale_bounds = (0.001, 0.2),
    verbose = True,
    optimizer_options = {
        'eps': 5e-3,
        'disp': True,
        'maxiter': 15000,
        #'ftol': 1e-1,
        #'gtol': 1e-1,
        #'factr': 1e12 # <-- Not working?
    },
    plot = True, #force_optimize_gpr,
    plot_data = False
)

### Implausibility ############################################################
print "="*80, "\nImplausibility\n", "="*80
###############################################################################
hm.calc_and_plot_implausibility(plot=True, do_plot_data=True, plot_data_highlight=pd.DataFrame()) # plot_data_highlight=hm.training_data.loc['8c7e4af7-1120-e711-9400-f0921c16849c.003328']

hm.training_data.to_excel(os.path.join('Cuts', cut_name, 'train_data.xlsx'))
hm.test_data.to_excel(os.path.join('Cuts', cut_name, 'test_data.xlsx'))

print 'Good'

