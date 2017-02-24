import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import statsmodels.api as sm
from scipy.special import logit
from scipy.stats import logistic
from pyDOE import lhs

import sys
sys.path.insert(0, '/home/dklein/GIT/Typhoid_Santiago_Fitting/Scripts/Meta_Modeling') # <-- Impossibly bad
from datamanager import DataManager
from glm import GLM
from gpr import GPR
from gpr_GPy import GPR_GPy

iteration = 0
iterdir = '0a329116-66fa-e611-9400-f0921c16849c'
training_fraction = 0.75

implausibility_threshold = 3    # Z score
force_optimize_glm = False
force_optimize_gpr = False or force_optimize_glm
discrepancy_var = 30**2

glm_fit_maxiter = 100000
second_order_basis_terms = True
third_order_basis_terms = False
fourth_order_basis_terms = False
fifth_order_basis_terms = False
higher_order_basis_terms = False

gpr_K_folds = 5

bhmdir = os.path.join(iterdir, 'BHM_CV')
if not os.path.exists( bhmdir):
    os.mkdir( bhmdir )

glmdir = os.path.join(bhmdir, 'GLM')
if not os.path.exists( glmdir):
    os.mkdir( glmdir )

gprdir = os.path.join(bhmdir, 'GPR')
if not os.path.exists( gprdir):
    os.mkdir( gprdir )

jointdir = os.path.join(bhmdir, 'Joint')
if not os.path.exists( jointdir):
    os.mkdir( jointdir )

glm_model_fn = os.path.join(glmdir, 'model.json')
mean_params_fn = os.path.join(glmdir, 'params.p')
gpr_model_fn = os.path.join(gprdir, 'model.json')
gpr_model_with_test_fn = os.path.join(gprdir, 'model_with_test_data.json')

# Data
samples_fn = 'Samples.xlsx'
results_fn = os.path.join(iterdir, 'Results.xlsx')

reference_fn = os.path.join('..', '..', 'Data', 'Zimbabwe.xlsx')

params = pd.read_excel(samples_fn, sheetname='Params')
Xcols_all = params['Name'].tolist()

#print 'Available X-columns:\n', '\n\t'.join(Xcols_all)
#Xcols = Xcols_all
Xcols = [
    #'Risk Reduction Fraction',
    #'Risk Ramp Rate',
    #'Risk Ramp MidYear',
    'Base Infectivity',
    'Trans Form Rate',
    'Informal Form Rate',
    'Marital Form Rate',
    'Male To Female Young',
    'Male To Female Old',
    #'Trns Condom Mid',
    'Trns Condom Rate',
    #'Infrml Condom Mid',
    #'Infrml Condoms Rate',
    #'Mrtl Condom Max',
    #'Mrtl Condom Mid',
    'Mrtl Condom Rate',
    #'Bulawayo: Trns Condoms Max',
    #'Bulawayo: Infmrl Condoms Max',
    'Bulawayo: LOW Risk',
    #'Harare: Trns Condoms Max',
    #'Harare: Infmrl Condoms Max',
    'Harare: LOW Risk',
    #'Manicaland: Trns Condoms Max',
    #'Manicaland: Infmrl Condoms Max',
    'Manicaland: LOW Risk',
    #'Mashonaland: Trns Condoms Max',
    #'Mashonaland: Infmrl Condoms Max',
    'Mashonaland: LOW Risk',
    #'Masvingo: Trns Condoms Max',
    #'Masvingo: Infmrl Condoms Max',
    'Masvingo: LOW Risk',
    #'Matabeleland: Trns Condoms Max',
    #'Matabeleland: Infmrl Condoms Max',
    'Matabeleland: LOW Risk',
    #'Midlands: Trns Condoms Max',
    #'Midlands: Infmrl Condoms Max',
    'Midlands: LOW Risk',
    #'Risk Assortivity',
    'Pr Ex Trns Male LOW',
    'Pr Ex Trns Male MED',
    'Pr Ex Trns Fem LOW',
    'Pr Ex Trns Fem MED',
    #'Pr Ex Infmrl Male LOW',
    'Pr Ex Infmrl Male MED',
    'Pr Ex Infmrl Fem LOW',
    'Pr Ex Infmrl Fem MED',
    'Max Trns M&F LOW',
    'Max Infmrl M&F LOW',
    'Max Trns M&F MED',
    'Max Infmrl M&F MED',
    #'Max Mrtl M&F MED',
    #'HCT Uptake Post Debut Max',
    #'HCT Uptake Post Debut Mid',
]

Ycol = 'Infected_Count'

Year = 2016
Source = 'ZIMPHIA'
Gender = 'Male'
AgeBin = '[40, 45)'

raw = pd.read_excel(reference_fn, 'NationalPrevalence')[['Year', 'Source', 'Gender', 'AgeBin', 'NationalPrevalence', 'Count']]
raw.set_index(['Source', 'Year', 'Gender', 'AgeBin'], inplace=True)
national_prevalence_reference_data = raw.loc[Source, Year, Gender, AgeBin]
reference_value = 1/100. * national_prevalence_reference_data['NationalPrevalence'] * national_prevalence_reference_data['Count']

### DATA MANAGER ##############################################################
print "Loading data ..."
dm = DataManager(samples_fn, results_fn, 'NationalPrevalenceAnalyzer', training_fraction)

# Fix names for picky statsmodels patsy:
params.set_index('Name', drop=True, inplace=True)
newXcols = []
newXcols_all = []
for i,xc in enumerate(Xcols_all):
    if ':' in xc or '&' in xc:
        new_xc = xc.replace(':', '').replace('&',' ')
        dm.rename(xc, new_xc) # statsmodels var names can't have space in formula
        #print '\t', xc, '-->', new_xc
        params.rename(index={xc:new_xc}, inplace=True)
        newXcols_all.append(new_xc)
        if xc in Xcols:
            newXcols.append(new_xc)
    else:
        newXcols_all.append(xc)
        if xc in Xcols:
            newXcols.append(xc)

Xcols_all_orig = Xcols_all
Xcols_all = newXcols_all
Xcols_orig = Xcols
Xcols = newXcols

# Interesting idea, not currently in use:
#dm.transform( 'Typhoid_Environmental_Peak_Multiplier', lambda x: np.log(x), 'LOG' )

# Now extract train and test data:
train_data = dm.get_training_data()
test_data = dm.get_test_data()

# TODO: Careful, what if some simulations for sample zero faded out?
nRep = train_data.query('Sample == 0')['Sim_Id'].count()

def compute_result(data):
    tmp = data.reset_index(drop=True).set_index(['Year', 'Gender', 'AgeBin', 'Sample', 'Sim_Id']).loc[Year, Gender, AgeBin]
    #tmp['Prevalence'] = tmp['Infected_Unscaled'] / tmp['Population_Unscaled']
    tmp['Infected_Count'] = tmp['Infected_Unscaled'] * national_prevalence_reference_data['Count'] / tmp['Population_Unscaled']
    #tmp['Logit_Prevalence'] = logit(tmp['Prevalence']/100.)
    return tmp

def filter_entries(data, lower=np.NaN, upper=np.NaN):
    if not np.isnan(lower):
        data = data.loc[ data[Ycol] > lower, :]
        print 'NOTE: Keeping > %f.' % lower

    if not np.isnan(upper):
        data = data.loc[ data[Ycol] < upper, :]
        print 'NOTE: Keeping only data < %f.' % upper

    return data


### DATA PRE-PROCESSING ###############################################
# COMPUTE RESULT
train = compute_result(train_data)
test = compute_result(test_data)

# Remove zeros
#train = filter_entries(train_data, lower=0, upper=np.NaN).reset_index(drop=True).set_index(['Sample', 'Sim_Id'])
#test = filter_entries(test_data, lower=0, upper=np.NaN).set_index(['Sample', 'Sim_Id'])

# Average over the non-zero replicates
train_mean = train.reset_index().groupby(['Sample']).mean()
test_mean = test.reset_index().groupby(['Sample']).mean()

#print 'WARNING: LIMITING TEST DATA!!!!\n'*10
#test_mean = test_mean.iloc[0:25]


#writer = pd.ExcelWriter('Data.xlsx')
#train.to_excel(writer, 'Train')
#test.to_excel(writer, 'Test')
#writer.save()

### GLM ###############################################################
print "Generalized Linear Modeling"
print "-"*80

if not force_optimize_glm and os.path.isfile(glm_model_fn) and os.path.isfile(mean_params_fn):
    print "Loading GLM from", glm_model_fn, ", with model params from", mean_params_fn
    glm_model = GLM.from_config(glm_model_fn, mean_params_fn)
else:
    glm_model = GLM( Xcols = Xcols,
                        Ycol = Ycol,
                        training_data = train_mean,
                        reference_value = reference_value,
                        family = 'Poisson', # Poisson, Gaussian
                        #family = sm.genmod.families.links.Logit,
                        #family = sm.genmod.families.Binomial(link=sm.genmod.families.links.logit),
                        second_order_basis_terms = second_order_basis_terms,
                        third_order_basis_terms = third_order_basis_terms,
                        fourth_order_basis_terms = fourth_order_basis_terms,
                        fifth_order_basis_terms = fifth_order_basis_terms,
                        higher_order_basis_terms = higher_order_basis_terms)
    print "Fitting the GLM"
    glm_model.fit(maxiter=glm_fit_maxiter)
    glm_model.save(glm_model_fn, mean_params_fn)


print 'Evaluating training and test data'
train_mean['Yglm'] = glm_model.evaluate(train_mean)
test_mean['Yglm'] = glm_model.evaluate(test_mean)

fig = glm_model.plot_errors(train_mean.reset_index(), test_mean.reset_index());
fig.savefig( os.path.join(glmdir, 'errors.pdf') );             plt.close(fig)

if False:
    print('Plotting')

    if False:
        #cp = pd.DataFrame()
        print test_mean.loc[[2110]]
        cp = test_mean.loc[[2110]]
        figs = glm_model.plot_data(circle_points=cp);
        pairdir = os.path.join(glmdir, 'PairwiseResults')
        if not os.path.exists( pairdir):
            os.mkdir( pairdir )
        for fn,fig in figs.iteritems():
            fig.savefig( os.path.join(pairdir, fn) ); plt.close(fig)

    fig = glm_model.plot_fitted_vs_observed();  fig.savefig( os.path.join(glmdir, 'fitted_vs_observed.pdf') ); plt.close(fig)
    fig = glm_model.plot_pearson_residuals();   fig.savefig( os.path.join(glmdir, 'pearson_residuals.pdf') );  plt.close(fig)
    fig = glm_model.plot_deviance_redisuals();  fig.savefig( os.path.join(glmdir, 'deviance_redisuals.pdf') ); plt.close(fig)
    fig = glm_model.plot_QQ();                  fig.savefig( os.path.join(glmdir, 'QQ.pdf') );                 plt.close(fig)
    fig = glm_model.plot_histogram();           fig.savefig( os.path.join(glmdir, 'histogram.pdf') );          plt.close(fig)
    fig = glm_model.plot_fit();                 fig.savefig( os.path.join(glmdir, 'fit.pdf') );                plt.close(fig)

train = train.join(train_mean['Yglm'])
train['Yerr'] = train[Ycol] - train['Yglm']

#print 'Best and worst training errors:\n', train.sort_values(by='Yerr')

test = test.join(test_mean['Yglm'])
test['Yerr'] = test[Ycol] - test['Yglm']

train_mean = train.reset_index().groupby(['Sample']).mean()
test_mean = test.reset_index().groupby(['Sample']).mean()

#print 'Best and worst test errors:\n', test.sort_values(by='Yerr')


### GPR ###############################################################
print "Gaussian Process Regression"
print "-"*80
###############################################################################

if not force_optimize_gpr and os.path.isfile(gpr_model_fn):
    print "Loading GPR from", gpr_model_fn
    gpr_model = GPR.from_config(gpr_model_fn)
else:
    gpr_model = GPR(    Xcols = Xcols,
                        Ycol = 'Yerr',
                        training_data = train,  # Use full training data here
                        param_info = params,
                        verbose = True  )

    #param_x0 = [0.1*(v['Max'] - v['Min']) for k,v in params.iterrows() if k in Xcols]
    #param_bounds = tuple( (0.01*(v['Max']-v['Min']), 1.5*(v['Max']-v['Min'])) for k,v in params.iterrows() if k in Xcols )
    param_x0 = len(Xcols)*[0.1]
    param_bounds = tuple( len(Xcols)*((0.001, 1.0),) )

    print "Fitting the GPR"
    gpr_model.optimize_hyperparameters(
        x0 = np.append(np.array([2, 0.5]), param_x0),
        bounds = ((0.005,10),)+((0.01,10),) + param_bounds,
        K=gpr_K_folds
    )
    gpr_model.save(gpr_model_fn)

print 'Evaluating training and test data'
ret = gpr_model.evaluate(train_mean)
train_mean['Mean_Err'] = ret['Mean']
train_mean['Mean_Estimate'] = train_mean['Yglm'] + train_mean['Mean_Err']
train_mean['Var_Err_Predictive'] = ret['Var_Predictive']
train_mean['Var_Err_Latent'] = ret['Var_Latent']
train = train.reset_index().join(train_mean[['Mean_Err', 'Mean_Estimate', 'Var_Err_Predictive', 'Var_Err_Latent']], on='Sample')
train.set_index(['Sample', 'Sim_Id'], inplace=True)

ret = gpr_model.evaluate(test_mean)
test_mean['Mean_Err'] = ret['Mean']
test_mean['Mean_Estimate'] = test_mean['Yglm'] + test_mean['Mean_Err']
test_mean['Var_Err_Predictive'] = ret['Var_Predictive']
test_mean['Var_Err_Latent'] = ret['Var_Latent']
test = test.reset_index().join(test_mean[['Mean_Err', 'Mean_Estimate', 'Var_Err_Predictive', 'Var_Err_Latent']], on='Sample')
test.set_index(['Sample', 'Sim_Id'], inplace=True)

if False:
    print('Plotting')
    fig = gpr_model.plot_errors(train.reset_index(), test.reset_index(), 'Mean_Err', 'Var_Err_Predictive', 'Var_Err_Latent');
    fig.savefig( os.path.join(gprdir, 'errors.pdf') );             plt.close(fig)

    if False:
        #circle_samples = train.sort_values(by='Yerr').iloc[[0, -1]].reset_index()['Sample'].values
        circle_samples = pd.DataFrame()
        figs = gpr_model.plot_data(samples_to_circle=circle_samples);
        pairdir = os.path.join(gprdir, 'PairwiseResults')
        if not os.path.exists( pairdir):
            os.mkdir( pairdir )
        for fn,fig in figs.iteritems():
            fig.savefig( os.path.join(pairdir, fn) ); plt.close(fig)


    if False: # Slow while other stuff is running, but a really nice plot!
        # TODO: Fix parameter ranges
        mu = train[Xcols].mean()
        #mu = train.loc[146][Xcols].mean(); print mu
        (fig_mean, fig_std_latent) = gpr_model.plot(mu, res=25);
        fig_mean.savefig( os.path.join(gprdir, 'plot_mean.pdf') );    plt.close(fig_mean) # SLOW
        fig_std_latent.savefig( os.path.join(gprdir, 'plot_std_latent.pdf') );    plt.close(fig_std_latent) # SLOW

    fig = gpr_model.plot_histogram();
    fig.savefig( os.path.join(gprdir, 'histogram.pdf') );
    plt.close(fig)

###############################################################################
print "Joint plot"
###############################################################################
def joint_plot(data, data_mean, log_x = False):
    fig = plt.figure(figsize=(16,32))

    data_mean_reset = data_mean.reset_index()
    data_reset = data.reset_index()
    first_sample = data.reset_index('Sim_Id').index.unique().min()
    last_sample = data.reset_index('Sim_Id').index.unique().max()

    plt.plot( 2 * [reference_value], [first_sample, last_sample], 'y-', linewidth=0.1) # , axes=axes[0,0]

    sim_cases_range = data.reset_index().groupby('Sample')[Ycol].agg({'Min':np.min, 'Max':np.max, 'Mean':np.mean})
    sim_cases_range = sim_cases_range.join(data_mean['Yglm'])
    for idx,s in sim_cases_range.iterrows():
        plt.plot( [s['Min'], s['Max']], [idx,idx], 'b-', linewidth=0.5 )
        #plt.plot( [s['Mean'], s['Fitted_Model_Mean']], [idx,idx], 'g-', linewidth=0.25 )
        plt.plot( [s['Mean'], s['Yglm']], [idx,idx], 'g:', linewidth=0.25 )
    plt.plot(
        [
            data_mean_reset['Mean_Estimate'] - 2*np.sqrt(data_mean_reset['Var_Err_Latent']),
            data_mean_reset['Mean_Estimate'] + 2*np.sqrt(data_mean_reset['Var_Err_Latent'])
        ],
        [
            data_mean_reset['Sample'],
            data_mean_reset['Sample']
        ],
        'm-', linewidth=1
    )
    plt.plot(
        [
            data_mean_reset['Mean_Estimate'] - 2*np.sqrt(data_mean_reset['Var_Err_Predictive']),
            data_mean_reset['Mean_Estimate'] + 2*np.sqrt(data_mean_reset['Var_Err_Predictive'])
        ],
        [
            data_mean_reset['Sample'],
            data_mean_reset['Sample']
        ],
        'c-', linewidth=0.5
    )

    plt.scatter(data_reset.query('Implausible==False')[Ycol], data_reset.query('Implausible==False')['Sample'], c='k', s=10, marker='|', alpha=1, linewidth=0.1, zorder=100)
    plt.scatter(data_reset.query('Implausible==True')[Ycol], data_reset.query('Implausible==True')['Sample'], c='r', s=10, marker='|', alpha=1, linewidth=0.2, zorder=100)

    plt.scatter(data_mean_reset['Yglm'], data_mean_reset['Sample'], c='g', s=13, marker='|', alpha=1, linewidth=0.1, zorder=90)
    plt.scatter(data_mean_reset['Mean_Estimate'], data_mean_reset['Sample'], c='m', s=13, marker='|', alpha=0.2, linewidth=0.5, zorder=91)
    plt.scatter(data_mean_reset['Mean_Estimate'], data_mean_reset['Sample'], c='c', s=15, marker='|', alpha=1, linewidth=0.1, zorder=101)

    plt.autoscale()
    plt.ylim(ymin=first_sample, ymax=last_sample)
    plt.xlabel('Y')
    plt.ylabel('Sample')
    if log_x:
        plt.xscale("log", nonposx='clip')

    return fig


def plot_errors(train, test):

    sns.set_style('whitegrid')

    train['Z_Noisy'] = (train[Ycol] - train['Mean_Estimate']) / np.sqrt(train['Var_Err_Predictive'])
    train['Z_Noiseless'] = (train[Ycol] - train['Mean_Estimate']) / np.sqrt(train['Var_Err_Latent'])
    test['Z_Noisy'] = (test[Ycol] - test['Mean_Estimate']) / np.sqrt(test['Var_Err_Predictive'])
    test['Z_Noiseless'] = (test[Ycol] - test['Mean_Estimate']) / np.sqrt(test['Var_Err_Latent'])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex='col', figsize=(16,10)) # , sharex='col', sharey='row')

    ax = ax1
    ax.errorbar(x=test[Ycol], y=test['Mean_Estimate'], yerr=2*np.sqrt(test['Var_Err_Predictive']), fmt='o', ms=3, c='m', lw=0.5)
    ax.errorbar(x=train[Ycol], y=train['Mean_Estimate'], yerr=2*np.sqrt(train['Var_Err_Predictive']), fmt='o', ms=3, c='c', lw=0.5)
    ax.margins(x=0,y=0.05)
    xlim = ax.get_xlim()
    ax.plot( [xlim[0],xlim[1]], [xlim[0], xlim[1]], 'r-')

    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')

    ax.set_xlabel(Ycol)
    ax.set_ylabel('Predicted (Noisy)')

    ax = ax2
    ax.scatter(x=train['Sample'], y=train[Ycol], c='c', marker='_', s=25, alpha=1, linewidths=1, zorder=50)
    ax.scatter(x=test['Sample'], y=test[Ycol], c='m', marker='_', s=25, alpha=1, linewidths=1, zorder=50)
    ax.errorbar(x=train['Sample'], y=train['Mean_Estimate'], yerr=2*np.sqrt(train['Var_Err_Predictive']), fmt='.', ms=5, linewidth=0.25, c='k')
    ax.errorbar(x=test['Sample'], y=test['Mean_Estimate'], yerr=2*np.sqrt(test['Var_Err_Predictive']), fmt='.', ms=5, linewidth=0.25, c='k')
    ax.margins(x=0,y=0.05)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel(Ycol)
    ax.set_yscale("log", nonposy='clip')

    a=0.05
    ax = ax4
    ax.scatter(x=train['Sample'], y=train['Z_Noisy'], c='c', marker='_', alpha=0.5, linewidth=1)
    ax.scatter(x=test['Sample'], y=test['Z_Noisy'], c='m', marker='_', alpha=0.5, linewidth=1)

    ax.margins(x=0,y=0.05)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.add_patch( patches.Rectangle( (0, -2), xlim[1], 4, alpha=a, color='g' ) )
    ax.add_patch( patches.Rectangle( (0, -3), xlim[1], 3, alpha=a, color='#FFA500' ) )
    ax.add_patch( patches.Rectangle( (0, 2), xlim[1], 3, alpha=a, color='#FFA500' ) )
    ax.add_patch( patches.Rectangle( (0, ylim[0]), xlim[1], abs(ylim[0])-3, alpha=a, color='r' ) )
    ax.add_patch( patches.Rectangle( (0, 3), xlim[1], abs(ylim[1])-3, alpha=a, color='r' ) )
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Z-Score')

    ax = ax3
    ax.scatter(x=train[Ycol], y=train['Z_Noisy'], facecolor='c', marker='.', lw=1, alpha=0.5, s=50)
    ax.scatter(x=test[Ycol], y=test['Z_Noisy'], facecolor='m', marker='.', lw=1, alpha=0.5, s=50)
    ax.set_xlabel(Ycol)
    ax.set_ylabel('Z-Score')
    ax.margins(x=0,y=0.05)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.add_patch( patches.Rectangle( (0, -2), xlim[1], 4, alpha=a, color='g' ) )
    ax.add_patch( patches.Rectangle( (0, -3), xlim[1], 3, alpha=a, color='#FFA500' ) )
    ax.add_patch( patches.Rectangle( (0, 2), xlim[1], 3, alpha=a, color='#FFA500' ) )
    ax.add_patch( patches.Rectangle( (0, ylim[0]), xlim[1], abs(ylim[0])-3, alpha=a, color='r' ) )
    ax.add_patch( patches.Rectangle( (0, 3), xlim[1], abs(ylim[1])-3, alpha=a, color='r' ) )

    ax.plot( [reference_value, reference_value], ylim, 'r-', lw=2)

    plt.tight_layout()

    return fig


train['Implausibility'] = \
            abs( train['Mean_Estimate'] - reference_value ) / \
            np.sqrt(train['Var_Err_Predictive'] + discrepancy_var)
train['Implausible'] = train[ 'Implausibility' ] > implausibility_threshold

test['Implausibility'] = \
            abs( test['Mean_Estimate'] - reference_value ) / \
            np.sqrt(test['Var_Err_Predictive'] + discrepancy_var)
test['Implausible'] = test[ 'Implausibility' ] > implausibility_threshold

if False:
    fig = joint_plot(train, train_mean);    fig.savefig( os.path.join(jointdir, 'train.pdf') ); plt.close(fig)
    fig = joint_plot(test, test_mean);      fig.savefig( os.path.join(jointdir, 'test.pdf') );  plt.close(fig)

    fig = joint_plot(train, train_mean, log_x=True);    fig.savefig( os.path.join(jointdir, 'train_log.pdf') ); plt.close(fig)
    fig = joint_plot(test, test_mean, log_x=True);      fig.savefig( os.path.join(jointdir, 'test_log.pdf') );  plt.close(fig)

    fig = plot_errors(train_mean.reset_index(), test_mean.reset_index()); fig.savefig( os.path.join(jointdir, 'errors.pdf') );  plt.close(fig)

###############################################################################
print "OKAY, TIME TO CUT", '#'*65
###############################################################################

# ADD TEST DATA TO GPR TRAINING
gpr_model.set_training_data(pd.concat([train, test]))
gpr_model.save(gpr_model_with_test_fn)

def plot_implausibility(data, column, thresh, save_fn=None):
    scaled = data[column] / data[column].max()
    good = data[column] < thresh
    D = len(Xcols)
    fig = plt.figure(figsize=(128,128))
    for row in range(D):
        for col in range(D):
            if col > row:
                gs = gridspec.GridSpec(D-1, D-1)
                ax = fig.add_subplot(gs[col-1,row])
                #x = data[ Xcols[row] ]; y = data[ Xcols[col] ]
                #plt.scatter(x, y, s=np.maximum(5, 50*scaled), c=scaled, cmap='jet', lw=0) #, s=area, c=colors, alpha=0.5)
                xg = data.loc[good, Xcols[row]]; yg = data.loc[good, Xcols[col]]; sg = scaled[good]
                plt.scatter(xg, yg, s=np.maximum(3, 20*sg), lw=0, c='g', alpha=0.5) #, facecolors='none', edgecolors='g'
                xb = data.loc[good==False, Xcols[row]]; yb = data.loc[good==False, Xcols[col]]; sb = scaled[good==False]
                plt.scatter(xb, yb, s=np.maximum(3, 20*sb), lw=0, c='r', alpha=0.5) #, facecolors='none', edgecolors='r'
                plt.autoscale(tight=True)
                if col == D-1:
                    plt.xlabel( Xcols[row] )
                if row == 0:
                    plt.ylabel( Xcols[col] )
    plt.tight_layout()
    if save_fn is not None:
        print 'Saving figure to %s' % save_fn
        plt.savefig(save_fn)
    return fig

def plot_implausibility_by_iter(data, save_fn=None):
    fig = plt.figure(figsize=(20,20))

    for it in range(iteration+2):
        col_first_only = 'c'
        col_second_only = 'y'
        col_neither = 'r'
        col_both = 'g'

        first_only = ~data['Implausible_0'] & data['Implausible_1']
        second_only = data['Implausible_0'] & ~data['Implausible_1']
        neither = data['Implausible_0'] & data['Implausible_1']
        both = ~data['Implausible_0'] & ~data['Implausible_1']

        size = 10

        D = len(Xcols)
        for row in range(D):
            for col in range(D):
                if col > row:
                    gs = gridspec.GridSpec(D-1, D-1)
                    ax = fig.add_subplot(gs[col-1,row])

                    x = data.loc[first_only, Xcols[row]]; y = data.loc[first_only, Xcols[col]];
                    h1 = plt.scatter(x, y, s=size, lw=0, c=col_first_only, alpha=0.5) #, facecolors='none', edgecolors='g'

                    x = data.loc[second_only, Xcols[row]]; y = data.loc[second_only, Xcols[col]];
                    h2 = plt.scatter(x, y, s=size, lw=0, c=col_second_only, alpha=0.5) #, facecolors='none', edgecolors='g'

                    x = data.loc[neither, Xcols[row]]; y = data.loc[neither, Xcols[col]];
                    h3 = plt.scatter(x, y, s=size, lw=0, c=col_neither, alpha=0.5) #, facecolors='none', edgecolors='g'

                    x = data.loc[both, Xcols[row]]; y = data.loc[both, Xcols[col]];
                    h4 = plt.scatter(x, y, s=size, lw=0, c=col_both, alpha=0.5) #, facecolors='none', edgecolors='g'

                    plt.autoscale(tight=True)
                    if col == D-1:
                        plt.xlabel( Xcols[row] )
                    if row == 0:
                        plt.ylabel( Xcols[col] )

    plt.figlegend((h1,h2,h3,h4), ('first only', 'second only', 'neither', 'both'), 'upper right', fontsize=16)

    plt.tight_layout()

    if save_fn is not None:
        print 'Saving figure to %s' % save_fn
        plt.savefig(save_fn)
    return fig


def histogram_implausibility(data, column, thresh=None, save_fn=None):
    fig, ax = plt.subplots()
    sns.distplot( data[column], rug=True, ax = ax)
    yl = ax.get_ylim()
    if thresh is not None:
        plt.plot([thresh,thresh], yl, 'r-')
    if save_fn is not None:
        print 'Saving figure to %s' % save_fn
        plt.savefig(save_fn)


print "Looking for ", reference_value

num_desired_candidates = 5000
num_good_candidates = 0
candidates = pd.DataFrame( columns=Xcols )

search_step = 0

glm_all = {iteration: glm_model}
gpr_all = {iteration: gpr_model}
for it in range(iteration): # Loop over previous iterations
    iterdir = os.path.join('..', 'iter%d'%it, 'BHM')
    glm_all[it] = GLM.from_config(os.path.join(iterdir, 'GLM', 'model.json'), os.path.join(iterdir, 'GLM', 'params.p'))
    gpr_all[it] = GPR.from_config(os.path.join(iterdir, 'GPR', 'model.json'))

print 'Looking for candidates, step', search_step; search_step+=1
while num_good_candidates < num_desired_candidates:
    print '-'*80
    # Likeliy need to over-sample, this could take a long time if p is low
    # Min here to avoid running out of GPU ram!
    lhs_sample = lhs( len(Xcols_all), samples=min(2500, num_desired_candidates-num_good_candidates))

    # WARNING: HARD CODING SCALE!!!!!!!!
    # ALL PARAMS [0,1] EXCEPT LAST (LOG Environmental Peak Multiplier)
    for i, xc in enumerate(Xcols_all):
        v = params.loc[xc]
        lhs_sample[:, i] = (v['Max'] - v['Min']) * lhs_sample[:, i] + (v['Min'])

    new_candidates = pd.DataFrame( lhs_sample, columns=Xcols_all) # Note _orig :)
    new_candidates['Implausible'] = False

    for it in range(iteration+1):
        print('Processing iteration %d'%it)
        new_candidates['Yglm'] = glm_model.evaluate(new_candidates)
        ret = gpr_all[it].evaluate(new_candidates)
        new_candidates['Mean_Estimate'] = new_candidates['Yglm'] + ret['Mean']
        new_candidates['Var_Predictive'] = ret['Var_Predictive']

        new_candidates[ 'Implausibility_%d'%it ] = \
            abs( new_candidates['Mean_Estimate'] - reference_value ) / \
            np.sqrt(new_candidates['Var_Predictive'] + discrepancy_var)

        new_candidates[ 'Implausible_%d'%it ] = new_candidates[ 'Implausibility_%d'%it ] > implausibility_threshold
        new_candidates['Implausible'] |= new_candidates[ 'Implausible_%d'%it ]

    candidates = candidates.append(new_candidates)
    num_new_good_candidates = sum(new_candidates['Implausible'] == False)
    num_good_candidates += num_new_good_candidates

    #print new_candidates
    del new_candidates

    print 'Candidates: New = %d, Tot = %d' % (num_new_good_candidates, num_good_candidates)

# Put back orig parameter names
candidates.rename(columns={new:orig for (new,orig) in zip(Xcols_all, Xcols_all_orig)}, inplace=True)

print 'Rejected %.1f%%' % (100 * sum(candidates['Implausible']) / float(candidates.shape[0]))

non_implausible_candidates = candidates.loc[ candidates['Implausible'] == False, :]
writer = pd.ExcelWriter('Candidates_for_iter%d.xlsx'%(iteration+1))
non_implausible_candidates[Xcols_all_orig].to_excel(writer, sheet_name='Values', index=False)
non_implausible_candidates.set_index(Xcols_all_orig).to_excel(writer, sheet_name='NonImplausible')
candidates.set_index(Xcols_all_orig).to_excel(writer, sheet_name='All')

store = pd.HDFStore('candidates_for_iter%d.h5'%(iteration+1))
store['candidates'] = candidates
store.close()

exit()
for it in range(iteration+1):
    candidates['Log_Implausibility_%d'%it] = np.log(candidates['Implausibility_%d'%it])
    implausibility_hist_fn = os.path.join(bhmdir, 'Implausibility_Histogram_%d.pdf'%it)
    histogram_implausibility(candidates, 'Implausibility_%d'%it, thresh=implausibility_threshold, save_fn=implausibility_hist_fn)

    implausibility_fn = os.path.join(bhmdir, 'Log_Implausibility_%d.pdf'%it)
    plot_implausibility(candidates, 'Log_Implausibility_%d'%it, thresh=np.log(implausibility_threshold), save_fn=implausibility_fn)

#implausibility_by_iter_fn = os.path.join(bhmdir, 'Implausibility_by_Iter.pdf')
#plot_implausibility_by_iter(candidates, save_fn=implausibility_by_iter_fn)
'''
        candidates['Log_Implausibility_%d'%it] = np.log(candidates['Implausibility_%d'%it])

        implausibility_fn = os.path.join(bhmdir, 'Implausibility_%d.pdf'%it)
        plot_implausibility(candidates, 'Implausibility_%d'%it, thresh=implausibility_threshold, save_fn = implausibility_fn)

        implausibility_hist_fn = os.path.join(bhmdir, 'Implausibility_Histogram_%d.pdf'%it)
        histogram_implausibility(candidates, 'Log_Implausibility_%d'%it, thresh=np.log(implausibility_threshold), save_fn = implausibility_hist_fn)
'''

#plt.show()

