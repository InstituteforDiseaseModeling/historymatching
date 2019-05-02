import os, copy, re
import math
import random
import pandas as pd
import numpy as np
from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from simtools.ModBuilder import ModBuilder, ModFn
from dtk.utils.builders.TemplateHelper import TemplateHelper
from dtk.utils.builders.ConfigTemplate import ConfigTemplate
from dtk.utils.builders.TaggedTemplate import CampaignTemplate, DemographicsTemplate

N_new_samples_around_each_sample = 20

sample_ids = [
    4482,
    3818,
    4624,
    3758,
    4268,
]

std_factor = 0.01

exp_id = 'primary_exp'
iteration = int(re.search(r'iter(\d+)', os.getcwd()).group(1))
N_rep_per_sample = 1
campaign_fn = 'campaign_baseline.json' # 'campaign_baseline.json' or 'campaign_more_seeding.json'

plugin_files_dir = os.path.join('..', '..', '..', 'InputFiles')
cfg = ConfigTemplate.from_file( os.path.join(plugin_files_dir, 'config.json') )
cpn = CampaignTemplate.from_file( os.path.join(plugin_files_dir, campaign_fn) )

# Set static parameters
static_params = {
    'Base_Population_Scale_Factor' : 0.03, # 0.03
    'Typhoid_Carrier_Removal_Year': 2050,
    'Inset_Chart_Reporting_Start_Year': 1900
}
# 'Inset_Chart_Reporting_Start_Year': 1900
cfg.set_params( static_params )

templates = TemplateHelper()
table_base = {
    'ACTIVE_TEMPLATES': [cfg, cpn],
    'TAGS': {'BayesianHistoryMatching':None, 'Iteration': iteration}
}

# Standard DTKConfigBuilder
config_builder = DTKConfigBuilder.from_files(
    os.path.join(plugin_files_dir, 'config.json'),
    os.path.join(plugin_files_dir, campaign_fn)
)

def map_sample_to_model_input(config_builder, idx, replicate_idx, sample):
    table = copy.deepcopy(table_base)
    table['TAGS'].update({
        '__sample_index__': idx[0],
        '__parent_index__': idx[1],
        '__variate_index__': idx[2],
        '__replicate_index__': replicate_idx
    })
    table['Run_Number'] = random.randint(0, 1e6)

    if 'LOG Acute Infectiousness' in sample:
        value = sample.pop('LOG Acute Infectiousness')
        table['Typhoid_Acute_Infectiousness'] = math.exp(value)

    if 'LOG Contact Exposure Period' in sample:
        value = sample.pop('LOG Contact Exposure Period')
        table['Typhoid_Contact_Exposure_Rate'] = 1.0/math.exp(value)

    if 'LOG Environmental Exposure Period' in sample:
        value = sample.pop('LOG Environmental Exposure Period')
        table['Typhoid_Environmental_Exposure_Rate'] = 1.0/math.exp(value)

    if 'LOG Node Contagion Decay Period' in sample:
        value = sample.pop('LOG Node Contagion Decay Period')
        table['Node_Contagion_Decay_Rate'] = 1.0/math.exp(value)

    if 'Exposure Age Median' in sample:
        value = sample.pop('Exposure Age Median')
        table['Typhoid_Exposure_Lambda'] = 20.0/value - 2.0

    if 'Carrier Probability' in sample:
        value = sample.pop('Carrier Probability')
        table['Typhoid_Carrier_Probability_Male'] = value
        table['Typhoid_Carrier_Probability_Female'] = value

    for param_name,p in params.iterrows():
        if param_name in sample and 'MapTo' in p:
            if isinstance(p['MapTo'],float) and math.isnan(p['MapTo']):
                continue
            table[p['MapTo']] = sample.pop( param_name )

    for name,value in sample.iteritems():
        print 'UNUSED PARAMETER:', name
    assert( len(sample) == 0 ) # All params used

    return templates.mod_dynamic_parameters(config_builder, table)

def sample_around(sample, cov_mat):
    variates_near_sample = pd.DataFrame(np.random.multivariate_normal(mean=sample.values[0], cov=cov_mat, size=N_new_samples_around_each_sample), columns=sample.columns)

    variates_near_sample.index.name = 'Variate'

    return variates_near_sample

def choose_samples():
    params = pd.read_excel(os.path.join('..', 'Params.xlsx'), sheetname='Params').set_index('Name')
    samples = pd.read_excel(os.path.join(exp_id, 'Samples.xlsx'), sheetname='Samples')
    samples.set_index('Sample', inplace=True)
    sample_ids.sort()

    params['Stdev'] = std_factor * (params['Max'] - params['Min'])
    params['Var'] = params['Stdev']**2
    N = params.shape[0]
    cov_mat = np.diag( params.loc[samples.columns, 'Var'] )

    center_points = samples.loc[sample_ids]

    from functools import partial
    samples_to_run = center_points.groupby(level='Sample').apply(partial(sample_around, cov_mat=cov_mat))

    # Constrain:
    for param_name in samples_to_run.columns:
        param = params.loc[param_name]
        samples_to_run.loc[ samples_to_run[param_name] < param['Min'], param_name ] = param['Min']
        samples_to_run.loc[ samples_to_run[param_name] > param['Max'], param_name ] = param['Max']

    samples_to_run.reset_index(inplace=True)
    samples_to_run.rename(columns={'Sample': 'Parent'}, inplace=True)
    samples_to_run.index.name = 'Sample'
    samples_to_run = samples_to_run.reset_index().set_index(['Sample', 'Parent', 'Variate'])

    N_samples = samples_to_run.shape[0]
    ret = raw_input('About to commission %d simulations (%d samples x %d reps per sample).  Okay?  y/N: ' % (N_samples*N_rep_per_sample, N_samples, N_rep_per_sample))
    if ret.lower() != 'y':
        exit()

    return (samples_to_run, params)

(samples, params) = choose_samples()

samples.to_excel('Samples_Around.xlsx', sheet_name='Samples', merge_cells=False)

exp_builder = ModBuilder.from_combos(
    [
        ModFn(map_sample_to_model_input,
            sample[0],  # <-- sample index, parent, variate
            rep,        # <-- replicate index
            {k:v for k,v in zip(samples.columns.values, sample[1:])})
        for sample in samples.itertuples() for rep in range(N_rep_per_sample)
    ])

run_sim_args =  {'config_builder': config_builder,
                 'exp_builder': exp_builder,
                 'exp_name': 'Typhoid BHMv4 SampleAround Iter%d'%iteration}

