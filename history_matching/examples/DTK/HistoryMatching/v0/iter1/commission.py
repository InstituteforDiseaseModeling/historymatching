import os, copy, re
import math
import random
import pandas as pd
from pyDOE import lhs
import numpy as np

from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from simtools.ModBuilder import ModBuilder, ModFn
from dtk.utils.builders.TemplateHelper import TemplateHelper
from dtk.utils.builders.ConfigTemplate import ConfigTemplate
from dtk.utils.builders.TaggedTemplate import CampaignTemplate, DemographicsTemplate
from history_matching import quick_read

iteration = int(re.search(r'iter(\d+)', os.getcwd()).group(1))
N_rep_per_sample = 1
N_samples = 50

params = quick_read( os.path.join('..', 'Params.xlsx'), 'Params').set_index('Name')

plugin_files_dir = os.path.join('..', 'InputFiles')
cfg = ConfigTemplate.from_file( os.path.join(plugin_files_dir, 'config.json') )
cpn = CampaignTemplate.from_file( os.path.join(plugin_files_dir, 'campaign.json') )

# Set static parameters
static_params = {
    'Enable_Aging' : 1, # 0
    'Enable_Demographics_Birth': 0
}
cfg.set_params( static_params )

templates = TemplateHelper()
table_base = {
    'ACTIVE_TEMPLATES': [cfg, cpn],
    'TAGS': {'BayesianHistoryMatching':None, 'Iteration': iteration}
}

# Standard DTKConfigBuilder
config_builder = DTKConfigBuilder.from_files(
    os.path.join(plugin_files_dir, 'config.json'),
    os.path.join(plugin_files_dir, 'campaign.json')
)

def map_sample_to_model_input(config_builder, sample_idx, replicate_idx, sample):
    table = copy.deepcopy(table_base)
    table['TAGS'].update({'__sample_index__': sample_idx, '__replicate_index__': replicate_idx})
    table['Run_Number'] = random.randint(0, 1e6)

    if 'LOG Base Infectivity' in sample:
        value = sample.pop('LOG Base Infectivity')
        table['Base_Infectivity'] = math.exp(value)

    for param_name,p in params.iterrows():
        if param_name in sample and 'MapTo' in p:
            if isinstance(p['MapTo'],float) and math.isnan(p['MapTo']):
                continue
            table[p['MapTo']] = sample.pop( param_name )

    for name,value in sample.items():
        print('UNUSED PARAMETER:', name)
    assert( len(sample) == 0 ) # All params used

    return templates.mod_dynamic_parameters(config_builder, table)

def choose_and_scale_samples_unconstrained(num_samples):
    N_dim = params.shape[0]
    samples = pd.DataFrame(lhs(N_dim, samples = num_samples), columns=params.index.tolist())

    for param_name in samples.columns.values:
        pmin,pmax = (params.loc[param_name,'Min'], params.loc[param_name,'Max'])
        samples[param_name] = pmin + samples[param_name]*(pmax-pmin)
    samples.index.name = 'Sample'
    return samples


def choose_and_scale_samples(num_samples):
    if iteration == 0:
        samples_unconstrained = choose_and_scale_samples_unconstrained(num_samples)
        # Here is an example of how to impose a simple constraint:
        #samples_unconstrained['Days'] = samples_unconstrained[['Env Ramp Up', 'Env Ramp Down', 'Env Cutoff']].sum(axis=1)
        #samples = samples_unconstrained.loc[ samples_unconstrained['Days'] < 365, :]
        samples = samples_unconstrained.copy()

        remaining = num_samples - samples.shape[0]
        while remaining > 0:
            assert(False) # Should never get here in this example because there is no constraint!
            samples_unconstrained = choose_and_scale_samples_unconstrained(num_samples)
            # Here is an example of how to impose a simple constraint:
            samples_unconstrained['Days'] = samples_unconstrained[['Env Ramp Up', 'Env Ramp Down', 'Env Cutoff']].sum(axis=1)
            samples = pd.concat([samples, samples_unconstrained.loc[ samples_unconstrained['Days'] < 365, :] ], ignore_index=True)
            remaining = num_samples - samples.shape[0]

        samples.index.name = 'Sample'

        #return samples.iloc[:num_samples,:].drop('Days', axis=1) # <-- Use this with the constraint to remove 'Days'
        return samples.iloc[:num_samples,:]
    else:
        #samples = pd.read_excel(os.path.join('..', 'iter%d'%(iteration-1), 'Candidates_for_iter%d.xlsx'%iteration), sheetname='Values')
        samples = pd.read_hdf(os.path.join('..', 'iter%d'%(iteration-1), 'Candidates_for_iter%d.hd5'%iteration), key='values')

        samples.index.name = 'Sample'
        #assert( pd.Series.all( samples[['Env Ramp Up', 'Env Ramp Down', 'Env Cutoff']].sum(axis=1) < 365 ))
        return samples.iloc[:num_samples,:]


if os.path.exists('Samples.xlsx'):
    samples = pd.read_excel('Samples.xlsx', sheet_name='Samples') \
        .set_index('Sample')
else:
    samples = choose_and_scale_samples(N_samples)

writer = pd.ExcelWriter('Samples.xlsx')
samples.to_excel(writer, sheet_name='Samples')
params.to_excel(writer, sheet_name='Params')
writer.save()

exp_builder = ModBuilder.from_combos(
    [
        ModFn(map_sample_to_model_input,
            sample[0],  # <-- sample index
            rep,        # <-- replicate index
            {k:v for k,v in zip(samples.columns.values, sample[1:])})
        for sample in samples.itertuples() for rep in range(N_rep_per_sample)
    ])

run_sim_args =  {'config_builder': config_builder,
                 'exp_builder': exp_builder,
                 'exp_name': 'Generic BHM Demo v0 Iter%d'%iteration}
