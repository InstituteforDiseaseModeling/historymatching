import os
import pandas as pd
from pyDOE import lhs
import numpy as np
import re, time

from history_matching.quick_read import quick_read

iteration = int(re.search(r'[+-]?\d+', os.getcwd()).group())
experiment_name = 'Data_%s'%time.strftime("%Y%m%d_%H%M%S")
N_samples = 25 # <-- Only applies to iteration 0

params = quick_read( os.path.join('..', 'Params.xlsx'), 'Params').set_index('Name')

def choose_samples(num_samples):
    if iteration == 0:
        N_dim = params.shape[0]
        samples = pd.DataFrame(lhs(N_dim, samples = num_samples), columns=params.index.tolist())

        for param_name in samples.columns.values:
            pmin,pmax = (params.loc[param_name,'Min'], params.loc[param_name,'Max'])
            samples[param_name] = pmin + samples[param_name]*(pmax-pmin)
        samples.index.name = 'Sample'
    else:
        samples = pd.read_excel(os.path.join('..', 'iter%d'%(iteration-1), 'Candidates_for_iter%d.xlsx'%iteration), sheetname='Values')
        samples.index.name = 'Sample'
    return samples


samples = choose_samples(N_samples)

# 2-norm with added random noise for this example.  Just one "replicate" per sample here.
mu=0; sigma=1
results = np.sqrt(np.square(samples).sum(axis=1)).to_frame(name='Sim_Result')
results['Sim_Result'] += np.random.normal(mu, sigma, results.shape[0])

# Every Sample must have a unique simulation id to differentiate it from other replicates, if replicates are used
results['Sim_Id'] = results.index.values

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

writer = pd.ExcelWriter( os.path.join(experiment_name,'Samples.xlsx') )
samples.to_excel(writer, sheet_name='Samples')
params.to_excel(writer, sheet_name='Params')
writer.save()

results.to_excel( os.path.join(experiment_name, 'Results.xlsx') )

