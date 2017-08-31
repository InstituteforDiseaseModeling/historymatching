import os
import sys
import pandas as pd
import numpy as np
#from ../../../newlib.history_matching.newlib.quick_read import quick_read

if len(sys.argv) != 4:
    raise Exception('Usage: generate_results.py <case> <iteration_number> <data_source_to_make>')

case_directory = str(sys.argv[1].strip())
iteration_number = int(sys.argv[2])
iteration_directory = os.path.join(case_directory, 'iter%d' % iteration_number)
data_source = str(sys.argv[3].strip())
data_source_directory = os.path.join(iteration_directory, 'Data', data_source)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','newlib'))
from quick_read import quick_read
# data_source_directory = os.path.join('Data', 'Data_%s'%time.strftime("%Y%m%d_%H%M%S"))

param_file = os.path.join(case_directory, 'Params.xlsx')
params = quick_read(param_file, 'Params').set_index('Name')

def choose_samples(samples_filename):
    samples = pd.read_excel(samples_filename, sheetname='Values')
    samples.index.name = 'id'
    return samples

samples_filename = os.path.join(data_source_directory, 'Samples.xlsx')
samples = choose_samples(samples_filename)

# 2-norm with added random noise for this example.  Just one "replicate" per sample here.
mu=0; sigma=1
results = np.sqrt(np.square(samples).sum(axis=1)).to_frame(name='Sim_Result')
results['Sim_Result'] += np.random.normal(mu, sigma, results.shape[0])

# Every Sample must have a unique simulation id to differentiate it from other replicates, if replicates are used
results['Sim_Id'] = results.index.values

if not os.path.exists(data_source_directory):
    os.makedirs(data_source_directory)

#writer = pd.ExcelWriter( os.path.join(experiment_name,'Samples.xlsx') )
#samples.to_excel(writer, sheet_name='Values')
#params.to_excel(writer, sheet_name='Params')
#writer.save()
from shutil import copyfile
#copyfile('Samples.xlsx', os.path.join(data_source_directory, 'Samples.xlsx'))

results.to_excel(os.path.join(data_source_directory, 'Results.xlsx'), sheet_name='Values')

