import sys, os
import seaborn as sns
import numpy as np
import pandas as pd
from history_matching import quick_read
import matplotlib.pyplot as plt
from matplotlib import collections as mc

from scipy.stats import beta
from functools import partial
from pandas.tools.plotting import parallel_coordinates

if len(sys.argv) < 2:
    print('Usage: plot_prevalence.py path/to/Results1.csv path/to/Results1.csv ...')
    exit()

cmap=plt.get_cmap('Set1')

reference = quick_read( os.path.join('..', '..', '..', 'Data', 'CountryData.xlsx'), 'Prevalence' )

def make_collection_distribution(d, col):
    return list(zip(d['Timestep'], d[col]))

fig, ax = plt.subplots( 1, 1, figsize=(10,4) )

for fni, fn in enumerate(sys.argv[1:]):
    results = pd.read_csv(fn, skipinitialspace=True) \
        [['Sim_Id', 'Timestep', 'Prevalence']]

    N = len(results['Sim_Id'].unique())

    alpha = 0.1 + 0.9 * np.exp(-0.01*N) # 0.04

    lc = mc.LineCollection( results.groupby('Sim_Id').apply(
        partial(make_collection_distribution, col='Prevalence')
    ), linewidths=0.5, edgecolor=cmap(fni), label='Iteration %d'%fni)
    lc.set_alpha(alpha)
    ax.add_collection(lc)

    '''
    sns.tsplot(data=data, time='Year', value='Prevalence', unit='Sim_Id', err_style=['ci_band'], ax=ax)
    '''

    ax.plot(reference['Timestep'], reference['Prevalence'], marker='.', color='k', ms=15, label='Data')

    for idx, year_row in reference.iterrows():
        p = year_row['Prevalence']
        std = year_row['Stdev']
        low_high = [p-2*std, p+2*std]
        ax.plot([year_row['Timestep'], year_row['Timestep']], low_high, color='k')

    ax.autoscale()
    ax.margins(x=0, y=0.1)
    ax.set_ylim(ymin=0)

    ax.set_ylabel('Prevalence')
    ax.set_xlabel('Timestep')

plt.legend()
fig.tight_layout()

fig.savefig('Prevalence.png')

#plt.show()
#plt.close(fig)

