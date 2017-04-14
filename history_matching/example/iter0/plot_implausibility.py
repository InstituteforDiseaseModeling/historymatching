import os
import copy
import pandas as pd
import numpy as np
from history_matching import quick_read
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm

iteration = 0
res = 5
sns.set_style("whitegrid", {'axes.grid' : False})

params_file = os.path.join('..', 'Params.xlsx')
param_info = quick_read(params_file, 'Params').set_index('Name')
Xcols = param_info.index.tolist()

candidates = quick_read('Candidates_for_iter%d.xlsx'%(iteration+1), 'All')

gs = gridspec.GridSpec(len(Xcols), len(Xcols))

cuts_dir = os.path.join('..', 'iter%d'%iteration, 'Cuts')

for cut_name in [name for name in os.listdir(cuts_dir) if os.path.isdir(os.path.join(cuts_dir, name))]:
    print('Working on cut %s'%cut_name)

    implausible_col = 'Implausible_%d_%s'%(iteration, cut_name)
    implausibility_col = 'Implausibility_%d_%s'%(iteration, cut_name)

    fig = plt.figure(figsize=(6,6))
    for row in range(len(Xcols)):
        for col in range(len(Xcols)):
            if col == row:
                ax = plt.subplot(gs[row, col])
                ax.text(0.5,0.5,Xcols[row].replace(' ', '\n'), rotation=45, ha='center', va='center', fontsize=12)
                ax.set_axis_off()
            elif col > row:
                xc = Xcols[row]
                yc = Xcols[col]

                cc = candidates[[xc,yc,implausible_col,implausibility_col]]

                bins = []
                centers = []
                for var in [xc,yc]:
                    binname = 'Binned %s'%var
                    bins.append(binname)
                    edges = np.linspace(param_info.loc[var,'Min'], param_info.loc[var,'Max'], res+1)
                    centers.append( [(a+b)/2. for a,b in zip(edges[:-1], edges[1:])] )
                    cc[binname] = pd.cut(cc[var], edges, labels=centers[-1], include_lowest=True, right=False)

                all_inds = pd.DataFrame(index=pd.MultiIndex.from_tuples([(x, y) for x in centers[0] for y in centers[1]]))
                #all_inds['Junk'] = 0
                all_inds.index.rename(bins, level=[0,1], inplace=True)

                cxy_gb = cc.groupby(bins)

                p = 1 - cxy_gb[implausible_col].sum() / cxy_gb[implausible_col].count()
                impl_frac = np.log10(p)#/(1-p))
                #impl_frac.name = 'Non-Implausibility Fraction'
                impl_frac = all_inds.merge(impl_frac.to_frame(), left_index=True, right_index=True, how='left')
                Z = impl_frac.values.reshape((res,res))

                ax = plt.subplot(gs[row, col])
                masked_array = np.ma.array(Z, mask=np.isnan(Z))
                cmap = copy.copy(cm.cool_r)
                cmap.set_bad('gray',1.) # color, alpha
                # nearest, bicubic
                ax.imshow(masked_array, interpolation='nearest', cmap=cmap, aspect='equal', origin='lower') # , vmin=0, vmax=1
                ax.set_axis_off()

                impl_min = cxy_gb[implausibility_col].min()
                #impl_min.name = 'Min Implausibility'
                impl_min = all_inds.merge(impl_min.to_frame(), left_index=True, right_index=True, how='left')
                Z = impl_min.values.reshape((res,res))#.transpose()

                ax = plt.subplot(gs[col, row])
                masked_array = np.ma.array(Z, mask=np.isnan(Z))
                cmap = copy.copy(cm.autumn)
                cmap.set_bad('gray',1.) # color, alpha
                # nearest, bicubic
                ax.imshow(masked_array, interpolation='nearest', cmap=cmap, aspect='equal', vmin=0, vmax=1, origin='lower')
                ax.set_axis_off()

    fig.savefig('Implausibility_%s.pdf'%cut_name)
    plt.close(fig)
