import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as patches
import seaborn as sns
from history_matching.basis import Basis
import os

def plot_implausibility(data, Xcols, column, thresh):
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
    return fig

def plot_implausibility_by_iter(data, Xcols):
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

            return fig


def joint_plot(data, data_mean, Ycol, desired_result, log_x = False):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,10), dpi=300)

    data_mean_reset = data_mean.reset_index()
    data_reset = data.reset_index()
    first_sample = data.reset_index('Sim_Id').index.unique().min()
    last_sample = data.reset_index('Sim_Id').index.unique().max()

    plt.plot( 2 * [desired_result], [first_sample, last_sample], 'y-', linewidth=0.1) # , axes=axes[0,0]

    sim_cases_range = data.reset_index().groupby('Sample')[Ycol].agg({'Min':np.min, 'Max':np.max, 'Mean':np.mean})
    if 'Yglm' in data_mean.columns:
        sim_cases_range = sim_cases_range.join(data_mean['Yglm'])
    for idx,s in sim_cases_range.iterrows():
        plt.plot( [s['Min'], s['Max']], [idx,idx], 'b-', linewidth=0.5 )
        #plt.plot( [s['Mean'], s['Fitted_Model_Mean']], [idx,idx], 'g-', linewidth=0.25 )
        if 'Yglm' in s:
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

    if 'Yglm' in data_mean_reset:
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


def plot_errors(train, test, Ycol, desired_result):

    test_impl = test.set_index('Implausible')
    train_impl = train.set_index('Implausible')

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,8), sharex='row', sharey='row')

    a=0.05
    for data, ax, label in zip( [train_impl, test_impl], [ax1,ax2], ['Train', 'Test']):
        if True in data.index:
            ax.scatter(x=data.loc[True, Ycol], y=data.loc[True, 'Z_Noisy'], marker='x', facecolor='r', lw=1, alpha=0.5, s=25)
        if False in data.index:
            ax.scatter(x=data.loc[False, Ycol], y=data.loc[False, 'Z_Noisy'], marker='.', facecolor='k', lw=1, alpha=0.5, s=25)
        ax.set_xlabel(Ycol)
        ax.set_ylabel('Z-Score')
        ax.margins(x=0,y=0.05)
        ax.set_title(label)

    fig.tight_layout()

    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    for ax in [ax1, ax2]:
        ax.add_patch( patches.Rectangle( (0, -2), xlim[1], 4, alpha=a, color='g' ) )
        ax.add_patch( patches.Rectangle( (0, -3), xlim[1], 1, alpha=a, color='#FFA500' ) )
        ax.add_patch( patches.Rectangle( (0, 2), xlim[1], 1, alpha=a, color='#FFA500' ) )
        ax.add_patch( patches.Rectangle( (0, ylim[0]), xlim[1], abs(ylim[0])-3, alpha=a, color='r' ) )
        ax.add_patch( patches.Rectangle( (0, 3), xlim[1], abs(ylim[1])-3, alpha=a, color='r' ) )

        ax.plot( [desired_result, desired_result], ylim, 'y-', lw=2)

    return fig


def plot_errors_old(train, test, Ycol, desired_result):

    train['Z_Noisy'] = (train[Ycol] - train['Mean_Estimate']) / np.sqrt(train['Var_Err_Predictive'])
    train['Z_Noiseless'] = (train[Ycol] - train['Mean_Estimate']) / np.sqrt(train['Var_Err_Latent'])
    test['Z_Noisy'] = (test[Ycol] - test['Mean_Estimate']) / np.sqrt(test['Var_Err_Predictive'])
    test['Z_Noiseless'] = (test[Ycol] - test['Mean_Estimate']) / np.sqrt(test['Var_Err_Latent'])

    test_impl = test.set_index('Implausible')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex='col', figsize=(16,10)) # , sharex='col', sharey='row')

    ax = ax1
    ax.errorbar(x=test_impl.loc[True, Ycol], y=test_impl.loc[True, 'Mean_Estimate'], yerr=2*np.sqrt(test_impl.loc[True, 'Var_Err_Predictive']), fmt='o', ms=3, c='r', lw=0.5)
    ax.errorbar(x=test_impl.loc[False, Ycol], y=test_impl.loc[False, 'Mean_Estimate'], yerr=2*np.sqrt(test_impl.loc[False, 'Var_Err_Predictive']), fmt='o', ms=3, c='m', lw=0.5)
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
    #ax.scatter(x=test[Ycol], y=test['Z_Noisy'], facecolor='m', marker='.', lw=1, alpha=0.5, s=50)
    ax.scatter(x=test_impl.loc[True, Ycol], y=test_impl.loc[True, 'Z_Noisy'], facecolor='m', marker='.', lw=1, alpha=0.5, s=50, edgecolors='r')
    ax.scatter(x=test_impl.loc[False, Ycol], y=test_impl.loc[False, 'Z_Noisy'], facecolor='m', marker='.', lw=1, alpha=0.5, s=50, edgecolors='k')
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

    ax.plot( [desired_result, desired_result], ylim, 'r-', lw=2)

    plt.tight_layout()

    return fig


def histogram_implausibility(data, column, thresh=None):
    fig, ax = plt.subplots()
    sns.distplot( data[column], rug=True, ax = ax)
    yl = ax.get_ylim()
    if thresh is not None:
        plt.plot([thresh,thresh], yl, 'r-')
    return fig


def plot_data(data, Ycol, param_info, circle_points=pd.DataFrame(), saveto_dir = None, log_scale=True, desired_result=None):

    td = data.reset_index().set_index('Implausible')
    ticks = np.array([0,0.25,0.5,0.75,1])
    if desired_result is not None:
        desired_tick = (desired_result-td[Ycol].min()) / (td[Ycol].max()-td[Ycol].min())
        ticks = np.append(ticks, desired_tick)

    scaled = (td[Ycol]-td[Ycol].min()) / (td[Ycol].max()-td[Ycol].min())
    tick_labels = ticks * (td[Ycol].max()-td[Ycol].min()) + td[Ycol].min()
    vmin = scaled.min()
    vmax = scaled.max()
    if log_scale:
        scaled = np.log( 10*scaled+1 )
        tick_labels = (np.exp(ticks)-1)/10.0 * (td[Ycol].max()-td[Ycol].min()) + td[Ycol].min()

    figs = {}

    basis = Basis.identity_basis(params=param_info.index.unique().tolist(), param_info=param_info)
    Xcols = basis.get_terms()

    reverse_param_dict = {v:k for k,v in basis.param_dict.items()}
    Xcols = [reverse_param_dict[xc] for xc in Xcols]

    for row in range(len(Xcols)):
        for col in range(len(Xcols)):
            if col > row:
                fn = '%s-%s.pdf' % (Xcols[row], Xcols[col])
                fig = plt.figure(figsize=(6,6)) #GPy.plotting.plotting_library().figure()

                for true_or_false, ec in zip([False, True], ['k', 'r']):
                    if true_or_false not in td.index:
                        continue

                    x = td.loc[ true_or_false, [Xcols[row]] ].values
                    y = td.loc[ true_or_false, [Xcols[col]] ].values
                    s = scaled.loc[ true_or_false ]
                    if not isinstance(s, np.float64): # If only one value
                        s = s.values[:,None]
                    else:
                        s = np.array(s)

                    sc = plt.scatter(x, y, s=np.maximum(1, 25*s), c=s, cmap='jet', linewidths=1, alpha=0.5, edgecolors=ec, vmin=vmin, vmax=vmax)
                    if true_or_false == False:
                        cbar = plt.colorbar(sc, ticks=ticks)
                        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        cbar.ax.set_yticklabels(tick_labels)  # vertically oriented colorbar

                if circle_points.shape[0] > 0:
                    for idx, pt in circle_points.iterrows():
                        plt.scatter(pt[ Xcols[row] ], pt[ Xcols[col] ], s=65, facecolors='none', edgecolors='k', alpha=1, linewidths=1.0, marker='s')

                plt.autoscale(tight=True)
                plt.xlabel( Xcols[row] )
                plt.ylabel( Xcols[col] )
                plt.tight_layout()
                if saveto_dir is not None:
                    fig.savefig( os.path.join(saveto_dir, fn) ); plt.close(fig)
                else:
                    figs[fn] = fig

    return figs

