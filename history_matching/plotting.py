def joint_plot(data, data_mean, log_x = False):
    fig = plt.figure(figsize=(16,32))

    data_mean_reset = data_mean.reset_index()
    data_reset = data.reset_index()
    first_sample = data.reset_index('Sim_Id').index.unique().min()
    last_sample = data.reset_index('Sim_Id').index.unique().max()

    plt.plot( 2 * [self.desired_result], [first_sample, last_sample], 'y-', linewidth=0.1) # , axes=axes[0,0]

    sim_cases_range = data.reset_index().groupby('Sample')[self.Ycol].agg({'Min':np.min, 'Max':np.max, 'Mean':np.mean})
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

    plt.scatter(data_reset.query('Implausible==False')[self.Ycol], data_reset.query('Implausible==False')['Sample'], c='k', s=10, marker='|', alpha=1, linewidth=0.1, zorder=100)
    plt.scatter(data_reset.query('Implausible==True')[self.Ycol], data_reset.query('Implausible==True')['Sample'], c='r', s=10, marker='|', alpha=1, linewidth=0.2, zorder=100)

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

    train['Z_Noisy'] = (train[self.Ycol] - train['Mean_Estimate']) / np.sqrt(train['Var_Err_Predictive'])
    train['Z_Noiseless'] = (train[self.Ycol] - train['Mean_Estimate']) / np.sqrt(train['Var_Err_Latent'])
    test['Z_Noisy'] = (test[self.Ycol] - test['Mean_Estimate']) / np.sqrt(test['Var_Err_Predictive'])
    test['Z_Noiseless'] = (test[self.Ycol] - test['Mean_Estimate']) / np.sqrt(test['Var_Err_Latent'])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex='col', figsize=(16,10)) # , sharex='col', sharey='row')

    ax = ax1
    ax.errorbar(x=test[self.Ycol], y=test['Mean_Estimate'], yerr=2*np.sqrt(test['Var_Err_Predictive']), fmt='o', ms=3, c='m', lw=0.5)
    ax.errorbar(x=train[self.Ycol], y=train['Mean_Estimate'], yerr=2*np.sqrt(train['Var_Err_Predictive']), fmt='o', ms=3, c='c', lw=0.5)
    ax.margins(x=0,y=0.05)
    xlim = ax.get_xlim()
    ax.plot( [xlim[0],xlim[1]], [xlim[0], xlim[1]], 'r-')

    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')

    ax.set_xlabel(self.Ycol)
    ax.set_ylabel('Predicted (Noisy)')

    ax = ax2
    ax.scatter(x=train['Sample'], y=train[self.Ycol], c='c', marker='_', s=25, alpha=1, linewidths=1, zorder=50)
    ax.scatter(x=test['Sample'], y=test[self.Ycol], c='m', marker='_', s=25, alpha=1, linewidths=1, zorder=50)
    ax.errorbar(x=train['Sample'], y=train['Mean_Estimate'], yerr=2*np.sqrt(train['Var_Err_Predictive']), fmt='.', ms=5, linewidth=0.25, c='k')
    ax.errorbar(x=test['Sample'], y=test['Mean_Estimate'], yerr=2*np.sqrt(test['Var_Err_Predictive']), fmt='.', ms=5, linewidth=0.25, c='k')
    ax.margins(x=0,y=0.05)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel(self.Ycol)
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
    ax.scatter(x=train[self.Ycol], y=train['Z_Noisy'], facecolor='c', marker='.', lw=1, alpha=0.5, s=50)
    ax.scatter(x=test[self.Ycol], y=test['Z_Noisy'], facecolor='m', marker='.', lw=1, alpha=0.5, s=50)
    ax.set_xlabel(self.Ycol)
    ax.set_ylabel('Z-Score')
    ax.margins(x=0,y=0.05)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.add_patch( patches.Rectangle( (0, -2), xlim[1], 4, alpha=a, color='g' ) )
    ax.add_patch( patches.Rectangle( (0, -3), xlim[1], 3, alpha=a, color='#FFA500' ) )
    ax.add_patch( patches.Rectangle( (0, 2), xlim[1], 3, alpha=a, color='#FFA500' ) )
    ax.add_patch( patches.Rectangle( (0, ylim[0]), xlim[1], abs(ylim[0])-3, alpha=a, color='r' ) )
    ax.add_patch( patches.Rectangle( (0, 3), xlim[1], abs(ylim[1])-3, alpha=a, color='r' ) )

    ax.plot( [self.desired_result, self.desired_result], ylim, 'r-', lw=2)

    plt.tight_layout()

    return fig


