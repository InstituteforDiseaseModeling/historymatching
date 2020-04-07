import matplotlib.pyplot as plt

def _plot_data_multi(train_x, train_y, figsize=None, log_scale=False, cmap='viridis', alpha=0.5):
    """Generates many pair-wise scatter plots of the training data.

    Args:
        log_scale: Transforms size and color using log(10 * normalized_y_value + 1)

    Returns: a dictionary of matplotlib figure handles with keys indicating
    the parameter names via the filename which would be used to save the
    figure.
    """
    C = len(train_x.columns)

    individual_plots = dict()
    collective_fig, collective_ax = plt.subplots(nrows=C, ncols=C)

    for rowi, row in enumerate(train_x):
        for coli, col in enumerate(train_x):
            fn = (row,col)
            x  = train_x[row]
            y  = train_x[col]

            fig, ax = plt.subplots()
            ax.scatter(x, y, c=train_y, cmap=cmap, alpha=alpha)
            ax.set_xlabel(row)
            ax.set_ylabel(col)
            fig.tight_layout()
            individual_plots[fn] = fig

            collective_ax[rowi,coli].scatter(x, y, c=train_y, cmap=cmap, alpha=alpha)
            collective_ax[rowi,coli].set_xlabel(row)
            collective_ax[rowi,coli].set_ylabel(col)

            #TODO
            # if circle_points.shape[0] > 0: 
            #     for _, pt in cp_dmat.iterrows():
            #         plt.scatter(pt[row], pt[col], s=50, c='k', alpha=1, linewidths=2.0, marker='x') #, s=area, c=colors, alpha=0.5)

    individual_plots['all'] = collective_fig

    return individual_plots
