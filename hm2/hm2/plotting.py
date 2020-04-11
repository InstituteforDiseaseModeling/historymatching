import random

import matplotlib.pyplot as plt
import plotnine as pn


#TODO: Check if works
class WrappedFigure:
    """Class for repeatedly displaying a figure with the `print` command"""
    def __init__(self, fig):
        self.fig=fig
    def __repr__(self):
        """Called if class instance is typed in REPL"""
        return self.fig
    def __str__(self):
        """Called with `print`; displays the figure"""
        # create a dummy figure and use its manager to display "fig"
        dummy = plt.figure()
        new_manager = dummy.canvas.manager
        new_manager.canvas.figure = self.fig
        self.fig.set_canvas(new_manager.canvas)
        plt.show()


def plot_data_multi(train_x, train_y, figsize=None, log_scale=False, cmap='viridis', alpha=0.5):
    """Generates many pair-wise scatter plots of the training data.

    Args:
        log_scale (bool): Transforms size and color using log(10 * normalized_y_value + 1)

    Returns: 
        dict: A dictionary of matplotlib figure handles with keys indicating
        the parameter names via the filename which would be used to save the
        figure.
    """
    C = len(train_x.columns)

    plots = dict()
    collective_fig, collective_ax = plt.subplots(nrows=C, ncols=C)

    for rowi, row in enumerate(train_x):
        for coli, col in enumerate(train_x):
            fn = (row,col)
            x  = train_x[row]
            y  = train_x[col]

            #TODO: Use plotnine
            fig, ax = plt.subplots()
            ax.scatter(x, y, c=train_y, cmap=cmap, alpha=alpha)
            ax.set_xlabel(row)
            ax.set_ylabel(col)
            fig.tight_layout()
            plots[fn] = fig

            collective_ax[rowi,coli].scatter(x, y, c=train_y, cmap=cmap, alpha=alpha)
            collective_ax[rowi,coli].set_xlabel(row)
            collective_ax[rowi,coli].set_ylabel(col)

            #TODO
            # if circle_points.shape[0] > 0: 
            #     for _, pt in cp_dmat.iterrows():
            #         plt.scatter(pt[row], pt[col], s=50, c='k', alpha=1, linewidths=2.0, marker='x') #, s=area, c=colors, alpha=0.5)

    plots['all'] = collective_fig

    plots = {k:WrappedFigure(v) for k,v in plots.items()}

    return plots


def plot_runs_time_series(runs, param_id=None, samples=None):
    """Plots all the observations from a model in time series graphs.

    Args:
        runs (list): A list of :ref:`TimeSimFrame`.
        param_id (int): Filter to this param_id. `None` implies no filtering.
        samples (int): Randomly choose this many runs to display. `None` implies all.

    Returns: 
        A plotnine image
    """
    if isinstance(runs[0], tuple):
        runs = [x[0] for x in runs]
    if samples is not None:
        runs=random.sample(runs,samples)
    for x in runs:
        ValidateTimeSimFrame(x)
    runs = pd.concat(runs, ignore_index=True)
    if param_id is not None:
        runs = runs[runs["param_id"] == param_id]
    return (
        pn.ggplot(runs, pn.aes("time", "value", group="replicate"))
        + pn.geom_line()
        + pn.facet_wrap("~observation", scales="free_y")
    )
# TODO: Include observations above
# for i,obs in observations.iterrows():
#     ax.plot(obs['Times'], obs['Prevalence'], 'ko')
#     ax.plot(
#         [obs['Times'],obs['Times']],
#         [obs['Prevalence']-2*obs['Stdev'],obs['Prevalence']+2*obs['Stdev']],
#         'k-')
