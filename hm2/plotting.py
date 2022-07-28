import random

import matplotlib.pyplot as plt
import plotnine as pn

from .data_validation import *



class WrappedFigure:
    """Class for repeatedly displaying a figure with the `print` command."""
    def __init__(self, fig):
        """Wrap a figure
        Args:
            fig: A figure from, e.g. `plt.subplots()`
        """
        self.fig=fig
        plt.close(fig)

    def __repr__(self):
        """Called if class instance is typed in REPL"""
        return self.fig.__repr__()

    def __str__(self):
        """Called with `print`; displays the figure"""
        # create a dummy figure and use its manager to display "fig"
        dummy = plt.figure()
        new_manager = dummy.canvas.manager
        new_manager.canvas.figure = self.fig
        self.fig.set_canvas(new_manager.canvas)
        plt.show()
        return self.__repr__()


def plot_pairwise(X, color=None, figsize=None, cmap='viridis', alpha=0.5):
    """Generates many pairwise scatter plots of the columns of X.

    Args:
        X (DataFrame): Plot each column for X against every other one
        color (array): Color for each point for X (or None)
        figsize: Size of figure; passed to PyPlot.
        cmap: Colormap to use for `color`.
        alpha (float): Value in the range [0,1] indicating transparency

    Returns:
        dict: A dictionary of matplotlib figure handles with keys indicating
        the parameter names via the filename which would be used to save the
        figure.
    """
    #Automagically remove some common columns that we don't want to plot as data
    if 'param_id' in X:
        X = X.drop(columns='param_id')

    #TODO: Add log scaling
    C = len(X.columns)

    #Stop plots from showing before user prints them
    was_interactive = plt.isinteractive()
    plt.ioff()

    #Hold individual views for close-ups
    plots = dict()
    #Hold collective view for handy group display
    collective_fig, collective_ax = plt.subplots(nrows=C, ncols=C, figsize=figsize)

    for rowi, row in enumerate(X):
        for coli, col in enumerate(X):
            fn = (row,col)
            x  = X[row]
            y  = X[col]

            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(x, y, c=color, cmap=cmap, alpha=alpha)
            ax.set_xlabel(row)
            ax.set_ylabel(col)
            fig.tight_layout()
            plt.close(fig)
            plots[fn] = fig

            collective_ax[rowi,coli].scatter(x, y, c=color, cmap=cmap, alpha=alpha)
            collective_ax[rowi,coli].set_xlabel(row)
            collective_ax[rowi,coli].set_ylabel(col)

            #TODO
            # if circle_points.shape[0] > 0:
            #     for _, pt in cp_dmat.iterrows():
            #         plt.scatter(pt[row], pt[col], s=50, c='k', alpha=1, linewidths=2.0, marker='x') #, s=area, c=colors, alpha=0.5)

    plt.close(collective_fig)
    plots['all'] = collective_fig

    #Wrap figures so we can show them repeatedly
    plots = {k:WrappedFigure(v) for k,v in plots.items()}

    if was_interactive:
        plt.ion()

    return plots


def plot_runs_time_series(runs, param_id=None, samples=None, real_observations=None):
    """Plots all the observations from a model in time series graphs.

    Args:
        runs (list): A list of :ref:`SimFrame`.
        param_id (int): Filter to this param_id. `None` implies no filtering.
        samples (int): Randomly choose this many runs to display. `None` implies all.
        observations (:ref:`ObservationsFrame`): Observations to show. Only time obserations are shown.

    Returns:
        A plotnine image
    """
    #TODO: Only show time values

    #If we have pairs of (Time,Summary), get the times
    if isinstance(runs[0], tuple):
        runs = [x[0] for x in runs]
    #Filter to a particular parameter value
    if param_id is not None:
        runs = [x for x in runs if param_id in x['param_id'].values]
    #Randomly choose runs if we have too many
    if samples is not None and len(runs)>samples:
        runs=random.sample(runs,samples)
    #Validate the runs we're gonna plot
    for x in runs:
        ValidateSimFrame(x)
    #Make a big dataframe
    runs = pd.concat(runs, ignore_index=True)
    #Make column combining parameter+replicate info for colouring each combo uniquely
    runs['param_replicate'] = runs['param_id'].astype(str) + "_" + runs['replicate'].astype(str)
    #Plot it!
    p = ( pn.ggplot(runs, pn.aes("time", "value", group="param_replicate", color="param_replicate"))
            + pn.geom_line(show_legend=False)
            + pn.facet_wrap("~observation", scales="free_y")
        )
    if real_observations is not None:
        real_observations = ValidateObservationsFrame(real_observations)
        real_observations['param_replicate'] = "NA"
        real_observations['ymin'] = real_observations['value'] - real_observations['stdev']
        real_observations['ymax'] = real_observations['value'] + real_observations['stdev']
        p += pn.geom_errorbar(
            data=real_observations,
            mapping=pn.aes(x='time', y='value', group="param_replicate", ymin='ymin', ymax='ymax'),
            color='black',
            size=2,
            show_legend=False)
    return p
