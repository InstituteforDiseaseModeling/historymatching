"""Library of functions to analyze and select features (i.e., summary statistics) to be used in history matching iterations. This library includes a subset of functions that compute summary statistics from time series."""
import inspect
import warnings
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import scipy.stats




class Diagnostics:
    """
    Data analysis utilities. Methods in this class follow a model y=f(x), where x
    are the inputs to a (black-box) model and y are the outputs of said model.

    Attributes:
        x : Predictor data. Pandas dataframe with columns representing independent
            data or inputs to a model whose output is y.
        y : Response data. Pandas dataframe with columns representing features or 
            output data that is dependent on the entries in x. 
    """

    
    def __init__(self, x: Optional[pd.DataFrame] = None, y: Optional[pd.DataFrame] = None):
        """Initialize the data diagnostics class.

        Args:
            x : Predictor data. Pandas dataframe with columns representing independent
                data or inputs to a model whose output is y.
            y : Response data. Pandas dataframe with columns representing features or 
                output data that is dependent on the entries in x. 

        Returns:
            None
        """
        self.x = x
        self.y = y

        return

    
    def interactive(self):
        """Generate interactive plots in a Jupyter Notebook.
        """

        # Don't do anything if _not_ running in an interactive environment
        if not self._is_jupyter_notebook():
            warnings.warn( 'Diagnostics.interactive() only runs when called within a notebook.' )
            return
        
        # Library imports and initialization
        import bokeh
        from bokeh import io
        from bokeh import plotting
        from bokeh import models
        from bokeh import layouts
        bokeh.io.output_notebook()    # Output to notebook ---we could also just open a new browser, hence relaxing 
                                      # the notebook condition.

        # Convert data to dictionaries for use in CustomJS
        x_dict = self.x.to_dict('list')
        y_dict = self.y.to_dict('list')

        # Generate summary statistics for all columns in self.y
        summary_stats_raw = self.y.describe()
        summary_stats = pd.concat( [ pd.DataFrame( [ summary_stats_raw.columns], 
                                                     columns=summary_stats_raw.columns, 
                                                     index=['name'] ),
                                     summary_stats_raw,
                                     pd.DataFrame( [self.y.var()], columns=self.y.columns, index=['variance'] )
                                    ] 
                                  )
        summary_stats['index'] = summary_stats.index

        # Select data to plot
        x1_select = bokeh.models.Select( title='Predictor A', value=self.x.columns[0], options=list(self.x.columns) )
        y1_select = bokeh.models.Select( title='Response A' , value=self.y.columns[0], options=list(self.y.columns) )
        controls_1 = bokeh.layouts.row( x1_select, y1_select )

        x2_select = bokeh.models.Select( title='Predictor B', value=self.x.columns[0], options=list(self.x.columns) )
        y2_select = bokeh.models.Select( title='Response B' , value=self.y.columns[0], options=list(self.y.columns) )
        controls_2 = bokeh.layouts.row( x2_select, y2_select )

        # Plot first pair of predictor and response
        source_1 = bokeh.models.ColumnDataSource( data = dict( x=x_dict[ self.x.columns[0] ],
                                                               y=y_dict[ self.y.columns[0] ]  )
                                                 )
        plot_title_1 = f'{y1_select.value} vs {x1_select.value}' 
        plot_1 = bokeh.plotting.figure( title=plot_title_1, height=300, width=500 )
        plot_1.xaxis.axis_label = 'Predictor A'
        plot_1.yaxis.axis_label = 'Response A'
        plot_1.scatter( 'x', 'y', source=source_1 )

        # Display summary statistics of first response selection
        table_formatter = bokeh.models.HTMLTemplateFormatter( template='<div style="color: black;"><%= value %></div>' )
        summary_source_1 = models.ColumnDataSource( data=dict( index = summary_stats['index'].to_list(),    
                                                               Value = summary_stats[y1_select.value].to_list() )
                                                  )
        columns_1 = [ models.TableColumn(field='index', title='Statistic', formatter=table_formatter),
                      models.TableColumn(field='Value', title='Value', formatter=table_formatter)
                    ]
        data_table_1 = models.DataTable(source=summary_source_1, columns=columns_1, width=400, height=280)

        # Plot second pair of predictor and response
        source_2 = bokeh.models.ColumnDataSource( data = dict( x=x_dict[ self.x.columns[0] ],
                                                               y=y_dict[ self.y.columns[0] ]  )
                                                 )
        plot_title_2 = f'{y2_select.value} vs {x2_select.value}' 
        plot_2 = bokeh.plotting.figure( title=plot_title_2, height=300, width=500 )
        plot_2.xaxis.axis_label = 'Predictor B'
        plot_2.yaxis.axis_label = 'Response B'
        plot_2.scatter( 'x', 'y', source=source_2 )

        # Display summary statistics of second response selection
        summary_source_2 = models.ColumnDataSource( data=dict( index = summary_stats['index'].to_list(),    
                                                               Value = summary_stats[y2_select.value].to_list() )
                                                  )
        columns_2 = [ models.TableColumn(field='index', title='Statistic', formatter=table_formatter),
                      models.TableColumn(field='Value', title='Value', formatter=table_formatter)
                    ]
        data_table_2 = models.DataTable(source=summary_source_2, columns=columns_2, width=400, height=280)

        # Scatter plot of selected response variables
        source_3 = bokeh.models.ColumnDataSource( data = dict( x=y_dict[ self.y.columns[0] ],
                                                               y=y_dict[ self.y.columns[0] ]  )
                                                 )
        plot_title_3 = f'{y2_select.value} vs {y1_select.value}'
        plot_3 = bokeh.plotting.figure( title=plot_title_3, height=400, width=400 )
        plot_3.xaxis.axis_label = 'Response A'
        plot_3.yaxis.axis_label = 'Response B'
        plot_3.scatter( 'x', 'y', source=source_3 )
        
        # Combined callback for all plots
        callback = bokeh.models.CustomJS( args = dict( x_dict    = x_dict, 
                                                       y_dict    = y_dict, 
                                                       source_1  = source_1, 
                                                       x1_select = x1_select, 
                                                       y1_select = y1_select,
                                                       plot_1    = plot_1,
                                                       source_2  = source_2, 
                                                       x2_select = x2_select, 
                                                       y2_select = y2_select,
                                                       plot_2    = plot_2,
                                                       source_3  = source_3,
                                                       plot_3    = plot_3,
                                                       summary_source_1 = summary_source_1,
                                                       summary_source_2 = summary_source_2,
                                                       summary_stats    = summary_stats.to_dict(orient='list')
                                                     ),
                                          code = '''
                                                   // Read data
                                                   const data_1 = source_1.data;
                                                   const xname_1 = x1_select.value;
                                                   const yname_1 = y1_select.value;
                                                   
                                                   const data_2 = source_2.data;
                                                   const xname_2 = x2_select.value;
                                                   const yname_2 = y2_select.value;

                                                   const data_3 = source_3.data;

                                                   const summary_data_1 = summary_source_1.data;
                                                   const summary_data_2 = summary_source_2.data;
                                                   
                                                   // Update data sources with selected data
                                                   data_1['x'] = x_dict[xname_1];
                                                   data_1['y'] = y_dict[yname_1];

                                                   data_2['x'] = x_dict[xname_2];
                                                   data_2['y'] = y_dict[yname_2];

                                                   data_3['x'] = y_dict[yname_1];
                                                   data_3['y'] = y_dict[yname_2];

                                                   // Update title and labels
                                                   plot_1.title.text = `${yname_1} vs ${xname_1}`;
                                                   plot_2.title.text = `${yname_2} vs ${xname_2}`;
                                                   plot_3.title.text = `${yname_2} vs ${yname_1}`;

                                                   // Update summary stats tables
                                                   summary_data_1['Value'] = summary_stats[yname_1];
                                                   summary_data_2['Value'] = summary_stats[yname_2];

                                                   // Apply changes
                                                   source_1.change.emit();
                                                   source_2.change.emit();
                                                   source_3.change.emit();
                                                   summary_source_1.change.emit();
                                                   summary_source_2.change.emit();
                                                 '''
                                        )

        # Attach callback to all relevant select widgets
        x1_select.js_on_change( 'value', callback )
        y1_select.js_on_change( 'value', callback )
        x2_select.js_on_change( 'value', callback )
        y2_select.js_on_change( 'value', callback )

        # Render the figure
        hline_1 = bokeh.models.Div( text='<div style="width: 750px; height: 2px; background-color: CornflowerBlue;"></div>' )
        hline_2 = bokeh.models.Div( text='<div style="width: 750px; height: 2px; background-color: CornflowerBlue;"></div>' )
        hline_3 = bokeh.models.Div( text='<div style="width: 750px; height: 2px; background-color: CornflowerBlue;"></div>' )

        title = bokeh.models.Div( text='<h1>Data Diagnostics</h1>', width=400 )
        intro = bokeh.models.Div( text = '''<p>These are some basic diagnostic plots and tables that can help
                                            analyze data resulting from a model. The dropdown menus allow you
                                            to select two predictor-response (or x-y) pairs. The selected response
                                            variables will be plotted against the selected predictors, and a 
                                            simple data description table will be created for each of them. </p>''',
                                  width = 600
                                )

        controls_1 = bokeh.layouts.row( x1_select, y1_select )
        lineplots_1 = bokeh.layouts.row( plot_1, data_table_1 )

        controls_2 = bokeh.layouts.row( x2_select, y2_select )
        lineplots_2 = bokeh.layouts.row( plot_2, data_table_2 )

        layout = bokeh.layouts.column( title,
                                       intro,
                                       hline_1,
                                       controls_1,
                                       lineplots_1,
                                       hline_2,
                                       controls_2,
                                       lineplots_2,
                                       hline_3,
                                       plot_3 
                                      )

        bokeh.io.show( layout )

        return


    def _is_jupyter_notebook(self):
        """Check if the code is called from a notebook."""
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True  # Jupyter notebook or JupyterLab
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (unknown)
        except NameError:
            return False  # Probably standard Python interpreter




def select_features( simulatedFeatures: pd.DataFrame, 
                     observedFeatures: pd.DataFrame, 
                     featureStatistics: pd.DataFrame, 
                     metric: str, 
                     iteration: int, 
                     history: Optional[List] = None
                    ) -> Tuple[str, Union[int, float, np.number], pd.DataFrame]:
    """
    Select target feature for history matching.

    Args:
      simulatedFeatures: DataFrame of features (columns) and their simulated values (rows)
      observedFeatures: DataFrame of features (columns) and their observed values (one row)
      featureStatistics: DataFrame of statistics (columns) and their values for each feature (rows)
      metric: name of statistic to use for assessment, e.g. "var" or "fano"
      iteration: current history matching iteration/wave
      history: list of features recently used in previous iterations/waves, implicitly in order from earliest used to most recent

    Returns:
      Tuple of selected feature name, observed value for that feature, and simulated values for that feature
    """






class DerivedFeatures:

    """Library of functions to compute derived features from time series."""

    @staticmethod
    def derivative_cauchyFit(x, *args):
        """
        Returns the parameters of a Cauchy distribution that fits the
        derivative of the input time series. The output is a pandas dataframe.
        """

        dx = np.gradient(x, axis=1)
        n = len(dx)
        loc = np.zeros(n)
        scale = np.zeros(n)
        for i in range(0, n):
            loc[i], scale[i] = scipy.stats.cauchy.fit(dx[i, :])
        dxCauchyFit_df = pd.DataFrame(
            {
                "dx_cauchy_loc": loc,
                "dx_cauchy_scale": scale,
            }
        )
        return dxCauchyFit_df

    @staticmethod
    def derivative_gaussianFit(x, *args):
        """
        Returns the parameters of a Gaussian distribution that fits the
        derivative of the input time series. The output is a pandas dataframe.
        """

        dx = np.gradient(x, axis=1)
        mean = np.mean(dx, axis=1)
        var = np.var(dx, axis=1)
        dxGaussianFit_df = pd.DataFrame(
            {
                "dx_mean": mean,
                "dx_var": var,
            }
        )
        return dxGaussianFit_df

    @staticmethod
    def derivative_laplaceFit(x, *args):
        """
        Returns the parameters of a Laplace distribution that fits the
        derivative of the input time series. The output is a pandas dataframe.
        """

        dx = np.gradient(x, axis=1)
        n = len(dx)
        mean = np.zeros(n)
        var = np.zeros(n)
        for i in range(0, n):
            mean[i], var[i] = scipy.stats.laplace.fit(dx[i, :])
        dxLaplaceFit_df = pd.DataFrame(
            {
                "dx_laplace_mean": mean,
                "dx_laplace_var": var,
            }
        )
        return dxLaplaceFit_df

    @staticmethod
    def derivative(x, *args):
        """
        Returns the derivative of time series as a pandas dataframe.
        """

        dx = np.gradient(x, axis=1)
        dx_df = pd.DataFrame(dx)
        for i in dx_df:
            dx_df.rename(columns={dx_df.columns[i]: f"dx_{i}"}, inplace=True)
        return dx_df

    @staticmethod
    def derivative2_cauchyFit(x, *args):
        """
        Returns the parameters of a Cauchy distribution that fits the second
        derivative of the input time series. The output is a pandas dataframe.
        """

        dx = np.gradient(x, axis=1)
        dx2 = np.gradient(dx, axis=1)
        n = len(dx2)
        loc = np.zeros(n)
        scale = np.zeros(n)
        for i in range(0, n):
            loc[i], scale[i] = scipy.stats.cauchy.fit(dx2[i, :])
        dx2CauchyFit_df = pd.DataFrame(
            {
                "dx2_cauchy_loc": loc,
                "dx2_cauchy_scale": scale,
            }
        )
        return dx2CauchyFit_df

    @staticmethod
    def derivative2_gaussianFit(x, *args):
        """
        Returns the parameters of a Gaussian distribution that fits the second
        derivative of the input time series. The output is a pandas dataframe.
        """

        dx = np.gradient(x, axis=1)
        dx2 = np.gradient(dx, axis=1)
        mean = np.mean(dx2, axis=1)
        var = np.var(dx2, axis=1)
        dx2GaussianFit_df = pd.DataFrame(
            {
                "dx2_mean": mean,
                "dx2_var": var,
            }
        )
        return dx2GaussianFit_df

    @staticmethod
    def derivative2_laplaceFit(x, *args):
        """
        Returns the parameters of a Laplace distribution that fits the second
        derivative of the input time series. The output is a pandas dataframe.
        """

        dx = np.gradient(x, axis=1)
        dx2 = np.gradient(dx, axis=1)
        n = len(dx2)
        mean = np.zeros(n)
        var = np.zeros(n)
        for i in range(0, n):
            mean[i], var[i] = scipy.stats.laplace.fit(dx2[i, :])
        dx2LaplaceFit_df = pd.DataFrame(
            {
                "dx2_laplace_mean": mean,
                "dx2_laplace_var": var,
            }
        )
        return dx2LaplaceFit_df

    @staticmethod
    def derivative2(x, *args):
        """
        Returns the second derivative of time series as a pandas dataframe.
        """

        dx = np.gradient(x, axis=1)
        dx2 = np.gradient(dx, axis=1)
        dx2_df = pd.DataFrame(dx2)
        for i in dx2_df:
            dx2_df.rename(columns={dx2_df.columns[i]: f"dx2_{i}"}, inplace=True)
        return dx2_df

    @staticmethod
    def diff_L1(x, xref):
        """
        Returns the L1 norm of the difference between each time series in x and
        xref. The output is a pandas dataframe.
        """

        return __diffL__(x, xref, order=1, column="diff_L1")

    @staticmethod
    def diff_L2(x, xref):
        """
        Returns the L2 norm of the difference between each time series in x and
        xref. The output is a pandas dataframe.
        """

        return __diffL__(x, xref, order=2, column="diff_L2")

    @staticmethod
    def diff_Linf(x, xref):
        """
        Returns the L_{\\inf} norm of the difference between each time series in x
        and xref. The output is a pandas dataframe.
        """

        return __diffL__(x, xref, order=np.inf, column="diff_Linf")

    @staticmethod
    def diff(x, xref):
        """
        Returns the difference between each time series in x and xref. The output
        is a pandas dataframe.
        """

        m = len(x)
        diff = np.add(x, -np.repeat(xref, m, axis=0))
        diff_df = pd.DataFrame(diff)
        for i in diff_df:
            diff_df.rename(columns={diff_df.columns[i]: f"diff_{i}"}, inplace=True)

        return diff_df

    @staticmethod
    def log10(x, *args):
        """
        Returns the logarithm in base 10 of the input time series as a pandas
        dataframe.
        """

        np.seterr(divide="ignore")
        xLog10 = np.log10(x)
        np.seterr(divide="warn")
        xLog10_df = pd.DataFrame(xLog10)
        for i in xLog10_df:
            xLog10_df.rename(columns={xLog10_df.columns[i]: f"xLog10_{i}"}, inplace=True)
        return xLog10_df

    @staticmethod
    def partialSum2(x, *args):
        """
        Returns the time series obtained from adding up groups of 2 values from the
        input time series. The output is a pandas dataframe.
        """

        return __partialSum__(x, intervalSize=2)

    @staticmethod
    def partialSum7(x, *args):
        """
        Returns the time series obtained from adding up groups of 7 values from the
        input time series. The output is a pandas dataframe.
        """

        return __partialSum__(x, intervalSize=7)

    @staticmethod
    def partialSum10(x, *args):
        """
        Returns the time series obtained from adding up groups of 10 values from the
        input time series. The output is a pandas dataframe.
        """

        return __partialSum__(x, 10)

    @staticmethod
    def partialSum15(x, *args):
        """
        Returns the time series obtained from adding up groups of 15 values from the
        input time series. The output is a pandas dataframe.
        """

        return __partialSum__(x, 15)

    @staticmethod
    def partialSum30(x, *args):
        """
        Returns the time series obtained from adding up groups of 30 values from the
        input time series. The output is a pandas dataframe.
        """

        return __partialSum__(x, 30)

    @staticmethod
    def sum_log10(x, *args):
        """
        Returns Log10 of the sum of elements of each the time series as a pandas
        dataframe.
        """

        sum = x.sum(axis=1)
        sum_df = pd.DataFrame({"sumLog10_x": np.log10(sum)})
        return sum_df

    @staticmethod
    def sum(x, *args):
        """
        Returns the sum of elements of each the time series as a pandas dataframe.
        """

        sum = x.sum(axis=1)
        sum_df = pd.DataFrame({"sum_x": sum})
        return sum_df

    @staticmethod
    def passthrough(x, *args):
        return pd.DataFrame(x)

    @staticmethod
    def series(x, *args):
        """
        Returns the array of time series as a pandas dataframe.
        """

        x_df = pd.DataFrame(x)
        for i in x_df:
            x_df.rename(columns={x_df.columns[i]: f"x_{i}"}, inplace=True)
        return x_df


def __diffL__(x, xref, order, column: str) -> pd.DataFrame:
    """Common code for L1, L2, and Linf norms."""

    m = len(x)
    diff = np.add(x, -np.repeat(xref, m, axis=0))
    diff_L = np.linalg.norm(diff, ord=order, axis=1)
    diff_L_df = pd.DataFrame({column: diff_L})
    return diff_L_df


def __partialSum__(x, intervalSize: int) -> pd.DataFrame:
    """Common code for partialSum2, partialSum7, partialSum10, partialSum15, and partialSum30."""

    n = x.shape[1]
    nIntervals = int(np.floor((n - 1) / intervalSize))
    partialSum = np.full((len(x), nIntervals + 1), np.nan)
    for i in range(0, nIntervals):
        xSample = x[:, i * intervalSize : (i + 1) * intervalSize]
        partialSum[:, i] = xSample.sum(axis=1)
    xSample = x[:, nIntervals * intervalSize : n]
    partialSum[:, nIntervals] = xSample.sum(axis=1)
    partialSum_df = pd.DataFrame(partialSum)
    for i in partialSum_df:
        columns = {partialSum_df.columns[i]: f"partialSum{intervalSize}_{i}"}
        partialSum_df.rename(columns=columns, inplace=True)

    return partialSum_df


class Statistics:

    """Functions for feature statistics."""

    @staticmethod
    def fano(f: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the Fano factor of each column of the input dataframe. The Fano
        factor is defined as (var/mean).
        """

        np.seterr(divide="ignore", invalid="ignore")
        fano = f.var(axis=0) / f.mean(axis=0)
        np.seterr(divide="warn", invalid="warn")

        fano_df = pd.DataFrame({"fano": fano})
        return fano_df

    @staticmethod
    def mean(f: pd.DataFrame) -> pd.DataFrame:
        """
        Returns mean of each column of the input dataframe.
        """

        return __og_stats__(f, lambda f: np.mean(f, axis=0), "mean")

    @staticmethod
    def qcd(f: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the Quartile Coefficient of Dispersion (QCD) of each column of the
        input dataframe.
        """

        q3 = f.quantile(q=0.75, axis=0)
        q1 = f.quantile(q=0.25, axis=0)

        np.seterr(divide="ignore", invalid="ignore")
        qcd_df = pd.DataFrame({"qcd": (q3 - q1) / (q3 + q1)})
        np.seterr(divide="warn", invalid="warn")

        return qcd_df

    @staticmethod
    def rsd(f: pd.DataFrame) -> pd.DataFrame:
        """
        Returns Relative Standard Deviation (RSD) of each column of the input
        dataframe. The RSD is defined as sqrt(var/mean).
        """

        np.seterr(divide="ignore", invalid="ignore")
        rsd = np.sqrt(f.var(axis=0) / f.mean(axis=0))
        np.seterr(divide="warn", invalid="warn")

        rsd_df = pd.DataFrame({"rsd": rsd})
        return rsd_df

    @staticmethod
    def skew(f: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the unbiased skewness of each column of the input dataframe.
        """

        # return pd.DataFrame(scipy.stats.skew(f), columns=["skew"], index=f.columns)
        skew = f.skew(axis=0)  # use Pandas DataFrame implementation
        skew_df = pd.DataFrame({"skew": skew})
        return skew_df

    @staticmethod
    def std(f: pd.DataFrame) -> pd.DataFrame:
        """
        Returns standard deviation of each column of the input dataframe.
        """

        # return __og_stats__(f, np.std, "std")
        return __og_stats__(f, lambda f: np.std(f, axis=0, ddof=1), "std")

    @staticmethod
    def var(f: pd.DataFrame) -> pd.DataFrame:
        """
        Returns variance of each column of the input dataframe.
        """

        return __og_stats__(f, lambda f: np.var(f, axis=0, ddof=1), "var")


def __og_stats__(data, fn, column) -> pd.DataFrame:
    """Common code for mean, std, and var."""

    stat = fn(data)
    df = pd.DataFrame({column: stat})

    return df


def getFeatures(simulationOutputs: pd.DataFrame, observations: pd.DataFrame, active_features: Optional[set] = None, active_statistics: Optional[set] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gets (derived) features and selected statistics for those features from the current simulation outputs.

    Args:
        simulationOutputs: values for (source) features as calculated by the simulator
        observations: observed values for recorded (source) features
        active_features: derivation functions to run on simulator outputs to derive feature values
        active_statistics: statistics to calculate for each feature, e.g. "variance" or "mean"

    Returns:
        Tuple of derived feature values (DataFrame) and their corresponding statistics (DataFrame)
    """

    derivedFeatures = getDerivedFeatures(simulationOutputs, observations, active_features)
    featureStatistics = getFeatureStatistics(derivedFeatures, active_statistics)

    return derivedFeatures, featureStatistics


def getDerivedFeatures(simulationOutputs: pd.DataFrame, observations: pd.DataFrame, active_features: Optional[set] = None) -> pd.DataFrame:
    """Gets derived features from the current simulation outputs."""

    simulationOutputs_np = simulationOutputs.to_numpy(copy=True)
    observations_np = observations.to_numpy(copy=True)

    if active_features is None:
        # all features, _ for unused function value
        active_features = {name for name, _ in inspect.getmembers(DerivedFeatures, inspect.isfunction)}

    # compute derived features
    derivedFeatures = pd.DataFrame()
    for function in [function for name, function in inspect.getmembers(DerivedFeatures, inspect.isfunction) if name in active_features]:
        feature_df = function(simulationOutputs_np, observations_np)
        derivedFeatures = pd.concat([derivedFeatures, feature_df], axis=1)

    return derivedFeatures


def getFeatureStatistics(features: pd.DataFrame, active_statistics: Optional[set] = None) -> pd.DataFrame:
    """Gets statistics for derived features."""

    if active_statistics is None:
        # all statistics, _ for unused function value
        active_statistics = {name for name, _ in inspect.getmembers(Statistics, inspect.isfunction)}

    # compute feature statistics
    featureStatistics = pd.DataFrame()
    for function in [function for name, function in inspect.getmembers(Statistics, inspect.isfunction) if name in active_statistics]:
        statistic_df = function(features)
        featureStatistics = pd.concat([featureStatistics, statistic_df], axis=1)

    return featureStatistics


# TODO - move selected feature history into Situation(?)
__history__ = []


def select_features_old(
    simulatedFeatures: pd.DataFrame, observedFeatures: pd.DataFrame, featureStatistics: pd.DataFrame, metric: str, iteration: int, history: Optional[List] = None
) -> Tuple[str, Union[int, float, np.number], pd.DataFrame]:
    """
    Select target feature for history matching.

    Args:
      simulatedFeatures: DataFrame of features (columns) and their simulated values (rows)
      observedFeatures: DataFrame of features (columns) and their observed values (one row)
      featureStatistics: DataFrame of statistics (columns) and their values for each feature (rows)
      metric: name of statistic to use for assessment, e.g. "var" or "fano"
      iteration: current history matching iteration/wave
      history: list of features recently used in previous iterations/waves, implicitly in order from earliest used to most recent

    Returns:
      Tuple of selected feature name, observed value for that feature, and simulated values for that feature
    """

    # if history is None:
    #     history = []

    FEATURE_SELECTION_QUARANTINE_PERIOD = 8
    FEATURE_SELECTION_CLOSE_CORRELATION_THRESHOLD = 0.90

    # Get indices of features in order from largest absolute value of statistics to smallest.
    # E.g., features with large variance are more interesting than features with little variance.
    unsortedFeatureSelectionMetric = -np.abs(featureStatistics[metric].values)
    sortedFeatureIndices = np.argsort(unsortedFeatureSelectionMetric)

    nFeatures = len(simulatedFeatures.columns)
    for rankIndex in range(nFeatures):
        candidateIndex = sortedFeatureIndices[rankIndex]

        # Check that feature stats are neither NaN nor Inf (which would be the last indices in sortedFeatureIndices)
        if not np.isfinite(unsortedFeatureSelectionMetric[candidateIndex]):
            warnings.warn(f"Unable to find valid feature (stopping search at position {rankIndex} of {nFeatures} potential features)", stacklevel=1)
            candidateIndex = sortedFeatureIndices[0]
            break

        # Check that feature is not highly correlated with another already in the quarantine list (i.e., with a recently-selected feature)
        acceptCandidate = True
        candidateCorrelation = simulatedFeatures.corr(method="pearson").iloc[:, candidateIndex]

        # for recentFeature in history:
        for recentFeature in __history__:
            # only acceptble if candidate was not recently used
            acceptCandidate &= simulatedFeatures.columns[candidateIndex] != recentFeature
            # only acceptable if candidate does _not_ correlate highly with a recently used feature
            acceptCandidate &= np.abs(candidateCorrelation.loc[recentFeature]) <= FEATURE_SELECTION_CLOSE_CORRELATION_THRESHOLD

        if acceptCandidate:
            break

    feature_name = simulatedFeatures.columns[candidateIndex]

    # Extract values for the selected feature
    # We don't need these, they are available from the observations and simulator results
    # observedFeatureValue = observedFeatures[observedFeatures.feature == feature_name].mean
    # simulatedFeatureValues = simulatedFeatures[feature_name]

    # Add this feature to the list of recently used features
    # history.append(feature_name)
    __history__.append(feature_name)

    # Remove previously used features from history after quarantine period
    # while (len(history) > FEATURE_SELECTION_QUARANTINE_PERIOD) or (len(history) >= nFeatures):
    #     history.pop(0)
    while (len(__history__) > FEATURE_SELECTION_QUARANTINE_PERIOD) or (len(__history__) >= nFeatures):
        __history__.pop(0)

    # Finalize and return
    # simulatedFeatures.to_csv(f"features_iter_{iteration}.csv")
    # featureStatistics.to_csv(f"featureStats_iter_{iteration}.csv")

    return [feature_name]
