"""Library of functions to analyze and select features (i.e., summary 
statistics) to be used in history matching iterations. This library 
includes a subset of functions that compute summary statistics from 
time series.
"""
import inspect
import warnings
from typing import List, Optional, Tuple, Union

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




class DerivedFeatures:
    """Library of functions to compute derived features from time series."""

    @staticmethod
    def derivative_cauchy_fit(x, *args):
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
        dx_cauchy_fit_df = pd.DataFrame( { 'dx_cauchy_loc'  : loc,
                                         'dx_cauchy_scale': scale } )
        return dx_cauchy_fit_df

    @staticmethod
    def derivative_gaussian_fit(x, *args):
        """
        Returns the parameters of a Gaussian distribution that fits the
        derivative of the input time series. The output is a pandas dataframe.
        """
        dx = np.gradient(x, axis=1)
        mean = np.mean(dx, axis=1)
        var = np.var(dx, axis=1)
        dx_gaussian_fit_df = pd.DataFrame( { 'dx_mean': mean,
                                             'dx_var' : var  } )
        return dx_gaussian_fit_df

    @staticmethod
    def derivative_laplace_fit(x, *args):
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
        dx_laplace_fit_df = pd.DataFrame( { 'dx_laplace_mean' : mean,
                                            'dx_laplace_var'  : var   } )
        return dx_laplace_fit_df

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
    def derivative2_cauchy_fit(x, *args):
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
        dx2_cauchy_fit_df = pd.DataFrame( { 'dx2_cauchy_loc'  : loc,
                                            'dx2_cauchy_scale': scale } )
        return dx2_cauchy_fit_df

    @staticmethod
    def derivative2_gaussian_fit(x, *args):
        """
        Returns the parameters of a Gaussian distribution that fits the second
        derivative of the input time series. The output is a pandas dataframe.
        """
        dx = np.gradient(x, axis=1)
        dx2 = np.gradient(dx, axis=1)
        mean = np.mean(dx2, axis=1)
        var = np.var(dx2, axis=1)
        dx2_gaussian_fit_df = pd.DataFrame( { 'dx2_mean': mean,
                                              'dx2_var' : var  } )
        return dx2_gaussian_fit_df

    @staticmethod
    def derivative2_laplace_fit(x, *args):
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
        dx2_laplace_fit_df = pd.DataFrame( { 'dx2_laplace_mean': mean,
                                             'dx2_laplace_var' : var  } )
        return dx2_laplace_fit_df

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
        return self.__diffL__(x, xref, order=1, column='diff_L1')

    @staticmethod
    def diff_L2(x, xref):
        """
        Returns the L2 norm of the difference between each time series in x and
        xref. The output is a pandas dataframe.
        """
        return self.__diffL__(x, xref, order=2, column='diff_L2')

    @staticmethod
    def diff_Linf(x, xref):
        """
        Returns the L_{\\inf} norm of the difference between each time series in x
        and xref. The output is a pandas dataframe.
        """
        return self.__diffL__(x, xref, order=np.inf, column='diff_Linf')

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
            diff_df.rename(columns={diff_df.columns[i]: f'diff_{i}'}, inplace=True)
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
            x_log10_df.rename(columns={xLog10_df.columns[i]: f'x_log10_{i}'}, inplace=True)
        return x_log10_df

    @staticmethod
    def partial_sum_2(x, *args):
        """
        Returns the time series obtained from adding up groups of 2 values from the
        input time series. The output is a pandas dataframe.
        """
        return self.__partial_sum__(x, interval_size=2)

    @staticmethod
    def partial_sum_7(x, *args):
        """
        Returns the time series obtained from adding up groups of 7 values from the
        input time series. The output is a pandas dataframe.
        """
        return self.__partial_sum__(x, interval_size=7)

    @staticmethod
    def partial_sum_10(x, *args):
        """
        Returns the time series obtained from adding up groups of 10 values from the
        input time series. The output is a pandas dataframe.
        """
        return self.__partial_sum__(x, interval_size=10)

    @staticmethod
    def partial_sum_15(x, *args):
        """
        Returns the time series obtained from adding up groups of 15 values from the
        input time series. The output is a pandas dataframe.
        """
        return self.__partial_sum__(x, interval_size=15)

    @staticmethod
    def partial_sum_30(x, *args):
        """
        Returns the time series obtained from adding up groups of 30 values from the
        input time series. The output is a pandas dataframe.
        """
        return self.__partial_sum__(x, interval_size=30)

    @staticmethod
    def sum_log10(x, *args):
        """
        Returns Log10 of the sum of elements of each the time series as a pandas
        dataframe.
        """
        sum = x.sum(axis=1)
        sum_df = pd.DataFrame({'sum_Log10_x': np.log10(sum)})
        return sum_df

    @staticmethod
    def sum(x, *args):
        """
        Returns the sum of elements of each the time series as a pandas dataframe.
        """
        sum = x.sum(axis=1)
        sum_df = pd.DataFrame({'sum_x': sum})
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
            x_df.rename(columns={x_df.columns[i]: f'x_{i}'}, inplace=True)
        return x_df

    def __diffL__(x, xref, order, column: str) -> pd.DataFrame:
        """Common code for L1, L2, and Linf norms."""
        m = len(x)
        diff = np.add(x, -np.repeat(xref, m, axis=0))
        diff_L = np.linalg.norm(diff, ord=order, axis=1)
        diff_L_df = pd.DataFrame({column: diff_L})
        return diff_L_df

    def __partial_sum__(x, interval_size: int) -> pd.DataFrame:
        """Common code for partialSum2, partialSum7, partialSum10, partialSum15, and partialSum30."""
        n = x.shape[1]
        n_intervals = int(  np.floor( (n-1)/interval_size )  )
        partial_sum = np.full( (len(x), n_intervals+1), np.nan )
        for i in range(0, n_intervals):
            x_sample = x[:, i * interval_size : (i + 1) * interval_size]
            partial_sum[:, i] = x_sample.sum(axis=1)
        x_sample = x[:, n_intervals*interval_size : n]
        partial_sum[:, n_intervals] = x_sample.sum(axis=1)
        partial_sum_df = pd.DataFrame(partial_sum)
        for i in partial_sum_df:
            columns = {partial_sum_df.columns[i]: f'partial_sum{intervalSize}_{i}'}
            partial_sum_df.rename(columns=columns, inplace=True)
        return partial_sum_df




class Statistics:
    """Functions for feature statistics."""

    @staticmethod
    def fano(f: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the Fano factor of each column of the input dataframe. The Fano
        factor is defined as (var/mean).
        """
        np.seterr(divide='ignore', invalid='ignore')
        fano = f.var(axis=0) / f.mean(axis=0)
        np.seterr(divide='warn', invalid='warn')
        fano_df = pd.DataFrame({'fano': fano})
        return fano_df

    @staticmethod
    def mean(f: pd.DataFrame) -> pd.DataFrame:
        """
        Returns mean of each column of the input dataframe.
        """
        return self.__og_stats__(f, lambda f: np.mean(f, axis=0), 'mean')

    @staticmethod
    def qcd(f: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the Quartile Coefficient of Dispersion (QCD) of each column of the
        input dataframe.
        """
        q3 = f.quantile(q=0.75, axis=0)
        q1 = f.quantile(q=0.25, axis=0)
        np.seterr(divide='ignore', invalid='ignore')
        qcd_df = pd.DataFrame({'qcd': (q3 - q1) / (q3 + q1)})
        np.seterr(divide='warn', invalid='warn')
        return qcd_df

    @staticmethod
    def rsd(f: pd.DataFrame) -> pd.DataFrame:
        """
        Returns Relative Standard Deviation (RSD) of each column of the input
        dataframe. The RSD is defined as sqrt(var/mean).
        """
        np.seterr(divide='ignore', invalid='ignore')
        rsd = np.sqrt(f.var(axis=0) / f.mean(axis=0))
        np.seterr(divide='warn', invalid='warn')
        rsd_df = pd.DataFrame({'rsd': rsd})
        return rsd_df

    @staticmethod
    def skew(f: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the unbiased skewness of each column of the input dataframe.
        """
        skew = f.skew(axis=0)  # use Pandas DataFrame implementation
        skew_df = pd.DataFrame({'skew': skew})
        return skew_df

    @staticmethod
    def std(f: pd.DataFrame) -> pd.DataFrame:
        """
        Returns standard deviation of each column of the input dataframe.
        """
        return self.__og_stats__(f, lambda f: np.std(f, axis=0, ddof=1), 'std')

    @staticmethod
    def var(f: pd.DataFrame) -> pd.DataFrame:
        """
        Returns variance of each column of the input dataframe.
        """
        return self.__og_stats__(f, lambda f: np.var(f, axis=0, ddof=1), 'var')

    def __og_stats__(data, fn, column) -> pd.DataFrame:
        """Common code for mean, std, and var."""
        stat = fn(data)
        df = pd.DataFrame({column: stat})
        return df




def compute_stats(features: pd.DataFrame, active_statistics: Optional[set] = None) -> pd.DataFrame:
    """Gets a set of basic statistics for all features or columns in the features dataframe."""

    if active_statistics is None:
        active_statistics = {name for name, _ in inspect.getmembers(Statistics, inspect.isfunction) if not name.startswith('__')}

    # compute feature statistics
    features_stats = pd.DataFrame()
    for function in [function for name, function in inspect.getmembers(Statistics, inspect.isfunction) if name in active_statistics]:
        statistic_df = function(features)
        features_stats = pd.concat([features_stats, statistic_df], axis=1)

    return features_stats




def feature_selection( features: pd.DataFrame, 
                       features_stats: pd.DataFrame, 
                       feature_selection_metric: str,
                       cooldown_period: int = 5,
                       correlation_threshold: float = 0.8
                      ) -> str:
    """
    Selects target features for history matching.

    Args:
      features                : DataFrame of features (columns) for a set 
                                of simulations (rows).
      features_stats          : DataFrame of statistics (columns) for each
                                of the columns in `features`.
      feature_selection_metric: name of the statistic to be used for the 
                                target selection. It must be the name of a 
                                column in `features_stats`.
      cooldown_period         : the number of recent selections to track, 
                                preventing re-selection of the same or similar
                                features within this limit. Higher values
                                increase the minimum time before a previously
                                selected feature can be chosen again.
      correlation_threshold   : The maximum allowed correlation between a 
                                candidate feature and any recently selected 
                                features. If the correlation between a candidate
                                and a recent selection exceeds this threshold,
                                the candidate will be excluded from selection
                                to reduce redundancy.
      
    Returns:
      Name of the selected feature.
    """

    # Create a history attribute to keep track of what features have been already used
    if not hasattr( feature_selection, '_history' ):
        feature_selection._history = []

    
    # Get indices of features sorted from largest absolute value of statistics to smallest.
    # E.g., features with large variance are more interesting than features with little variance.
    unsorted_feature_metric = -np.abs(features[metric].values)  # negative so that the 
                                                                # sorting below starts with the
                                                                # largest number
    sorted_feature_metric_index = np.argsort(unsorted_feature_metric)

    
    # Find the best feature (i.e., the one with the largest metric and that is valid)
    for candidate_index in sorted_feature_metric_index:

        accept_candidate = True    # Assume that the candidate will be accepted; this value
                                   # changes if a candidate should be rejected

        # Reject candidates with non-numeric values (NaN or Inf)
        if not np.isfinite( unsorted_feature_metric[candidate_index] ):
            accept_candidate = False
            continue
        
        # Reject candidates that were recently used
        if features.columns[candidate_index] in feature_selection._history:
            accept_candidate = False
            continue

        # Reject candidates that are highly correlated with recent candidates
        for recent_feature in feature_selection._history:
            candidate_correlation = features[recent_feature].corr(method='pearson').iloc[:, candidate_index]
            if abs(candidate_correlation) >= correlation_threshold:
                accept_candidate = False
                break

        if accept_candidate:    # The candidate passed all test; we can exit this loop now
            break
            
    if accept_candidate:    # A valid candidate was found, let's save it
        feature_name = features.columns[candidate_index]
    else:
        feature_name = features.columns[sorted_feature_metric_index[0]]
        warnings.warn( f'Unable to find a valid feature; using {feature_name}' )

    
    # Update history and return
    feature_selection._history.append( feature_name )
    while(    ( len(feature_selection._history) > cooldown_period )     \
           or ( len(feature_selection._history) >= len(features)  ) ):
        feature_selection._history.pop(0)

    return [feature_name]





#-----------------------------------------

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

