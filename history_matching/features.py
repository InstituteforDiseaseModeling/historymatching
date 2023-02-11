import inspect
from typing import List, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import scipy.stats


class DerivedFeatures:

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
        dxCauchyFit_df = pd.DataFrame({
            "dx_cauchy_loc": loc,
            "dx_cauchy_scale": scale,
        })
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
        dxGaussianFit_df = pd.DataFrame({
            "dx_mean": mean,
            "dx_var": var,
        })
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
        dxLaplaceFit_df = pd.DataFrame({
            "dx_laplace_mean": mean,
            "dx_laplace_var": var,
        })
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
        dx2CauchyFit_df = pd.DataFrame({
            "dx2_cauchy_loc": loc,
            "dx2_cauchy_scale": scale,
        })
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
        dx2GaussianFit_df = pd.DataFrame({
            "dx2_mean": mean,
            "dx2_var": var,
        })
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
        dx2LaplaceFit_df = pd.DataFrame({
            "dx2_laplace_mean": mean,
            "dx2_laplace_var": var,
        })
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

    m = len(x)
    diff = np.add(x, -np.repeat(xref, m, axis=0))
    diff_L = np.linalg.norm(diff, ord=order, axis=1)
    diff_L_df = pd.DataFrame({column: diff_L})
    return diff_L_df


def __partialSum__(x, intervalSize: int) -> pd.DataFrame:

    n = x.shape[1]
    nIntervals = int(np.floor((n - 1) / intervalSize))
    partialSum = np.full((len(x), nIntervals + 1), np.nan)
    for i in range(0, nIntervals):
        xSample = x[:, i * intervalSize:(i + 1) * intervalSize]
        partialSum[:, i] = xSample.sum(axis=1)
    xSample = x[:, nIntervals * intervalSize:n]
    partialSum[:, nIntervals] = xSample.sum(axis=1)
    partialSum_df = pd.DataFrame(partialSum)
    for i in partialSum_df:
        columns = {partialSum_df.columns[i]: f"partialSum{intervalSize}_{i}"}
        partialSum_df.rename(columns=columns, inplace=True)

    return partialSum_df


class Statistics:

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
        skew = f.skew(axis=0)   # use Pandas DataFrame implementation
        skew_df = pd.DataFrame({"skew": skew})
        return skew_df

    @staticmethod
    def std(f: pd.DataFrame) -> pd.DataFrame:
        """
        Returns standard deviation of each column of the input dataframe.
        """

        # return __og_stats__(f, np.std, "std")
        return __og_stats__(f, lambda f: np.std(f, ddof=1), "std")

    @staticmethod
    def var(f: pd.DataFrame) -> pd.DataFrame:
        """
        Returns variance of each column of the input dataframe.
        """

        return __og_stats__(f, lambda f: np.var(f, ddof=1), "var")


def __og_stats__(data, fn, column) -> pd.DataFrame:

    stat = fn(data)
    df = pd.DataFrame({column: stat})

    return df


def getFeatures(simulationOutputs: pd.DataFrame, observations: pd.DataFrame, active_features: set = None, active_statistics: set = None) -> Tuple[pd.DataFrame, pd.DataFrame]:

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


def getDerivedFeatures(simulationOutputs: pd.DataFrame, observations: pd.DataFrame, active_features: set = None) -> pd.DataFrame:

    simulationOutputs_np = simulationOutputs.to_numpy(copy=True)
    observations_np = observations.to_numpy(copy=True)

    if active_features is None:
        # all features, _ for unused function value
        active_features = set([name for name, _ in inspect.getmembers(DerivedFeatures, inspect.isfunction)])

    # compute derived features
    derivedFeatures = pd.DataFrame()
    for function in [function for name, function in inspect.getmembers(DerivedFeatures, inspect.isfunction) if name in active_features]:
        feature_df = function(simulationOutputs_np, observations_np)
        derivedFeatures = pd.concat([derivedFeatures, feature_df], axis=1)

    return derivedFeatures


def getFeatureStatistics(features: pd.DataFrame, active_statistics: set = None) -> pd.DataFrame:

    if active_statistics is None:
        # all statistics, _ for unused function value
        active_statistics = set([name for name, _ in inspect.getmembers(Statistics, inspect.isfunction)])

    # compute feature statistics
    featureStatistics = pd.DataFrame()
    for function in [function for name, function in inspect.getmembers(Statistics, inspect.isfunction) if name in active_statistics]:
        statistic_df = function(features)
        featureStatistics = pd.concat([featureStatistics, statistic_df], axis=1)

    return featureStatistics


def select_features(simulatedFeatures: pd.DataFrame, observedFeatures: pd.DataFrame, featureStatistics: pd.DataFrame, metric: str, iteration: int, history: List = None) -> Tuple[str, Union[int, float, np.number], pd.DataFrame]:

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

    if history is None:
        history = []

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

            warnings.warn(f"Unable to find valid feature (stopping search at position {rankIndex} of {nFeatures} potential features)")
            candidateIndex = sortedFeatureIndices[0]
            break

        # Check that feature is not highly correlated with another already in the quarantine list (i.e., with a recently-selected feature)
        acceptCandidate = True
        candidateCorrelation = simulatedFeatures.corr(method="pearson").iloc[:, candidateIndex]

        for recentFeature in history:

            # only acceptble if candidate was not recently used
            acceptCandidate &= (simulatedFeatures.columns[candidateIndex] != recentFeature)
            # only acceptable if candidate does _not_ correlate highly with a recently used feature
            acceptCandidate &= (np.abs(candidateCorrelation.loc[recentFeature]) <= FEATURE_SELECTION_CLOSE_CORRELATION_THRESHOLD)

        if acceptCandidate:
            break

    feature_name = simulatedFeatures.columns[candidateIndex]

    # Extract values for the selected feature
    observedFeatureValue = observedFeatures[feature_name][0]
    simulatedFeatureValues = simulatedFeatures[feature_name]

    # Add this feature to the list of recently used features
    history.append(feature_name)

    # Remove previously used features from history after quarantine period
    while len(history) > FEATURE_SELECTION_QUARANTINE_PERIOD:
        history.pop(0)

    # Finalize and return
    # simulatedFeatures.to_csv(f"features_iter_{iteration}.csv")
    # featureStatistics.to_csv(f"featureStats_iter_{iteration}.csv")

    return feature_name, observedFeatureValue, simulatedFeatureValues
