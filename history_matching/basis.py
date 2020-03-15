import itertools
import re

from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from history_matching.error import *

class Basis():
    """Class to support polynomial basis, data matrix generation, and parameter name handling.
    """

    def __init__(self, order, intercept, param_info=None, verbose=False):
        """Create and instance of the Basis class.

        Args:
            param_info:  (Pandas dataframe) used to normalize data
                Columns include:
                * Name: The name of the parameter, must match column name in training_data.
                * Min: Minimum value of parameter.
                * Max: Maximum value of parameter.
                * MapTo: (optional) For use in commissioning script to assist in mapping the parameter to model input.
                * Source: (optional) Source from which parameter ranges came from
            verbose: (bool)
        """
        self.order = order
        self.intercept = intercept
        # self.D = len(self.model_terms)
        self.param_info = param_info # To normalize data to [0,1].  Should be here?
        self.verbose = verbose

    @classmethod
    def make_identity_basis(cls, params, param_info=None):
        """This constructor builds the `identity` basis, in which X1-->x1, X2-->x2, ...

        Args:
            params: (str) List of parameters, by name, to include in the basis.
            param_info:  (Pandas dataframe)
                Columns include:
                * Name: The name of the parameter, must match column name in training_data.
                * Min: Minimum value of parameter.
                * Max: Maximum value of parameter.
                * MapTo: (optional) For use in commissioning script to assist in mapping the parameter to model input.
                * Source: (optional) Source from which parameter ranges came from

        Returns: Instance of Basis class.
        """

        return cls(order=1, intercept=False, param_info=param_info)

    @classmethod
    def make_polynomial_basis(cls,
            params,
            intercept = True,
            order = 1,
            param_info = None,
            verbose = False
    ):
        """This constructor builds a `polynmial` basis, in which [X1, X1, ...]-->[X1, X1^2, X2, X2^2, X1*X2, ...]

        Args:
            params: (str) List of parameters, by name, to include in the basis.
            intercept: (bool) Set True to include a constant in the basis.
            order: (int) A number 0-6 indicating what order of polynomial should
                         be used for fitting.
                         6 includes only some sixth and seventh order terms.
            param_info:  (Pandas dataframe)
                Columns include:
                * Name: The name of the parameter, must match column name in training_data.
                * Min: Minimum value of parameter.
                * Max: Maximum value of parameter.
                * MapTo: (optional) For use in commissioning script to assist in mapping the parameter to model input.
                * Source: (optional) Source from which parameter ranges came from
            verbose: (bool)

        Returns: Instance of Basis class.
        """

        return cls(order=order, intercept=intercept, param_info=param_info, verbose=verbose)

    def scale_data(self, data):
        """ Helper to scale data.

        The transformation for data x is:
            y = [x-min] / [max-min]
        where parameter ranges come from param_info that was passed in previously.
        """
        if self.param_info is None:
            raise HistoryMatchingError("Basis must have param_info to do scaling!")

        for col in data.columns.tolist():
            if col in self.param_info.index:
                data[col] = (data[col] - self.param_info.loc[col,'Min'])/(self.param_info.loc[col,'Max']-self.param_info.loc[col,'Min'])
            elif self.verbose:
                print('Basis: Unable to scale', col)
        return data

    def generate_dmatrix(self, data, scaleX = False):
        """Generate data matrix.

        Args:
            data: (DataFrame) The data to transform.
            scaleX: (bool) When True, the scale_data will be called.

        Return: data matrix
        """

        data = data.copy()
        if scaleX:
            data = self.scale_data(data)

        if self.order==0:
            return pd.DataFrame({'1':np.ones(len(data.columns))})

        polyfit = PolynomialFeatures(self.order, interaction_only=False, include_bias=self.intercept)
        try:
            dmatrix = pd.DataFrame(polyfit.fit_transform(data))
        except ValueError as e:
            if pd.isnull(data).any().any():
                print(data[data.isnull().any(axis=1)])
                print('Data contains Null/None/NaN, see data above.')
                raise HistoryMatchingError("Input contains NaN, infinity or a value too large for float64")

        dmatrix.columns = polyfit.get_feature_names(data.columns)
        dmatrix = dmatrix.reindex(sorted(dmatrix.columns), axis=1)

        return dmatrix

    def generate_dmatrices(self, data, Ycol, scaleX = False):
        """Generates the data and response matrices.

        Args:
            data: (DataFrame) The data to transform.
            Ycol: (str) The column in the data that contains the responses
            scaleX: (bool) When True, the scale_data will be called.

        Return: (response matrix, data matrix)
        """

        response_matrix = data[[Ycol]]
        data_matrix = self.generate_dmatrix(data, scaleX = scaleX)
        return response_matrix, data_matrix

    def get_terms(self):
        """Returns the basis terms.

        Returns: A list of terms.
        """

        md = ModelDesc([], self.model_terms)
        terms = [c.strip() for c in md.describe().split('+')]
        terms[0] = terms[0][2:] # Remove '~ ' from beginning of first column

        if '0' in terms: # NO INTERCEPT
            terms.remove('0')
        else:
            if '1' in terms:
                print('Found "1" in terms, removing as this is likely a stored representation of the intercept.')
                terms.remove('1')
            terms = ['Intercept'] + terms

        return terms

    def regularize(self, inputs, results, alpha, scaleX = False):
        """Performs a lasso L1 regularization to select important terms.

        Args:
            inputs: (DataFrame) The training data.
            results: (Series) The result values.
            alpha: (float) Penalty weight.
            scaleX: (bool) When True, the scale_data will be called.

        Returns: The predicted results at the inputs.
        """

        print(f'User selected alpha = {alpha}')

        if scaleX:
            inputs = self.scale_data(inputs.copy())

        Ycol = 'Sim_Result'
        my_results = results.copy()
        my_results.name = Ycol
        data = pd.merge(inputs.reset_index(), my_results.reset_index(), on='Sample_Id').set_index(['Sample_Id', 'Sim_Id']).sort_index()

        response_matrix, data_matrix = self.generate_dmatrices(data, Ycol)
        model = sm.OLS(response_matrix, data_matrix)

        if alpha > 0:
            fit = model.fit_regularized(alpha=alpha, refit=True)
        else:
            fit = model.fit()

        if self.verbose:
            print('SUMMARY:\n', fit.summary())
            print('AIC:', fit.aic)
            print('BIC:', fit.bic)

        params = pd.Series(fit.params, index=data_matrix.columns)
        params = params[abs(params)>0]
        print('Non-Zero:', len(params), 'of', self.D)

        if len(params) == 0:
            raise HistoryMatchingError('In regularize, no parameters had a non-zero coefficient.  Try making alpha smaller.')

        terms = params.index.values.tolist()
        if 'Intercept' in terms:
            intercept_term = [Term([])]
            terms.remove('Intercept')
        else:
            intercept_term = []

        self.model_terms = intercept_term + [Term([EvalFactor(t)]) for t in terms]
        self.param_dict = Basis.make_param_dict(inputs.columns.tolist())
        self.D = len(self.model_terms)

        return fit.predict(data_matrix)

    def fit(self, inputs, results, scaleX = False):
        """Fits an ordinary least-squares model.

        Args:
            inputs: (DataFrame) The training data.
            results: (Series) The result values.
            scaleX: (bool) When True, the scale_data will be called.

        Returns: The predicted results at the inputs.
        """

        return self.regularize(inputs, results, 0, scaleX)

    def plot_regularize(self, inputs, results, alpha, scaleX = False, title = None):

        if scaleX:
            inputs = self.scale_data(inputs.copy())

        Ycol = 'Sim_Result'
        my_results = results.copy()
        my_results.name = Ycol
        data = pd.merge(inputs.reset_index(), my_results.reset_index(), on='Sample_Id')

        response_matrix, data_matrix = self.generate_dmatrices(data, Ycol)
        model = sm.OLS(response_matrix, data_matrix)

        num_params = np.zeros_like(alpha)
        bic = np.zeros_like(alpha)
        for i,a in enumerate(alpha):
            print(f'Regularize: {i} of {len(alpha)}')
            fit = model.fit_regularized(alpha=a, refit=True)

            params = pd.Series(fit.params, index=data_matrix.columns)
            params = params[abs(params)>0]

            num_params[i] = len(params)
            bic[i] = fit.bic

        lns = []
        fig, ax1 = plt.subplots()
        lns += ax1.plot(alpha, bic, 'ro-', label='BIC')
        ax1.set_xscale('log')
        ax1.set_xlabel('alpha')

        ax2 = ax1.twinx()
        lns += ax2.plot(alpha, num_params, 'bo-', label='N')
        ax2.set_ylabel('N')

        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc=0)
        if title is not None:
            plt.title(title)
        fig.tight_layout()
        plt.show()

        return fig
