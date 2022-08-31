import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import patsy # TODO: Cleanup
from patsy import ModelDesc, Term, LookupFactor, EvalFactor, dmatrices
import itertools
# For regularized selection:
import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging

logger = logging.getLogger(__name__)

class Basis():
    """Class to support polynomial basis, data matrix generation, and parameter name handling.
    """

    def __init__(self, model_terms, param_dict, param_info=None):
        """Create and instance of the Basis class.

        Args:
            model_terms: (list) basis model terms.
                This argument is typically only used in deserialization.
            param_dict: (dict) Parameters in a dictionary form that maps original parameter names to patsy-safe parameter names.  The helper function make_param_dict generates this mapping.
            param_info:  (Pandas dataframe) used to normalize data
                Columns include:
                * Name: The name of the parameter, must match column name in training_data.
                * Min: Minimum value of parameter.
                * Max: Maximum value of parameter.
                * MapTo: (optional) For use in commissioning script to assist in mapping the parameter to model input.
                * Source: (optional) Source from which parameter ranges came from
        """

        self.model_terms = model_terms
        self.param_dict = param_dict
        self.D = len(self.model_terms)
        self.param_info = param_info # To normalize data to [0,1].  Should be here?


    @staticmethod
    def make_param_dict(param_names):
        """Static helper method to transform parameter names into a dictionary in which the keys are the original parameter names and the values are patsy-safe strings

        Args:
            param_names: (list) The original parameter names.
        """

        # Return mapping from original parameter name to patsy-safe name
        return {p:p.replace(':','').replace('&',' ').replace(' ', '_').replace('-','_') for p in param_names}


    @classmethod
    def identity_basis(cls, params, param_info=None):
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

        param_dict = Basis.make_param_dict(params)
        params_patsy = param_dict.values()
        model_terms = [Term([LookupFactor(x)]) for x in params_patsy] # X
        return cls(model_terms, param_dict, param_info)


    @classmethod
    def polynomial_basis(cls,
            params,
            intercept = True,
            first_order = True,
            second_order = False,
            third_order = False,
            fourth_order = False,
            fifth_order = False,
            higher_order = False,
            param_info = None
    ):
        """This constructor builds a `polynmial` basis, in which [X1, X1, ...]-->[X1, X1^2, X2, X2^2, X1*X2, ...]

        Args:
            params: (str) List of parameters, by name, to include in the basis.
            intercept: (bool) Set True to include a constant in the basis.
            first_order: (bool) Set True to include first-order terms.
            second_order: (bool) Set True to include second-order terms.
            third_order: (bool) Set True to include third-order terms.
            fourth_order: (bool) Set True to include fourth-order terms.
            fifth_order: (bool) Set True to include fifth-order terms.
            higher_order: (bool) Set True to include some sixth and seventh order terms.
            param_info:  (Pandas dataframe)
                Columns include:
                * Name: The name of the parameter, must match column name in training_data.
                * Min: Minimum value of parameter.
                * Max: Maximum value of parameter.
                * MapTo: (optional) For use in commissioning script to assist in mapping the parameter to model input.
                * Source: (optional) Source from which parameter ranges came from

        Returns: Instance of Basis class.
        """

        param_dict = Basis.make_param_dict(params)
        params_patsy = param_dict.values()

        # Intercept
        if intercept:
            model_terms = [Term([])]
        else:
            model_terms = []

        # First order
        if first_order:
            model_terms += [Term([LookupFactor(x)]) for x in params_patsy] # X

        # Second order
        if second_order:
            model_terms += [Term([EvalFactor('%s**2'%x)]) for x in params_patsy] # X^2
            model_terms += [Term([EvalFactor('%s*%s'%x)]) for x in itertools.combinations(params_patsy, 2)] # X*Y

        # Third order
        if third_order:
            model_terms += [Term([EvalFactor('%s**3'%x)]) for x in params_patsy] # X^3

            model_terms += [Term([EvalFactor('%s*%s**2'%x)]) for x in itertools.combinations(params_patsy, 2)] # X*Y^2
            model_terms += [Term([EvalFactor('%s**2*%s'%x)]) for x in itertools.combinations(params_patsy, 2)] # X^2*Y

            model_terms += [Term([EvalFactor('%s*%s*%s'%x)]) for x in itertools.combinations(params_patsy, 3)] # X*Y*Z

        # Fourth order
        if fourth_order:
            model_terms += [Term([EvalFactor('%s**4'%x)]) for x in params_patsy] # X^4
            model_terms += [Term([EvalFactor('%s**3*%s'%x)]) for x in itertools.combinations(params_patsy, 2)] # X^3*Y
            model_terms += [Term([EvalFactor('%s*%s**3'%x)]) for x in itertools.combinations(params_patsy, 2)] # X*Y^3

            model_terms += [Term([EvalFactor('%s**2*%s**2'%x)]) for x in itertools.combinations(params_patsy, 2)] # X^2*Y^2

            model_terms += [Term([EvalFactor('%s**2*%s*%s'%x)]) for x in itertools.combinations(params_patsy, 3)] # X^2*Y*Z
            model_terms += [Term([EvalFactor('%s*%s**2*%s'%x)]) for x in itertools.combinations(params_patsy, 3)] # X*Y^2*Z
            model_terms += [Term([EvalFactor('%s*%s*%s**2'%x)]) for x in itertools.combinations(params_patsy, 3)] # X*Y*Z^2

            model_terms += [Term([EvalFactor('%s*%s*%s*%s'%x)]) for x in itertools.combinations(params_patsy, 4)] # W*X*Y*Z

        # Fifth order
        if fifth_order:
            model_terms += [Term([EvalFactor('%s**5'%x)]) for x in params_patsy] # X^5
            model_terms += [Term([EvalFactor('%s**4*%s'%x)]) for x in itertools.combinations(params_patsy, 2)] # X^4*Y
            model_terms += [Term([EvalFactor('%s*%s**4'%x)]) for x in itertools.combinations(params_patsy, 2)] # X*Y^4

            model_terms += [Term([EvalFactor('%s**3*%s**2'%x)]) for x in itertools.combinations(params_patsy, 2)] # X^3*Y^2
            model_terms += [Term([EvalFactor('%s**2*%s**3'%x)]) for x in itertools.combinations(params_patsy, 2)] # X^2*Y^3

            model_terms += [Term([EvalFactor('%s**3*%s*%s'%x)]) for x in itertools.combinations(params_patsy, 3)] # X^3*Y*Z
            model_terms += [Term([EvalFactor('%s*%s**3*%s'%x)]) for x in itertools.combinations(params_patsy, 3)] # X*Y^3*Z
            model_terms += [Term([EvalFactor('%s*%s*%s**3'%x)]) for x in itertools.combinations(params_patsy, 3)] # X*Y*Z^3

            model_terms += [Term([EvalFactor('%s**2*%s*%s*%s'%x)]) for x in itertools.combinations(params_patsy, 4)] # W^2*X*Y*Z
            model_terms += [Term([EvalFactor('%s*%s**2*%s*%s'%x)]) for x in itertools.combinations(params_patsy, 4)] # W*X^2*Y*Z
            model_terms += [Term([EvalFactor('%s*%s*%s**2*%s'%x)]) for x in itertools.combinations(params_patsy, 4)] # W*X*Y^2*Z
            model_terms += [Term([EvalFactor('%s*%s*%s*%s**2'%x)]) for x in itertools.combinations(params_patsy, 4)] # W*X*Y*Z^2

            model_terms += [Term([EvalFactor('%s**2*%s**2*%s'%x)]) for x in itertools.combinations(params_patsy, 3)] # X^2*Y^2*Z
            model_terms += [Term([EvalFactor('%s**2*%s*%s**2'%x)]) for x in itertools.combinations(params_patsy, 3)] # X^2*Y*Z^2
            model_terms += [Term([EvalFactor('%s*%s**2*%s**2'%x)]) for x in itertools.combinations(params_patsy, 3)] # X*Y^2*Z^2

            model_terms += [Term([EvalFactor('%s*%s*%s*%s*%s'%x)]) for x in itertools.combinations(params_patsy, 5)] # V*W*X*Y*Z

        if higher_order:
            # Some sixth order
            model_terms += [Term([EvalFactor('%s**6'%x)]) for x in params_patsy] # X^6

            model_terms += [Term([EvalFactor('%s**5*%s'%x)]) for x in itertools.combinations(params_patsy, 2)] # X^5*Y
            model_terms += [Term([EvalFactor('%s*%s**5'%x)]) for x in itertools.combinations(params_patsy, 2)] # X*Y^5

            model_terms += [Term([EvalFactor('%s**3*%s*%s*%s'%x)]) for x in itertools.combinations(params_patsy, 4)] # W^3*X*Y*Z
            model_terms += [Term([EvalFactor('%s*%s**3*%s*%s'%x)]) for x in itertools.combinations(params_patsy, 4)] # W*X^3*Y*Z
            model_terms += [Term([EvalFactor('%s*%s*%s**3*%s'%x)]) for x in itertools.combinations(params_patsy, 4)] # W*X*Y^3*Z
            model_terms += [Term([EvalFactor('%s*%s*%s*%s**3'%x)]) for x in itertools.combinations(params_patsy, 4)] # W*X*Y*Z^3

            # Some seventh?! order
            model_terms += [Term([EvalFactor('%s**7'%x)]) for x in params_patsy] # X^7

            model_terms += [Term([EvalFactor('%s**6*%s'%x)]) for x in itertools.combinations(params_patsy, 2)] # X^6*Y
            model_terms += [Term([EvalFactor('%s*%s**6'%x)]) for x in itertools.combinations(params_patsy, 2)] # X*Y^6

        return cls(model_terms, param_dict, param_info)


    @classmethod
    def deserialize(cls, state):
        """ Helper to read basis from file.

        Args:
            state: (dict) Contents of file produced by serialize.

        Returns: Instance of Basis class.
        """

        terms = state['Terms']

        if 'Intercept' in terms:
            intercept_term = [Term([])]
            terms.remove('Intercept')
        else:
            intercept_term = []

        return cls(
            model_terms = intercept_term + [Term([EvalFactor(t)]) for t in terms],
            param_dict = state['Param_Dict'],
            param_info = pd.read_json( state['Param_Info'], orient='split' ).set_index('Name')
        )


    def serialize(self):
        """ Helper to write Basis to file.
        """

        return {
            'Terms' : self.get_terms(),
            'Param_Dict' : self.param_dict,
            'Param_Info' : self.param_info.reset_index().to_json(orient='split')
        }


    def scale_data(self, data):
        """ Helper to scale data.

        The transformation for data x is:
            y = [x-min] / [max-min]
        where parameter ranges come from param_info that was passed in previously.
        """

        for col in data.columns.tolist():
            if col in self.param_info.index:
                data[col] = (data[col] - self.param_info.loc[col,'Min'])/(self.param_info.loc[col,'Max']-self.param_info.loc[col,'Min'])
            else:
                logger.info(f'Basis: Unable to scale {col}')
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
            assert(self.param_info is not None)
            data = self.scale_data(data)
        data = data.rename(columns=self.param_dict)

        md = ModelDesc([], self.model_terms)
        try:
            dmat = patsy.dmatrix(md, data = data, return_type = 'dataframe', NA_action="raise")
        except Exception as e:
            print(str(e))
            if pd.isnull(data).any().any():
                #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(data[data.isnull().any(axis=1)])
                print('Data contains Null/None/NaN, see data above.')
                exit()
        return dmat

    def generate_dmatrices(self, data, Ycol, scaleX = False):
        """Generates the data and response matrices.

        Args:
            data: (DataFrame) The data to transform.
            Ycol: (str) The column in the data that contains the responses
            scaleX: (bool) When True, the scale_data will be called.

        Return: (response matrix, data matrix)
        """

        response_terms = [Term([LookupFactor(Ycol)])]

        data = data.copy()
        if scaleX:
            assert(self.param_info is not None)
            data = self.scale_data(data)

        data = data.rename(columns=self.param_dict)
        md = ModelDesc(response_terms, self.model_terms)

        (response_matrix, data_matrix) = dmatrices(md, data=data, return_type='dataframe')
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
                logger.info('Found "1" in terms, removing as this is likely a stored representation of the intercept.')
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

        print('User selected alpha = %f' % alpha)

        if scaleX:
            assert(self.param_info is not None)
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

        logger.info(f'SUMMARY:\n{fit.summary()}')
        logger.info(f'AIC:{fit.aic}')
        logger.info(f'BIC:{fit.bic}')

        params = pd.Series(fit.params, index=data_matrix.columns)
        params = params[abs(params)>0]
        print('Non-Zero:', len(params), 'of', self.D)

        if len(params) == 0:
            raise ValueError('In regularize, no parameters had a non-zero coefficient.  Try making alpha smaller.')

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


    def plot_regularize(self, inputs, results, alpha, scaleX = False, title = None, fig_file = None):

        if scaleX:
            assert(self.param_info is not None)
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
            print('Regularize: ', i,' of ', len(alpha))
            fit = model.fit_regularized(alpha=a, refit=True)

            params = pd.Series(fit.params, index=data_matrix.columns)
            params = params[abs(params)>0]

            num_params[i] = len(params)
            bic[i] = fit.bic

        lns = []
        fig, ax1 = plt.subplots(figsize=(12, 9), dpi=300)
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
        if fig_file: fig.savefig(fig_file)
        plt.show()

        return fig
