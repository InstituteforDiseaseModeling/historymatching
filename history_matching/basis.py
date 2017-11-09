import patsy # TODO: Cleanup
from patsy import ModelDesc, Term, LookupFactor, EvalFactor, dmatrices
import itertools
# For regularized selection:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

class Basis():
    def __init__(self, model_terms, param_dict, param_info=None, verbose=False):
        self.model_terms = model_terms
        self.param_dict = param_dict
        self.D = len(self.model_terms)
        self.param_info = param_info # To normalize data to [0,1].  Should be here?
        self.verbose = verbose


    @staticmethod
    def make_param_dict(param_names):
        # Return mapping from original parameter name to patsy-safe name
        return {p:p.replace(':','').replace('&',' ').replace(' ', '_').replace('-','_') for p in param_names}


    @classmethod
    def identity_basis(cls, params, param_info=None):
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
            param_info = None,
            verbose = False
    ):
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

        return cls(model_terms, param_dict, param_info, verbose)


    @classmethod
    def deserialize(cls, state):
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
        return {
            'Terms' : self.get_terms(),
            'Param_Dict' : self.param_dict,
            'Param_Info' : self.param_info.reset_index().to_json(orient='split')
        }


    def scale_data(self, data):
        for col in data.columns.tolist():
            if col in self.param_info.index:
                data[col] = (data[col] - self.param_info.loc[col,'Min'])/(self.param_info.loc[col,'Max']-self.param_info.loc[col,'Min'])
            elif self.verbose:
                print('Basis: Unable to scale %s'%col)
        return data


    def generate_dmatrix(self, data, scaleX = False):

        data = data.copy()
        if scaleX:
            assert(self.param_info is not None)
            data = self.scale_data(data)
        data = data.rename(columns=self.param_dict)

        md = ModelDesc([], self.model_terms)
        try:
            dmat = patsy.dmatrix(md, data = data, return_type = 'dataframe', NA_action="raise")
        except Exception as e:
            print str(e)
            if pd.isnull(data).any().any():
                #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print data[data.isnull().any(axis=1)]
                print 'Data contains Null/None/NaN, see data above.'
                exit()
        return dmat

    def generate_dmatrices(self, data, Ycol, scaleX = False):
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
        md = ModelDesc([], self.model_terms)
        terms = [c.strip() for c in md.describe().split('+')]
        terms[0] = terms[0][2:] # Remove '~ ' from beginning of first column

        if '0' in terms: # NO INTERCEPT
            terms.remove('0')
        else:
            terms = ['Intercept'] + terms

        return terms


    def fit(self, inputs, results, scaleX = False):
        # TODO: TEST

        if scaleX:
            assert(self.param_info is not None)
            inputs = self.scale_data(inputs.copy())

        Ycol = 'Sim_Result'
        my_results = results.copy()
        my_results.name = Ycol

        data = pd.merge(inputs.reset_index(), my_results.reset_index(), on=['Sample_Id', 'Exp_Id', 'Sample']).set_index(['Sample_Id', 'Exp_Id', 'Sample', 'Sim_Id']).sort_index()

        response_matrix, data_matrix = self.generate_dmatrices(data, Ycol)
        model = sm.OLS(response_matrix, data_matrix)

        fit = model.fit()
        if self.verbose:
            print 'SUMMARY:\n', fit.summary()
        print 'AIC:', fit.aic
        print 'BIC:', fit.bic
        params = pd.Series(fit.params, index=data_matrix.columns)
        params = params[abs(params)>0]
        #print 'FV:\n', fit.fittedvalues
        print 'Non-Zero:', len(params), 'of', self.D

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


    def regularize(self, inputs, results, alpha, scaleX = False):

        if scaleX:
            assert(self.param_info is not None)
            inputs = self.scale_data(inputs.copy())

        Ycol = 'Sim_Result'
        my_results = results.copy()
        my_results.name = Ycol
        #data = pd.merge(inputs.reset_index(), my_results.reset_index(), on=['Sample_Id', 'Exp_Id', 'Sample']).set_index(['Sample_Id', 'Exp_Id', 'Sample', 'Sim_Id']).sort_index()
        data = pd.merge(inputs.reset_index(), my_results.reset_index(), on='Sample_Id').set_index(['Sample_Id', 'Sim_Id']).sort_index()

        response_matrix, data_matrix = self.generate_dmatrices(data, Ycol)
        model = sm.OLS(response_matrix, data_matrix)

        if alpha > 0:
            fit = model.fit_regularized(alpha=alpha, refit=True)
        else:
            fit = model.fit()
        if self.verbose:
            print 'SUMMARY:\n', fit.summary()
        print 'AIC:', fit.aic
        print 'BIC:', fit.bic
        params = pd.Series(fit.params, index=data_matrix.columns)
        params = params[abs(params)>0]
        #print 'FV:\n', fit.fittedvalues
        print 'Non-Zero:', len(params), 'of', self.D
        #print alpha, len(params), fit.bic

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

    def plot_regularize(self, inputs, results, alpha, scaleX = False, title = None):

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
            print('Regularize: %d of %d' % (i, len(alpha)))
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
