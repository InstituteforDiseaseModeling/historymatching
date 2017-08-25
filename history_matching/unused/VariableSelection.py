from glm import GLM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.discrete.discrete_model as dm
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from patsy import ModelDesc, Term, LookupFactor, EvalFactor, dmatrices
import itertools

class VariableSelection():

    def __init__(self,
        param_info,         # Parameter definitions
        inputs,
        results,
        family = 'Poisson'
    ):

        self.inputs = inputs.copy()
        self.results = results.copy()

        self.results.name = 'Sim_Result'
        self.data = pd.merge(self.inputs.reset_index(), self.results.reset_index(), on='Sample').set_index(['Sample', 'Sim_Id']).sort_index()
        self.Xcols = inputs.columns.tolist()
        self.Ycol = self.results.name

        # Using all data for variable selection ...
        train_mean = self.data.reset_index().groupby(['Sample']).mean()

        self.glm_model = GLM(   Xcols = self.Xcols,
                                Ycol = self.Ycol,
                                training_data = train_mean,
                                family = family)

        #return self.stepwise_selection(max_var = 2)


    def build_basis(self):
        # TODO: IF KEEPING, DO NOT HAVE A COPY HERE AND IN GLM!
        Xcols = [s.replace(':','').replace('&',' ').replace(' ', '_') for s in self.Xcols]

        self.response_terms = [Term([LookupFactor(self.Ycol)])]
        self.model_terms = [Term([])] # Intercept

        # First order
        if self.first_order_basis_terms:
            self.model_terms += [Term([LookupFactor(x)]) for x in Xcols] # X

        # Second order
        if self.second_order_basis_terms:
            self.model_terms += [Term([EvalFactor('%s**2'%x)]) for x in Xcols] # X^2
            self.model_terms += [Term([EvalFactor('%s*%s'%x)]) for x in itertools.combinations(Xcols, 2)] # X*Y

        if self.third_order_basis_terms:
            # Third order
            self.model_terms += [Term([EvalFactor('%s**3'%x)]) for x in Xcols] # X^3

            self.model_terms += [Term([EvalFactor('%s*%s**2'%x)]) for x in itertools.combinations(Xcols, 2)] # X*Y^2
            self.model_terms += [Term([EvalFactor('%s**2*%s'%x)]) for x in itertools.combinations(Xcols, 2)] # X^2*Y

            self.model_terms += [Term([EvalFactor('%s*%s*%s'%x)]) for x in itertools.combinations(Xcols, 3)] # X*Y*Z

        if self.fourth_order_basis_terms:
            # Fourth order
            self.model_terms += [Term([EvalFactor('%s**4'%x)]) for x in Xcols] # X^4
            self.model_terms += [Term([EvalFactor('%s**3*%s'%x)]) for x in itertools.combinations(Xcols, 2)] # X^3*Y
            self.model_terms += [Term([EvalFactor('%s*%s**3'%x)]) for x in itertools.combinations(Xcols, 2)] # X*Y^3

            self.model_terms += [Term([EvalFactor('%s**2*%s**2'%x)]) for x in itertools.combinations(Xcols, 2)] # X^2*Y^2

            self.model_terms += [Term([EvalFactor('%s**2*%s*%s'%x)]) for x in itertools.combinations(Xcols, 3)] # X^2*Y*Z
            self.model_terms += [Term([EvalFactor('%s*%s**2*%s'%x)]) for x in itertools.combinations(Xcols, 3)] # X*Y^2*Z
            self.model_terms += [Term([EvalFactor('%s*%s*%s**2'%x)]) for x in itertools.combinations(Xcols, 3)] # X*Y*Z^2

            self.model_terms += [Term([EvalFactor('%s*%s*%s*%s'%x)]) for x in itertools.combinations(Xcols, 4)] # W*X*Y*Z

        if self.fifth_order_basis_terms:
            # Fifth order
            self.model_terms += [Term([EvalFactor('%s**5'%x)]) for x in Xcols] # X^5
            self.model_terms += [Term([EvalFactor('%s**4*%s'%x)]) for x in itertools.combinations(Xcols, 2)] # X^4*Y
            self.model_terms += [Term([EvalFactor('%s*%s**4'%x)]) for x in itertools.combinations(Xcols, 2)] # X*Y^4

            self.model_terms += [Term([EvalFactor('%s**3*%s**2'%x)]) for x in itertools.combinations(Xcols, 2)] # X^3*Y^2
            self.model_terms += [Term([EvalFactor('%s**2*%s**3'%x)]) for x in itertools.combinations(Xcols, 2)] # X^2*Y^3

            self.model_terms += [Term([EvalFactor('%s**3*%s*%s'%x)]) for x in itertools.combinations(Xcols, 3)] # X^3*Y*Z
            self.model_terms += [Term([EvalFactor('%s*%s**3*%s'%x)]) for x in itertools.combinations(Xcols, 3)] # X*Y^3*Z
            self.model_terms += [Term([EvalFactor('%s*%s*%s**3'%x)]) for x in itertools.combinations(Xcols, 3)] # X*Y*Z^3

            self.model_terms += [Term([EvalFactor('%s**2*%s*%s*%s'%x)]) for x in itertools.combinations(Xcols, 4)] # W^2*X*Y*Z
            self.model_terms += [Term([EvalFactor('%s*%s**2*%s*%s'%x)]) for x in itertools.combinations(Xcols, 4)] # W*X^2*Y*Z
            self.model_terms += [Term([EvalFactor('%s*%s*%s**2*%s'%x)]) for x in itertools.combinations(Xcols, 4)] # W*X*Y^2*Z
            self.model_terms += [Term([EvalFactor('%s*%s*%s*%s**2'%x)]) for x in itertools.combinations(Xcols, 4)] # W*X*Y*Z^2

            self.model_terms += [Term([EvalFactor('%s**2*%s**2*%s'%x)]) for x in itertools.combinations(Xcols, 3)] # X^2*Y^2*Z
            self.model_terms += [Term([EvalFactor('%s**2*%s*%s**2'%x)]) for x in itertools.combinations(Xcols, 3)] # X^2*Y*Z^2
            self.model_terms += [Term([EvalFactor('%s*%s**2*%s**2'%x)]) for x in itertools.combinations(Xcols, 3)] # X*Y^2*Z^2

            self.model_terms += [Term([EvalFactor('%s*%s*%s*%s*%s'%x)]) for x in itertools.combinations(Xcols, 5)] # V*W*X*Y*Z

        if self.higher_order_basis_terms:
            # Some sixth order
            self.model_terms += [Term([EvalFactor('%s**6'%x)]) for x in Xcols] # X^6

            self.model_terms += [Term([EvalFactor('%s**5*%s'%x)]) for x in itertools.combinations(Xcols, 2)] # X^5*Y
            self.model_terms += [Term([EvalFactor('%s*%s**5'%x)]) for x in itertools.combinations(Xcols, 2)] # X*Y^5

            self.model_terms += [Term([EvalFactor('%s**3*%s*%s*%s'%x)]) for x in itertools.combinations(Xcols, 4)] # W^3*X*Y*Z
            self.model_terms += [Term([EvalFactor('%s*%s**3*%s*%s'%x)]) for x in itertools.combinations(Xcols, 4)] # W*X^3*Y*Z
            self.model_terms += [Term([EvalFactor('%s*%s*%s**3*%s'%x)]) for x in itertools.combinations(Xcols, 4)] # W*X*Y^3*Z
            self.model_terms += [Term([EvalFactor('%s*%s*%s*%s**3'%x)]) for x in itertools.combinations(Xcols, 4)] # W*X*Y*Z^3

            # Some seventh?! order
            self.model_terms += [Term([EvalFactor('%s**7'%x)]) for x in Xcols] # X^7

            self.model_terms += [Term([EvalFactor('%s**6*%s'%x)]) for x in itertools.combinations(Xcols, 2)] # X^6*Y
            self.model_terms += [Term([EvalFactor('%s*%s**6'%x)]) for x in itertools.combinations(Xcols, 2)] # X*Y^6


    def OLS_regularized_selection(self, param_info,
            alpha=0,
            first_order_basis_terms = True,
            second_order_basis_terms = True,
            third_order_basis_terms = False,
            fourth_order_basis_terms = False,
            fifth_order_basis_terms = False,
            higher_order_basis_terms = False
        ):

        data = self.data.copy()

        for xc in self.Xcols:
            data[xc] = (data[xc] - param_info.loc[xc,'Min']) / float(param_info.loc[xc,'Max'] - param_info.loc[xc,'Min'])

        data[self.Ycol] = data[self.Ycol] - data[self.Ycol].mean()
        data[self.Ycol] = data[self.Ycol].apply(np.round).astype(int)

        self.first_order_basis_terms = first_order_basis_terms
        self.second_order_basis_terms = second_order_basis_terms
        self.third_order_basis_terms = third_order_basis_terms
        self.fourth_order_basis_terms = fourth_order_basis_terms
        self.fifth_order_basis_terms = fifth_order_basis_terms
        self.higher_order_basis_terms = higher_order_basis_terms

        self.build_basis()
        data = data.rename(columns={s:s.replace(':','').replace('&',' ').replace(' ', '_') for s in self.Xcols})
        md = ModelDesc(self.response_terms, self.model_terms)
        (response_matrix, data_matrix) = dmatrices(md, data=data, return_type='dataframe')

        model = sm.OLS(response_matrix, data_matrix)
        #fit = model.fit_regularized(alpha=alpha)
        #for alpha in np.logspace(5,1,10):
        fit = model.fit_regularized(alpha=alpha, refit=True)
        print 'SUMMARY:\n', fit.summary()
        print 'AIC:', fit.aic
        print 'BIC:', fit.bic
        params = pd.Series(fit.params, index=data_matrix.columns)
        params = params[params>0]
        #print 'FV:\n', fit.fittedvalues
        print 'Non-Zero:', len(params), 'of', len(self.Xcols)
        #print alpha, len(params), fit.bic

        # Dang you patsy!
        invdict = {s.replace(':','').replace('&',' ').replace(' ', '_'):s for s in self.Xcols}
        param_list = []

        for p,_ in params.iteritems():
            print p
            if '*' in p:
                p_orig = [invdict[t] if t in invdict else t for t in map(str.strip, p.split('*'))]
            else:
                p_orig = invdict[p]
            param_list.append(p_orig)

        return param_list


    def stepwise_selection(self,
        max_vars=3,
        first_order_basis_terms = True,
        second_order_basis_terms = False,
        third_order_basis_terms = False,
        fourth_order_basis_terms = False,
        fifth_order_basis_terms = False,
        higher_order_basis_terms = False
    ):
        from itertools import combinations

        self.glm_model.first_order_basis_terms = first_order_basis_terms
        self.glm_model.second_order_basis_terms = second_order_basis_terms
        self.glm_model.third_order_basis_terms = third_order_basis_terms
        self.glm_model.fourth_order_basis_terms = fourth_order_basis_terms
        self.glm_model.fifth_order_basis_terms = fifth_order_basis_terms
        self.glm_model.higher_order_basis_terms = higher_order_basis_terms

        Xcols_all = self.Xcols[:] # Copy
        verbose = self.glm_model.verbose
        self.glm_model.verbose = False

        selected_X = []
        bic = np.zeros(max_vars)

        for i in range(max_vars+1):
            best_new_X = None
            lowest_bic = np.NaN
            for X in Xcols_all:
                self.glm_model.Xcols = selected_X + [X]
                self.glm_model.build_basis()
                self.glm_model.fit()
                #print self.fitted_model.bic, ':', self.Xcols
                if best_new_X is None or np.isnan(lowest_bic) or self.glm_model.fitted_model.bic < lowest_bic:
                    best_new_X = X
                    lowest_bic = self.glm_model.fitted_model.bic

            bic[i] = lowest_bic
            #print 'BEST_X:', best_new_X, ' with BIC =', lowest_bic
            selected_X.append(best_new_X)
            Xcols_all.remove(best_new_X)
            print 'Selected:', selected_X, 'BIC =',lowest_bic


        fig = plt.figure()
        plt.plot(range(max_vars), bic, 'ko-')
        plt.xlabel('Number of Parameters')
        plt.ylabel('BIC')
        plt.show()

        self.glm_model.verbose = verbose

        return selected_X

    def penalized_selection(self, param_info, alpha=0):
        #data = self.data.loc[ self.data[self.Ycol] < 25, : ]
        data = self.data.copy()
        #print 'MAX:', np.max(data[self.Ycol].values)

        for xc in self.Xcols:
            data[xc] = (data[xc] - param_info.loc[xc,'Min']) / float(param_info.loc[xc,'Max'] - param_info.loc[xc,'Min'])

        data[self.Ycol] = data[self.Ycol] - data[self.Ycol].mean()
        data[self.Ycol] = data[self.Ycol].apply(np.round).astype(int)

        mod = dm.Poisson(endog=data[self.Ycol], exog=add_constant(data[self.Xcols]))
        #res = mod.fit(method='bfgs', maxiter=100, disp=True)
        res = mod.fit_regularized(alpha=alpha, full_output=True, disp=True, qc_verbose=True, maxiter=10000)
        '''
            method='l1_cvxopt_cp',
            abstol=1e-3,
            reltol=1e-3,
            feastol=1e03,
            refinement=1) # , acc=1e-6, trim_mode='size'
        '''

        print(res.summary())
        return res.params[res.params>0].index.unique().values.tolist()

