from glm import GLM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.discrete.discrete_model as sm
from statsmodels.tools.tools import add_constant

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

        for i in range(max_vars):
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

        '''
        plt.figure()
        plt.plot(range(max_vars), bic, 'ko-')
        plt.xlabel('Number of Parameters')
        plt.ylabel('BIC')
        plt.show()
        '''

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

        mod = sm.Poisson(endog=data[self.Ycol], exog=add_constant(data[self.Xcols]))
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

