# http://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/background/BayesianLinearRegression.ipynb

import json
import patsy
import os, StringIO

import statsmodels.api as sm
import statsmodels.formula.api as smf

from basis import Basis

import numpy as np, GPy, pandas as pd, seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

class GLM(object):

    def __init__(self, basis, Ycol,
            training_data = None,
            reference_value = 0,
            family = 'Poisson', # 'Poisson', 'NegativeBinomial', 'Gaussian'
            #first_order_basis_terms = True,
            #second_order_basis_terms = True,
            #third_order_basis_terms = False,
            #fourth_order_basis_terms = False,
            #fifth_order_basis_terms = False,
            #higher_order_basis_terms = False,
            fitted_model = None,
            verbose = True
        ):

        self.training_data = training_data
        self.reference_value = reference_value
        #self.Xcols = Xcols
        self.basis = basis
        self.Ycol = Ycol
        self.D = self.basis.D
        self.family = family

        self.verbose = verbose

        self.fitted_model = fitted_model

        #sns.set_style("whitegrid")

        if family == 'Poisson':
            self.glmfam = sm.families.Poisson()
            print 'Using Poisson family'
        elif family == 'NegativeBinomial':
            alpha = 1.9
            self.glmfam = sm.families.NegativeBinomial(alpha=alpha) # Does strange things with float vs int values of alpha!
            print 'Using NegativeBinomial family, alpha = %f' % alpha
        else:
            self.glmfam = sm.families.Gaussian()
            print 'Using Gaussian family'

        if self.fitted_model is not None:
            print self.fitted_model.summary()
            print 'AIC:', self.fitted_model.aic
            print 'BIC:', self.fitted_model.bic
            print 'ITERATION:', self.fitted_model.fit_history['iteration']


    @classmethod
    def from_config(cls, meta_fn, fitted_fn):
        print "from_config:", meta_fn, fitted_fn
        try:
            fitted_model = sm.load(fitted_fn)
            with open(os.path.join(meta_fn)) as data_file:
                config = json.load( data_file )

                return cls(
                    #config['Xcols'], # TODO: DESERIALIZE BASIS!
                    basis = Basis.deserialize(config['Basis']),
                    Ycol = config['Ycol'],
                    training_data = pd.read_json( config['Training_Data'], orient='split' ).set_index('Sample'),
                    reference_value = config['Reference_Value'],
                    family = config['Family'],

                    #first_order_basis_terms = config['First_Order_Basis_Terms'],
                    #second_order_basis_terms = config['Second_Order_Basis_Terms'],
                    #third_order_basis_terms = config['Third_Order_Basis_Terms'],
                    #fourth_order_basis_terms = config['Fourth_Order_Basis_Terms'],
                    #fifth_order_basis_terms = config['Fifth_Order_Basis_Terms'],
                    #higher_order_basis_terms = config['Higher_Order_Basis_Terms'],

                    fitted_model = fitted_model
                )
        except EnvironmentError:
            print "Unable to load GLM from_config file", meta_fn, fitted_fn
            raise


    def save(self, save_meta_to, save_fitted_to):
        self.fitted_model.save(save_fitted_to)
        with open(save_meta_to, 'w') as fout:
            json.dump(
                {
                    'Basis'         : self.basis.serialize(),
                    'Ycol'          : self.Ycol,
                    'Training_Data' : self.training_data.reset_index().to_json(orient='split'), # [self.Xcols + [self.Ycol]]
                    'Reference_Value': self.reference_value,
                    'D'             : self.D,
                    'Family'        : self.family,
                }, fout, indent=4)

    def evaluate(self, data):
        dmat = self.basis.generate_dmatrix(data, scaleX=True)
        mean = self.fitted_model.predict( dmat, transform=False )

        return mean


    def fit(self, maxiter=100):
        (response_matrix, data_matrix) = self.basis.generate_dmatrices(self.training_data, self.Ycol, scaleX=True)

        self.model = sm.GLM(response_matrix, data_matrix, family=self.glmfam)

        if self.verbose:
            print 'Fitting the model, please wait ...'
        self.fitted_model = self.model.fit(maxiter=maxiter)

        if self.verbose:
            print self.fitted_model.summary()
            print 'AIC:', self.fitted_model.aic
            print 'BIC:', self.fitted_model.bic
            print 'ITERATION:', self.fitted_model.fit_history['iteration']

    def plot_fitted_vs_observed(self):
        fig, ax = plt.subplots()
        y = self.training_data[self.Ycol]
        ax.scatter(y, self.fitted_model.mu, marker='+')
        #line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()
        #abline_plot(model_results=line_fit, ax=ax)

        ax.set_title('Model Fit Plot')
        ax.set_xlabel('Observed values')
        ax.set_ylabel('Fitted values');

        return fig


    def plot_pearson_residuals(self):
        fig, ax = plt.subplots()
        ax.scatter(self.fitted_model.mu, self.fitted_model.resid_pearson, marker='+')
        #ax.hlines(0, 0, 1)
        #ax.set_xlim(0, 1)
        ax.set_title('Residual Dependence Plot')
        ax.set_ylabel('Pearson Residuals')
        ax.set_xlabel('Fitted values')

        return fig


    def plot_deviance_redisuals(self):
        from scipy import stats
        fig, ax = plt.subplots()
        resid = self.fitted_model.resid_deviance.copy()
        resid_std = stats.zscore(resid)
        ax.hist(resid_std)#, bins=25)
        ax.set_title('Standardized deviance residuals');

        return fig

    def plot_QQ(self):
        from statsmodels import graphics
        import scipy
        fig = graphics.gofplots.qqplot(self.fitted_model.resid_deviance, line='45', fit=True)

        return fig


    def plot_data_multiD(self, circle_points=[]):
        #scaled = self.training_data[self.Ycol] / self.training_data[self.Ycol].max()
        scaled = np.log(1+self.training_data[self.Ycol])# / self.training_data[self.Ycol].max()

        figs = {}

        Xcols = basis.get_terms()
        for row in range(self.D):
            for col in range(self.D):
                if col > row:
                    fn = '%s-%s.pdf' % (Xcols[row], Xcols[col])
                    figs[fn] = plt.figure(figsize=(6,6)) #GPy.plotting.plotting_library().figure()

                    x = self.training_data[ Xcols[row] ]
                    y = self.training_data[ Xcols[col] ]

                    plt.scatter(x, y, s=np.maximum(1, 5*scaled), c=scaled, cmap='jet', linewidths=0.1, alpha=0.5, edgecolors='k') #, s=area, c=colors, alpha=0.5)

                    for idx, pt in circle_points.iterrows():
                        plt.scatter(pt[ Xcols[row] ], pt[ Xcols[col] ], s=50, c='k', alpha=1, linewidths=2.0, marker='x') #, s=area, c=colors, alpha=0.5)
                        #scl = np.log(1+pt[self.Ycol])# / self.training_data[self.Ycol].max()
                        #plt.scatter(pt[ Xcols[row] ], pt[ Xcols[col] ], s=10*scl, alpha=1, linewidths=2.0, facecolors="None", edgecolors='k') #, s=area, c=colors, alpha=0.5)

                    plt.autoscale(tight=True)
                    plt.xlabel( Xcols[row] )
                    plt.ylabel( Xcols[col] )
                    plt.tight_layout()

        return figs


    def plot_data_1D(self, circle_points=[]):
        scaled = np.log(1+self.training_data[self.Ycol])# / self.training_data[self.Ycol].max()

        Xcols = basis.get_terms()[0] # Not tested!
        fig = plt.figure(figsize=(6,8)) #GPy.plotting.plotting_library().figure()
        x = self.training_data[ Xcols ]
        y = self.training_data[self.Ycol]

        plt.scatter(x, y, s=15, c=scaled, cmap='jet', linewidths=0.1, alpha=0.5, edgecolors='k') #, s=area, c=colors, alpha=0.5)

        for idx, pt in circle_points.iterrows():
            plt.scatter(pt[ Xcols ], pt[ self.Ycol ], s=25, c='k', alpha=1, linewidths=2.0, marker='x') #, s=area, c=colors, alpha=0.5)

        plt.autoscale(tight=True)
        plt.xlabel( Xcols )
        plt.ylabel( self.Ycol )
        plt.tight_layout()

        return {Xcols: fig}


    def plot_data(self, **kwargs):
        if self.D > 1:
            return self.plot_data_multiD(**kwargs)
        return self.plot_data_1D(**kwargs)

    def plot_histogram(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sns.distplot(self.training_data[self.Ycol], rug=True, ax = ax)

        return fig

    def plot_fit(self):
        fig, axes = plt.subplots(figsize=(16, 16))
        #sns.despine(left=True)

        d = self.training_data.reset_index()
        d_by_sample = self.training_data.reset_index().set_index('Sample')
        n_samples = len(d_by_sample.index.unique())

        axes.plot( 2 * [self.reference_value], [0,n_samples], 'r-') # , axes=axes[0,0]

        sim_cases_range = self.training_data.reset_index().groupby('Sample')[self.Ycol].agg({'Min':np.min, 'Max':np.max, 'Mean':np.mean})
        sim_cases_range['Fitted_Model_Mean'] = self.fitted_model.mu
        for idx,s in sim_cases_range.iterrows():
            axes.plot( [s['Min'], s['Max']], [idx,idx], 'b-', linewidth=0.5 )
            axes.plot( [s['Mean'], s['Fitted_Model_Mean']], [idx,idx], 'g-', linewidth=0.25 )
        axes.scatter(d[self.Ycol], d['Sample'], c='k', marker='|', alpha=1, linewidths=0.5)

        axes.scatter(self.fitted_model.mu, d['Sample'], c='g', marker='+', alpha=1, linewidths=0.5)

        # TODO: Vectorize
        '''
        for idx,s in d.iterrows():
            k = s['Ref_Cases']
            n = s['Ref_Population']
            a = s['Sim_Cases_Unscaled']+1
            b = s['Sim_Population_Unscaled']+1

            mean = n*a / (a+b)
            var = n*a*b*(a+b+n) / ((a+b)**2 * (a+b+1))

            axes[0].errorbar(s['Sim_Cases'], int(float(s['Sample'])), xerr=2*np.sqrt(var), marker='|', markersize=20, ecolor='k', mew=1)
        '''
        plt.autoscale()
        axes.set_ylim(ymin=0, ymax=n_samples)
        #axes.set_xlabel('LOG(1+Y)')
        axes.set_xlabel('Y')
        axes.set_ylabel('Sample')

        return fig


    def plot_errors(self, train, test):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex='col', figsize=(16,10)) # , sharex='col', sharey='row')

        ax = ax1
        ax.plot(train['Yglm'], train[self.Ycol], 'c+', ms=10, mew=1)
        ax.plot(test['Yglm'], test[self.Ycol], 'm+', ms=10, mew=1)
        ax.margins(x=0,y=0.05)
        xlim = ax.get_xlim()
        ax.plot( [xlim[0],xlim[1]], [xlim[0], xlim[1]], 'r-')
        ax.set_xlabel('Predicted')
        ax.set_ylabel(self.Ycol)

        ax = ax2
        ax.scatter(x=train['Sample'], y=train[self.Ycol], c='c', marker='_', s=25, alpha=1, linewidths=1, zorder=50)
        ax.scatter(x=test['Sample'], y=test[self.Ycol], c='m', marker='_', s=25, alpha=1, linewidths=1, zorder=50)
        ax.plot(train['Sample'], train['Yglm'], 'k.', ms=5, linewidth=1)
        ax.plot(test['Sample'], test['Yglm'], 'k.', ms=5, linewidth=1)
        ax.margins(x=0,y=0.05)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(self.Ycol)

        return fig
