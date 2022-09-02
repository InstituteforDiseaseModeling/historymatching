# http://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/background/BayesianLinearRegression.ipynb
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import json
import patsy
import os

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels import graphics

from history_matching.basis import Basis

import numpy as np, pandas as pd, seaborn as sns
import scipy
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class GLM(object):
    """Generalized Linear Modeling (GLM).

    This class implementes Generalized Linear Modeling using statsmodels as the engine.
    """

    def __init__(self,
            basis,
            Ycol,
            training_data = None,
            reference_value = 0,
            family = 'Poisson', # 'Poisson', 'NegativeBinomial', 'Gaussian'
            fig_type = 'pdf',
            fitted_model = None
        ):
        """Initialize the GLM class.

        Args:
            basis: (basis)
                Provide an instance of a basis class which determines the parameters and computes the data matrix for the GLM.
            Ycol:  (str)
                The name of the column in training_data that contains the model output values.  Ycol must be a column in training_data
            training_data:  (Pandas dataframe)
                Columns must include:
                * Sample_Id: A unique string that identifies each sample.
                * Sim_Id: A unique string the identifies each simulation, typically the COMPS simulation ID.
                * Sample: (optional?) The sample index.
                * Exp_Id: (optional?) The name of the experiment.
                * PARAMETER NAMES: One column for each parameter name.
            reference_value: (float) The reference value from data, used in plotting only
            family: (str) The family of generalized linear model to use.  Options include 'Poisson', 'Binomial', 'Gamma', 'NegativeBinomial', and 'Gaussian'.  Note that NegativeBinomial is currently hard-coded to use alpha=1.9.
            fitted_model: (GLM) When restoring from cache, this enables file-based configuration.
        """

        self.training_data = training_data
        self.reference_value = reference_value
        self.basis = basis
        self.Ycol = Ycol
        self.D = self.basis.D
        self.family = family
        self.fig_type = fig_type

        self.fitted_model = fitted_model

        if family == 'Poisson':
            logger.info('Using Poisson family')
            self.glmfam = sm.families.Poisson()
        elif family == 'Binomial':
            logger.info('Using Binomial family')
            self.glmfam = sm.families.Binomial()
        elif family == 'Gamma':
            logger.info('Using Gamma family')
            self.glmfam = sm.families.Gamma()
        elif family == 'NegativeBinomial':
            alpha = 1.9
            logger.info(f'Using NegativeBinomial family, alpha = {alpha}')
            self.glmfam = sm.families.NegativeBinomial(alpha=alpha) # Does strange things with float vs int values of alpha!
        else:
            logger.info('Using Gaussian family')
            self.glmfam = sm.families.Gaussian()

        if self.fitted_model is not None:
            logger.info(self.fitted_model.summary()) # Should work, but was causing errors with some versions of statsmodels.
            logger.info(f'AIC:       {self.fitted_model.aic}')
            logger.info(f'BIC (LLF): {self.fitted_model.bic_llf}')
            #logger.info(f'ITERATION: {self.fitted_model.fit_history["iteration"]}')


    @classmethod
    def from_config(cls, meta_fn, fitted_fn):
        """Restore a GLM instance from a saved configuration files.

        Args:
            meta_fn: (str)
                JSON file containing configuration such as the serialized basis, column names, order, etc.
            fitted_fn: (str)
                Contains the saved (pickled) statsmodel.
        """

        try:
            fitted_model = sm.load(fitted_fn)
            with open(os.path.join(meta_fn)) as data_file:
                config = json.load( data_file )

                if 'Basis' in config:
                    basis = Basis.deserialize(config['Basis'])
                else:
                    # Backwards compatibility
                    Xcols = config['Xcols']
                    basis = Basis.polynomial_basis(
                        params = Xcols,
                        intercept = True,
                        first_order = config['First_Order_Basis_Terms'],
                        second_order = config['Second_Order_Basis_Terms'],
                        third_order = config['Third_Order_Basis_Terms'],
                        fourth_order = config['Fourth_Order_Basis_Terms'],
                        fifth_order = config['Fifth_Order_Basis_Terms'],
                        higher_order = config['Higher_Order_Basis_Terms'],
                        param_info = pd.read_json( config['Param_Info'], orient='split' ).set_index('Name')
                    )

                return cls(
                    basis = basis,
                    Ycol = config['Ycol'],
                    training_data = pd.read_json( config['Training_Data'], orient='split' ).set_index('Sample_Id'),
                    reference_value = config['Reference_Value'],
                    family = config['Family'],
                    fitted_model = fitted_model
                )
        except EnvironmentError:
            logger.critical(f"Unable to load GLM from_config file {meta_fn} {fitted_fn}")
            raise


    def save(self, save_meta_to, save_fitted_to):
        """Save a GLM instance to configuration files.

        Args:
            save_meta_to: (str)
                JSON filename to contain configuration such as the serialized basis, column names, order, etc.
            save_fitted_to: (str)
                Filename to contains the saved (pickled) statsmodel.
        """

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
        """Evaluate the GLM and return the mean prediction.

        Args:
            data: (Pandas DataFrame)
                Data frame of points similar to training_data.

        Returns:
            Predicted outputs at the inputs specified by data.
        """

        dmat = self.basis.generate_dmatrix(data, scaleX=True)
        mean = self.fitted_model.predict( dmat, transform=False )

        return mean


    def fit(self, maxiter=1000):
        """Fit the GLM.

        Args:
            maxiter: (int)
                maxiter parameter passed to the statsmodels `fit` function.
        """

        (response_matrix, data_matrix) = self.basis.generate_dmatrices(self.training_data, self.Ycol, scaleX=True)
        self.model = sm.GLM(response_matrix, data_matrix, family=self.glmfam)

        logger.info('Fitting the model, please wait ...')
        self.fitted_model = self.model.fit(maxiter=maxiter)

        logger.info(self.fitted_model.summary())
        logger.info(f'AIC:       {self.fitted_model.aic}')
        logger.info(f'BIC (LLF): {self.fitted_model.bic_llf}')
        #logger.info(f'ITERATION: {self.fitted_model.fit_history["iteration"]}')

    def plot_fitted_vs_observed(self):
        """Generates a plot of the fitted values vs the observed values from the training data.

        Returns: A matplotlib figure handle.
        """

        fig, ax = plt.subplots(figsize=(16,6), dpi=300)
        y = self.training_data[self.Ycol]
        ax.scatter(y, self.fitted_model.mu, marker='+')
        #line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()
        #abline_plot(model_results=line_fit, ax=ax)

        ax.set_title('Model Fit Plot')
        ax.set_xlabel('Observed values')
        ax.set_ylabel('Fitted values')

        return fig


    def plot_pearson_residuals(self):
        """Generates a plot of the peasron residuals.

        Returns: A matplotlib figure handle.
        """

        fig, ax = plt.subplots(figsize=(16,12), dpi=300)
        ax.scatter(self.fitted_model.mu, self.fitted_model.resid_pearson, marker='+')
        #ax.hlines(0, 0, 1)
        #ax.set_xlim(0, 1)
        ax.set_title('Residual Dependence Plot')
        ax.set_ylabel('Pearson Residuals')
        ax.set_xlabel('Fitted values')

        return fig


    def plot_deviance_residuals(self):
        """Generates a plot of the deviance residuals.

        Returns: A matplotlib figure handle.
        """

        fig, ax = plt.subplots(figsize=(16,12), dpi=300)
        resid = self.fitted_model.resid_deviance.copy()
        resid_std = stats.zscore(resid)
        ax.hist(resid_std)#, bins=25)
        ax.set_title('Standardized deviance residuals')

        return fig


    def plot_QQ(self):
        """Generates a QQ plot.

        Returns: A matplotlib figure handle.
        """

        fig = graphics.gofplots.qqplot(self.fitted_model.resid_deviance, line='45', fit=True)

        return fig


    def plot_data_multiD(self, circle_points=pd.DataFrame(), saveto_dir = None, log_scale=True):
        """Generates many pair-wise scatter plots of the training data.

        Args:
            circle_points: (Pandas DataFrame)
                A data frame like training_data.  Each entry will be marked with a black x's in the figures.  Good for debugging large Z scores.
            saveto_dir: (str)
                If not None, figures will be saved to this directory.  The user may need to create the output directory.
            log_scale:  (boolean, default is False) transforms size and color using log(10 * normalized_y_value + 1)

        Returns: a dictionary of matplotlib figure handles with keys indicating the parameter names via the filename which would be used to save the figure.
        """

        scaled = (self.training_data[self.Ycol]-self.training_data[self.Ycol].min()) / (self.training_data[self.Ycol].max()-self.training_data[self.Ycol].min())
        if log_scale:
            scaled = np.log( 10*scaled+1 )

        figs = {}

        basis = Basis.identity_basis(params=self.basis.param_info.index.unique().tolist(), param_info=self.basis.param_info)
        Xcols = basis.get_terms()
        dmat = basis.generate_dmatrix(self.training_data, scaleX=True)

        if circle_points.shape[0] > 0:
            cp_dmat = basis.generate_dmatrix(circle_points, scaleX=True)

        reverse_param_dict = {v:k for k,v in basis.param_dict.items()}

        for row in range(len(Xcols)):
            for col in range(len(Xcols)):
                if col > row:
                    fn = '%s-%s' % (Xcols[row], Xcols[col]) + '.'+self.fig_type
                    fig = plt.figure(figsize=(16, 12), dpi=300)

                    x_name = reverse_param_dict[ Xcols[row] ]
                    y_name = reverse_param_dict[ Xcols[col] ]
                    x = dmat[ Xcols[row] ] * (basis.param_info.loc[x_name]['Max'] - basis.param_info.loc[x_name]['Min']) + basis.param_info.loc[x_name]['Min']
                    y = dmat[ Xcols[col] ] * (basis.param_info.loc[y_name]['Max'] - basis.param_info.loc[y_name]['Min']) + basis.param_info.loc[y_name]['Min']

                    plt.scatter(x, y, 100*scaled, c=100*scaled, cmap='jet', linewidths=0.1, alpha=0.5, edgecolors='k')

                    if circle_points.shape[0] > 0:
                        for idx, pt in cp_dmat.iterrows():
                            plt.scatter(pt[ Xcols[row] ], pt[ Xcols[col] ], s=50, c='k', alpha=1, linewidths=2.0, marker='x') #, s=area, c=colors, alpha=0.5)

                    #plt.autoscale(tight=True)
                    plt.xlim(basis.param_info.loc[x_name][['Min', 'Max']])
                    plt.ylim(basis.param_info.loc[y_name][['Min', 'Max']])
                    plt.xlabel( x_name )
                    plt.ylabel( y_name )
                    plt.tight_layout()
                    if saveto_dir is not None:
                        fig.savefig( os.path.join(saveto_dir, fn) ); plt.close(fig)
                    else:
                        figs[fn] = fig

        return figs


    def plot_data_1D(self, circle_points=pd.DataFrame(), saveto_dir = None, log_scale=True):
        """For 1D data, plots a scatter of output (y) vs input (x).

        Args:
            circle_points: (Pandas DataFrame)
                A data frame like training_data.  Each entry will be marked with a black x's in the figures.  Good for debugging large Z scores.
            saveto_dir: (str)
                If not None, figures will be saved to this directory.  The user may need to create the output directory.
            log_scale:  (boolean, default is False) transforms size and color using log(10 * normalized_y_value + 1)

        Returns: a dictionary of matplotlib figure handles with keys indicating the parameter names via the filename which would be used to save the figure.
        """

        # TODO: Save and log scale!
        scaled = np.log(1+self.training_data[self.Ycol])# / self.training_data[self.Ycol].max()

        Xcols = self.basis.get_terms()[0] # Not tested!
        fig = plt.figure(figsize=(16,12), dpi=300)
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
        """Helper to call plot_data_1D or plot_data_multiD depending on the number of independent variables.

        kwargs are required by the respective functions, although not called out explicitly here.

        Args:
            circle_points: (Pandas DataFrame)
                A data frame like training_data.  Each entry will be marked with a black x's in the figures.  Good for debugging large Z scores.
            saveto_dir: (str)
                If not None, figures will be saved to this directory.  The user may need to create the output directory.
            log_scale:  (boolean, default is False) transforms size and color using log(10 * normalized_y_value + 1)

        Returns: a dictionary of matplotlib figure handles with keys indicating the parameter names via the filename which would be used to save the figure.
        """

        '''
        if self.D > 1:
            return self.plot_data_multiD(**kwargs)
        return self.plot_data_1D(**kwargs)
        '''

        # 1D not working, do multiD:
        return self.plot_data_multiD(**kwargs)

    def plot_histogram(self):
        """Plots a histogram of the outputs.

        Returns: matplotlib figure handle.
        """

        fig = plt.figure(figsize=(16, 12), dpi=300)
        ax = fig.add_subplot(111)
        sns.displot(self.training_data[self.Ycol], rug=True)

        return fig

    def plot_fit(self):
        """Plots each output predicted by the GLM on X agains sample index is on Y.

        If there are multiple replicates per Sample_ID, a blue line will connect the Min to the Max.  A vertical red line is drawn at the reference value.  The green line is at the mean of the fitted model.  Finally, the black `|` is the true value(s) from the simulation.

        Returns: matplotlib figure handle.
        """

        fig, axes = plt.subplots(figsize=(16,12), dpi=300)
        #sns.despine(left=True)

        d = self.training_data.reset_index()
        d_by_sample = self.training_data.reset_index().set_index('Sample_Id')
        n_samples = len(d_by_sample.index.unique())

        axes.plot( 2 * [self.reference_value], [0,n_samples], 'r-') # , axes=axes[0,0]

        sim_cases_range = self.training_data.reset_index().groupby('Sample_Id')[self.Ycol].agg({'Min':np.min, 'Max':np.max, 'Mean':np.mean})
        sim_cases_range.loc[:,'Fitted_Model_Mean'] = self.fitted_model.mu
        for idx,s in sim_cases_range.iterrows():
            axes.plot( [s['Min'], s['Max']], [idx,idx], 'b-', linewidth=0.5 )
            axes.plot( [s['Mean'], s['Fitted_Model_Mean']], [idx,idx], 'g-', linewidth=0.25 )
        axes.scatter(d[self.Ycol], d['Sample_Id'], c='k', marker='|', alpha=1, linewidths=0.5)

        axes.scatter(self.fitted_model.mu, d['Sample_Id'], c='g', marker='+', alpha=1, linewidths=0.5)

        plt.autoscale()
        axes.set_ylim(ymin=0, ymax=n_samples)
        axes.set_xlabel('Y')
        axes.set_ylabel('Sample Id')

        return fig


    def plot_errors(self, train, test):
        """Generates several plots on a single figure, one for each unique experiment ID.

        The upper plot shows GLM prediction on Y as a function of the true Y-values on X.  The lower panel shows Z-score on Y and the true Y-values on X.

        In both panels, training data is cyan and test data is magenta.

        Args:
            train: (Pandas DataFrame) training data like training_data.
            test: (Pandas DataFrame) test data like training_data.

        Returns: Dictionary of matplotlib figure handles.
        """

        figs = {}

        _tr = train.reset_index()
        _ts = test.reset_index()

        first_sample_id = _tr.iloc[0]['Sample_Id']
        if isinstance(first_sample_id, str) and '.' in first_sample_id:
            _tr['Exp_Id'] = _tr['Sample_Id'].apply(lambda x: x.split('.')[0])
            _tr['Sample'] = _tr['Sample_Id'].apply(lambda x: int(x.split('.')[1]))

            _ts['Exp_Id'] = _ts['Sample_Id'].apply(lambda x: x.split('.')[0])
            _ts['Sample'] = _ts['Sample_Id'].apply(lambda x: int(x.split('.')[1]))

            _tr.set_index(['Exp_Id', 'Sample'], inplace=True)
            _ts.set_index(['Exp_Id', 'Sample'], inplace=True)

        else:
            _tr['Exp_Id'] = 0
            _tr['Sample'] = _tr['Sample_Id']

            _ts['Exp_Id'] = 0
            _ts['Sample'] = _ts['Sample_Id']

            _tr.set_index(['Exp_Id', 'Sample'], inplace=True)
            _ts.set_index(['Exp_Id', 'Sample'], inplace=True)

        train_exps = _tr.index.get_level_values(_tr.index.names.index('Exp_Id')).unique().tolist()
        test_exps = _ts.index.get_level_values(_tr.index.names.index('Exp_Id')).unique().tolist()
        exp_ids = list(set(train_exps + test_exps))

        fig, ax = plt.subplots(figsize=(16,12), dpi=300)
        ax.plot(train[self.Ycol], train['Yglm'], 'c+', ms=10, mew=1)
        ax.plot(test[self.Ycol], test['Yglm'], 'm+', ms=10, mew=1)
        ax.margins(x=0,y=0.05)
        xlim = ax.get_xlim()
        ax.plot( [xlim[0],xlim[1]], [xlim[0], xlim[1]], 'r-')
        ax.set_xlabel('Simulation Result')
        ax.set_ylabel('Predicted')

        figs['GLM Predicted vs Actual'] = fig

        for i, exp_id in enumerate(exp_ids):
            fig, ax = plt.subplots(figsize=(16,12), dpi=300)
            data_all = []
            cols = []
            if exp_id in train_exps: 
                data_all.append(_tr.loc[exp_id])
                cols.append('c')
            if exp_id in test_exps:
                data_all.append(_ts.loc[exp_id])
                cols.append('m')

            for data, col in zip(data_all, cols):
                data = data.reset_index()
                ax.scatter(x=data['Sample'], y=data[self.Ycol], c=col, marker='_', s=25, alpha=1, linewidths=1, zorder=50)
                ax.plot(data['Sample'], data['Yglm'], 'k.', ms=5, linewidth=1)
                ax.set_title(exp_id)

            ax.margins(x=0,y=0.05)
            ax.set_xlabel('Sample')

            figs['GLM expId ' + str(exp_id)] = fig

        #ax.set_ylabel(self.Ycol)
        #plt.tight_layout()

        return figs
