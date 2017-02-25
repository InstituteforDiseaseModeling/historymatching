import pandas as pd
import numpy as np
import os, errno
from glm import GLM

import matplotlib.pyplot as plt

class HistoryMatching():

    def __init__(self,
        param_info,         # Parameter definitions
        inputs,
        results,
        desired_result,
        training_fraction = 0.75,
        iteration = 0  # Current iteration, needed?
    ):
        """
        :param DataFrame param_info: Parameter info with index named 'Name' containing parameter name, and columns of 'Min' and 'Max'
        :param DataFrame inputs: Model inputs with index named 'Sample' containing parameter names.  All Names from param_info must be columns in this data frame, but it can have other columns as well.  Data contents are model input parameter values.
        :param Series results: Series with MultiIndex of 'Sample' and 'Sim_Id'.  Data are simulation results.
        :param float desired_result: The desired result to match.
        :param int iteration: The current iteration.  Work will be saved to iter[iteration]
        """

        print('Welcome to IDM History Matching!')

        self.param_info = param_info.copy()
        self.desired_result = desired_result
        self.training_fraction = training_fraction
        self.iteration = iteration

        Xcols_all = inputs.columns

        results.name = 'Sim_Result'
        self.data = pd.merge(inputs.reset_index(), results.reset_index(), on='Sample').set_index(['Sample', 'Sim_Id']).sort_index()
        self.Ycol = results.name

        # TODO: Verify that all Xcols are columns of data

        # Fix names for picky statsmodels patsy, statsmodels var names can't have space in formula
        Xcols = self.param_info.index.unique()
        newXcols = []
        newXcols_all = []
        rename_dict = {}
        for i,xc in enumerate(Xcols_all):
            if ':' in xc or '&' in xc:
                new_xc = xc.replace(':', '').replace('&',' ')
                rename_dict[xc] = new_xc
                newXcols_all.append(new_xc)
                if xc in Xcols:
                    newXcols.append(new_xc)
            else:
                newXcols_all.append(xc)
                if xc in Xcols:
                    newXcols.append(xc)

        self.data.rename(columns=rename_dict, inplace=True)
        self.param_info.rename(index=rename_dict, inplace=True)

        self.Xcols_all_orig = Xcols_all
        self.Xcols_all = newXcols_all
        self.Xcols_orig = Xcols
        self.Xcols = newXcols

        # Train/test split
        nSamp = len( self.data.index.levels[0] )
        nTrain = int(round(self.training_fraction * nSamp))
        nTest = nSamp - nTrain
        nRep = self.data.loc[0].shape[0]

        self.training_data = self.data.loc[:nTrain-1]
        self.test_data = self.data.loc[nTrain:]

        print "Found %d unique parameter configurations, each of which is repeated %d time(s)." % (nSamp, nRep)
        print "--> Training with %d unique parameter configurations (%d simulations including replicates)"  % (nSamp-nTest, (nSamp-nTest)*nRep)
        print "--> Testing  with %d unique parameter configurations (%d simulations including replicates)" % (nTest, nTest*nRep)

        # Dir prep
        self.glmdir = HistoryMatching.mkdir_if_needed('GLM')
        self.gprdir = HistoryMatching.mkdir_if_needed('GPR')


    @classmethod
    def from_file():
        pass

    def save(self):
        pass

    @staticmethod
    def mkdir_if_needed(path):
        # TODO: Move to helper
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise
        return path


    def filter_data(self, source='Train', lower=np.NaN, upper=np.NaN):
        print 'Filtering data:'
        assert( source in ['Train', 'Test', 'Both'] )

        if not np.isnan(lower):
            if source in ['Train', 'Both']:
                self.training_data = self.training_data.loc[ self.training_data[self.Ycol] > lower, :]
                print '\tFilter keeping training data > %f.' % lower
            if source in ['Test', 'Both']:
                self.test_data = self.test_data.loc[ self.test_data[self.Ycol] > lower, :]
                print '\tFilter keeping test data > %f.' % lower

        if not np.isnan(upper):
            if source in ['Train', 'Both']:
                self.training_data = self.training_data.loc[ self.training_data[self.Ycol] < uppser, :]
                print '\tFilter keeping only training data < %f.' % upper
            if source in ['Test', 'Both']:
                self.test_data = self.test_data.loc[ self.test_data[self.Ycol] < upper, :]
                print '\tFilter keeping only test data < %f.' % upper

        print 'Done filtering data'


    def get_initial_samples(self, Nsamples):
        pass

    def step(self, samples, results,
            training_fraction = 0.75,
            force_optimize_glm = False,
            force_optimize_gpr = False,
            implausibility_threshold = 3,
            discrepancy_var = 30**2
        ):

        pass

    def glm(self,
            force_optimize_glm = False,
            glm_fit_maxiter = 100000,
            second_order_basis_terms = True,
            third_order_basis_terms = False,
            fourth_order_basis_terms = False,
            fifth_order_basis_terms = False,
            higher_order_basis_terms = False,
            family = 'Poisson', # e.g. Poisson, Gaussian
        ):

            glm_model_fn = os.path.join(self.glmdir, 'model.json')
            mean_params_fn = os.path.join(self.glmdir, 'params.p')

            # TODO: Ask user if they want mean, although I'm not sure statsmodels works without it!
            train_mean = self.training_data.reset_index().groupby(['Sample']).mean()
            test_mean = self.test_data.reset_index().groupby(['Sample']).mean()

            #print train_mean[self.Ycol].head()
            print self.training_data.head()
            exit()

            if not force_optimize_glm and os.path.isfile(glm_model_fn) and os.path.isfile(mean_params_fn):
                print "Loading GLM from", glm_model_fn, ", with model params from", mean_params_fn
                glm_model = GLM.from_config(glm_model_fn, mean_params_fn)
            else:
                glm_model = GLM(    Xcols = self.Xcols,
                                    Ycol = self.Ycol,
                                    training_data = train_mean,
                                    reference_value = self.desired_result,
                                    family = family,
                                    #family = sm.genmod.families.links.Logit,
                                    #family = sm.genmod.families.Binomial(link=sm.genmod.families.links.logit),
                                    second_order_basis_terms = second_order_basis_terms,
                                    third_order_basis_terms = third_order_basis_terms,
                                    fourth_order_basis_terms = fourth_order_basis_terms,
                                    fifth_order_basis_terms = fifth_order_basis_terms,
                                    higher_order_basis_terms = higher_order_basis_terms)
                print "Fitting the GLM"
                glm_model.fit(maxiter=glm_fit_maxiter)
                glm_model.save(glm_model_fn, mean_params_fn)

            print 'Evaluating training and test data'
            train_mean['Yglm'] = glm_model.evaluate(train_mean)
            test_mean['Yglm'] = glm_model.evaluate(test_mean)

            fig = glm_model.plot_errors(train_mean.reset_index(), test_mean.reset_index());
            fig.savefig( os.path.join(self.glmdir, 'errors.pdf') );             plt.close(fig)

            if False:
                print('Plotting')

                if False:
                    #cp = pd.DataFrame()
                    print test_mean.loc[[2110]]
                    cp = test_mean.loc[[2110]]
                    figs = glm_model.plot_data(circle_points=cp);
                    pairdir = os.path.join(self.glmdir, 'PairwiseResults')
                    if not os.path.exists( pairdir):
                        os.mkdir( pairdir )
                    for fn,fig in figs.iteritems():
                        fig.savefig( os.path.join(pairdir, fn) ); plt.close(fig)

                fig = glm_model.plot_fitted_vs_observed();  fig.savefig( os.path.join(self.glmdir, 'fitted_vs_observed.pdf') ); plt.close(fig)
                fig = glm_model.plot_pearson_residuals();   fig.savefig( os.path.join(self.glmdir, 'pearson_residuals.pdf') );  plt.close(fig)
                fig = glm_model.plot_deviance_redisuals();  fig.savefig( os.path.join(self.glmdir, 'deviance_redisuals.pdf') ); plt.close(fig)
                fig = glm_model.plot_QQ();                  fig.savefig( os.path.join(self.glmdir, 'QQ.pdf') );                 plt.close(fig)
                fig = glm_model.plot_histogram();           fig.savefig( os.path.join(self.glmdir, 'histogram.pdf') );          plt.close(fig)
                fig = glm_model.plot_fit();                 fig.savefig( os.path.join(self.glmdir, 'fit.pdf') );                plt.close(fig)

            self.training_data = self.training_data.join(train_mean['Yglm'])
            self.training_data['Yerr'] = self.training_data[self.Ycol] - self.training_data['Yglm']

            #print 'Best and worst training errors:\n', train.sort_values(by='Yerr')

            self.test_data = self.test_data.join(test_mean['Yglm'])
            self.test_data['Yerr'] = self.test_data[self.Ycol] - self.test_data['Yglm']

            #train_mean = self.training_data.reset_index().groupby(['Sample']).mean()
            #test_mean = self.test_data.reset_index().groupby(['Sample']).mean()

            #print 'Best and worst test errors:\n', test.sort_values(by='Yerr')


    def gpr(self,
        method = 'CrossValidation'
    ):
        pass

    def joint(self):
        pass

    def implausibility(self):
        pass
