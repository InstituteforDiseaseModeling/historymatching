import pandas as pd
import numpy as np
import os, errno
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pyDOE import lhs

from glm import GLM
from gpr import GPR
from plotting import joint_plot, plot_errors # <-- TODO: Fix names

# TODO: Error plot
# TODO: Reference plot

class HistoryMatching():

    def __init__(self,
        cut_name,           # Name for this cut
        param_info,         # Parameter definitions
        inputs,
        results,
        desired_result,
        iteration,  # Current iteration, needed?
        implausibility_threshold = 3,
        discrepancy_var = 0,
        training_fraction = 0.75
    ):
        """
        :param DataFrame param_info: Parameter info with index named 'Name' containing parameter name, and columns of 'Min' and 'Max'
        :param DataFrame inputs: Model inputs with index named 'Sample' containing parameter names.  All Names from param_info must be columns in this data frame, but it can have other columns as well.  Data contents are model input parameter values.
        :param Series results: Series with MultiIndex of 'Sample' and 'Sim_Id'.  Data are simulation results.
        :param float desired_result: The desired result to match.
        :param int iteration: The current iteration.  Work will be saved to iter[iteration]
        """

        print('Welcome to IDM History Matching!')

        sns.set_style('whitegrid')

        self.cut_name = cut_name
        self.param_info = param_info.copy()
        self.inputs = inputs.copy()
        self.results = results.copy()
        self.implausibility_threshold = implausibility_threshold
        self.discrepancy_var = discrepancy_var
        self.desired_result = desired_result
        self.training_fraction = training_fraction
        self.iteration = iteration

        Xcols_all = self.param_info.index.unique()

        self.results.name = 'Sim_Result'
        self.data = pd.merge(inputs.reset_index(), self.results.reset_index(), on='Sample').set_index(['Sample', 'Sim_Id']).sort_index()
        self.Ycol = self.results.name

        # TODO: Verify that all Xcols are columns of data

        # Fix names for picky statsmodels patsy, statsmodels var names can't have space in formula
        Xcols = inputs.columns
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
        self.cutdir = HistoryMatching.mkdir_if_needed(os.path.join('Cuts',cut_name) )
        self.glmdir = HistoryMatching.mkdir_if_needed(os.path.join(self.cutdir, 'GLM') )
        self.gprdir = HistoryMatching.mkdir_if_needed(os.path.join(self.cutdir, 'GPR') )
        self.combineddir = HistoryMatching.mkdir_if_needed(os.path.join(self.cutdir, 'Implausibility') )


    @classmethod
    def from_file(cls, cut_dir, cut_name):
        config_fn = os.path.join(cut_dir, cut_name, 'history_matching_config.xlsx')
        with pd.ExcelFile(config_fn) as xls:
            hm_params = pd.read_excel(xls, 'History_Matching_Params', index_col=0, na_values=['NA'])
            cut_name = hm_params.loc['cut_name'].values[0]
            implausibility_threshold = hm_params.loc['implausibility_threshold'].values[0]
            discrepancy_var = hm_params.loc['discrepancy_var'].values[0]
            training_fraction = hm_params.loc['training_fraction'].values[0]
            desired_result = hm_params.loc['desired_result'].values[0]
            iteration = hm_params.loc['iteration'].values[0]

            inputs = pd.read_excel(xls, 'Inputs', index_col=0)
            results = pd.read_excel(xls, 'Results', index_col=[0,1]) # NOTE: was series, now DF
            param_info = pd.read_excel(xls, 'Param_Info', index_col=0)

        return cls(
            cut_name = cut_name,
            param_info = param_info,
            inputs = inputs,
            results = results,
            desired_result = desired_result,
            iteration = iteration,
            implausibility_threshold = implausibility_threshold,
            discrepancy_var = discrepancy_var,
            training_fraction   = training_fraction
        )


    def save(self):
        config_fn = os.path.join(self.cutdir, 'history_matching_config.xlsx')
        hm_params = pd.Series({
            'cut_name'                  : self.cut_name,
            'implausibility_threshold'  : self.implausibility_threshold,
            'discrepancy_var'           : self.discrepancy_var,
            'training_fraction'         : self.training_fraction,
            'desired_result'            : self.desired_result,
            'iteration'                 : self.iteration
        }, name='Value')
        hm_params.index.name = 'Parameter'

        with pd.ExcelWriter(config_fn) as writer:
            hm_params.to_frame().to_excel(writer, sheet_name='History_Matching_Params')
            self.inputs.to_excel(writer, sheet_name='Inputs')
            self.results.to_frame().to_excel(writer, sheet_name='Results', merge_cells=False)
            self.param_info.to_excel(writer, sheet_name='Param_Info')
            #writer.save()


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


    def glm(self,
            force_optimize_glm = False,
            glm_fit_maxiter = 100000,
            second_order_basis_terms = True,
            third_order_basis_terms = False,
            fourth_order_basis_terms = False,
            fifth_order_basis_terms = False,
            higher_order_basis_terms = False,
            family = 'Poisson', # e.g. Poisson, Gaussian
            plot = True
        ):

            glm_model_fn = os.path.join(self.glmdir, 'model.json')
            mean_params_fn = os.path.join(self.glmdir, 'params.p')

            # TODO: Ask user if they want mean, although I'm not sure statsmodels works without it!
            train_mean = self.training_data.reset_index().groupby(['Sample']).mean()
            test_mean = self.test_data.reset_index().groupby(['Sample']).mean()

            if not force_optimize_glm and os.path.isfile(glm_model_fn) and os.path.isfile(mean_params_fn):
                print "Loading GLM from", glm_model_fn, ", with model params from", mean_params_fn
                self.glm_model = GLM.from_config(glm_model_fn, mean_params_fn)
            else:
                self.glm_model = GLM(    Xcols = self.Xcols,
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
                self.glm_model.fit(maxiter=glm_fit_maxiter)
                self.glm_model.save(glm_model_fn, mean_params_fn)

            print 'Evaluating training and test data'
            train_mean['Yglm'] = self.glm_model.evaluate(train_mean)
            test_mean['Yglm'] = self.glm_model.evaluate(test_mean)

            fig = self.glm_model.plot_errors(train_mean.reset_index(), test_mean.reset_index());
            fig.savefig( os.path.join(self.glmdir, 'errors.pdf') );             plt.close(fig)

            if plot:
                print('Plotting')

                if False:
                    # TODO: Save plots as they are made in GLM class
                    cp = pd.DataFrame()
                    #print test_mean.loc[[2110]]
                    #cp = test_mean.loc[[2110]]
                    figs = self.glm_model.plot_data(circle_points=cp);
                    pairdir = os.path.join(self.glmdir, 'PairwiseResults')
                    if not os.path.exists( pairdir):
                        os.mkdir( pairdir )
                    for fn,fig in figs.iteritems():
                        fig.savefig( os.path.join(pairdir, fn) ); plt.close(fig)

                fig = self.glm_model.plot_fitted_vs_observed();  fig.savefig( os.path.join(self.glmdir, 'fitted_vs_observed.pdf') ); plt.close(fig)
                fig = self.glm_model.plot_pearson_residuals();   fig.savefig( os.path.join(self.glmdir, 'pearson_residuals.pdf') );  plt.close(fig)
                fig = self.glm_model.plot_deviance_redisuals();  fig.savefig( os.path.join(self.glmdir, 'deviance_redisuals.pdf') ); plt.close(fig)
                fig = self.glm_model.plot_QQ();                  fig.savefig( os.path.join(self.glmdir, 'QQ.pdf') );                 plt.close(fig)
                fig = self.glm_model.plot_histogram();           fig.savefig( os.path.join(self.glmdir, 'histogram.pdf') );          plt.close(fig)
                fig = self.glm_model.plot_fit();                 fig.savefig( os.path.join(self.glmdir, 'fit.pdf') );                plt.close(fig)

            self.training_data = self.training_data.join(train_mean['Yglm'])
            self.training_data['Yerr'] = self.training_data[self.Ycol] - self.training_data['Yglm']

            #print 'Best and worst training errors:\n', train.sort_values(by='Yerr')

            self.test_data = self.test_data.join(test_mean['Yglm'])
            self.test_data['Yerr'] = self.test_data[self.Ycol] - self.test_data['Yglm']

            #train_mean = self.training_data.reset_index().groupby(['Sample']).mean()
            #test_mean = self.test_data.reset_index().groupby(['Sample']).mean()

            #print 'Best and worst test errors:\n', test.sort_values(by='Yerr')


    def gpr(self,
        force_optimize_gpr = True,
        K_folds = 5,
        eps = 1e-2,
        method = 'CrossValidation',
        verbose = False,
        plot = True
    ):

        assert( method in ['CrossValidation'] ) # Supporint only CV for now

        gpr_model_fn = os.path.join(self.gprdir, 'model.json')

        if not force_optimize_gpr and os.path.isfile(gpr_model_fn):
            print "Loading GPR from", gpr_model_fn
            self.gpr_model = GPR.from_config(gpr_model_fn)
        else:
            self.gpr_model = GPR(    Xcols = self.Xcols,
                                Ycol = 'Yerr',
                                training_data = self.training_data,
                                param_info = self.param_info,
                                kernel_mode = 'RBF',
                                kernel_params = None,
                                verbose = verbose,
                                debug = False   )

            print "Fitting the GPR"
            self.gpr_model.optimize_hyperparameters(
                x0 = np.array([2, 0.10] + len(self.Xcols)*[0.1]),
                bounds = ((0.005,10),)+((0.01,10),) + len(self.Xcols)*((0.01,1),),
                eps = eps,
                K = K_folds
            )
            self.gpr_model.save(gpr_model_fn)


        train_mean = self.training_data.reset_index().groupby(['Sample']).mean()
        test_mean = self.test_data.reset_index().groupby(['Sample']).mean()

        print 'GPR evaluating training data'
        ret = self.gpr_model.evaluate(train_mean)
        train_mean['Mean_Err'] = ret['Mean']
        train_mean['Mean_Estimate'] = train_mean['Yglm'] + train_mean['Mean_Err']
        train_mean['Var_Err_Predictive'] = ret['Var_Predictive']
        train_mean['Var_Err_Latent'] = ret['Var_Latent']
        self.training_data = self.training_data.reset_index().join(train_mean[['Mean_Err', 'Mean_Estimate', 'Var_Err_Predictive', 'Var_Err_Latent']], on='Sample')
        self.training_data.set_index(['Sample', 'Sim_Id'], inplace=True)

        print 'GPR evaluating test data'
        ret = self.gpr_model.evaluate(test_mean)
        test_mean['Mean_Err'] = ret['Mean']
        test_mean['Mean_Estimate'] = test_mean['Yglm'] + test_mean['Mean_Err']
        test_mean['Var_Err_Predictive'] = ret['Var_Predictive']
        test_mean['Var_Err_Latent'] = ret['Var_Latent']
        self.test_data = self.test_data.reset_index().join(test_mean[['Mean_Err', 'Mean_Estimate', 'Var_Err_Predictive', 'Var_Err_Latent']], on='Sample')
        self.test_data.set_index(['Sample', 'Sim_Id'], inplace=True)

        if plot:
            print('Plotting')
            fig = self.gpr_model.plot_errors(self.training_data.reset_index(), self.test_data.reset_index(), 'Mean_Err', 'Var_Err_Predictive', 'Var_Err_Latent');
            fig.savefig( os.path.join(self.gprdir, 'errors.pdf') );             plt.close(fig)

            #circle_samples = train.sort_values(by='Yerr').iloc[[0, -1]].reset_index()['Sample'].values
            circle_samples = pd.DataFrame()
            #fig = self.gpr_model.plot_data(samples_to_circle=circle_samples);    fig.savefig( os.path.join(self.gprdir, 'data.pdf') );    plt.close(fig)
            if False: # TODO: Save plots are they are made in GPR class!
                cp = pd.DataFrame()
                #print test_mean.loc[[2110]]
                #cp = test_mean.loc[[2110]]
                figs = self.gpr_model.plot_data(samples_to_circle=circle_samples)
                pairdir = os.path.join(self.gprdir, 'PairwiseResults')
                if not os.path.exists( pairdir):
                    os.mkdir( pairdir )
                for fn,fig in figs.iteritems():
                    fig.savefig( os.path.join(pairdir, fn) ); plt.close(fig)

            if False:
                mu = self.training_data[self.Xcols].mean()
                #mu = train.loc[146][Xcols].mean(); print mu
                (fig_mean, fig_std_latent) = self.gpr_model.plot(mu, res=25);
                fig_mean.savefig( os.path.join(self.gprdir, 'plot_mean.pdf') );    plt.close(fig_mean) # SLOW
                fig_std_latent.savefig( os.path.join(self.gprdir, 'plot_std_latent.pdf') );    plt.close(fig_std_latent) # SLOW

            fig = self.gpr_model.plot_histogram();
            fig.savefig( os.path.join(self.gprdir, 'histogram.pdf') );
            plt.close(fig)


    def calc_and_plot_implausibility(self,
        plot = False
    ):

        self.training_data['Implausibility'] = \
                    abs( self.training_data['Mean_Estimate'] - self.desired_result ) / \
                    np.sqrt(self.training_data['Var_Err_Predictive'] + self.discrepancy_var)
        self.training_data['Implausible'] = self.training_data[ 'Implausibility' ] > self.implausibility_threshold

        self.test_data['Implausibility'] = \
                    abs( self.test_data['Mean_Estimate'] - self.desired_result ) / \
                    np.sqrt(self.test_data['Var_Err_Predictive'] + self.discrepancy_var)
        self.test_data['Implausible'] = self.test_data[ 'Implausibility' ] > self.implausibility_threshold

        if plot:
            train_mean = self.training_data.reset_index().groupby(['Sample']).mean()
            test_mean = self.test_data.reset_index().groupby(['Sample']).mean()

            fig = plot_errors(train_mean.reset_index(), test_mean.reset_index()); fig.savefig( os.path.join(self.combineddir, 'errors.pdf') );  plt.close(fig)

            fig = joint_plot(self.training_data, train_mean);    fig.savefig( os.path.join(self.combineddir, 'train.pdf') ); plt.close(fig)
            fig = joint_plot(self.test_data, test_mean);      fig.savefig( os.path.join(self.combineddir, 'test.pdf') );  plt.close(fig)

            fig = joint_plot(self.training_data, train_mean, log_x=True);    fig.savefig( os.path.join(self.combineddir, 'train_log.pdf') ); plt.close(fig)
            fig = joint_plot(self.test_data, test_mean, log_x=True);      fig.savefig( os.path.join(self.combineddir, 'test_log.pdf') );  plt.close(fig)



    def cut(self, num_desired_candidates = 5000):
        gpr_model_with_test_fn = os.path.join(self.gprdir, 'model_with_test_data.json')

        # Add test data to gpr training
        self.gpr_model.set_training_data(pd.concat([self.training_data, self.test_data]))
        self.gpr_model.save(gpr_model_with_test_fn)

        def plot_implausibility(data, column, thresh, save_fn=None):
            scaled = data[column] / data[column].max()
            good = data[column] < thresh
            D = len(self.Xcols)
            fig = plt.figure(figsize=(128,128))
            for row in range(D):
                for col in range(D):
                    if col > row:
                        gs = gridspec.GridSpec(D-1, D-1)
                        ax = fig.add_subplot(gs[col-1,row])
                        #x = data[ Xcols[row] ]; y = data[ Xcols[col] ]
                #plt.scatter(x, y, s=np.maximum(5, 50*scaled), c=scaled, cmap='jet', lw=0) #, s=area, c=colors, alpha=0.5)
                xg = data.loc[good, self.Xcols[row]]; yg = data.loc[good, self.Xcols[col]]; sg = scaled[good]
                plt.scatter(xg, yg, s=np.maximum(3, 20*sg), lw=0, c='g', alpha=0.5) #, facecolors='none', edgecolors='g'
                xb = data.loc[good==False, self.Xcols[row]]; yb = data.loc[good==False, self.Xcols[col]]; sb = scaled[good==False]
                plt.scatter(xb, yb, s=np.maximum(3, 20*sb), lw=0, c='r', alpha=0.5) #, facecolors='none', edgecolors='r'
                plt.autoscale(tight=True)
                if col == D-1:
                    plt.xlabel( self.Xcols[row] )
                if row == 0:
                    plt.ylabel( self.Xcols[col] )
            plt.tight_layout()
            if save_fn is not None:
                print 'Saving figure to %s' % save_fn
                plt.savefig(save_fn)
            return fig

        def plot_implausibility_by_iter(data, save_fn=None):
            fig = plt.figure(figsize=(20,20))

            for it in range(iteration+2):
                col_first_only = 'c'
                col_second_only = 'y'
                col_neither = 'r'
                col_both = 'g'

                first_only = ~data['Implausible_0'] & data['Implausible_1']
                second_only = data['Implausible_0'] & ~data['Implausible_1']
                neither = data['Implausible_0'] & data['Implausible_1']
                both = ~data['Implausible_0'] & ~data['Implausible_1']

                size = 10

                D = len(self.Xcols)
                for row in range(D):
                    for col in range(D):
                        if col > row:
                            gs = gridspec.GridSpec(D-1, D-1)
                            ax = fig.add_subplot(gs[col-1,row])

                            x = data.loc[first_only, self.Xcols[row]]; y = data.loc[first_only, self.Xcols[col]];
                            h1 = plt.scatter(x, y, s=size, lw=0, c=col_first_only, alpha=0.5) #, facecolors='none', edgecolors='g'

                            x = data.loc[second_only, self.Xcols[row]]; y = data.loc[second_only, self.Xcols[col]];
                            h2 = plt.scatter(x, y, s=size, lw=0, c=col_second_only, alpha=0.5) #, facecolors='none', edgecolors='g'

                            x = data.loc[neither, self.Xcols[row]]; y = data.loc[neither, self.Xcols[col]];
                            h3 = plt.scatter(x, y, s=size, lw=0, c=col_neither, alpha=0.5) #, facecolors='none', edgecolors='g'

                            x = data.loc[both, self.Xcols[row]]; y = data.loc[both, self.Xcols[col]];
                            h4 = plt.scatter(x, y, s=size, lw=0, c=col_both, alpha=0.5) #, facecolors='none', edgecolors='g'

                            plt.autoscale(tight=True)
                            if col == D-1:
                                plt.xlabel( self.Xcols[row] )
                            if row == 0:
                                plt.ylabel( self.Xcols[col] )

            plt.figlegend((h1,h2,h3,h4), ('first only', 'second only', 'neither', 'both'), 'upper right', fontsize=16)

            plt.tight_layout()

            if save_fn is not None:
                print 'Saving figure to %s' % save_fn
                plt.savefig(save_fn)
            return fig


        def histogram_implausibility(data, column, thresh=None, save_fn=None):
            fig, ax = plt.subplots()
            sns.distplot( data[column], rug=True, ax = ax)
            yl = ax.get_ylim()
            if thresh is not None:
                plt.plot([thresh,thresh], yl, 'r-')
            if save_fn is not None:
                print 'Saving figure to %s' % save_fn
                plt.savefig(save_fn)


        print "Looking for ", self.desired_result

        num_good_candidates = 0
        candidates = pd.DataFrame( columns=self.Xcols )

        search_step = 0

        hm_params = {(self.iteration, self.cut_name): {
            'desired_result':self.desired_result,
            'discrepancy_var':self.discrepancy_var,
            'implausibility_threshold':self.implausibility_threshold,
        }}
        glm_all = {(self.iteration, self.cut_name): self.glm_model}
        gpr_all = {(self.iteration, self.cut_name): self.gpr_model}
        cuts = [(self.iteration, self.cut_name)]
        for it in range(self.iteration): # Loop over previous iterations
            cuts_dir = os.path.join('..', 'iter%d'%it, 'Cuts')

            for cut_name in [name for name in os.listdir(cuts_dir) if os.path.isdir(os.path.join(cuts_dir, name))]:
                print('Reading cut from %s' % cut_name)
                hm = HistoryMatching.from_file(cuts_dir, cut_name)
                print '\t Discrepancy Var:', hm.discrepancy_var
                print '\t Desired Result:', hm.desired_result
                print '\t Imp Thresh:', hm.implausibility_threshold
                hm_params[(it, cut_name)] = {
                    'desired_result':hm.desired_result,
                    'discrepancy_var':hm.discrepancy_var,
                    'implausibility_threshold':hm.implausibility_threshold,
                }

                glm_all[(it, cut_name)] = GLM.from_config(os.path.join(cuts_dir, cut_name, 'GLM', 'model.json'), os.path.join(cuts_dir, cut_name, 'GLM', 'params.p'))
                gpr_all[(it, cut_name)] = GPR.from_config(os.path.join(cuts_dir, cut_name, 'GPR', 'model.json'))
                cuts.append((it, cut_name))

        print 'Looking for candidates, step', search_step; search_step+=1
        while num_good_candidates < num_desired_candidates:
            print '-'*80
            # Likeliy need to over-sample, this could take a long time if p is low
            # Min here to avoid running out of GPU ram!
            lhs_sample = lhs( len(self.Xcols_all), samples=min(2500, num_desired_candidates-num_good_candidates))

            for i, xc in enumerate(self.Xcols_all):
                v = self.param_info.loc[xc]
                lhs_sample[:, i] = (v['Max'] - v['Min']) * lhs_sample[:, i] + (v['Min'])

            new_candidates = pd.DataFrame( lhs_sample, columns=self.Xcols_all) # Note _orig :)
            new_candidates['Implausible'] = False

            for (it, cut_name) in cuts:
                print('Processing iteration %d, cut %s'%(it, cut_name))
                new_candidates['Yglm'] = glm_all[(it,cut_name)].evaluate(new_candidates)
                ret = gpr_all[(it,cut_name)].evaluate(new_candidates)
                new_candidates['Mean_Estimate'] = new_candidates['Yglm'] + ret['Mean']
                new_candidates['Var_Predictive'] = ret['Var_Predictive']

                new_candidates[ 'Implausibility_%d_%s'%(it, cut_name) ] = \
                    abs( new_candidates['Mean_Estimate'] - hm_params[(it, cut_name)]['desired_result'] ) / \
                    np.sqrt(new_candidates['Var_Predictive'] + hm_params[(it, cut_name)]['discrepancy_var'] )

                new_candidates[ 'Implausible_%d_%s'%(it, cut_name) ] = new_candidates[ 'Implausibility_%d_%s'%(it, cut_name) ] > hm_params[(it, cut_name)]['implausibility_threshold']
                new_candidates['Implausible'] |= new_candidates[ 'Implausible_%d_%s'%(it, cut_name) ]

            candidates = candidates.append(new_candidates)
            num_new_good_candidates = sum(new_candidates['Implausible'] == False)
            num_good_candidates += num_new_good_candidates

            #print new_candidates
            del new_candidates

            print 'Candidates: New = %d, Tot = %d' % (num_new_good_candidates, num_good_candidates)

# Put back orig parameter names
        candidates.rename(columns={new:orig for (new,orig) in zip(self.Xcols_all, self.Xcols_all_orig)}, inplace=True)

        rejected_percent = (100 * sum(candidates['Implausible']) / float(candidates.shape[0]))
        print 'Rejected %.1f%%' % rejected_percent

        return (candidates, rejected_percent)

        non_implausible_candidates = candidates.loc[ candidates['Implausible'] == False, :]
        writer = pd.ExcelWriter('Candidates_for_iter%d.xlsx'%(self.iteration+1))
        non_implausible_candidates[self.Xcols_all_orig].to_excel(writer, sheet_name='Values', index=False)
        non_implausible_candidates.set_index(self.Xcols_all_orig).to_excel(writer, sheet_name='NonImplausible')
        candidates.set_index(self.Xcols_all_orig).to_excel(writer, sheet_name='All')
        writer.save()

        store = pd.HDFStore('candidates_for_iter%d.h5'%(self.iteration+1))
        store['candidates'] = candidates
        store.close()
