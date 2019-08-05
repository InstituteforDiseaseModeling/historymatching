import pandas as pd
import numpy as np
import os, errno
import matplotlib.pyplot as plt
import seaborn as sns
from pyDOE import lhs
from shutil import copyfile
import datetime

from history_matching.glm import GLM
from history_matching.gpr import GPR
from history_matching.plotting import plot_data, joint_plot, plot_errors, plot_implausibility, plot_implausibility_by_iter, histogram_implausibility # <-- TODO: Fix names

# TODO: Error plot
# TODO: Reference plot

class HistoryMatching():
    """Main class to support history matching.
    """

    def __init__(self,
        cut_name,           # Name for this cut
        param_info,         # Parameter definitions
        inputs,
        results,
        desired_result,
        iteration,  # Current iteration, needed?
        implausibility_threshold = 3,
        discrepancy_var = 0,
        desired_result_var = 0,
        training_fraction = 0.75,
        fig_type = 'pdf',
        use_glm = True,      # Disable the glm by setting to False
        verbose = False
    ):
        """ Initialize a history matching object.

        Args:
            cut_name: (str) Name for this cut.  A directory in the `Cuts` folder will be generated.
            param_info: (DataFrame) Parameter info with index named 'Name' containing parameter name, and columns of 'Min' and 'Max'.  Other columns will be ignored.
            inputs: (DataFrame) Model inputs with index named 'Sample_Id' containing parameter names.  All Names from param_info must be columns in this data frame, but it can have other columns as well.  Data contents are model input parameter values.
            results: (Series) Series with MultiIndex of 'Sample_Id' and 'Sim_Id'.  Data are simulation results.
            desired_result: (float) The desired result to match.
            iteration: (int) The current iteration.  Work will be saved to iter[iteration].
            implausibility_threshold: (float) The threshold to use for determining if a point is implausible.
            discrepancy_var: (float) Constant variance to include in implausibility calculations for discrepancy.
            desired_result_var: (float) Constant variance to include in implausibility calculations for variance in the desired result.  This typically comes from a confidence interval in survey data.
            discrepatraining_fraction: (float) The fraction of the inputs and results to use a training data. NOTE: You can also specify training data by including a boolean column named `Train` in the inputs or results data frames.
            use_glm: (bool) Set False to disable the GLM, in which case the results will be modeled purely using GPR.
            verbose: (bool) Set True to see more details.

        Returns:
            Class instance.
        """

        print('Welcome to IDM History Matching!')

        sns.set_style('whitegrid')

        self.cut_name = cut_name
        self.param_info = param_info.copy()
        self.inputs = inputs.copy()
        self.results = results.copy()
        self.implausibility_threshold = implausibility_threshold
        self.discrepancy_var = discrepancy_var
        self.desired_result_var = desired_result_var
        self.desired_result = desired_result
        self.training_fraction = training_fraction
        self.iteration = iteration
        self.use_glm = use_glm
        self.fig_type = fig_type
        self.verbose = verbose

        self.results.name = 'Sim_Result'
        self.Ycol = self.results.name
        if 'Train' in self.inputs.columns:
            self.data = pd.merge(self.inputs.reset_index(), self.results.reset_index(), on='Sample_Id')
            self.data['Train'] = self.data['Train'].astype(bool)  # Annoying that I have to cast this!
            self.data.set_index(['Train', 'Sample_Id', 'Sim_Id'], inplace=True)#.sort_index()
            self.training_data = self.data.loc[True]
            self.test_data = self.data.loc[False]
            print('Using train/test split as specified by user')
        else:
            self.data = pd.merge(self.inputs.reset_index(), self.results.reset_index(), on='Sample_Id').set_index(['Sample_Id', 'Sim_Id'])#.sort_index()

            # Train/test split
            nSamp = len(self.data.index.get_level_values('Sample_Id'))
            nTrain = int(round(self.training_fraction * nSamp))
            nTest = nSamp - nTrain

            # TODO: Fix REPLICATES!!!
            data_tmp = self.data.reset_index()
            data_tmp.rename(columns={'Sample_Id': 'Sample_Orig'}, inplace=True)
            data_tmp.index.name='Sample_Id'
            nRep = 1 if len(data_tmp.iloc[0].shape)==1 else data_tmp.iloc[0].shape[0]
            self.training_data = data_tmp.loc[:nTrain-1]
            self.test_data = data_tmp.loc[nTrain:]

            print("Found", nSamp, "unique parameter configurations, each of which is repeated", nRep, "time(s).")
            print("--> Training with",nSamp-nTest, "unique parameter configurations (", (nSamp-nTest)*nRep," simulations including replicates)")
            print("--> Testing  with", nTest," unique parameter configurations (", nTest*nRep, "simulations including replicates)")

        # Dir prep
        self.cutdir = HistoryMatching.mkdir_if_needed(os.path.join('..', 'iter%d'%self.iteration, 'Cuts',cut_name) )
        self.glmdir = HistoryMatching.mkdir_if_needed(os.path.join(self.cutdir, 'GLM') )
        self.gprdir = HistoryMatching.mkdir_if_needed(os.path.join(self.cutdir, 'GPR') )
        self.combineddir = HistoryMatching.mkdir_if_needed(os.path.join(self.cutdir, 'Implausibility') )


    @classmethod
    def from_file(cls, cut_dir, cut_name):
        """Load an instance of HistoryMatching from cache.  Used extensively during `cutting.`

        The configuration file from which the HistoryMatching instance will be created is cut_dir/cut_name/history_matching_config.xlsx.

        Args:
            cut_dir: (str) Directory containing all cuts
            cut_name: (str) The name of the specific cut to restore from.

        Returns: Instance of HistoryMatching
        """

        config_fn = os.path.join(cut_dir, cut_name, 'history_matching_config.xlsx')
        with pd.ExcelFile(config_fn) as xls:
            hm_params = pd.read_excel(xls, 'History_Matching_Params', index_col=0, na_values=['NA'])
            cut_name = hm_params.loc['cut_name'].values[0]
            implausibility_threshold = hm_params.loc['implausibility_threshold'].values[0]
            discrepancy_var = hm_params.loc['discrepancy_var'].values[0]
            desired_result_var = hm_params.loc['desired_result_var'].values[0] if 'desired_result_var' in hm_params else 0
            training_fraction = hm_params.loc['training_fraction'].values[0]
            desired_result = hm_params.loc['desired_result'].values[0]
            iteration = hm_params.loc['iteration'].values[0]

            inputs = pd.read_excel(xls, 'Inputs', index_col=0)
            results = pd.read_excel(xls, 'Results', index_col=[0,1]).squeeze() # NOTE: was series, now DF
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
            desired_result_var = desired_result_var,
            training_fraction   = training_fraction,
        )


    def save(self):
        """Save instance of HistoryMatching to cache.

        A configuration file will be saved to cut_dir/cut_name/history_matching_config.xlsx.
        """

        config_fn = os.path.join(self.cutdir, 'history_matching_config.xlsx')
        hm_params = pd.Series({
            'cut_name'                  : self.cut_name,
            'implausibility_threshold'  : self.implausibility_threshold,
            'discrepancy_var'           : self.discrepancy_var,
            'desired_result_var'        : self.desired_result_var,
            'training_fraction'         : self.training_fraction,
            'desired_result'            : self.desired_result,
            'iteration'                 : self.iteration
        }, name='Value')
        hm_params.index.name = 'Parameter'

        with pd.ExcelWriter(config_fn) as writer:
            hm_params.to_frame().to_excel(writer, sheet_name='History_Matching_Params')
            self.inputs.to_excel(writer, sheet_name='Inputs', merge_cells=False)
            self.results.to_frame().to_excel(writer, sheet_name='Results', merge_cells=False)
            self.param_info.to_excel(writer, sheet_name='Param_Info')
            #writer.save()


    @staticmethod
    def mkdir_if_needed(path):
        """Utility to make a directory, but only if needed.
        """

        # TODO: Move to helper
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise
        return path


    def filter(self, func, train=False, test=False):
        """Utility to allow the user to filter training and test data.  For example, you could keep only data where the result is greater than zero.

        Args:
            func: (lambda) Function to apply to the data
            train: (bool) Set True to apply fun to the training data
            test: (bool) Set True to apply fun to the test data
        """

        if train:
            self.training_data = func(self.training_data)
        if test:
            self.test_data = func(self.test_data)

    def glm(self,
        basis,
        force_optimize_glm = False,
        glm_fit_maxiter = 100000,
        family = 'Poisson', # e.g. Poisson, Gaussian
        plot = True,
        plot_data = False,
        **kwargs
    ):
        """Perform Generalized Linear Modeling (GLM).

        Note that the GLM will be performed on the mean of the training data if multiple replicates are provided for each Sample_Id.

        Args:
            basis: (Basis instance) Instance of basis allowing the inputs to be transformed into a data matrix.
            force_optimize_glm: (bool) Set True to force optimization of the GLM parameters even when results from a previous optimization exist.
            glm_fit_maxiter: (int) Maximum number of iterations during parameter optimization.
            family: (str) GLM family from statsmodels.  Examples include `Poisson` and `Gaussian.`
            plot: (bool) Set True to produce informative diagnostic plots.
            plot_data: (bool) Set True to visualize the data in many pairwise plots.  Note plot must also be true for plot_data to produce results.  Results will be saved to a folder named PairwiseResults.
        """

        if not self.use_glm:
            print('use_glm is False, why are you calling glm?')
            return

        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = self.verbose

        # Files to store the model and parameters
        glm_model_fn = os.path.join(self.glmdir, 'model.json')
        mean_params_fn = os.path.join(self.glmdir, 'params.p')

        # TODO: Ask user if they want mean, although I'm not sure statsmodels works without it!
        train_mean = self.training_data.reset_index().groupby('Sample_Id').mean()
        test_mean = self.test_data.reset_index().groupby('Sample_Id').mean()

        if not force_optimize_glm and os.path.isfile(glm_model_fn) and os.path.isfile(mean_params_fn):
            print("Loading GLM from", glm_model_fn, ", with model params from", mean_params_fn)
            self.glm_model = GLM.from_config(glm_model_fn, mean_params_fn)
        else:
            self.glm_model = GLM(
                basis = basis,
                Ycol = self.Ycol,
                training_data = train_mean,
                reference_value = self.desired_result,
                family = family,
                verbose = verbose)

            if self.verbose:
                print("Fitting the GLM")
            self.glm_model.fit(maxiter=glm_fit_maxiter)
            self.glm_model.save(glm_model_fn, mean_params_fn)

        if self.verbose:
            print('Evaluating training and test data') # Store results in Yglm
        train_mean['Yglm'] = self.glm_model.evaluate(train_mean)
        test_mean['Yglm'] = self.glm_model.evaluate(test_mean)

        # Plot the errors and save to errors_glm.pdf
        figs = self.glm_model.plot_errors(train_mean.reset_index(), test_mean.reset_index());
        for key, fig in figs.items():
            fig.savefig( os.path.join(self.glmdir, key+'.'+self.fig_type) );
            plt.close(fig)

        if plot:
            print('Plotting')

            if plot_data:
                # TODO: Save plots as they are made
                pairdir = os.path.join(self.glmdir, 'PairwiseResults')
                if not os.path.exists( pairdir):
                    os.mkdir( pairdir )
                cp = pd.DataFrame() # To not circle a point, pass in an empty data frame.
                #print(test_mean.loc[[2110]])
                #cp = test_mean.loc[[2110]]
                figs = self.glm_model.plot_data(circle_points=cp, saveto_dir = pairdir, log_scale=True)

            fig = self.glm_model.plot_fitted_vs_observed();  fig.savefig( os.path.join(self.glmdir, 'fitted_vs_observed'+'.'+self.fig_type) ); plt.close(fig)
            fig = self.glm_model.plot_pearson_residuals();   fig.savefig( os.path.join(self.glmdir, 'pearson_residuals'+'.'+self.fig_type) );  plt.close(fig)
            fig = self.glm_model.plot_deviance_redisuals();  fig.savefig( os.path.join(self.glmdir, 'deviance_redisuals'+'.'+self.fig_type) ); plt.close(fig)
            fig = self.glm_model.plot_QQ();                  fig.savefig( os.path.join(self.glmdir, 'QQ'+'.'+self.fig_type) );                 plt.close(fig)
            #SLOW: fig = self.glm_model.plot_histogram();           fig.savefig( os.path.join(self.glmdir, 'histogram'+'.'+self.fig_type) );          plt.close(fig)
            #SLOW: fig = self.glm_model.plot_fit();                 fig.savefig( os.path.join(self.glmdir, 'fit'+'.'+self.fig_type) );                plt.close(fig)


        train_mean = train_mean.reset_index().set_index('Sample_Id')
        test_mean = test_mean.reset_index().set_index('Sample_Id')

        # Compute Yerr as the difference between the training data and the GLM for training and test data
        if 'Yglm' in self.training_data:
            self.training_data.drop('Yglm', axis=1, inplace=True)
        self.training_data = self.training_data.join(train_mean['Yglm'])
        self.training_data['Yerr'] = self.training_data[self.Ycol] - self.training_data['Yglm']

        if 'Yglm' in self.test_data:
            self.test_data.drop('Yglm', axis=1, inplace=True)
        self.test_data = self.test_data.join(test_mean['Yglm'])
        self.test_data['Yerr'] = self.test_data[self.Ycol] - self.test_data['Yglm']


    def gpr(self, basis,
        force_optimize_gpr = True,
        method = 'CrossValidation',
        verbose = False,
        plot = True,
        plot_data = False,
        sigma2_f_guess = 2,
        sigma2_f_bounds = (0.005,10),
        sigma2_n_guess = 0.10,
        sigma2_n_bounds = (0.01,10),
        lengthscale_guess = 0.1, # Note, lengthscale is in a scaled range, training data to [0,1] for each parameter
        lengthscale_bounds = (0.01,1),
        normalize_y = True,
        optimize_sigma2_n = True,
        log_transform = False,
        optimizer_options= {},
        **kwargs
    ):
        """Perform Gaussian Process Regression modeling.

        Note that the GLM will be performed on the mean of the training data if multiple replicates are provided for each Sample_Id.

        By default, the GPR will be configured to use the `RBF` kernel.

        Args:
            force_optimize_gpr: (bool) Set True to force optimization of the GPR parameters even when results from a previous optimization exist.
            plot: (bool) Set True if you want to see diagnostic plots.
            plot_data: (bool) Set True to produce many pairwise plots of the inputs and results.  Within the GPR folder, they will appear in `PairwiseResults.`
            sigma2_f_guess: (float) The guess value for the signal variance. Note that when normalizing Y, a value of 1 correspons to the variance of the results.
            sigma2_f_bounds: (tuple) Lower and upper bounds for sigma2_f, e.g. like (0.005,10).
            sigma2_n_guess: (float) Initial guess value for observation noise variance.  Normalized like sigma2_f.
            sigma2_n_bounds: (tuple) Lower and upper bounds for sigma2_n, e.g. like (0.01,10).
            lengthscale_guess: (float or ndarray)
                If supplying a float, this value representes the kernel lengthscale guess and will be used for all lengthscales.  Note, lengthscale is in a scaled range, training data to [0,1] for each parameter.
                Alternatively, you can provide a ndarray with one entry for each parameter.
            lengthscale_bounds: (tuple) Range for lengthscale, e.g. (0.01,1).
            normalize_y: (bool) Set True to normalize the outputs (recommended).
            method: (str) Must be 'CrossValidation' for now.
            verbose: (bool) Set True to see lots of output.
            optimizer_options: (dict) Dictionary to be passed to the optimization algorithm within the GPR code.
            kwargs: (dict) Additional arguments to pass to the GPR class.
        """

        assert( method in ['CrossValidation'] ) # Supporint only CV for now

        gpr_model_fn = os.path.join(self.gprdir, 'model.json')

        if plot_data:
            pairdir = os.path.join(self.gprdir, 'PairwiseResults')
            if not os.path.exists( pairdir):
                os.mkdir( pairdir )

        if not force_optimize_gpr and os.path.isfile(gpr_model_fn):
            print("Loading GPR from", gpr_model_fn)
            self.gpr_model = GPR.from_config(gpr_model_fn)
            if plot_data:
                figs = self.gpr_model.plot_data(samples_to_circle=pd.DataFrame(), saveto_dir = pairdir, log_scale=True)
        else:
            if self.use_glm:
                Ycol = 'Yerr'
            else:
                Ycol = 'Sim_Result'

            self.gpr_model = GPR(
                basis = basis,
                Ycol = Ycol,
                training_data = self.training_data,
                param_info = self.param_info,
                kernel_mode = 'RBF',
                kernel_params = None,
                normalize_y = normalize_y,
                verbose = verbose,
                debug = False, # Debug is really for testing the code
                **kwargs)

            if isinstance(lengthscale_guess, int) or isinstance(lengthscale_guess, float):
                lengthscale_guess = basis.D*[lengthscale_guess]
            else:
                assert( isinstance(lengthscale_guess, list) )
                assert( len(lengthscale_guess) == basis.D )

            if os.path.isfile(gpr_model_fn):
                timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                backup_fn = os.path.join(self.gprdir, 'model_%s.json'%timestamp)
                print('Backing up gpr model to', backup_fn)
                copyfile(gpr_model_fn, backup_fn)

            #TODO: Check guess within bounds
            x0 = np.array([sigma2_f_guess, sigma2_n_guess] +  lengthscale_guess)
            self.gpr_model.theta = x0
            self.gpr_model.save(gpr_model_fn)

            if plot_data:
                figs = self.gpr_model.plot_data(samples_to_circle=pd.DataFrame(), saveto_dir = pairdir, log_scale=True)

            print("Fitting the GPR")
            self.gpr_model.optimize_hyperparameters(
                x0 = x0,
                bounds = (sigma2_f_bounds,)+(sigma2_n_bounds,) + basis.D*(lengthscale_bounds,),
                #eps = eps,
                optimize_sigma2_n = optimize_sigma2_n,
                log_transform = log_transform,
                optimizer_options = optimizer_options
            )
            self.gpr_model.save(gpr_model_fn) # Save the model to file


        # Taking the mean prior to evaluation because it is unnecessary to evaluate each point more than once as the GP output will always be the same
        train_mean = self.training_data.reset_index().groupby(['Sample_Id']).mean()
        test_mean = self.test_data.reset_index().groupby(['Sample_Id']).mean()

        print('GPR evaluating training data')
        ret = self.gpr_model.evaluate(train_mean)
        train_mean['Mean_Err'] = ret['Mean']
        train_mean['Mean_Estimate'] = train_mean['Mean_Err']
        if self.use_glm:
            train_mean['Mean_Estimate'] += train_mean['Yglm']
        train_mean['Var_Err_Predictive'] = ret['Var_Predictive']
        train_mean['Var_Err_Latent'] = ret['Var_Latent']

        merge_cols = ['Mean_Err', 'Mean_Estimate', 'Var_Err_Predictive', 'Var_Err_Latent']
        if 'Mean_Err' in self.training_data:
            self.training_data.drop(merge_cols, axis=1, inplace=True)
        self.training_data = self.training_data.reset_index().join(train_mean[merge_cols], on='Sample_Id')
        self.training_data.set_index(['Sample_Id', 'Sim_Id'], inplace=True)

        print('GPR evaluating test data')
        ret = self.gpr_model.evaluate(test_mean)
        test_mean['Mean_Err'] = ret['Mean']
        test_mean['Mean_Estimate'] = test_mean['Mean_Err']
        if self.use_glm:
            test_mean['Mean_Estimate'] += test_mean['Yglm']
        test_mean['Var_Err_Predictive'] = ret['Var_Predictive']
        test_mean['Var_Err_Latent'] = ret['Var_Latent']
        if 'Mean_Err' in self.test_data:
            self.test_data.drop(merge_cols, axis=1, inplace=True)
        self.test_data = self.test_data.reset_index().join(test_mean[['Mean_Err', 'Mean_Estimate', 'Var_Err_Predictive', 'Var_Err_Latent']], on='Sample_Id')
        self.test_data.set_index(['Sample_Id', 'Sim_Id'], inplace=True)

        # Add test data to gpr training
        gpr_model_with_test_fn = os.path.join(self.gprdir, 'model_with_test_data.json')
        self.gpr_model.set_training_data(pd.concat([self.training_data, self.test_data]))
        self.gpr_model.save(gpr_model_with_test_fn)

        if plot:
            print('Plotting')
            fig = self.gpr_model.plot_errors(self.training_data.reset_index(), self.test_data.reset_index(), 'Mean_Err', 'Var_Err_Predictive');
            fig.savefig( os.path.join(self.gprdir, 'gpr'+'.'+self.fig_type) );             plt.close(fig)

            '''' # Useful debugging
            if False:
                mu = self.training_data[self.Xcols_GPR].mean()
                #mu = train.loc[146][Xcols_GPR].mean(); print(mu)
                (fig_mean, fig_std_latent) = self.gpr_model.plot(mu, res=25);
                fig_mean.savefig( os.path.join(self.gprdir, 'plot_mean'+'.'+self.fig_type) );    plt.close(fig_mean) # SLOW
                fig_std_latent.savefig( os.path.join(self.gprdir, 'plot_std_latent'+'.'+self.fig_type) );    plt.close(fig_std_latent) # SLOW
            '''

            fig = self.gpr_model.plot_histogram();
            fig.savefig( os.path.join(self.gprdir, 'histogram'+'.'+self.fig_type) );
            plt.close(fig)

            Ymean = self.training_data['Mean_Err'] + self.training_data['Yglm']
            Yvar = self.training_data['Var_Err_Predictive']
            #self.training_data['Sim_Result']
            fig, ax = plt.subplots(figsize=(16,10))
            ax.errorbar(
                x=self.training_data['Sim_Result'], 
                y=self.training_data['Mean_Err'] + self.training_data['Yglm'], 
                yerr=2*np.sqrt(self.training_data['Var_Err_Predictive']),
                fmt='o', c='c', lw=0.5)
            ax.errorbar(
                x=self.test_data['Sim_Result'], 
                y=self.test_data['Mean_Err'] + self.test_data['Yglm'], 
                yerr=2*np.sqrt(self.test_data['Var_Err_Predictive']),
                fmt='o', c='m', lw=0.5)
            ax.margins(x=0,y=0.05)
            xlim = ax.get_xlim()
            ax.plot( [xlim[0],xlim[1]], [xlim[0], xlim[1]], 'r-')
            ax.set_xlabel('Simulation Result')
            ax.set_ylabel('Predicted')
            fig.savefig( os.path.join(self.gprdir, 'emulation'+'.'+self.fig_type) );
            plt.close(fig)



    def plot(self):
        fig, ax = plt.subplots(figsize=(10,6)) # , sharex='col', sharey='row')

        ax.errorbar(x=self.test_data[self.Ycol], y=self.test_data['Mean_Err'] + self.test_data['Yglm'], yerr=2*np.sqrt(self.test_data['Var_Err_Predictive']), fmt='o', c='m', lw=0.5)
        ax.errorbar(x=self.training_data[self.Ycol], y=self.training_data['Mean_Err'] + self.training_data['Yglm'], yerr=2*np.sqrt(self.training_data['Var_Err_Predictive']), fmt='o', c='c', lw=0.5)
        ax.margins(x=0,y=0.05)
        xlim = ax.get_xlim()
        ax.plot( [xlim[0],xlim[1]], [xlim[0], xlim[1]], 'r-')
        ax.set_xlabel(self.Ycol)
        ax.set_ylabel('Predicted')

        #plt.tight_layout()

        fig.savefig( os.path.join(self.cutdir, 'emulation'+'.'+self.fig_type) ); plt.close(fig)


    def calc_and_plot_implausibility(self,
        plot = False,
        do_plot_data = False,
        plot_data_highlight = pd.DataFrame(),
        log_scale = False
    ):
        """Calculate and plot implausibility.

        Args:
            plot: (bool) Set True to produce plots.
            do_plot_data: (bool) Set True to produce many pairwise plots of the inputs and results.  Within the Implausibility folder, they will appear in `PairwiseResults` for both `Train` and `Test.`
            plot_data_highlight: (float) The guess value for the signal variance. Note that when normalizing Y, a value of 1 correspons to the variance of the results.
            log_scale: (tuple) Lower and upper bounds for sigma2_f, e.g. like (0.005,10).
        """

        self.training_data['Implausibility'] = \
                    abs( self.training_data['Mean_Estimate'] - self.desired_result ) / \
                    np.sqrt(self.training_data['Var_Err_Predictive'] + self.discrepancy_var + self.desired_result_var)
        self.training_data['Implausible'] = self.training_data[ 'Implausibility' ] > self.implausibility_threshold

        self.test_data['Implausibility'] = \
                    abs( self.test_data['Mean_Estimate'] - self.desired_result ) / \
                    np.sqrt(self.test_data['Var_Err_Predictive'] + self.discrepancy_var + self.desired_result_var)
        self.test_data['Implausible'] = self.test_data[ 'Implausibility' ] > self.implausibility_threshold

        self.training_data['Z_Noisy'] = (self.training_data[self.Ycol] - self.training_data['Mean_Estimate']) / np.sqrt(self.training_data['Var_Err_Predictive'])
        self.training_data['Z_Noiseless'] = (self.training_data[self.Ycol] - self.training_data['Mean_Estimate']) / np.sqrt(self.training_data['Var_Err_Latent'])

        self.test_data['Z_Noisy'] = (self.test_data[self.Ycol] - self.test_data['Mean_Estimate']) / \
            np.sqrt(self.test_data['Var_Err_Predictive'] + self.discrepancy_var + self.desired_result_var)
        self.test_data['Z_Noiseless'] = (self.test_data[self.Ycol] - self.test_data['Mean_Estimate']) / \
            np.sqrt(self.test_data['Var_Err_Latent'] + self.discrepancy_var + self.desired_result_var)

        if plot:
            train_mean = self.training_data.reset_index().groupby(['Sample_Id']).mean()
            test_mean = self.test_data.reset_index().groupby(['Sample_Id']).mean()

            fig = plot_errors(train_mean.reset_index(), test_mean.reset_index(), Ycol=self.Ycol, desired_result = self.desired_result);
            fig.savefig( os.path.join(self.combineddir, 'implausibility'+'.'+self.fig_type) );  plt.close(fig)

            if do_plot_data:
                pairdir = HistoryMatching.mkdir_if_needed(os.path.join(self.combineddir, 'PairwiseResults', 'Train'))
                plot_data(train_mean.reset_index(), Ycol=self.Ycol, param_info=self.param_info, circle_points=plot_data_highlight, saveto_dir=pairdir, log_scale=log_scale, desired_result=self.desired_result)

                pairdir = HistoryMatching.mkdir_if_needed(os.path.join(self.combineddir, 'PairwiseResults', 'Test'))
                plot_data(test_mean.reset_index(), Ycol=self.Ycol, param_info=self.param_info, circle_points=plot_data_highlight, saveto_dir=pairdir, log_scale=log_scale)

