
#TODO: Finish
class GLM_GPR_Emulator(EmulatorBase):
    """Emulator that trains a GLM on data and a GPR on the residuals.
    """
    def __init__(
            self,
            glm_basis,
            gpr_basis,
            family = 'poisson',
    ):
        """Initialize the Emulator

        Args:
            polyorder: Order of polynomial expansion of the data features
            intercept: Whether to add an intercept feature
            family: (str) The family of generalized linear model to use. 
                          Options include 'poisson', 'binomial', 'gamma', 
                          'negativebinomial', and 'gaussian'. 
        """
        self.model = None

        self.glm = GLM(basis=glm_basis, family=family)
        self.gpr = GPR(basis=gpr_basis)

    #TODO(r-barnes): Differentiate glm and gpr maxiter?
    def fit(self, data, endog, maxiter=1000):
        """Fit the emulator.

        Args:
            maxiter: (int)
                maxiter parameter passed to the statsmodels `fit` function.
        """
        self.glm.fit(data, endog, maxiter)
        residuals = self.glm.residuals(data, endog)
        self.glm.fit(data, residuals, maxiter)

    def predict(self, data):
        """Evaluate the emulator and return the mean prediction.

        Args:
            data: (Pandas DataFrame)
                Data frame of points similar to training_data.

        Returns:
            Predicted outputs at the inputs specified by data.
        """
        return self.glm.predict(data)+self.glm.predict(data)

    def residuals(self, data, endog):
        return self.predict(data) - endog











#  _______ .______   .______         .______    __        ______   .___________.    _______.
# /  _____||   _  \  |   _  \        |   _  \  |  |      /  __  \  |           |   /       |
#|  |  __  |  |_)  | |  |_)  |       |  |_)  | |  |     |  |  |  | `---|  |----`  |   (----`
#|  | |_ | |   ___/  |      /        |   ___/  |  |     |  |  |  |     |  |        \   \
#|  |__| | |  |      |  |\  \----.   |  |      |  `----.|  `--'  |     |  |    .----)   |
# \______| | _|      | _| `._____|   | _|      |_______| \______/      |__|    |_______/


    def plot_data(self, samples_to_circle=pd.DataFrame(), saveto_dir = None, log_scale = False):
        """Make pairwise plots of data.

        TODO: Make the scaling function a lambda.

        Size is maximum of 1 and 25*normalized_y_value.

        Args:
            samples_to_circle: (Pandas DataFrame, similar cols to training_data) Plot size 50 black x's, one for each row
            saveto_dir: (str, default None) Directory name where resulting figures should be saved.  None disables saving.
            log_scale:  (boolean, default is False) transforms size and color using log(10 * normalized_y_value + 1)

        Returns: Dictionary of matplotlib figure handle.  Keys are like VarName1-VarName2.pdf for each pair of variables.
        """

        scaled = (self.training_data[self.Ycol]-self.training_data[self.Ycol].min()) / (self.training_data[self.Ycol].max()-self.training_data[self.Ycol].min())
        if log_scale:
            scaled = np.log( 10*scaled+1 )

        figs = {}

        X = self.basis.generate_dmatrix( self.training_data, scaleX = True)
        Xcols = X.columns.tolist()

        if samples_to_circle.shape[0] > 0:
            samples_to_circle_dmat = self.basis.generate_dmatrix( samples_to_circle, scaleX = True)

        for row in range(self.D):
            for col in range(self.D):
                if col > row:
                    fn = '%s-%s' % (Xcols[row], Xcols[col]) +'.'+self.fig_type
                    fig = plt.figure(figsize=(6,6)) #GPy.plotting.plotting_library().figure()

                    x = X[Xcols[row]]
                    y = X[Xcols[col]]

                    plt.scatter(x, y, s=100*scaled, c=100*scaled, cmap='jet', linewidths=0.1, alpha=0.5, edgecolors='k') #, s=area, c=colors, alpha=0.5)

                    # Circle some interesting samples
                    if samples_to_circle.shape[0] > 0:
                        for _, pt in samples_to_circle_dmat.iterrows():
                            plt.scatter(pt[ Xcols[row] ], pt[ Xcols[col] ], s=50, c='k', alpha=1, linewidths=2.0, marker='x') #, s=area, c=colors, alpha=0.5)

                    plt.autoscale(tight=True)
                    plt.xlabel( Xcols[row] )
                    plt.ylabel( Xcols[col] )
                    plt.tight_layout()

                    if saveto_dir is not None:
                        fig.savefig( os.path.join(saveto_dir, fn) ); plt.close(fig)
                    else:
                        figs[fn] = fig

        return figs


    def plot_histogram(self):
        """Plots histograms of the training data using Seaborn's distplot routine.

        Returns: Matplotlib figure handle
        """

        fig, ax = plt.subplots(nrows=1, ncols=1) # , figsize=(5,5), sharex='col', sharey='row')
        sns.distplot(self.training_data[self.Ycol], rug=True, ax = ax)

        return fig


    def plot(self, Xcenter, res=10):
        """Plots 2D contour slices through the output GPR.

        When evaluating sweeping two parameteres at a time, the other parameters are fixed at Xcenter.

        Args:
            Xcenter: (1D ndarray similar to x0) These are the 'baseline' values unless modified in a 2D sweep.
            res: (int) number of grid points per dimension.  res*res points will be evaluated to generate each pairwise plot.

        Returns: Tuple of matplotlib figure handles.  The first element is for the mean and the second is for the latent standard deviation.
        """
        Xmu = np.repeat( np.array([Xcenter]), res*res, axis=0)

        fig = plt.figure(figsize=(4*(self.D-1),4*(self.D-1)))
        fig_std_latent = plt.figure(figsize=(4*(self.D-1),4*(self.D-1)))
        for row in range(self.D):
            for col in range(self.D):
                if col > row:
                    gs = gridspec.GridSpec(self.D-1, self.D-1)
                    ax = fig.add_subplot(gs[col-1,row]) # , projection='3d'
                    ax_std_latent = fig_std_latent.add_subplot(gs[col-1,row]) # , projection='3d'

                    fixed_inputs = [ (x,mean) for (i, (x,mean)) in enumerate(zip(range(self.D), Xcenter)) if row is not i and col is not i]
                    print(row, col, row*self.D+col, fixed_inputs)

                    row_min, row_max = self.training_data[self.Xcols[row]].min(), self.training_data[self.Xcols[row]].max()
                    col_min, col_max = self.training_data[self.Xcols[col]].min(), self.training_data[self.Xcols[col]].max()
                    x1 = np.linspace(row_min, row_max, res)
                    x2 = np.linspace(col_min, col_max, res)
                    X1, X2 = np.meshgrid(x1, x2)

                    X = Xmu.copy()
                    X[:,row] = X1.flatten()
                    X[:,col] = X2.flatten()

                    Xdf = pd.DataFrame(X, columns=self.Xcols)

                    self.debug=False
                    self.verbose=False

                    ret = self.evaluate( Xdf )

                    Y_mean = np.reshape(ret['Mean'], [res,res])
                    Y_std_latent = np.reshape( np.sqrt(ret['Var_Latent']), [res, res])
                    #Y_std_predictive = np.reshape( np.sqrt(ret['Var_Predictive']), [res, res])

                    try:
                        CS = ax.contour(X1, X2, Y_mean, zorder=100)
                        ax.clabel(CS, inline=1, fontsize=10, zorder=100)
                    except:
                        print('Unable to plot mean contour')

                    ax.scatter(self.training_data[self.Xcols[row]], self.training_data[self.Xcols[col]], c=self.training_data[self.Ycol], s=25, cmap='jet')

                    try:
                        CS = ax_std_latent.contour(X1, X2, Y_std_latent, zorder=100)
                        ax_std_latent.clabel(CS, inline=1, fontsize=10, zorder=100)
                    except:
                        print('Unable to plot std contour')

                    if col == self.D-1:
                        ax.set_xlabel( self.Xcols[row] )
                    if row == 0:
                        ax.set_ylabel( self.Xcols[col] )
        #plt.tight_layout()
        return (fig, fig_std_latent)


    def plot_errors(self, train, test, mean_col, var_col):
        """Generates two plots on a single figure.

        The upper plot shows GP prediction on Y as a function of the true Y-values on X.  The lower panel shows Z-score on Y and the true Y-values on X.

        In both panels, training data is cyan and test data is magenta.

        Args:
            train: (Pandas DataFrame) training data like training_data.
            test: (Pandas DataFrame) test data like training_data.
            mean_col: (str) Column name of predicted mean in train and test dataframes.
            var_col: (str) Column name of variance (latent or predictive, you pick) in train and test dataframes.

        Returns: Matplotlib figure handle.
        """

        train['Z_Score'] = (train[self.Ycol_orig] - train[mean_col]) / np.sqrt(train[var_col])
        test['Z_Score'] = (test[self.Ycol_orig] - test[mean_col]) / np.sqrt(test[var_col])

        fig, ax = plt.subplots(nrows=1, ncols=1, sharex='col', figsize=(16,10)) # , sharex='col', sharey='row')

        ax.errorbar(x=test[self.Ycol_orig], y=test[mean_col], yerr=2*np.sqrt(test[var_col]), fmt='o', c='m', lw=0.5)
        ax.errorbar(x=train[self.Ycol_orig], y=train[mean_col], yerr=2*np.sqrt(train[var_col]), fmt='o', c='c', lw=0.5)
        ax.margins(x=0,y=0.05)
        xlim = ax.get_xlim()
        ax.plot( [xlim[0],xlim[1]], [xlim[0], xlim[1]], 'r-')
        ax.set_xlabel(self.Ycol_orig)
        ax.set_ylabel('Predicted')

        #TODO(dklein): Can the below be removed?
        '''
        ax = ax2
        ax.scatter(x=train[self.Ycol_orig], y=train['Z_Score'], facecolor='c', marker='.', lw=1, alpha=0.5, s=50)
        ax.scatter(x=test[self.Ycol_orig], y=test['Z_Score'], facecolor='m', marker='.', lw=1, alpha=0.5, s=50)
        ax.set_xlabel(self.Ycol_orig)
        ax.set_ylabel('Z-Score')
        ax.margins(x=0,y=0.05)
        '''

        plt.tight_layout()

        return fig












#   ________    __  ___   ____  __    ____  ___________
#  / ____/ /   /  |/  /  / __ \/ /   / __ \/_  __/ ___/
# / / __/ /   / /|_/ /  / /_/ / /   / / / / / /  \__ \
#/ /_/ / /___/ /  / /  / ____/ /___/ /_/ / / /  ___/ /
#\____/_____/_/  /_/  /_/   /_____/\____/ /_/  /____/



    def plot_data_1D(self, circle_points=None, saveto_dir=None, log_scale=True):
        """For 1D data, plots a scatter of output (y) vs input (x).

        Args:
            circle_points: (Pandas DataFrame)
                A data frame like training_data.  Each entry will be marked with a black x's in the figures.  Good for debugging large Z scores.
            saveto_dir: (str)
                If not None, figures will be saved to this directory.  The user may need to create the output directory.
            log_scale:  (boolean, default is False) transforms size and color using log(10 * normalized_y_value + 1)

        Returns: a dictionary of matplotlib figure handles with keys indicating the parameter names via the filename which would be used to save the figure.
        """
        if circle_points is None:
            circle_points = pd.DataFrame()

        # TODO: Save and log scale!
        scaled = np.log(1+self.training_data[self.Ycol])# / self.training_data[self.Ycol].max()

        Xcols = self.basis.get_terms()[0] # Not tested!
        fig = plt.figure(figsize=(6, 8)) # GPy.plotting.plotting_library().figure()
        x = self.training_data[Xcols]
        y = self.training_data[self.Ycol]

        plt.scatter(x, y, s=15, c=scaled, cmap='jet', linewidths=0.1, alpha=0.5, edgecolors='k') #, s=area, c=colors, alpha=0.5)

        for _, pt in circle_points.iterrows():
            plt.scatter(pt[Xcols], pt[self.Ycol], s=25, c='k', alpha=1, linewidths=2.0, marker='x') #, s=area, c=colors, alpha=0.5)

        plt.autoscale(tight=True)
        plt.xlabel(Xcols)
        plt.ylabel(self.Ycol)
        plt.tight_layout()

        return {Xcols: fig}








    def plot_fit(self, figsize=(16,16)):
        """Plots each output predicted by the GLM on X againsT sample index on Y.

        If there are multiple replicates per Sample_ID, a blue line will connect
        the Min to the Max. A vertical red line is drawn at the reference value.
         The green line is at the mean of the fitted model. Finally, the black
        `|` is the true value(s) from the simulation.

        Returns: matplotlib figure handle.
        """

        fig, axes = plt.subplots(figsize=figsize)

        d = self.training_data.reset_index()
        d_by_sample = self.training_data.reset_index().set_index('Sample_Id')
        n_samples = len(d_by_sample.index.unique())

        axes.plot(2*[self.reference_value], [0, n_samples], 'r-') # , axes=axes[0,0]

        sim_cases_range = self.training_data.reset_index().groupby('Sample_Id')[self.Ycol].agg({'Min':np.min, 'Max':np.max, 'Mean':np.mean})
        sim_cases_range['Fitted_Model_Mean'] = self.fitted_model.mu
        for idx, s in sim_cases_range.iterrows():
            axes.plot([s['Min'], s['Max']], [idx, idx], 'b-', linewidth=0.5)
            axes.plot([s['Mean'], s['Fitted_Model_Mean']], [idx, idx], 'g-', linewidth=0.25)
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

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.plot(train[self.Ycol], train['Yglm'], 'c+', ms=10, mew=1)
        ax.plot(test[self.Ycol], test['Yglm'], 'm+', ms=10, mew=1)
        ax.margins(x=0, y=0.05)
        xlim = ax.get_xlim()
        ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], 'r-')
        ax.set_xlabel('Simulation Result')
        ax.set_ylabel('Predicted')

        figs['GLM Predicted vs Actual'] = fig

        for exp_id in exp_ids:
            fig, ax = plt.subplots(figsize=(16, 10))
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

            ax.margins(x=0, y=0.05)
            ax.set_xlabel('Sample')

            figs['GLM expId ' + str(exp_id)] = fig

        #ax.set_ylabel(self.Ycol)
        #plt.tight_layout()

        return figs
