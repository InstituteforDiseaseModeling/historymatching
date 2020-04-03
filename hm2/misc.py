
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
