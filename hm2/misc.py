
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
