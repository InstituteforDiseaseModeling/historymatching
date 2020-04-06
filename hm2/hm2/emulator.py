import abc

from hm2.basis import BasisBase
from hm2.data_validation import *
import hm2.glm
import hm2.gpr



class EmulatorBase(abc.ABC):
    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass



class TorchGPREmulator(EmulatorBase):
    def __init__(self, basis):
        if not isinstance(basis,BasisBase):
            raise HistoryMatchingError("`basis` must inherit from BasisBase!")
        self.basis = basis
        self.gpr = hm2.gpr.TorchGPR()

    def fit(self, train_x, train_y, stdev_y, maxiter):
        """Fit the GPR.

        Args:
            train_x: Training data
            train_y: Correct outputs
            stdev_y: Standard deviation of Y values (uncertainty)
            maxiter: Maximum number of training iterations

        Returns: None
        """
        #Extract the relevant parameter sample values
        self.gpr.fit(
            train_x = self.basis.fit_transform(train_x),
            train_y = train_y,
            stdev_y = stdev_y,
            maxiter = maxiter
        )

    def predict(self, test_x):
        test_x = self.basis.fit_transform(test_x)
        return self.gpr.predict(test_x)



class SkGPREmulator(EmulatorBase):
    def __init__(self, basis):
        if not isinstance(basis,BasisBase):
            raise HistoryMatchingError("`basis` must inherit from BasisBase!")
        self.basis = basis
        self.gpr = hm2.gpr.SkGPR()

    def fit(self, train_x, train_y, stdev_y, maxiter):
        """Fit the GPR.

        Args:
            train_x: Training data
            train_y: Correct outputs
            stdev_y: Standard deviation of Y values (uncertainty)
            maxiter: Maximum number of training iterations

        Returns: None
        """
        #Extract the relevant parameter sample values
        self.gpr.fit(
            train_x = self.basis.fit_transform(train_x),
            train_y = train_y,
            stdev_y = stdev_y,
            maxiter = maxiter
        )

    def predict(self, test_x):
        return self.gpr.predict(self.basis.fit_transform(test_x))



class GLM_GPR_Emulator(EmulatorBase):
    """Emulator that trains a GLM on data and a GPR on the residuals.
    """
    def __init__(
            self,
            glm_basis,
            gpr_basis,
            family = 'gaussian',
    ):
        """Initialize the Emulator"""
        if not isinstance(glm_basis,BasisBase):
            raise HistoryMatchingError("`glm_basis` must inherit from BasisBase!")
        if not isinstance(gpr_basis,BasisBase):
            raise HistoryMatchingError("`gpr_basis` must inherit from BasisBase!")

        self.glm_basis = glm_basis
        self.gpr_basis = gpr_basis

        self.glm = hm2.glm.GLM(family=family)
        self.gpr = hm2.gpr.SkGPR()

    def fit(self, train_x, train_y, stdev_y, glm_maxiter=1000, gpr_maxiter=1000):
        """Fit the GPR.

        Args:
            train_x: Training data
            train_y: Correct outputs
            stdev_y: Standard deviation of Y values (uncertainty)
            glm_maxiter: Maximum number of training iterations in GLM fitting
            gpr_maxiter: Maximum number of training iterations in GLM fitting

        Returns: None
        """
        #Extract the relevant parameter sample values
        train_x_glm = self.glm_basis.fit_transform(train_x)
        train_x_gpr = self.gpr_basis.fit_transform(train_x)

        self.glm.fit(train_x_glm, train_y, maxiter=glm_maxiter)

        residuals = train_y-self.glm.predict(train_x_glm)

        self.gpr.fit(train_x_gpr, residuals, stdev_y, maxiter=gpr_maxiter)

    def predict(self, test_x):
        """Evaluate the emulator and return the mean prediction.

        Args:
            test_x: (Pandas DataFrame)
                Data frame of points similar to training_data.

        Returns:
            Predicted outputs at the inputs specified by data.
        """
        test_x_glm = self.glm_basis.fit_transform(test_x)
        test_x_gpr = self.gpr_basis.fit_transform(test_x)

        return self.glm.predict(test_x_glm) + self.gpr.predict(test_x_gpr)
