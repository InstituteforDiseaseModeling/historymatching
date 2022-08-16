from typing import Optional, Type
import abc

import numpy as np

from hm2.basis import BasisBase
from hm2.data_validation import *
import hm2.glm
import hm2.gpr
from hm2.plotting import *


class EmulatorBase(abc.ABC):  # pragma: no cover
    def _prep_fitting_data(self,
        train_x: pd.DataFrame,
        train_y,
        stdev_y: np.array
    ):
        train_x = ValidateParameterSamplesFrame(train_x)
        train_x = train_x.drop(columns='param_id')
        self._train_x = train_x
        self._train_y = train_y.copy()
        return train_x, train_y, stdev_y

    def _prep_prediction_data(self, test_x):
        if "param_id" in test_x.columns:
            test_x = test_x.drop(columns="param_id")
        return test_x

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    @abc.abstractmethod
    def plot_data(self, *args, **kwargs):
        pass



class SkGPREmulator(EmulatorBase):
    """Use the Sklearn GPR as the emulator"""
    def __init__(self, basis:EmulatorBase):
        if not isinstance(basis, BasisBase):
            raise HistoryMatchingError("`basis` must inherit from BasisBase!")
        self.basis = basis
        self.gpr = hm2.gpr.SkGPR()

    def fit(self,
        train_x: pd.DataFrame,
        train_y,
        stdev_y,
        maxiter:int
    ):
        """Fit the GPR.

        Args:
            train_x: Training data. A :ref:`ParameterSamplesFrame`.
            train_y: Correct outputs
            stdev_y: Standard deviation of Y values (uncertainty)
            maxiter (int): Maximum number of training iterations

        Returns:
            None
        """
        assert isinstance(maxiter,int) and maxiter>=0

        train_x, train_y, stdev_y = self._prep_fitting_data(train_x, train_y, stdev_y)

        # Extract the relevant parameter sample values
        self.gpr.fit(
            train_x=self.basis(train_x),
            train_y=train_y,
            stdev_y=stdev_y,
            maxiter=maxiter,
        )
        return self

    def predict(self, test_x):
        """Evaluate the emulator and return its prediction.

        Args:
            test_x: Data frame of points similar to training_data.

        Returns:
            Predicted outputs at the inputs specified by data.
        """
        test_x = self._prep_prediction_data(test_x)
        return self.gpr.predict(self.basis(test_x))

    def plot_data(self, *args, **kwargs):
        return plot_pairwise(
            self.basis(self._train_x), self._train_y, *args, **kwargs
        )


class GLM_GPR_Emulator(EmulatorBase):
    """Emulator that trains a GLM on data and a GPR on the residuals.
    """

    def __init__(
        self,
        glm_basis: Type[BasisBase],
        gpr_basis: Type[BasisBase],
        family:str="gaussian",
    ):
        """Initialize the Emulator"""
        if not isinstance(glm_basis, BasisBase):
            raise TypeError("`glm_basis` must inherit from BasisBase!")
        if not isinstance(gpr_basis, BasisBase):
            raise TypeError("`gpr_basis` must inherit from BasisBase!")

        self.glm_basis = glm_basis
        self.gpr_basis = gpr_basis

        self.glm = hm2.glm.GLM(family=family)
        self.gpr = hm2.gpr.SkGPR()

    def fit(self,
        train_x: pd.DataFrame,
        train_y,
        stdev_y: Optional[np.array]=None,
        glm_maxiter:int=1000,
        gpr_maxiter:int=1000,
        glm_seed:Optional[int]=None
    ):
        """Fit the GPR.

        Args:
            train_x: Training data. A :ref:`ParameterSamplesFrame`.
            train_y: Correct outputs
            stdev_y: Standard deviation of Y values (uncertainty). If `None`,
                     then zero uncertainty is assumed.
            glm_maxiter (int): Maximum number of training iterations in GLM fitting
            gpr_maxiter (int): Maximum number of training iterations in GLM fitting
            glm_seed: Random seed for initializing GPR centers. `None`
                      chooses a random seed.

        Returns:
            None
        """
        assert isinstance(glm_maxiter,int) and glm_maxiter>=0
        assert isinstance(gpr_maxiter,int) and gpr_maxiter>=0

        train_x, train_y, stdev_y = self._prep_fitting_data(train_x, train_y, stdev_y)

        # Extract the relevant parameter sample values
        train_x_glm = self.glm_basis(train_x)
        train_x_gpr = self.gpr_basis(train_x)

        self.glm.fit(train_x_glm, train_y, maxiter=glm_maxiter)

        residuals = train_y - self.glm.predict(train_x_glm)

        self.gpr.fit(train_x_gpr, residuals, stdev_y, maxiter=gpr_maxiter, random_state=glm_seed)

        return self

    def predict(
        self,
        test_x: pd.DataFrame
    ):
        """Evaluate the emulator and return its prediction.

        Args:
            test_x: Data frame of points similar to training_data.

        Returns:
            Predicted outputs at the inputs specified by data.
        """
        test_x = self._prep_prediction_data(test_x)

        test_x_glm = self.glm_basis(test_x)
        test_x_gpr = self.gpr_basis(test_x)

        glm_prediction = self.glm.predict(test_x_glm)
        gpr_prediction, gpr_stdev = self.gpr.predict(test_x_gpr)

        return glm_prediction + gpr_prediction, gpr_stdev

    def plot_data(self, *args, **kwargs):
        """Plots the basisified training data against itself in pairwise plots
        with colour determined by the y value"""
        return plot_pairwise(
            self.glm_basis(self._train_x), self._train_y, *args, **kwargs
        )

    def plot_emulated_vs_predicted(self, figsize=(10.0, 8.0)):
        #TODO: docstring
        predicted, stdev = self.predict(self._train_x)

        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        ax.errorbar(self._train_y, predicted, yerr=stdev, fmt='o', ms=3, lw=0.5)
        ax.set_title('Model Output vs Predicted')
        ax.set_xlabel('Model Output')
        ax.set_ylabel('Predicted')

        return WrappedFigure(fig)