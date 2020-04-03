import abc

from hm2.basis import BasisBase
from hm2.data_validation import *
import hm2.gpr



class EmulatorBase(abc.ABC):
    def __init__(self, basis):
        if not isinstance(basis,BasisBase):
            raise HistoryMatchingError("`basis` must inherit from BasisBase!")
        self.basis=basis

    def fit(self, param_samples, model_output, maxiter=1000):
        """Fit the Emulator

        Args:
            param_samples - ParameterSamplesFrame
            model_output - A TimeStandardAnalysisFrame or 
                           SummaryStandardAnalysisFrame built using parameters
                           from param_samples
            maxiter - Number of training iterations

        Returns: None
        """
        param_samples = ValidateParameterSamplesFrame(param_samples)
        model_output  = ValidateEmulatorInput(model_output)

        #Extract the relevant parameter sample values
        train_x = param_samples.iloc[model_output['param_id']]
        train_x = train_x.drop(columns=['param_id'])
        train_x = self.basis.fit_transform(train_x).to_numpy()

        train_y = model_output['value'].to_numpy()
        stdev_y = model_output['stdev'].to_numpy()

        print(train_x)
        print(train_y)
        print(stdev_y)

        self._fit(train_x, train_y, stdev_y, maxiter)

    def predict(self, test_x):
        """Get GPR's predictions for test_x

        Args:
            test_x - TODO

        Returns: TODO
        """
        test_x = self.basis.fit_transform(test_x).to_numpy()
        print(test_x)
        return self._predict(test_x)



class TorchGPREmulator(EmulatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpr = hm2.gpr.TorchGPR()

    def _fit(self, train_x, train_y, stdev_y, maxiter):
        self.gpr.fit(train_x, train_y, stdev_y, maxiter)

    def _predict(self, test_x):
        return self.gpr.predict(test_x)

class SkGPREmulator(EmulatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpr = hm2.gpr.SkGPR()

    def _fit(self, train_x, train_y, stdev_y, maxiter):
        self.gpr.fit(train_x, train_y, stdev_y, maxiter)

    def _predict(self, test_x):
        return self.gpr.predict(test_x)
