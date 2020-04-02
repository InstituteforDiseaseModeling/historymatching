import abc

from hm2.gpr import TorchGPR
from hm2.data_validation import *

class EmulatorBase(abc.ABC):
    pass

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
        self._fit(param_samples, model_output, maxiter)

    def predict(self, test_x):
        """Get GPR's predictions for test_x

        Args:
            test_x - TODO

        Returns: TODO
        """
        return self._predict(test_x)

class TorchGPREmulator(EmulatorBase):
    def __init__(self, basis):
        self.gpr = TorchGPR(basis=basis)

    def _fit(self, param_samples, model_output, maxiter):
        self.gpr.fit(
            train_x = param_samples.iloc[model_output['param_id']],
            train_y = model_output[['value']],
            stdev_y = model_output[['stdev']],
            maxiter = maxiter
        )

    def _predict(self, test_x):
        return self.gpr.predict(test_x)
