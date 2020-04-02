import abc

from hm2.gpr import GPR
from hm2.data_validation import *

class SingleEmulatorBase(abc.ABC):
    pass



class GPR_SingleEmulator(SingleEmulatorBase):
    def __init__(self, basis):
        self.gpr = GPR(basis=basis)

    def fit(self, param_samples, model_output, maxiter=1000):
        """Fit the GPR_SingleEmulator

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

        self.gpr.fit(
            train_x = param_samples.iloc[model_output['param_id']],
            train_y = model_output[['value']],
            stdev_y = model_output[['stdev']],
            maxiter = maxiter
        )

    def predict(self, test_x):
        """Get GPR's predictions for test_x

        Args:
            test_x - TODO

        Returns: TODO
        """
        return self.gpr.predict(test_x)
