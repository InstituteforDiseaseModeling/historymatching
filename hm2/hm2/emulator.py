import abc
# import hm2.GLM
from hm2.gpr import GPR

class EmulatorBase(abc.ABC):
    pass



class GPR_Emulator(EmulatorBase):
    def __init__(self, basis):
        self.gpr = GPR(basis=basis)

    def fit(self, data, endog, maxiter=1000):
        return self.gpr.fit(data, endog, maxiter)

    def predict(self, data):
        return self.gpr.predict(data)
