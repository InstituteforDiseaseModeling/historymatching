import unittest
import numpy as np
import pandas as pd
from history_matching.gpc import GPC
import history_matching.kernels as kernels

class KernelTest(unittest.TestCase):

    def test_kernel_xp_cpu_gpu_same(self):
        X     = np.random.random((1000,30))
        P     = np.random.random((200, 30))
        theta = np.random.random(31)

        cpu_result = kernels.kernel_xp(X=X, P=P, sigma2_f=theta[0], theta=theta[1:], mode="cpu")
        gpu_result = kernels.kernel_xp(X=X, P=P, sigma2_f=theta[0], theta=theta[1:], mode="gpu")

        self.assertTrue(np.allclose(cpu_result,gpu_result))
