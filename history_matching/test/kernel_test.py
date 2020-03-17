import unittest
import numpy as np
import pandas as pd
from history_matching.gpc import GPC
import history_matching.kernels as kernels

class KernelTest(unittest.TestCase):

    def test_cpu_gpu_same(self):
        np.random.seed(10)

        #N = 20+30+10
        N = 100
        x = np.linspace(0,2*np.pi,N)
        #f = (np.sin(x) + 1)/2.
        f = 1 / (1 + np.exp(-3*(x-np.pi)))
        y = 2 * (np.random.rand(N) < f) - 1

        data = pd.DataFrame({
            'x': x,
            'y': y
            })
        data.index.name = 'Sample'

        param_info = pd.DataFrame({
            'Name': ['x'],
            'Min': [0], #[-9], #[0],
            'Max': [2*np.pi], #[5], #[2*np.pi]
        }).set_index('Name')

        g = GPC(['x'], 'y', data, param_info,
                    kernel_mode = 'RBF',
                    kernel_params = [40, 0.14], # Sigma_f^2 and lengthscale^2
                    verbose = False,
                    debug = False
                )

        X     = np.random.random((10,30))
        P     = np.random.random((5,30))
        theta = np.random.random(31)

        print("theta", theta)
        print("sigma2_f", theta[0])
        print("theta cut", theta[1:])

        gpu_result = g.kxp_gpu_wrapper(X=X, P=P, theta=theta)
        # cpu_result = g.kernel_xp(X=X, P=P, theta=theta)
        cpu_result2 = kernels.kernel_xp(X=X, P=P, sigma2_f=theta[0], theta=theta[1:], mode="cpu")
        gpu_result2 = kernels.kernel_xp(X=X, P=P, sigma2_f=theta[0], theta=theta[1:], mode="gpu")

        print("\ngpu_result\n",gpu_result)
        print("\ncpu_result2\n",cpu_result2)
        print("\ngpu_result2\n",gpu_result2)

        self.assertTrue(np.allclose(gpu_result,cpu_result2))
        self.assertTrue(np.allclose(gpu_result2,cpu_result2))
        # self.assertTrue(np.allclose(cpu_result2,cpu_result))
