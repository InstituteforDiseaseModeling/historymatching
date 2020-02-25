import unittest
import numpy as np
import pandas as pd
from history_matching.gpc import GPC

class GPCTest(unittest.TestCase):

    def test_1d(self):
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
                    #kernel_params = [0.001, 0.04],
                    kernel_params = [40, 0.14], # Sigma_f^2 and lengthscale^2
                    verbose = False,
                    debug = False
                )

        ### Test find posterior mode against eq 3.17
        theta = [4, 0.2]
        ret = g.find_posterior_mode(theta, f_guess=None, tol_grad=1e-6, maxiter=10000)
        self.assertTrue( np.allclose(ret['f_hat'], np.dot(ret['K'], ret['d_df_log_p_y_given_f']), atol=1e-5) )

    def test_2d(self):
        np.random.seed(10)

        f = lambda x,y: 1/4 * (np.tanh(10*x-5)+1) * (np.tanh(3*y-0.9)+1)
        target = 0.7
        implausibility_threshold = 3

        N = 100

        pts = np.random.rand(N,2)
        data = pd.DataFrame({
            'x': pts[:,0],
            'y': pts[:,1],
            })
        data['f'] = data.apply( lambda d: f(d['x'], d['y']), axis=1)
        data['z'] = 2 * (np.random.rand(N) < data['f']) - 1
        data.index.name = 'Sample'

        param_info = pd.DataFrame({
            'Name': ['x', 'y'],
            'Min': [0, 0],
            'Max': [1, 1],
        }).set_index('Name')

        g = GPC(['x', 'y'], 'z', data, param_info,
                    kernel_mode = 'RBF',
                    kernel_params = [20.57666683, 0.20004966, 1.96484556], # Sigma_f^2 and lengthscale_x^2 lengthscale_y^2
                    verbose = False,
                    debug = False
                )

        ### Test find posterior mode against eq 3.17
        theta = [2, 0.1, 0.1]
        ret = g.find_posterior_mode(theta, f_guess=None, tol_grad=1e-6, maxiter=10000)
        self.assertTrue( np.allclose(ret['f_hat'], np.dot(ret['K'], ret['d_df_log_p_y_given_f']), atol=1e-5) )


unittest.main()
