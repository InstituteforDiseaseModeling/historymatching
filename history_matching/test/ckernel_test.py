import unittest

import numpy as np

import ckernels
import time

def kernel_xp(X, P, theta):
    """Original XP kernel

    Args:
        X: (2D ndarray) points of dimension N x D
        P: (2D ndarray) points of dimension P x D
        theta: (1D ndarray) hyperparameters of length D
    """
    sigma2_f = theta[0]

    Nx = X.shape[0]
    Np = P.shape[0]
    D = X.shape[1]

    kxp = np.zeros([Nx,Np])
    for i in range(Nx):
        for j in range(Np):
            dX = X[i,:]-P[j,:]
            r2 = 0
            for d in range(D):
                r2 += dX[d] * dX[d]/theta[2+d]
            kxp[i,j] = sigma2_f * np.exp( -r2 / 2. )
    return kxp

class TestCKernels(unittest.TestCase):
    def test_kernel_xp(self):
        Nx    = 200
        Np    = 400
        D     = 30
        X     = np.random.random((Nx,D))
        P     = np.random.random((Np,D))
        theta = np.random.random(D+2)

        t0 = time.time()
        pret = kernel_xp(X,P,theta)
        t1 = time.time()
        print(t1-t0)

        t0 = time.time()
        cret = ckernels.kernel_xp(X,P,theta)
        t1 = time.time()
        print(t1-t0)

        self.assertIsNone(np.testing.assert_array_equal(pret, cret))
