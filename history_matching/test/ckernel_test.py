import itertools
import time
import unittest

import numpy as np

import ckernels

def timeify(x,msg):
    t0 = time.time()
    ret = x()
    t1 = time.time()
    print(f"Time for {msg}: {t1-t0}s")
    return ret

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

def kernel_xx(X, theta, sigma2_n, add_sigma2_n, deriv):
    """Original XX kernel
    """
    Nx  = X.shape[0]
    D   = X.shape[1]
    Kxx = np.zeros([Nx,Nx], dtype=np.float64)
    sigma2_f = theta[0]
    for i in range(Nx):
        # Diagonal:
        if deriv <= 1:
            Kxx[i,i] = sigma2_f # theta[0] or 1, see above
        else:
            Kxx[i,i] = 0

        # Off-diagonal
        for j in range(i+1,Nx):
            dX = X[i,:]-X[j,:]
            r2 = 0
            for d in range(D):
                r2 += dX[d] * dX[d]/theta[2+d]
            Kxx[i,j] = sigma2_f * np.exp( -r2 / 2. )

            if (deriv > 1): # Lengthscale derivatives
                d = deriv-2;
                Kxx[i,j] *= 0.5 * (dX[d] * dX[d]) / (theta[2+d] * theta[2+d]);

            Kxx[j,i] = Kxx[i,j]

    if add_sigma2_n:
        # Add sigma_n^2 to the diagonal, observation noise
        Kxx[np.diag_indices(Nx)] += sigma2_n

    return Kxx


class TestCKernels(unittest.TestCase):
    def test_kernel_xp(self):
        Nx    = 200
        Np    = 400
        D     = 30
        X     = np.random.random((Nx,D))
        P     = np.random.random((Np,D))
        theta = np.random.random(D+2)

        pret = timeify(lambda:kernel_xp(X,P,theta), "kernel_xp")
        cret = timeify(lambda:ckernels.kernel_xp(X,P,theta), "ckernels.kernel_xp")
        self.assertIsNone(np.testing.assert_array_equal(pret, cret))

    def test_kernel_xx(self):
        Nx       = 200
        D        = 30
        X        = np.random.random((Nx,D))
        sigma2_n = np.random.random(Nx)
        theta    = np.random.random(D+2)
        deriv    = 2

        for add_sigma2_n, deriv in itertools.product([True,False],[-1,0,1,2,3,4]):
            pret = timeify(lambda:kernel_xx(X,theta,sigma2_n,True,deriv), "kernel_xx")
            cret = timeify(lambda:ckernels.kernel_xx(X,theta,sigma2_n,True,deriv), "ckernels.kernel_xx")
            self.assertIsNone(np.testing.assert_array_equal(pret, cret))
