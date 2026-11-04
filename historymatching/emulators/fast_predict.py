"""
Fast GPR prediction using numba, with no TensorFlow call at prediction time.

Extracts kernel hyperparameters from a trained GPflow model, precomputes
alpha = (K + noise I)^-1 (y - c) and the inverse Cholesky factor once, then
predicts the SE-kernel GP mean and variance in numpy/numba.

Not currently used by the engine, which filters through `GPR.predict`. Measured
against `GPR.predict` on 20,000 points (Apple M1 Max), it is 2.6x faster with
180 training points in 4 dimensions and 1.1x with 1,800 in 21, so it is kept
for a future, faster `GPR.predict` rather than as a separate filter.

Usage:

```python
from historymatching.emulators.fast_predict import FastGPRPredictor

fast = FastGPRPredictor.from_emulator(emulator)   # a trained GPR emulator
mean, var = fast.predict(X_test)                  # numpy arrays; var includes the noise
```
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import numba

    @numba.njit(parallel=True, cache=True)
    def _se_kernel_predict(X_test, X_train, ls, kernel_var, alpha, mean_c):
        """Predict GP mean for test points using SE kernel.

        Fused loop avoids allocating the (n_test × n_train × n_dims)
        intermediate array.  Parallelized across test points via prange.
        """
        n_test = X_test.shape[0]
        n_train = X_train.shape[0]
        n_dims = X_test.shape[1]
        mean = np.empty(n_test)
        for i in numba.prange(n_test):
            s = 0.0
            for j in range(n_train):
                sq_dist = 0.0
                for d in range(n_dims):
                    diff = (X_test[i, d] - X_train[j, d]) / ls[d]
                    sq_dist += diff * diff
                s += kernel_var * np.exp(-0.5 * sq_dist) * alpha[j]
            mean[i] = s + mean_c
        return mean

    @numba.njit(parallel=True, cache=True)
    def _se_kernel_predict_with_var(X_test, X_train, ls, kernel_var, alpha,
                                     mean_c, noise_var, L_inv):
        """Predict GP mean and variance for test points.

        Variance = kernel_var + noise_var - ||L_inv @ k_star||²
        where L_inv is the inverse of the Cholesky factor (precomputed).

        The L_inv @ k_row product is a simple dot (not triangular solve),
        so the cost per test point is O(n_train²) — same as O(n_train) for
        the mean when n_train is the dominant term.
        """
        n_test = X_test.shape[0]
        n_train = X_train.shape[0]
        n_dims = X_test.shape[1]
        mean = np.empty(n_test)
        var = np.empty(n_test)
        for i in numba.prange(n_test):
            # Compute K_star row and mean simultaneously
            k_row = np.empty(n_train)
            s = 0.0
            for j in range(n_train):
                sq_dist = 0.0
                for d in range(n_dims):
                    diff = (X_test[i, d] - X_train[j, d]) / ls[d]
                    sq_dist += diff * diff
                k_row[j] = kernel_var * np.exp(-0.5 * sq_dist)
                s += k_row[j] * alpha[j]
            mean[i] = s + mean_c

            # Variance: kernel_var + noise_var - ||L_inv @ k_row||²
            # L_inv is full (not triangular), so just dot products
            var_reduction = 0.0
            for j in range(n_train):
                v_j = 0.0
                for jj in range(n_train):
                    v_j += L_inv[j, jj] * k_row[jj]
                var_reduction += v_j * v_j
            var[i] = kernel_var - var_reduction + noise_var
            if var[i] < 0:
                var[i] = 0.0
        return mean, var

    _HAS_NUMBA = True

except ImportError:
    _HAS_NUMBA = False
    logger.info("numba not available — falling back to numpy for GPR prediction")


def _se_kernel_predict_numpy(X_test, X_train, ls, kernel_var, alpha, mean_c):
    """Numpy fallback (no numba)."""
    diffs = (X_test[:, None, :] - X_train[None, :, :]) / ls
    K_star = kernel_var * np.exp(-0.5 * np.sum(diffs ** 2, axis=-1))
    return K_star @ alpha + mean_c


class FastGPRPredictor:
    """Fast GPR predictor extracted from a trained GPflow model.

    Precomputes alpha = (K + σ²I)^{-1} (y - c) at construction time.
    Predictions are pure numpy/numba — no TensorFlow calls.
    """

    def __init__(self, X_train, kernel_var, kernel_ls, noise_var, mean_c,
                 alpha, L_inv, x_min, x_range, y_mean, y_std):
        self.X_train = np.ascontiguousarray(X_train, dtype=np.float64)
        self.kernel_var = float(kernel_var)
        self.kernel_ls = np.ascontiguousarray(kernel_ls, dtype=np.float64)
        self.noise_var = float(noise_var)
        self.mean_c = float(mean_c)
        self.alpha = np.ascontiguousarray(alpha, dtype=np.float64)
        self.L_inv = np.ascontiguousarray(L_inv, dtype=np.float64)
        self.x_min = np.asarray(x_min, dtype=np.float64)
        self.x_range = np.asarray(x_range, dtype=np.float64)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)

        if _HAS_NUMBA:
            # Warm up the JIT on a tiny input
            _se_kernel_predict(
                np.zeros((1, X_train.shape[1])), self.X_train,
                self.kernel_ls, self.kernel_var, self.alpha, self.mean_c,
            )

    @classmethod
    def from_emulator(cls, emulator) -> 'FastGPRPredictor':
        """Extract a fast predictor from a trained GPR emulator."""
        model = emulator.model
        kernel_var = float(model.kernel.variance.numpy())
        kernel_ls = model.kernel.lengthscales.numpy().astype(np.float64)
        noise_var = float(model.likelihood.variance.numpy())
        mean_c = float(model.mean_function.c.numpy().flat[0])
        X_train = model.data[0].numpy().astype(np.float64)
        Y_train = model.data[1].numpy().flatten().astype(np.float64)

        # Precompute alpha and L_inv
        n = len(X_train)
        diffs = (X_train[:, None, :] - X_train[None, :, :]) / kernel_ls
        K = kernel_var * np.exp(-0.5 * np.sum(diffs ** 2, axis=-1))
        K += noise_var * np.eye(n)
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train - mean_c))
        L_inv = np.linalg.solve(L, np.eye(n))

        return cls(
            X_train=X_train,
            kernel_var=kernel_var,
            kernel_ls=kernel_ls,
            noise_var=noise_var,
            mean_c=mean_c,
            alpha=alpha,
            L_inv=L_inv,
            x_min=emulator._x_min,
            x_range=emulator._x_range,
            y_mean=emulator._y_mean,
            y_std=emulator._y_std,
        )

    def predict_mean(self, X_raw):
        """Predict mean in original (un-normalized) output space.

        Args:
            X_raw: (n_test, n_dims) array in original parameter space.

        Returns:
            (n_test,) array of predicted means.
        """
        X_norm = (np.asarray(X_raw, dtype=np.float64) - self.x_min) / self.x_range
        if _HAS_NUMBA:
            mean_z = _se_kernel_predict(
                X_norm, self.X_train, self.kernel_ls,
                self.kernel_var, self.alpha, self.mean_c,
            )
        else:
            mean_z = _se_kernel_predict_numpy(
                X_norm, self.X_train, self.kernel_ls,
                self.kernel_var, self.alpha, self.mean_c,
            )
        return mean_z * self.y_std + self.y_mean

    def predict(self, X_raw):
        """Predict mean and variance in original output space.

        Args:
            X_raw: (n_test, n_dims) array in original parameter space.

        Returns:
            (mean, var) tuple of (n_test,) arrays.
        """
        X_norm = (np.asarray(X_raw, dtype=np.float64) - self.x_min) / self.x_range
        if _HAS_NUMBA:
            mean_z, var_z = _se_kernel_predict_with_var(
                X_norm, self.X_train, self.kernel_ls,
                self.kernel_var, self.alpha, self.mean_c,
                self.noise_var, self.L_inv,
            )
        else:
            # Numpy fallback, same quantities as the numba kernel:
            # var = kernel_var + noise_var - ||L_inv @ k_star||^2
            diffs = (X_norm[:, None, :] - self.X_train[None, :, :]) / self.kernel_ls
            K_star = self.kernel_var * np.exp(-0.5 * np.sum(diffs ** 2, axis=-1))
            mean_z = K_star @ self.alpha + self.mean_c
            v = self.L_inv @ K_star.T
            var_z = np.maximum(self.kernel_var + self.noise_var - np.sum(v ** 2, axis=0), 0.0)
        mean = mean_z * self.y_std + self.y_mean
        var = var_z * self.y_std ** 2
        return mean, var
