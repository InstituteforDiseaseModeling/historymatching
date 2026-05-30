"""
Bayes Linear emulator inspired by the hmer R package.

Uses a regression trend (OLS) plus a correlated residual process with
squared-exponential kernel.  Hyperparameters (correlation lengths) are
estimated by maximizing a concentrated log-likelihood on the residuals.

Pure numpy/scipy -- no TensorFlow or GPflow dependency.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

from .base import BaseEmulator
from .results import EmulationResults

logger = logging.getLogger(__name__)


class BayesLinear(BaseEmulator):
    """Bayes Linear emulator with OLS trend and squared-exponential residual correlation."""

    def __init__(self, x: Optional[pd.DataFrame] = None, y: Optional[pd.DataFrame] = None,
                 test_fraction=0.25, nugget=1e-6, ftol=1e-6, gtol=1e-4):
        super().__init__(x, y, test_fraction)
        self.nugget = nugget
        self.ftol = ftol
        self.gtol = gtol

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_x(self, x):
        """Normalize inputs to [0, 1] using training-set min/range."""
        return (np.asarray(x, dtype=np.float64) - self._x_min) / self._x_range

    @staticmethod
    def _design_matrix(X):
        """Build regression design matrix [1, x_1, ..., x_d]."""
        n = X.shape[0]
        return np.column_stack([np.ones(n), X])

    @staticmethod
    def _sq_exp_corr(X1, X2, theta):
        """Squared-exponential correlation matrix between two point sets.

        Parameters
        ----------
        X1 : (n, d)  X2 : (m, d)  theta : (d,) correlation lengths
        Returns
        -------
        R : (n, m) correlation matrix
        """
        # Scaled squared distances: sum_i (x1_i - x2_i)^2 / theta_i^2
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]  # (n, m, d)
        sq_dist = np.sum((diff / theta) ** 2, axis=2)        # (n, m)
        return np.exp(-sq_dist)

    def _concentrated_nll(self, log_theta, X_norm, residuals):
        """Concentrated negative log-likelihood for theta optimisation.

        With sigma^2 profiled out analytically, the NLL reduces to:
            NLL(theta) = 0.5 * (n * log(sigma^2_hat) + log|R + nug*I|)
        where sigma^2_hat = r^T (R + nug*I)^{-1} r / n.
        """
        theta = np.exp(log_theta)
        n = len(residuals)
        if hasattr(self, '_nll_eval_count'):
            self._nll_eval_count += 1
        R = self._sq_exp_corr(X_norm, X_norm, theta)
        R_nug = R + self.nugget * np.eye(n)

        try:
            L, lower = cho_factor(R_nug)
        except np.linalg.LinAlgError:
            return 1e10  # singular — reject this theta

        alpha = cho_solve((L, lower), residuals)
        sigma2_hat = float(residuals @ alpha) / n
        if sigma2_hat <= 0:
            return 1e10

        # log|R_nug| = 2 * sum(log(diag(L)))
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        nll = 0.5 * (n * np.log(sigma2_hat) + log_det)
        return nll

    # ------------------------------------------------------------------
    # BaseEmulator interface
    # ------------------------------------------------------------------

    def train(self):
        """Train the Bayes Linear emulator.

        Steps:
        1. Normalize inputs to [0, 1], standardize outputs.
        2. OLS regression for the trend.
        3. Optimize correlation lengths theta via concentrated log-likelihood.
        4. Pre-compute Cholesky factor and weight vector for fast prediction.
        """
        import time as _time
        t0 = _time.time()

        x_raw = np.asarray(self.X_train, dtype=np.float64)
        y_raw = np.asarray(self.y_train, dtype=np.float64).ravel()
        n_train, n_dims = x_raw.shape
        logger.info(f"    Training Bayes Linear: {n_train} points, {n_dims} dims")

        # --- Input normalization (min-max to unit box) ---
        self._x_min = x_raw.min(axis=0)
        self._x_range = x_raw.max(axis=0) - self._x_min
        self._x_range = np.maximum(self._x_range, 1e-12)

        # --- Output standardization (zero mean, unit variance) ---
        self._y_mean = float(np.mean(y_raw))
        self._y_std = float(np.std(y_raw))
        if self._y_std < 1e-12:
            self._y_std = 1.0

        X_norm = self._normalize_x(x_raw)
        y_std = (y_raw - self._y_mean) / self._y_std
        n, d = X_norm.shape

        # --- OLS trend ---
        logger.info(f"    OLS trend fit [{_time.time()-t0:.1f}s]")
        G = self._design_matrix(X_norm)  # (n, d+1)
        GtG = G.T @ G
        self._beta_hat = np.linalg.solve(GtG, G.T @ y_std)
        residuals = y_std - G @ self._beta_hat

        # --- Optimize theta (correlation lengths) ---
        logger.info(f"    Optimizing theta ({d} correlation lengths, {n}x{n} Cholesky per eval)...")
        self._nll_eval_count = 0
        self._nll_best = np.inf
        self._opt_t0 = _time.time()

        def _opt_callback(xk):
            nll = self._concentrated_nll(xk, X_norm, residuals)
            self._nll_best = min(self._nll_best, nll)
            elapsed = _time.time() - self._opt_t0
            logger.info(f"    L-BFGS-B iter {self._nll_eval_count}: NLL={self._nll_best:.4f} [{elapsed:.0f}s]")

        log_theta0 = np.zeros(d)  # initial guess: theta=1 in normalised space
        result = minimize(
            self._concentrated_nll,
            log_theta0,
            args=(X_norm, residuals),
            method='L-BFGS-B',
            bounds=[(-4, 4)] * d,  # theta in [~0.018, ~55] — wide range
            options={'maxiter': 200, 'ftol': self.ftol, 'gtol': self.gtol},
            callback=_opt_callback,
        )
        self._theta = np.exp(result.x)
        self._opt_result = result
        logger.info(f"    Theta optimization: {result.nit} L-BFGS iters, {self._nll_eval_count} NLL evals, "
                    f"final NLL={result.fun:.4f} [{_time.time()-t0:.1f}s]")

        if not result.success:
            logger.warning(
                "Bayes Linear theta optimization did not converge: %s. "
                "The fitted model may still be usable.", result.message
            )

        # --- Build correlation matrix and pre-compute Cholesky ---
        logger.info(f"    Building {n}x{n} correlation matrix and Cholesky...")
        R = self._sq_exp_corr(X_norm, X_norm, self._theta)
        R_nug = R + self.nugget * np.eye(n)
        self._L, self._lower = cho_factor(R_nug)
        self._alpha = cho_solve((self._L, self._lower), residuals)

        # Estimate residual variance (sigma^2)
        self._sigma2 = float(residuals @ self._alpha) / n

        # Store normalised training inputs and residuals for prediction
        self._X_train_norm = X_norm
        self._residuals = residuals

        self.training_complete = True
        logger.info(f"    Training complete — sigma²={self._sigma2:.4f}, "
                     f"theta range=[{self._theta.min():.3f}, {self._theta.max():.3f}] [{_time.time()-t0:.1f}s total]")

    def predict(self, x: pd.DataFrame) -> EmulationResults:
        """Predict using the trained Bayes Linear emulator."""

        X_new = self._normalize_x(x)
        n_new = X_new.shape[0]
        G_new = self._design_matrix(X_new)

        # --- Adjusted expectation ---
        # E_D[f(x*)] = g(x*)^T beta + c(x*, X) @ alpha
        c_new = self._sq_exp_corr(X_new, self._X_train_norm, self._theta)  # (n_new, n_train)
        mean_z = G_new @ self._beta_hat + c_new @ self._alpha

        # --- Adjusted variance ---
        # Var_D[f(x*)] = sigma^2 * (1 - c(x*, X) @ C^{-1} @ c(X, x*))
        v = cho_solve((self._L, self._lower), c_new.T)  # (n_train, n_new)
        var_reduction = np.sum(c_new.T * v, axis=0)      # diagonal of c @ C^{-1} @ c^T
        pred_var_z = np.maximum(self._sigma2 * (1.0 - var_reduction), 0.0)

        # Observation variance adds sigma^2 * nugget (noise term)
        obs_var_z = pred_var_z + self._sigma2 * self.nugget

        # --- Un-standardize ---
        ys, ym = self._y_std, self._y_mean
        mean = mean_z * ys + ym
        pred_var = pred_var_z * ys ** 2
        obs_var = obs_var_z * ys ** 2

        pred_std = np.sqrt(pred_var)
        obs_std = np.sqrt(obs_var)

        z = 1.96  # 95% CI
        additional = pd.DataFrame({
            'ci_obs_low': mean - z * obs_std,
            'ci_obs_high': mean + z * obs_std,
            'ci_pred_low': mean - z * pred_std,
            'ci_pred_high': mean + z * pred_std,
        }, index=x.index)

        return EmulationResults(
            mean=mean,
            std=obs_std,
            additional_data=additional,
        )

    def get_hyperparameters(self) -> dict:
        """Return Bayes Linear hyperparameters as a JSON-serializable dict."""
        if not self.training_complete:
            return {}

        param_names = list(self.X_df.columns) if self.X_df is not None else [f"x{i}" for i in range(len(self._theta))]

        # Un-standardize beta for display
        beta_raw = self._beta_hat.copy()
        beta_raw[0] = beta_raw[0] * self._y_std + self._y_mean
        beta_raw[1:] = beta_raw[1:] * self._y_std

        return {
            'type': 'bayes_linear',
            'nugget': float(self.nugget),
            'sigma_sq': float(self._sigma2 * self._y_std ** 2),
            'sigma_sq_standardized': float(self._sigma2),
            'theta': {name: float(t) for name, t in zip(param_names, self._theta)},
            'beta': {name: float(b) for name, b in zip(['intercept'] + param_names, beta_raw)},
            'optimizer_converged': bool(self._opt_result.success),
            'nll': float(self._opt_result.fun),
            'n_train': int(len(self.X_train)),
            'n_dims': int(len(self._theta)),
        }

    def print_emulator_description(self):
        """Display Bayes Linear emulator specifications."""
        hp = self.get_hyperparameters()
        if not hp:
            print("      Emulator has not been trained yet.")
            return

        print("      Bayes Linear Emulator")
        print(f"      Nugget:        {hp['nugget']}")
        print(f"      sigma^2:       {hp['sigma_sq']:.6f}")
        print(f"      NLL:           {hp['nll']:.4f}")
        print(f"      Optimizer:     {'converged' if hp['optimizer_converged'] else 'did NOT converge'}")
        print(f"      theta (correlation lengths):")
        for name, val in hp['theta'].items():
            print(f"        {name}: {val:.4f}")
        print(f"      beta (regression coefficients):")
        for name, val in hp['beta'].items():
            print(f"        {name}: {val:.6f}")
