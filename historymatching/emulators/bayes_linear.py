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
import sciris as sc
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .base import BaseEmulator
from .results import EmulationResults, PredictionBounds

logger = logging.getLogger(__name__)

# Bounds on log(theta) for the L-BFGS-B correlation-length search, in the
# normalised [0, 1] input space.  (-4, 4) corresponds to theta in
# [exp(-4), exp(4)] ~= [0.018, 55], a deliberately wide range that spans
# near-independent inputs (short lengths) to nearly-flat ones (long lengths).
LOG_THETA_BOUNDS = (-4.0, 4.0)

# Number of leading training points whose solve bounds the variance in
# `BayesLinear.predict_bounds`. At 3,750 training points the
# prefix costs about 2% of the full variance.
SCREENING_PREFIX = 500

# Starts for the `nugget='mle'` search; see `BayesLinear`.
DEFAULT_MLE_STARTS = ('warm', 'midpoint', 1e-2)

# Nugget at which the 'warm' start first fits the correlation lengths.
DEFAULT_WARM_NUGGET = 1e-2


class BayesLinear(BaseEmulator):
    """Bayes Linear emulator with OLS trend and squared-exponential residual correlation.

    This is the default emulator. It pairs an ordinary-least-squares regression
    trend with a correlated residual process (squared-exponential kernel), giving
    near-GPR-quality predictions and calibrated uncertainty at a fraction of the
    cost, using only numpy/scipy (no TensorFlow/GPflow).


    Args:
        x: Input data. Pandas DataFrame with columns representing parameter
            values.
        y: Output data. Pandas DataFrame with columns representing
            observations and rows representing samples. Each row must match
            the corresponding row in `x`.
        test_fraction: Fraction of `x` and `y` held out for testing
            (scalar between 0 and 1).
        nugget: Variance of the noise in a single simulator run, as a fraction
            of the residual process variance (`sigma_sq` in
            `get_hyperparameters()`). It is added to the diagonal of the
            training correlation matrix, and `sigma_sq * nugget` is added to
            every predictive variance, so implausibility already counts
            run-to-run noise through the emulator: give the targets their data
            uncertainty only. A number fixes the nugget; the default `1e-6` is
            a small jitter suited to deterministic simulators. `'mle'` learns
            one value, shared across the input space, by maximum likelihood;
            use it for stochastic simulators.
        nugget_bounds: Lower and upper bounds on the learned nugget when
            `nugget='mle'`; ignored otherwise.
        ftol: Function-value tolerance for the L-BFGS-B correlation-length
            optimization (convergence criterion).
        gtol: Gradient tolerance for the same optimization.
        mle_starts: Starting points for the `nugget='mle'` search; ignored
            otherwise. The likelihood often has more than one optimum, so
            each start runs its own L-BFGS-B search and the lowest NLL wins.
            Each entry is one of:

            - `'warm'`: fit the correlation lengths with the nugget fixed at
              `warm_nugget` (the same fit as `nugget=warm_nugget`), then free
              the nugget from that solution. The fixed-nugget solution is also
              a candidate, so the learned fit's NLL is never above the NLL of
              `nugget=warm_nugget` on the same training data.
            - `'midpoint'`: correlation lengths 1 and the nugget at the
              geometric middle of `nugget_bounds` (the only start before
              multistart was added).
            - A number: correlation lengths 1 and that nugget, clipped to
              `nugget_bounds`.

            The default, `('warm', 'midpoint', 1e-2)`, costs about three
            single fits. Every start's result is in
            `get_hyperparameters()['mle_starts']`.
        warm_nugget: Fixed nugget for the `'warm'` start, clipped to
            `nugget_bounds`. The default, `1e-2`, is run-to-run noise of 1% of
            the residual variance: inside the default bounds, large enough to
            keep the correlation matrix well conditioned, and small enough not
            to hide signal.

    Example:
        ```python
        import numpy as np, pandas as pd
        from historymatching.emulators.bayes_linear import BayesLinear
        x = pd.DataFrame({'beta': np.random.rand(50), 'gamma': np.random.rand(50)})
        y = pd.DataFrame({'peak': 3 * x['beta'] + x['gamma']})
        em = BayesLinear(x, y)
        em.train()
        pred = em.predict(x)
        pred.get_mean()  # predictive means; pred.get_variance() for uncertainty
        ```
    """

    def __init__(self, x: Optional[pd.DataFrame] = None, y: Optional[pd.DataFrame] = None,
                 test_fraction=0.25, nugget=1e-6, nugget_bounds=(1e-8, 1.0),
                 ftol=1e-6, gtol=1e-4, mle_starts=DEFAULT_MLE_STARTS,
                 warm_nugget=DEFAULT_WARM_NUGGET):
        self._learn_nugget = nugget == 'mle'
        self._validate_nugget(nugget)
        mle_starts = self._validate_mle_starts(mle_starts)
        if not warm_nugget > 0:
            raise ValueError("warm_nugget must be positive")

        super().__init__(x, y, test_fraction)
        self.nugget = nugget
        self.nugget_bounds = nugget_bounds
        self.ftol = ftol
        self.gtol = gtol
        self.mle_starts = mle_starts
        self.warm_nugget = warm_nugget
        self.mle_start_results = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_nugget(nugget):
        """Validate the requested nugget mode."""
        if isinstance(nugget, str):
            if nugget != 'mle':
                raise ValueError("nugget must be numeric or 'mle'")
            return

        if nugget < 0:
            raise ValueError("nugget must be non-negative")

    @staticmethod
    def _validate_mle_starts(mle_starts):
        """Validate the `nugget='mle'` start list and return it as a tuple."""
        if isinstance(mle_starts, (str, int, float)):
            mle_starts = (mle_starts,)
        mle_starts = tuple(mle_starts)
        if not mle_starts:
            raise ValueError("mle_starts must name at least one start")
        for start in mle_starts:
            if isinstance(start, str):
                if start not in ('warm', 'midpoint'):
                    raise ValueError(f"unknown mle_starts entry {start!r}: use 'warm', "
                                     "'midpoint' or a positive number")
            elif isinstance(start, bool) or not isinstance(start, (int, float, np.number)) \
                    or not start > 0:
                raise ValueError(f"mle_starts entry {start!r} must be 'warm', 'midpoint' "
                                 "or a positive number")
        return mle_starts

    def _normalize_x(self, x):
        """Normalize inputs to [0, 1] using training-set min/range."""
        return (np.asarray(x, dtype=np.float64) - self._x_min) / self._x_range

    @staticmethod
    def _design_matrix(X):
        """Build regression design matrix [1, x_1, ..., x_d]."""
        n = X.shape[0]
        return np.column_stack([np.ones(n), X])

    @staticmethod
    def _scaled(X, theta):
        """Inputs centred on the unit box and divided by the correlation lengths."""
        return (X - 0.5) / theta

    @classmethod
    def _sq_exp_corr(cls, X1, X2, theta):
        """Squared-exponential correlation matrix between two point sets.

        The scaled squared distances come from `|a|^2 + |b|^2 - 2 a.b`, one
        matrix product, instead of an `(n, m, d)` array of differences: memory
        is `O(n * m)` and the work runs as BLAS GEMM. Rounding moves each
        squared distance by at most about `(4d + 16) * eps/2 * (|a|^2 + |b|^2)`;
        centring the inputs on the unit box keeps `|a|^2` small. Tiny negative
        distances from rounding are clipped to zero. When `X1 is X2` the
        result is exactly symmetric with a unit diagonal.

        Args:
            X1: Normalized points, shape `(n, d)`.
            X2: Normalized points, shape `(m, d)`.
            theta: Correlation lengths, shape `(d,)`.

        Returns:
            Correlation matrix, shape `(n, m)`.
        """
        A = cls._scaled(X1, theta)
        B = A if X2 is X1 else cls._scaled(X2, theta)
        a2 = np.einsum('ij,ij->i', A, A)
        b2 = a2 if B is A else np.einsum('ij,ij->i', B, B)
        R = A @ B.T
        R *= -2.0
        R += a2[:, np.newaxis]
        R += b2[np.newaxis, :]
        if B is A:
            R += R.T
            R *= 0.5
            np.fill_diagonal(R, 0.0)
        np.maximum(R, 0.0, out=R)
        np.negative(R, out=R)
        np.exp(R, out=R)
        return R

    def _get_nugget_log_bounds(self):
        """Return nugget bounds in log-space for optimizer use."""
        lower, upper = self.nugget_bounds
        if lower <= 0:
            raise ValueError("nugget_bounds lower value must be positive")
        if upper <= lower:
            raise ValueError("nugget_bounds upper value must be greater than lower")

        return np.log(lower), np.log(upper)

    def _correlation_with_noise(self, X_norm, theta, nugget):
        """Return a correlation matrix with scalar observation noise."""
        R = self._sq_exp_corr(X_norm, X_norm, theta)
        return R + nugget * np.eye(len(X_norm))

    def _concentrated_nll(self, log_hyperparameters, X_norm, residuals):
        """Concentrated negative log-likelihood for theta optimisation.

        With sigma^2 profiled out analytically, the NLL reduces to:
            NLL(theta) = 0.5 * (n * log(sigma^2_hat) + log|R + nug*I|)
        where sigma^2_hat = r^T (R + nug*I)^{-1} r / n.

        `log_hyperparameters` is `log(theta)`, followed by `log(nugget)` when
        the nugget is learned.
        """
        if self._learn_nugget:
            return self._nll(log_hyperparameters[:-1], float(np.exp(log_hyperparameters[-1])),
                             X_norm, residuals)
        return self._nll(log_hyperparameters, self.nugget, X_norm, residuals)

    def _nll(self, log_theta, nugget, X_norm, residuals):
        """Concentrated NLL at `log(theta)` and a given nugget (see `_concentrated_nll`)."""
        theta = np.exp(log_theta)
        n = len(residuals)
        if hasattr(self, '_nll_eval_count'):
            self._nll_eval_count += 1
        R_nug = self._correlation_with_noise(X_norm, theta, nugget)

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

    def _lbfgsb(self, objective, x0, bounds, args, label, nll_scale=1.0):
        """One L-BFGS-B search, logging progress per iteration.

        `nll_scale` converts the objective back to the NLL for logging.
        """
        t0 = sc.timer()

        def _callback(intermediate_result):
            logger.debug(f"    L-BFGS-B ({label}) iter: NLL={intermediate_result.fun * nll_scale:.4f} "
                         f"[{t0.total:.0f}s]")

        result = minimize(
            objective,
            x0,
            args=args,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': self.ftol, 'gtol': self.gtol},
            callback=_callback,
        )
        result.seconds = t0.total
        return result

    def _fit_theta(self, X_norm, residuals, nugget):
        """Fit the correlation lengths with the nugget fixed: one start at theta = 1.

        This is the whole optimization for a numeric `nugget`, and the first
        stage of the `'warm'` start for `nugget='mle'`.

        Returns:
            The `scipy.optimize.OptimizeResult`; `fun` is the NLL.
        """
        d = X_norm.shape[1]
        return self._lbfgsb(lambda log_theta, X, r: self._nll(log_theta, nugget, X, r),
                            np.zeros(d), [LOG_THETA_BOUNDS] * d, (X_norm, residuals),
                            label=f'nugget fixed at {nugget:.3g}')

    def _fit_free(self, X_norm, residuals, x0, label):
        """Fit log(theta) and log(nugget) together from `x0`.

        The search minimizes the NLL per training point, NLL / n. With every
        variable bounded, L-BFGS-B's first trial step is the whole negative
        gradient, projected onto the bounds. The raw NLL's gradient grows with
        n, into the tens or hundreds, so that step used to put many log
        correlation lengths on their lower bound at once. There the training
        points are uncorrelated, the NLL no longer depends on theta or the
        nugget, and the search stopped after one iteration with a zero
        gradient, reporting success. Dividing by n keeps the first step
        short. `ftol` is a relative tolerance, so it means the same either
        way; `gtol` applies to the gradient of NLL / n.

        Returns:
            The `scipy.optimize.OptimizeResult`, with `fun` set to the NLL at `x`.
        """
        d = X_norm.shape[1]
        n = len(residuals)
        bounds = [LOG_THETA_BOUNDS] * d + [self._get_nugget_log_bounds()]

        def objective(z, X, r):
            return self._nll(z[:-1], float(np.exp(z[-1])), X, r) / n

        result = self._lbfgsb(objective, x0, bounds, (X_norm, residuals), label=label, nll_scale=n)
        result.fun = float(self._nll(result.x[:-1], float(np.exp(result.x[-1])), X_norm, residuals))
        return result

    @staticmethod
    def _start_record(start, stage, start_nugget, result, nugget):
        """JSON-serializable summary of one optimizer run."""
        return {
            'start': start,
            'stage': stage,
            'start_nugget': float(start_nugget),
            'nll': float(result.fun),
            'nugget': float(nugget),
            'nit': int(result.nit),
            'nfev': int(result.nfev),
            'success': bool(result.success),
            'message': str(result.message),
            'seconds': float(result.seconds),
        }

    def _fit_mle(self, X_norm, residuals):
        """Learn theta and the nugget from every start in `mle_starts`; keep the lowest NLL.

        Every candidate is a feasible point, so the chosen NLL is at most the
        NLL of every recorded run. In particular, the `'warm'` start's
        fixed-nugget stage is a candidate with the nugget exactly
        `warm_nugget`, so the chosen NLL is never above that of
        `BayesLinear(nugget=warm_nugget)` on the same training data.

        Returns:
            Tuple `(log_theta, nugget, result, records)`: the chosen
            hyperparameters, its `OptimizeResult` (with `fun` the NLL) and one
            record per optimizer run, the chosen one marked `'selected'`.
        """
        d = X_norm.shape[1]
        log_lo, log_hi = self._get_nugget_log_bounds()
        lo, hi = self.nugget_bounds
        records, candidates = [], []
        for start in self.mle_starts:
            if start == 'warm':
                w = float(np.clip(self.warm_nugget, lo, hi))
                fixed = self._fit_theta(X_norm, residuals, w)
                records.append(self._start_record('warm', 'nugget fixed', w, fixed, w))
                candidates.append((fixed.fun, fixed.x, w, fixed, records[-1]))
                x0 = np.append(fixed.x, np.clip(np.log(w), log_lo, log_hi))
                start_label = 'warm'
            elif start == 'midpoint':
                x0 = np.append(np.zeros(d), 0.5 * (log_lo + log_hi))
                start_label = 'midpoint'
            else:
                x0 = np.append(np.zeros(d), np.clip(np.log(float(start)), log_lo, log_hi))
                start_label = f'nugget={float(start):g}'
            start_nugget = float(np.exp(x0[-1]))
            free = self._fit_free(X_norm, residuals, x0, label=f'{start_label} start')
            nugget = float(np.exp(free.x[-1]))
            records.append(self._start_record(start_label, 'free nugget', start_nugget, free, nugget))
            candidates.append((free.fun, free.x[:-1], nugget, free, records[-1]))
            logger.info(f"    nugget='mle' start {start_label}: NLL={free.fun:.4f}, nugget={nugget:.3g}, "
                        f"{free.nit} iters, {free.message} [{free.seconds:.1f}s]")

        # The first of equal NLLs wins, so the outcome does not depend on rounding ties.
        best = min(range(len(candidates)), key=lambda i: (candidates[i][0], i))
        nll, log_theta, nugget, result, record = candidates[best]
        for r in records:
            r['selected'] = r is record
        return log_theta, nugget, result, records

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
        t0 = sc.timer()

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
        logger.info(f"    OLS trend fit [{t0.total:.1f}s]")
        G = self._design_matrix(X_norm)  # (n, d+1)
        GtG = G.T @ G
        try:
            self._beta_hat = np.linalg.solve(GtG, G.T @ y_std)
        except np.linalg.LinAlgError:
            self._beta_hat, _, _, _ = np.linalg.lstsq(G, y_std, rcond=None)
        residuals = y_std - G @ self._beta_hat

        # --- Optimize theta (correlation lengths) and optionally nugget ---
        self._nll_eval_count = 0
        if self._learn_nugget:
            logger.info(f"    Optimizing theta ({d} correlation lengths) and nugget "
                        f"from {len(self.mle_starts)} starts ({n}x{n} Cholesky per eval)...")
            log_theta, self.nugget, result, self.mle_start_results = self._fit_mle(X_norm, residuals)
        else:
            logger.info(f"    Optimizing theta ({d} correlation lengths, {n}x{n} Cholesky per eval)...")
            result = self._fit_theta(X_norm, residuals, self.nugget)
            log_theta = result.x
            self.mle_start_results = []

        self._theta = np.exp(log_theta)
        self._opt_result = result
        logger.info(f"    Theta optimization: {result.nit} L-BFGS iters, {self._nll_eval_count} NLL evals, "
                    f"final NLL={result.fun:.4f} [{t0.total:.1f}s]")

        if not result.success:
            logger.warning(
                "Bayes Linear theta optimization did not converge: %s. "
                "The fitted model may still be usable.", result.message
            )

        # --- Build correlation matrix and pre-compute Cholesky ---
        logger.info(f"    Building {n}x{n} correlation matrix and Cholesky...")
        R_nug = self._correlation_with_noise(X_norm, self._theta, self.nugget)
        self._L, self._lower = cho_factor(R_nug)
        self._alpha = cho_solve((self._L, self._lower), residuals)

        # Estimate residual variance (sigma^2)
        self._sigma2 = float(residuals @ self._alpha) / n

        # Store normalised training inputs and residuals for prediction
        self._X_train_norm = X_norm
        self._residuals = residuals

        self.training_complete = True
        logger.info(f"    Training complete — sigma²={self._sigma2:.4f}, "
                     f"theta range=[{self._theta.min():.3f}, {self._theta.max():.3f}] [{t0.total:.1f}s total]")

    def _mean_z(self, x):
        """Standardized adjusted expectation at `x`, and the correlations it used.

        Returns:
            Tuple `(mean_z, c_new)`: the mean on the standardized output scale,
            shape `(n_new,)`, and the correlations with the training points,
            shape `(n_new, n_train)`.
        """
        X_new = self._normalize_x(x)
        G_new = self._design_matrix(X_new)
        # E_D[f(x*)] = g(x*)^T beta + c(x*, X) @ alpha
        c_new = self._sq_exp_corr(X_new, self._X_train_norm, self._theta)  # (n_new, n_train)
        mean_z = G_new @ self._beta_hat + c_new @ self._alpha
        return mean_z, c_new

    def predict(self, x: pd.DataFrame) -> EmulationResults:
        """Predict using the trained Bayes Linear emulator."""

        mean_z, c_new = self._mean_z(x)

        # --- Adjusted variance ---
        # Var_D[f(x*)] = sigma^2 * (1 - c(x*, X) @ C^{-1} @ c(X, x*)).
        # With C = U^T U (or L L^T), c C^{-1} c^T = ||w||^2 for w = U^{-T} c^T,
        # so one triangular solve gives the diagonal, as a sum of squares (>= 0).
        w = solve_triangular(self._L, c_new.T, lower=self._lower,
                             trans='N' if self._lower else 'T')  # (n_train, n_new)
        var_reduction = np.einsum('ij,ij->j', w, w)
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
            index=x.index,
        )

    def _screening_slack(self):
        """Absolute slack on the variance reduction in the screening bound.

        Bounds the rounding error in a computed variance reduction `r`: about
        `2 * kappa * n * eps` for a triangular solve, where `kappa`, the
        condition number of the Cholesky factor, is at most
        `sqrt(n * (1 + nugget) / nugget)` because the correlation matrix's
        eigenvalues lie in `[nugget, n * (1 + nugget)]`. Doubled again to cover
        both the screening and the full computation. Pessimistic by design;
        with `nugget=1e-6` and 3,750 training points it is about `2e-7`.
        """
        n, nugget = len(self._X_train_norm), float(self.nugget)
        if nugget <= 0:
            return np.inf
        kappa = np.sqrt(n * (1.0 + nugget) / nugget)
        return max(4.0 * kappa * n * np.finfo(float).eps, 1e-12)

    def predict_bounds(self, x: pd.DataFrame) -> PredictionBounds:
        """Cheap bounds on the mean and observation variance `predict` returns.

        The variance needs a triangular solve against all `n_train` training
        points, about `n_train / d` times the work of the mean. Here it is
        replaced by bounds, so `historymatching.nroy_sampling.nroy_mask` can
        decide most points without calling `predict`:

        - Mean: computed as `predict` does. BLAS rounding for one row can
          depend on the other rows in the batch, so `mean_tol` bounds how far
          it can move between calls: the error bound on the correlations (see
          `_sq_exp_corr`), each relative to itself, summed against the weights,
          plus rounding in the products, with a safety factor of 4.
        - Variance, lower bound: `sigma2 * nugget`, since the predictive
          variance is never negative.
        - Variance, upper bound: the variance given only the first `m`
          training points. Conditioning on fewer points never reduces the
          variance more. With the Cholesky factor of the training correlations,
          that variance reduction is the partial sum over the first `m` rows
          of the solve `predict` does, so it costs about `(m / n_train)^2` of
          the full variance. `m` is `SCREENING_PREFIX` (500), or a quarter of
          the training points if that is fewer. The bound is widened for the
          correlation error and for rounding in the solve
          (`_screening_slack`), and is never above `sigma2 * (1 + nugget)`,
          the variance far from all training points.

        All bounds are on the output scale.

        Args:
            x: Input data, with the training parameter columns.

        Returns:
            A `PredictionBounds` with arrays of shape `(len(x),)`.
        """
        u = np.finfo(float).eps / 2
        mean_z, c = self._mean_z(x)
        X_new = self._normalize_x(x)
        n_new, d = X_new.shape
        n_train = len(self._X_train_norm)
        A = self._scaled(X_new, self._theta)
        B = self._scaled(self._X_train_norm, self._theta)
        a2 = np.einsum('ij,ij->i', A, A)
        b2 = np.einsum('ij,ij->i', B, B)
        G_new = self._design_matrix(X_new)

        # Error bound on each correlation. Any two evaluations of a scaled
        # squared distance (in different batches) lie within
        # E_ij = k u (|a_i|^2 + |b_j|^2) of each other,
        # so the correlations differ by at most c_ij (h_i E_ij + 3u), with
        # h_i >= expm1(E)/E for every E in row i, and 3u for rounding in exp.
        k = 4 * d + 16
        e_max = k * u * (a2 + b2.max())
        h = 1.001 * np.expm1(e_max) / np.where(e_max > 0, e_max, 1.0)
        h = np.maximum(h, 1.001)
        abs_alpha = np.abs(self._alpha)
        c_alpha = c @ abs_alpha
        corr_err = h * k * u * (a2 * c_alpha + c @ (b2 * abs_alpha)) + 3 * u * c_alpha
        # Rounding in the two mean computations themselves.
        gemv_err = 2 * (n_train + d + 2) * u * (
            np.abs(G_new) @ np.abs(self._beta_hat) + c_alpha + corr_err)
        ys, ym = self._y_std, self._y_mean
        mean = mean_z * ys + ym
        mean_tol = 4.0 * ((corr_err + gemv_err) * ys + 4 * u * (np.abs(mean_z) * ys + abs(ym)))

        s2, ys2 = self._sigma2, ys ** 2
        noise = s2 * self.nugget
        hi_z = np.full(n_new, s2 + noise)
        m = min(SCREENING_PREFIX, n_train // 4)
        if m > 0 and self.nugget > 0:
            c_m = c[:, :m]
            w = solve_triangular(self._L[:m, :m], c_m.T, lower=self._lower,
                                 trans='N' if self._lower else 'T')
            r_m = np.einsum('ij,ij->j', w, w)
            slack = self._screening_slack()
            # The correlation error moves sqrt(r_m) by at most
            # ||dc||_2 / sqrt(lambda_min) <= ||dc||_2 / sqrt(nugget).
            dc = (np.sqrt(np.einsum('ij,ij->i', c_m, c_m))
                  * (h * k * u * (a2 + b2[:m].max()) + 3 * u))
            shift = 4.0 * dc / np.sqrt(self.nugget)
            root = np.maximum(np.sqrt(np.maximum(r_m - slack, 0.0)) - shift, 0.0)
            prefix_z = s2 * np.maximum(1.0 - (root ** 2 - slack), 0.0) + noise
            hi_z = np.minimum(hi_z, prefix_z)

        # `predict` stores the std, and `get_variance()` squares it back, which can move the
        # variance by about one unit in the last place. Widen both bounds by 8 so they hold
        # exactly for `get_variance()`, not just up to rounding.
        widen = 8 * u
        return PredictionBounds(mean=mean, mean_tol=mean_tol,
                                var_lo=np.full(n_new, noise * ys2 * (1 - widen)),
                                var_hi=hi_z * ys2 * (1 + widen))

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
            'nugget_learned': bool(self._learn_nugget),
            'sigma_sq': float(self._sigma2 * self._y_std ** 2),
            'sigma_sq_standardized': float(self._sigma2),
            'theta': {name: float(t) for name, t in zip(param_names, self._theta)},
            'beta': {name: float(b) for name, b in zip(['intercept'] + param_names, beta_raw)},
            'optimizer_converged': bool(self._opt_result.success),
            'nll': float(self._opt_result.fun),
            'n_train': int(len(self.X_train)),
            'n_dims': int(len(self._theta)),
            'mle_starts': [dict(r) for r in self.mle_start_results],
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
        for r in hp['mle_starts']:
            mark = '*' if r['selected'] else ' '
            print(f"      {mark} start {r['start']} ({r['stage']}): NLL={r['nll']:.4f}, "
                  f"nugget={r['nugget']:.3g}, {r['nit']} iters")
        print(f"      theta (correlation lengths):")
        for name, val in hp['theta'].items():
            print(f"        {name}: {val:.4f}")
        print(f"      beta (regression coefficients):")
        for name, val in hp['beta'].items():
            print(f"        {name}: {val:.6f}")
