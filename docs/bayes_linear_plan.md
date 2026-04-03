# Plan: Add Bayes Linear Emulator to history_matching

## Context

The history_matching package currently has GPR (GPflow), LinearModel (statsmodels OLS), and GLM emulators. The user wants a **Bayes Linear** emulator inspired by the R package `hmer`. This fills a gap between the simple LinearModel (no spatial correlation) and the full GPR (heavy TensorFlow dependency, slow). Bayes Linear uses a regression trend + correlated residual structure, fitted with pure numpy/scipy -- no TF needed.

## Mathematical formulation

**Model:** `f(x) = g(x)^T beta + u(x)`

- `g(x)`: regression basis functions (1, x_1, ..., x_d) -- intercept + linear terms
- `beta`: regression coefficients (estimated via OLS on training data)
- `u(x)`: zero-mean residual process with variance sigma^2 and correlation function c(x, x')

**Correlation function (squared exponential / ARD):**
```
c(x, x') = exp( -sum_i (x_i - x'_i)^2 / theta_i^2 )
```

**Hyperparameter estimation (done sequentially, not joint):**
1. Fit OLS regression to get `beta_hat` and residuals `r = y - G @ beta_hat`
2. Estimate `sigma^2` from residual variance
3. Optimize `theta` (per-dimension correlation lengths) by maximizing the concentrated log-likelihood of the residual process
4. Nugget: small fixed value (default 1e-6) for numerical stability, or optionally estimated

**Prediction (Bayes Linear adjustment):**
```
E_D[f(x*)] = g(x*)^T beta_hat + c(x*, X) @ [C + nug*I]^{-1} @ r
Var_D[f(x*)] = sigma^2 * (1 - c(x*, X) @ [C + nug*I]^{-1} @ c(X, x*))
```

Where `C = sigma^2 * R` is the training covariance matrix and `R_ij = c(x_i, x_j)`.

This is essentially universal kriging with an OLS trend, but framed in the Bayes Linear paradigm (only expectations and variances are specified, no full distributional assumptions).

## Files to create/modify

### 1. Create: `history_matching/emulators/bayes_linear.py`

New `BayesLinear(BaseEmulator)` class with:

- **`__init__(self, x, y, test_fraction=0.25, nugget=1e-6)`** -- calls `super().__init__()`, stores nugget
- **`_normalize_x(self, x)`** -- min-max normalize inputs to [0,1] (same pattern as GPR)
- **`_build_design_matrix(self, X)`** -- returns `[1, x_1, ..., x_d]` matrix
- **`_sq_exp_correlation(self, X1, X2, theta)`** -- squared exponential correlation between two point sets
- **`_neg_log_likelihood(self, log_theta, R_func, residuals)`** -- concentrated negative log-likelihood for theta optimization
- **`train(self)`**:
  1. Normalize inputs to [0,1]
  2. Standardize outputs (zero mean, unit variance) -- same as GPR
  3. OLS fit: `beta_hat = (G^T G)^{-1} G^T y`, compute residuals
  4. Estimate `sigma^2` from residuals
  5. Optimize `theta` via `scipy.optimize.minimize` on concentrated log-likelihood
  6. Pre-compute `L = cholesky(C + nug*I)` and `alpha = L^{-T} L^{-1} r` for fast prediction
  7. Set `self.training_complete = True`
- **`predict(self, x)`**:
  1. Normalize inputs, build design matrix
  2. Compute adjusted mean and variance using stored Cholesky factor
  3. Un-standardize outputs
  4. Return `EmulationResults(mean, std, additional_data)` with `ci_obs_*` and `ci_pred_*`
- **`print_emulator_description(self)`** -- print beta, sigma^2, theta, nugget

### 2. Modify: `history_matching/emulators/__init__.py`

Add: `from .bayes_linear import BayesLinear  # noqa: F401 isort: skip`

### 3. Modify: `history_matching/emulators/factory.py`

- Import `BayesLinear`
- Add to registry: `'bayes_linear': BayesLinear`
- Add convenience function `create_bayes_linear_emulator()`

## Key design decisions

- **Pure numpy/scipy** -- no TensorFlow or GPflow dependency. This is a major advantage over GPR.
- **Input normalization + output standardization** -- follows the GPR pattern exactly so theta values are comparable across parameters.
- **Linear basis only** (no quadratic terms by default) -- keeps it simple; quadratic can be added later if needed.
- **Nugget as constructor parameter** -- defaults to 1e-6 for stability; can be increased for stochastic simulators.
- **Cholesky pre-computation** -- `L` and `alpha` are stored after training for O(n*m) prediction instead of O(n^3).

## Verification

1. Register in factory and confirm `EmulatorFactory.available_emulators()` includes `'bayes_linear'`
2. Run existing test patterns: create with data, train, predict, check EmulationResults has mean/std/additional_data
3. Compare predictions to GPR on a simple test case -- should be similar (same kernel family, same trend structure)
4. Verify diagnostics work: `emulator.test()`, `emulator.info()`, `emulator.plot_diagnostics()`
