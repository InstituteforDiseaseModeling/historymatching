# Emulators

Emulators are fast statistical surrogates of the user's simulator. History
matching trains one emulator per selected output feature each wave, then uses
their predictions (mean + variance) to decide which regions of parameter space
are implausible.

All emulators subclass [`BaseEmulator`](base.py) and are created by name through
[`EmulatorFactory`](factory.py) via the `emulator_type=` argument to
`HistoryMatching`.

| Module | Class | `emulator_type` | Best for | Cost |
|--------|-------|-----------------|----------|------|
| `bayes_linear.py` | `BayesLinear` | `'bayes_linear'` (default) | Near-GPR quality with calibrated uncertainty, no TensorFlow | Low |
| `linear.py` | `LinearModel` | `'linear'` | Near-linear responses, fast prototyping | Lowest |
| `glm.py` | `GLM` | `'glm'` | Count / non-Gaussian outputs (Poisson link) | Low |
| `gpr.py` | `GPR` | `'gpr'` | Nonlinear responses, small–medium data, best uncertainty | Higher |

Supporting modules:

- `base.py` — `BaseEmulator` abstract class (the extension contract).
- `results.py` — `EmulationResults`, the predicted mean/variance container returned by `predict`.
- `factory.py` — `EmulatorFactory`, maps `emulator_type` strings to classes.
- `fast_predict.py` — `FastGPRPredictor`, a numba/numpy fast path for GPR batch prediction.

To write your own emulator, see the
[Extending history matching](../../docs/extending.md) guide.
