# Glossary

Terms used throughout this documentation. History matching borrows vocabulary
from Bayesian calibration, design of experiments, and the emulation literature;
the definitions below assume general familiarity with calibration but not with
history matching itself.

### ARD — Automatic Relevance Determination { #ard }

A Gaussian Process kernel parameterization that fits a separate lengthscale to
each input parameter. A short lengthscale means the output varies quickly with
that parameter (it is "relevant"); a long one means the output barely depends on
it. Used by the [GPR](#gpr) emulator. (ARD reflects emulator sensitivity to the
current wave's target features; the `pairplot.png` diagnostic ranks parameters
differently — by overall constraint across all waves — so it does not use ARD.)

### Bayes linear emulator { #bayes-linear }

The default emulator (`emulator_type='bayes_linear'`). It combines a linear
regression *trend* with a correlated-residual process (a squared-exponential
kernel whose correlation lengths are fit by maximum likelihood) — effectively a
Gaussian process with a linear mean. In the Bayes linear tradition it tracks the
full predictive *uncertainty* (an adjusted expectation and variance) rather than
just a point estimate, giving calibrated error bars at a fraction of the cost of
[GPR](#gpr) and with no TensorFlow dependency. Implemented in pure NumPy/SciPy.

$$ y(x) = \underbrace{\beta_0 + \textstyle\sum_j \beta_j x_j}_{\text{linear trend}} + r(x), \qquad r(\cdot) \sim \mathcal{GP}\!\left(0,\, \sigma^2 k\right), \qquad k(x, x') = \exp\!\left(-\sum_j \frac{(x_j - x'_j)^2}{\theta_j^2}\right) $$

with trend coefficients $\beta$ (fit by OLS), residual variance $\sigma^2$, and a
per-parameter correlation length $\theta_j$ fit by maximum likelihood.

### Emulator (surrogate) { #emulator }

A fast statistical approximation of the (expensive) simulator, trained on a
modest number of simulation runs. The emulator predicts simulator output — *with
an uncertainty estimate* — at new parameter values for a tiny fraction of the
cost. History matching uses emulator predictions, not the simulator itself, to
rule out parameter space.

### Fano factor { #fano-factor }

The variance-to-mean ratio of an output across the sampled points. The automatic
feature-selection method `{'method': 'fano'}` uses it to pick the most
informative outputs to emulate: outputs whose value changes a lot relative to
their mean carry more signal for constraining parameters. See also
[mean_sq_z](#mean-sq-z) and [feature/output](#feature-output).

### Feature / output { #feature-output }

A scalar summary of a simulation run that you have a corresponding observation
for (e.g. peak incidence, attack rate, cases in week 1). The two words are used
interchangeably: "feature selection" chooses which model *outputs* to emulate in
a given wave. Each selected feature gets its own emulator.

### GLM — Generalized Linear Model { #glm }

An emulator (`emulator_type='glm'`) that fits a linear predictor through a link
function $g$:

$$ g\!\left(\mathbb{E}[y(x)]\right) = \beta_0 + \sum_j \beta_j x_j $$

With the default Gaussian family ($g$ = identity) it reduces to the
[linear emulator](#linear); with the Poisson family ($g = \log$,
`link='poisson'`) it models non-negative count outputs.

### GPR — Gaussian Process Regression { #gpr }

A flexible, nonlinear emulator (`emulator_type='gpr'`) built on
[GPflow](https://www.gpflow.org/)/TensorFlow with [ARD](#ard) kernels. Best for
nonlinear response surfaces and excellent uncertainty estimates, at a higher
computational cost than the linear emulators.

$$ y(x) \sim \mathcal{GP}(m,\, k), \qquad k(x, x') = \sigma_f^2 \exp\!\left(-\tfrac{1}{2}\sum_j \frac{(x_j - x'_j)^2}{\ell_j^2}\right) $$

with a constant mean $m$, signal variance $\sigma_f^2$, observation-noise
variance $\sigma_n^2$, and a separate lengthscale $\ell_j$ per parameter
([ARD](#ard)).

### Implausibility { #implausibility }

A score measuring how inconsistent a parameter value is with one observation,
expressed in standard deviations. For output $f$ at parameter $x$:

$$ I(x) = \frac{\left|\, \mathbb{E}[f(x)] - z \,\right|}{\sqrt{\operatorname{Var}_{\text{emulator}} + \operatorname{Var}_{\text{obs}} + \sigma_{\text{disc}}^2}} $$

where $z$ is the observed target and the denominator combines the emulator's
predictive variance, the observation's variance, and an optional **model
discrepancy** $\sigma_{\text{disc}}$ — a standard deviation accounting for
structural mismatch between the simulator and reality (`model_discrepancy`,
default `0`). With several outputs, the implausibility of a point is the
**maximum** over all outputs, so a point must be consistent with *every*
constraint to survive. A point is *ruled out* when its implausibility exceeds
the [threshold](#implausibility-threshold).

### Implausibility threshold { #implausibility-threshold }

The implausibility cutoff above which a parameter value is discarded
(`implausibility_threshold`, default `3.0`). The default of 3 follows the
**Vysochanskij–Petunin inequality** (popularized as Pukelsheim's "three-sigma
rule"): for *any* unimodal distribution with finite variance — not just a normal
one — at least ~95% of the probability mass lies within 3 standard deviations of
the mean. So an implausibility above 3 is strong, distribution-free evidence
that the point is inconsistent with the observation.

### LHS — Latin Hypercube Sampling { #lhs }

A space-filling experimental design that spreads sample points evenly across
every parameter's range. The default sampling strategy (`'lhs'`); better coverage
than uniform random sampling for the same number of points.

### Linear emulator { #linear }

The simplest emulator (`emulator_type='linear'`): an ordinary least-squares fit

$$ y(x) = \beta_0 + \sum_j \beta_j x_j + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2) $$

with predictive uncertainty taken from the OLS standard errors. Fast and
interpretable, but assumes the response is linear in the parameters — see
[GLM](#glm) for non-Gaussian outputs, or [Bayes linear](#bayes-linear) and
[GPR](#gpr) for nonlinear response surfaces.

### Maximin { #maximin }

A space-filling criterion that *maximizes the minimum distance* between sample
points, pushing them apart for even coverage. Used as an LHS `criterion`
(`{'type': 'lhs', 'criterion': 'maximin'}`) and as the final "thinning" stage of
the [`ray_resample`](#ray-resample) pipeline, where it selects a well-spread
subset from a larger candidate pool.

### mean_sq_z { #mean-sq-z }

The default automatic feature-selection method (`{'method': 'mean_sq_z'}`): mean
squared z-score, i.e. how far each output sits from its target in
standard-deviation units, averaged across samples. Outputs that are far from
their target carry the most information for ruling out parameter space. See also
[Fano factor](#fano-factor).

### NROY — Not Ruled Out Yet { #nroy }

The region of parameter space that history matching has **not** yet shown to be
[implausible](#implausibility) — the central output of the method. Instead of a
single best-fit point, history matching returns this *set* of all parameter
values that could plausibly have produced the observed data. The NROY region
shrinks wave by wave. **NROY samples** are points drawn from it (via
`get_nroy_samples()`), and the **NROY fraction** is the share of fresh prior
samples that fall inside it — a convergence diagnostic that falls toward zero as
the calibration tightens.

### Prior { #prior }

The initial parameter space — the bounds you start from, before any wave has
ruled anything out. History matching progressively carves the [NROY](#nroy)
region out of the prior.

### Ray sampling / `ray_resample` { #ray-resample }

A method for drawing [NROY](#nroy) samples efficiently when the NROY region is a
tiny fraction of the prior and plain rejection sampling becomes slow. The
`ray_resample` pipeline (inspired by the
[hmer](https://cran.r-project.org/package=hmer) R package) is four stages:
LHS → *ray sampling* (drawing points along lines connecting known NROY points) →
importance sampling → [maximin](#maximin) thinning. Faster than pure LHS at low
acceptance rates, but can over-represent region boundaries; see the
[NROY sampling methods tutorial](tutorials/06_nroy_sampling_methods.ipynb).

### Trajectory selection { #trajectory-selection }

A post-calibration step for stochastic models: having found which *parameters*
are plausible, select specific `(parameter set, random seed)` pairs whose
simulated trajectories match the observed data, using importance resampling
weighted by a pseudo-likelihood. See the
[trajectory selection tutorial](tutorials/05_trajectory_selection.ipynb).

### Wave { #wave }

One iteration of the history matching loop — sample, simulate, select features,
train emulators, filter. "Wave" and "iteration" are used interchangeably; the
on-disk output is organized into `wave1/`, `wave2/`, etc. Each wave starts from
the [NROY](#nroy) region left by the previous one, so the searched space
contracts with every wave.
