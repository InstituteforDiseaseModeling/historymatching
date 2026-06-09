# Glossary

Terms used throughout this documentation. History matching borrows vocabulary
from Bayesian calibration, design of experiments, and the emulation literature;
the definitions below assume general familiarity with calibration but not with
history matching itself.

### ARD — Automatic Relevance Determination { #ard }

A Gaussian Process kernel parameterization that fits a separate lengthscale to
each input parameter. A short lengthscale means the output varies quickly with
that parameter (it is "relevant"); a long one means the output barely depends on
it. Ranking parameters by ARD lengthscale shows which ones the data constrains
most — this is what the `pairplot.png` diagnostic uses to pick its "top 8"
parameters. Used by the [GPR](#gpr) emulator.

### Bayes linear emulator { #bayes-linear }

The default emulator (`emulator_type='bayes_linear'`). It fits a linear model
but, in the Bayes linear tradition, tracks the full predictive *uncertainty*
(expectation and variance) rather than just a point estimate — giving
Gaussian-Process-like calibrated error bars without the cost or the TensorFlow
dependency of [GPR](#gpr). Implemented in pure NumPy/SciPy.

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

### GPR — Gaussian Process Regression { #gpr }

A flexible, nonlinear emulator (`emulator_type='gpr'`) built on GPflow/
TensorFlow with [ARD](#ard) kernels. Best for nonlinear response surfaces and
excellent uncertainty estimates, at a higher computational cost than the linear
emulators.

### Implausibility { #implausibility }

A score measuring how inconsistent a parameter value is with one observation,
expressed in standard deviations. For output $f$ at parameter $x$:

$$ I(x) = \frac{\left|\, \mathbb{E}[f(x)] - z \,\right|}{\sqrt{\operatorname{Var}_{\text{emulator}} + \operatorname{Var}_{\text{obs}}}} $$

where $z$ is the observed target and the denominator combines the emulator's
predictive variance with the observation's variance. With several outputs, the
implausibility of a point is the **maximum** over all outputs, so a point must
be consistent with *every* constraint to survive. A point is *ruled out* when
its implausibility exceeds the [threshold](#implausibility-threshold).

### Implausibility threshold { #implausibility-threshold }

The implausibility cutoff above which a parameter value is discarded
(`implausibility_threshold`, default `3.0`). The default of 3 follows the
"3-sigma" rule: under a roughly normal error, a consistent point lies within ≈3
standard deviations of its target ~99.7% of the time, so a higher score is
strong evidence the point is implausible.

### LHS — Latin Hypercube Sampling { #lhs }

A space-filling experimental design that spreads sample points evenly across
every parameter's range. The default sampling strategy (`'lhs'`); better coverage
than uniform random sampling for the same number of points.

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
