# Overview

- temporarily putting new code into package `hm2` - we can replace existing `history_matching` implementation when we are ready.
- named some things somewhat whimsically based on [this post](https://betterprogramming.pub/software-component-names-should-be-whimsical-and-cryptic-ca260b013de0) - we can have more serious names if desired

----

## Example

- `one_parameter.py` tries to find the intercept for a linear equation `y=mx+b` with some noise in the observations
  - first it specifies a few configuration parameters and puts them in a `Config` object (see `config.py`)
  - second it creates an initial `Situation` object (see `situation.py`)\* with a parameter space for search, statistics about the observations, and some initial sample points in parameter space
  - third it creates a `Recipe` object (see `recipe.py`) which it customizes for running its model, generating an emulator (using @rnunez-IDM initial linear emulator), and choosing next points in the parameter space. We could use a dictionary with key:value pairs for steps of the "recipe" but I prefer an object to help catch spelling mistakes/typos.
  - finally it calls `do_step()` (see `step.py`) with the state, recipe, and config created above until `do_step()` returns `True` (== done)

----

  - parameter space samplers are in `samplers.py` - simple Latin Hypercube, fixed grid, and random
  - `utils.py` has two helper functions, one returns mean and variance for a set of raw observations while the other returns the feature names from an observations DataFrame.
  - `tests.py` has a set of tests for all the objects and functions used above (@tinghf)

- `test_template.py` tries to find four parameters for y = ax^3 + bx^2 + cx + d but I haven't debugged it enough to be sure it is working.

## TODO

- [ ] need to wrap `do_step()` with a "run calibration" loop which does basically what `one_parameter.py` does, calling `do_step()` until it returns `True`.
- [ ] need default functions in `Recipe` for 1) selecting features, 2) generating emulators, and 3) next point(s) generation
- [ ] need example choosing different features at different iterations in the calibration
- [ ] need example generating other emulator(s) than the default
- [ ] need to spec. emulator statistics returned from generating emulators (@rnunez-IDM) and spec. data structure for same (@clorton)
- [ ] need example with custom next point(s) generation algorithm
- [ ] need functions on `Situation` to save to/restore from disk
- [ ] need example hooking `end_step_callback` to output diagnostic information from each step of calibration
- [ ] need example hooking `end_step_callback` to use `Situation` save to disk functionality
- [ ] need example using `Situation` restore from disk functionality to re-start or branch a calibration
----
- [ ] need to update `history_matching` examples to use `hm2`, when ready
  - radius complete
  - deterministic SIR
  - stochastic SIR
- [ ] need to update `phylomodels` examples to use `hm2`, when ready
  - calibration_historyMatching_sir
  - calibration_historyMatching_featureSelectionForSIR