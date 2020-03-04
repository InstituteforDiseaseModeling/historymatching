# History Matching

Welcome to IDM History Matching!

This repository provice python-based source code for running history matching.  History matching finds regions of parameter space that are non-implausible with respect to observed data.

# Installation

First clone the repository,

```bash
git clone https://github.com/InstituteforDiseaseModeling/history_matching
```

Then install the package,
```python
python setup.py install
```

# Example

A simple example is provided in the history_matching/example folder.  This example has two input parameters, First Parameter and Second Parameter.  The result is simply the distance from the origin plus gaussian noise, and the desired results (observed data) is 15 with a stdev of 1.5.

The parameters are described in Params.xlsx, each parameter needs a Min and a Max bound.

On each iteration, beginning with iter0, run the python scripts in the following order:

|Order|Script|Purpose|
|-----|------|-------|
|1    |generate_samples_and_results.py|Generate inputs and outputs.  This script will make a folder called Data_DATE_TIME containing Samples.xlsx (inputs) and Results.xlsx (noisy outputs).|
|2|bhm.py|Build a statistical model to emulate the input-output relationship produced by the simulation.|
|3|cut.py|Use the emulator to regions of parameter space that are inconsistent with the data as implausible.  This will produce Candidates_for_iter#.xlsx (and hd5), which are points uniformly sampled from the non-implausible space for the next iteration.|

# Limitations

The package uses GPU acceleration based on CUDA, and thus only works on a computer with a modern NVIDIA graphics card.

# Setting up a dev environment

Install the necessary packages:

    pip3 install pre-commit pylint

Set up pre-commit hooks:

    pre-commit install
