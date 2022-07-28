# History Matching

Welcome to IDM History Matching!

This repository provides Python-based source code for running History Matching.  History Matching finds regions of a model's parameter space that are non-implausible with respect to observed data.

# Installation

First clone the repository,

```bash
git clone https://github.com/InstituteforDiseaseModeling/history_matching
```

Then install the package,
```python
python3 -m pip install -e .
```

# Documentation

For documentation see [http://historymatching.com](http://historymatching.com).

# Examples

Examples are available in the [examples/](examples/) directory.

# Installing pyCUDA on Windows
Instructions adapted from <a href="https://wiki.tiker.net/PyCuda/Installation/Windows"> this wiki</a>.
- Install python 2.7 or greater (e.g. conda)
- Install Visual Studio 2008
- Install the <a href="https://developer.nvidia.com/cuda-toolkit">CUDA Toolkit</a>
- Install pyCUDA    
```
python3 -m pip install pycuda
```

- Add to PATH variable: `C:\Program Files\Microsoft Visual Studio XX\VC\bin\;C:\Program Files\Microsoft Visual Studio XX\VC\bin\amd64;C:\Program Files\Microsoft Visual Studio XX\Common7\IDE` 

# Running tests

To run tests use:
```bash
./run_tests.sh
```