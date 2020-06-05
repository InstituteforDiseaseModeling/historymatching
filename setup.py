#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='hm2',
      version          = '0.1.0',
      description      = 'Support for model calibration using the History Matching algorithm',
      author           = 'Richard Barnes',
      author_email     = 'rijard.barnes@gmail.com',
      url              = 'https://github.com/InstituteforDiseaseModeling/history_matching',
      packages         = find_packages(),
      python_requires  =' >= 3.6',
      install_requires = [ # Required packages -- install via pip install -e .
        #TODO: Do fuzzy matching on package version
        "matplotlib>=3.1.3",
        "numpy>=1.18.1",
        "pandas>=1.0.1",
        "plotnine==0.6.0",
        "pyDOE==0.3.8",
        "sklearn",
        "statsmodels==0.11.0",
      ],
      classifiers=[
          "Programming Language :: Python :: 3 :: Only",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Operating System :: Unix",
          "Operating System :: MacOS",
          "Operating System :: POSIX",
          "Operating System :: POSIX :: Linux",
          #TODO: Need a License
          "Development Status :: 2 - Pre-Alpha",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Information Analysis"
      ]
)
