#!/usr/bin/env python

from setuptools import setup

setup(name='hm2',
      version          = '0.1',
      description      = 'Support for model calibration using the History Matching algorithm',
      author           = 'Richard Barnes',
      author_email     = 'rijard.barnes@gmail.com',
      url              = 'https://github.com/InstituteforDiseaseModeling/history_matching',
      packages         = ['hm2'],
      python_requires  =' >= 3.6',
      install_requires = [ # Required packages -- install via pip install -e .
        "matplotlib",
        "numpy",
        "pandas",
        "pyDOE",
      ],
      classifiers=[
          "Programming Language :: Python :: 3 :: Only",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: C++",
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

