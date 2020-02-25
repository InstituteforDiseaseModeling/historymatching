#!/usr/bin/env python

from setuptools import setup

setup(name='HistoryMatching',
      version='0.1',
      description='Support for model calibration using the History Matching algorithm',
      author='Daniel J. Klein',
      author_email='dklein@idmod.org',
      url='https://github.com/InstituteforDiseaseModeling/history_matching',
      packages=['history_matching'],
      python_requires='>=3.6',
      install_requires=[ # Required packages -- install via pip install -e .
        "matplotlib",
        "numpy",
        "pandas",
        "patsy",
        "pycuda",
        "pyDOE",
        "scipy",
        "scikit-cuda",
        "seaborn",
        "statsmodels",
        ],
        extras_require={ # Required for the Jupyter notebook only -- use pip install -e .[jupyter]
            "jupyter": [
                "jupyter",
                "wand",
                "xlrd",
                "openpyxl",
                ],
      },
      zip_safe=False,
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

