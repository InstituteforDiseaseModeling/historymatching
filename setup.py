#!/usr/bin/env python

from setuptools import setup

setup(name='HistoryMatching',
      version="0.1.1",  # if you change this, also change it manually in __init__.py until we start using bump2version
      description='Support for model calibration using the History Matching algorithm',
      author='Daniel J. Klein',
      author_email='dklein@idmod.org',
      url='https://github.com/InstituteforDiseaseModeling/history_matching',
      packages=['history_matching'],
      include_package_data=True,
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
        "pyDOE",
        "asdf",
        "pyarrow"
        ],
        extras_require={ # Required for the Jupyter notebook only -- use pip install -e .[jupyter]
            "jupyter": [
                "jupyter",
                "wand",
                "xlrd",
                "openpyxl",
                ],
      },
      zip_safe=False
     )

