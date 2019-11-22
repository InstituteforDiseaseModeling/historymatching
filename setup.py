#!/usr/bin/env python

from setuptools import setup
import history_matching

setup(name='HistoryMatching',
      version=history_matching.__version__,
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
        "pyDOE"
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

