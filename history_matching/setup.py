#!/usr/bin/env python

from setuptools import setup

setup(name='HistoryMatching',
      version='0.1',
      description='Support for model calibration using the History Matching algorithm',
      author='Daniel J. Klein',
      author_email='dklein@idmod.org',
      url='https://github.com/InstituteforDiseaseModeling/history_matching',
      packages=['history_matching'],
      install_requires=[
          'setuptools',
          'pandas',
          'pyDOE',
          'patsy',
          'matplotlib',
          'statsmodels',
          'seaborn',
          'xlrd',
          'tables',
          'openpyxl',
          'jupyter',
          'wand'
      ],
      zip_safe=False
     )

