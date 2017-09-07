#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='HistoryMatching',
      version='0.1',
      description='Support for model calibration using the History Matching algorithm',
      author='Daniel J. Klein',
      author_email='dklein@idmod.org',
      url='https://github.com/InstituteforDiseaseModeling/history_matching',
      packages=find_packages(), # ['history_matching'],
      package_data={'': ['newlib/kernel.c']},
     )

