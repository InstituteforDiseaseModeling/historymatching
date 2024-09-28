======
README
======

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.com/projects/institute-for-disease-modeling-history-matching/badge/?style=flat
    :target: https://docs.idmod.org/projects/history-matching/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/clorton/history_matching/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/clorton/history_matching/actions

.. |codecov| image:: https://codecov.io/gh/InstituteforDiseaseModeling/history_matching/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://app.codecov.io/github/InstituteforDiseaseModeling/history_matching

.. |version| image:: https://img.shields.io/pypi/v/bhm.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/bhm

.. |wheel| image:: https://img.shields.io/pypi/wheel/bhm.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/bhm

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/bhm.svg
    :alt: Supported versions
    :target: https://pypi.org/project/bhm

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/bhm.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/bhm

.. |commits-since| image:: https://img.shields.io/github/commits-since/clorton/history_matching/v0.9.0.svg
    :alt: Commits since latest release
    :target: https://github.com/clorton/history_matching/compare/v0.9.0...main



.. end-badges

A Python implementation of the Bayesian History Matching algorithm.

* Free software: MIT license

Installation
============

::

Option 1: Standard Installation

1. Clone the repository:
   ```
   git clone https://github.com/InstituteforDiseaseModeling/history_matching
   ```
2. Install the package:
   `
   python3 -m pip install .
   `
   This will install the package and its dependencies, allowing you to use it as a regular installed Python package.

Option 2: Editable mode (for development)

1. Clone the repository:
   `
   git clone https://github.com/InstituteforDiseaseModeling/history_matching
   `
2. Install the package in editable mode:
   `
   python3 -m pip install -e .
   `
   In this mode, any changes you make to the source code will be reflected immediately without needing to reinstall the package.


Documentation
=============


https://docs.idmod.org/projects/history-matching


Development
===========

To run all the tests run::

    tox

To run tests on your current environment run::

    tox -e tests

Other useful tox commands are::

    tox -e clean
    tox -e docs
    tox -e check
    tox -e py39-cover
    tox -e report

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
