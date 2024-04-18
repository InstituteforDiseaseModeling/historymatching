======
Readme
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

    On macOS need homebrew, brew install hdf5 c-blosc lzo bzip2, HDF5_DIR=/opt/homebrew/opt/hdf5, export HDF5_DIR

    pip install history-matching

You can also install the in-development version with::

    pip install https://github.com/clorton/history_matching/archive/main.zip


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
