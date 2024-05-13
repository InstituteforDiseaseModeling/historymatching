#!/usr/bin/env python
import re
from pathlib import Path

from setuptools import setup


def read(*names, **kwargs):
    with Path(__file__).parent.joinpath(*names).open(encoding=kwargs.get("encoding", "utf8")) as fh:
        return fh.read()


setup(
    name="history-matching",
    version="0.9.0",
    license="MIT",
    description="A Python implementation of the Bayesian History Matching algorithm.",
    long_description="{}\n{}".format(
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub("", read("README.rst")),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.rst")),
    ),
    author="Daniel J. Klein, Rafael C. Nunez, Richard Barnes, Christopher Lorton",
    author_email="Daniel.Klein@gatesfoundation.org",
    url="https://github.com/InstituteforDiseaseModeling/history_matching",
    py_modules=[path.stem for path in Path("src").glob("*.py")],
    packages=["history_matching"],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        # uncomment if you test on these interpreters:
        # "Programming Language :: Python :: Implementation :: IronPython",
        # "Programming Language :: Python :: Implementation :: Jython",
        # "Programming Language :: Python :: Implementation :: Stackless",
        "Topic :: Utilities",
    ],
    project_urls={
        "Documentation": "https://docs.idmod.org/projects/history-matching/",
        "Changelog": "https://docs.idmod.org/projects/history-matching/en/latest/changelog.html",
        "Issue Tracker": "https://github.com/InstituteforDiseaseModeling/history_matching/issues",
    },
    keywords=[
        "Bayesian History Matching",
        "History Matching",
        "Model Emulation",
        "Model Emulators",
        "Model Calibration",
        "Uncertainty Quantification"
    ],
    python_requires=">=3.7",
    install_requires=[
        # eg: "aspectlib==1.1.1", "six>=1.7",
        "numpy",
        "pandas",
        "matplotlib",
        "statsmodels",
        "scikit-learn",
        "tensorflow",
        "tf-keras",
        "gpflow",
        "numexpr!=2.8.5",  # until RE string issue gets fixed
    ],
    extras_require={
        # eg:
        #   "rst": ["docutils>=0.11"],
        #   ":python_version=="2.6"": ["argparse"],
        "notebooks": ["jupyterlab", "chardet"]
    },
    entry_points={
        "console_scripts": [
            "history-matching = history_matching.cli:main",
        ]
    },
)
