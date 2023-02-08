![Staging: history_matching]

# history_matching

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [User Installation](#user-installation)
  - [Pre-requisites](#pre-requisites)
- [Development Environment Setup](#development-environment-setup)
  - [First Time Setup](#first-time-setup)
  - [Development Tips](#development-tips)
  - [Building docs](#building-docs)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# User Installation

```bash
pip install history_matching --index-url=https://packages.idmod.org/api/pypi/pypi-production/simple
```

## Pre-requisites
- Python 3.6/3.7 x64


# Development Environment Setup

When setting up your environment for the first time, you can use the following instructions

## First Time Setup
1) Clone the repository:
   ```bash
   > git clone https://github.com/InstituteforDiseaseModeling/history_matching.git
   ```
2) Create a virtualenv. On Windows, please use venv to create the environment
   `python -m venv history_matching`
   On Unix(Mac/Linux) you can use venv or virtualenv
3) Activate the virtualenv
4) If you are on windows, run `pip install py-make --upgrade --force-reinstall`
5) Then run `python ./.dev_scripts/bootstrap.py`. This will install all the tools. 

## Development Tips

There is a Makefile file available for most common development tasks. Here is a list of commands
```bash
clean       -   Clean up temproary files
lint        -   Lint package and tests
test        -   Run All tests
coverage    -   Run tests and generate coverage report that is shown in browser
```
On Windows, you can use `pymake` instead of `make`


## Building docs

```make docs```