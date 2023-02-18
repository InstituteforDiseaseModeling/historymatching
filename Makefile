.PHONY: help clean lint test test-all test-failed test-smoke test-report coverage-report coverage coverage-smoke coverage-all coverage-report-view dist release-staging bump-release bump-release-dry-run bump-minor bump-minor-dry-run bump-major bump-major-dry-run bump-patch bump-patch-dry-run
.EXPORT_ALL_VARIABLES:
# Get dev scripts from any location
mkfile_path := $(lastword $(MAKEFILE_LIST))
mkfile_dir := $(dir $(mkfile_path))
# Platform Independent options for common commands
MV?=mv
RM?=rm
# Convenience function for running dev scripts
PDS=$(PY) ./.dev_scripts/
PY?=python
IPY=python -c
PDR=$(PDS)run.py
CLDIR=$(PDS)clean_dir.py
PYPI_URL?=https://packages.idmod.org/api/pypi/idm-pypi-staging/
PACKAGE_NAME=history_matching

# Makefile rules

help:
	help-from-makefile -f $(mkfile_path)

setup-dev: ## Install all dependencies needed for development
	$(PDS)bootstrap.py

clean: ## Clean most of the temp-data from the project
	$(MAKE) -C tests clean
	-rm -rf *.pyc *.pyo *.done *.log .coverage dist build **/__pycache__

clean-all: clean ## Deleting package info hides plugins so we only want to do that for packaging
	-rm -rf **/*.egg-info/


# Linting
lint: ## check style with flake8
	flake8 --ignore=E501,W291 $(PACKAGE_NAME)


## Tests
#############

test: ## Run our tests
	$(MAKE) -C tests $@

test-all: ## Run all our tests
	$(MAKE) -C tests $@

test-failed: ## Run only previously failed tests
	$(MAKE) -C tests $@

test-report: ## Launch test report in browser
	$(MAKE) -C tests $@

coverage-report: coverage ## Generate HTML report from coverage. Requires running coverage run first(coverage, coverage-smoke, coverage-all)
	$(MAKE) -C tests $@

coverage-report-view: coverage-report
	$(MAKE) -C tests $@

coverage: clean ## Generate a code-coverage report
	$(MAKE) -C tests $@

coverage-smoke: clean ## Generate a code-coverage report
	$(MAKE) -C tests $@

coverage-all: ## Generate a code-coverage report using all tests
	$(MAKE) -C tests $@


# Release related rules
#######################

dist: clean ## build our package
	python setup.py sdist

release-staging: dist ## perform a release to staging
	twine upload --verbose --repository-url $(PYPI_URL) dist/*

# versioning
bump-release: ## bump the release version.
	bump2version release --commit

# Use before release-staging-release-commit to confirm next version.
bump-release-dry-run: ## bump the release version. (dry run)
	bump2version release --dry-run --allow-dirty --verbose

bump-patch: ## bump the patch version
	bump2version patch --commit

bump-minor: ## bump the minor version
	bump2version minor --commit

bump-major: ## bump the major version
	bump2version major --commit

bump-patch-dry-run: ## bump the patch version(dry run)
	bump2version patch --dry-run --allow-dirty --verbose

bump-minor-dry-run: ## bump the minor version(dry run)
	bump2version minor --dry-run --allow-dirty --verbose

bump-major-dry-run: ## bump the major version(dry run)
	bump2version major --dry-run --allow-dirty --verbose


# Docs Related rules
#####################
docs: ## build docs
	$(PDR) -wd 'docs' -ex 'make html'

docs-server: docs ## builds docs and launch a webserver and watches for changes to documentation
	$(PDS)serve_docs.py


# Extra
licenses: ## Generate Notices.txt
	pip-licenses --output-file NOTICES.txt
