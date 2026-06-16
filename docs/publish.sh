#!/usr/bin/env bash
#
# Build the documentation and publish it to GitHub Pages.
#
# This runs `mkdocs gh-deploy`, which builds the site and pushes the result to
# the `gh-pages` branch of the origin remote. Notebooks are executed as part of
# the build (see the mkdocs-jupyter plugin config in mkdocs.yml), so this can
# take a few minutes.
#
# Usage:
#   ./docs/publish.sh        # build and deploy to gh-pages
#
set -euo pipefail

# Run from the repo root regardless of where the script is invoked from.
cd "$(dirname "$0")/.."

uv run mkdocs gh-deploy --force
