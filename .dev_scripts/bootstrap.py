#!/usr/bin/env python
import argparse
import logging
import os
import subprocess
import sys
import unicodedata
from logging import getLogger
from os.path import abspath, join, dirname

# on windows virtual env is not populated through pymake
from typing import List, Generator

if sys.platform == "win32" and 'VIRTUAL_ENV' in os.environ:
    sys.path.insert(0, os.environ['VIRTUAL_ENV'] + "\\Lib\\site-packages")

# This scripts aids in setup of development environments by installing all the local packages
# defined by packages using development installs.
#
# It is best used
# 1) After the creation of a new virtualenv
# 2) Installing new packages into an existing environment
# 3) Updating existing environments
#
# To use simply run
# python bootstrap.py

base_directory = abspath(join(dirname(__file__), '..'))

default_install = ['test']
data_class_default = default_install

escapes = ''.join([chr(char) for char in range(1, 32)])
translator = str.maketrans('', '', escapes)
logger = getLogger("bootstrap")


def execute(cmd: List['str'], cwd: str = base_directory, ignore_error: bool = False) -> Generator[str, None, None]:
    """
    Runs a command and filters output

    Args:
        cmd: Command to run
        cwd: Working directory - Defaults to current directory
        ignore_error: Should we ignore errors

    Returns:
        Generator returning each line

    Raises:
        CalledProcessError if the return code was not 0
    """
    logger.debug(f'Running {" ".join(cmd)}')
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, cwd=cwd)
    yield from iter(process.stdout.readline, "")
    process.stdout.close()
    return_code = process.wait()
    if return_code and not ignore_error:
        raise subprocess.CalledProcessError(return_code, cmd)


def process_output(line):
    """
    Filter output
    Args:
        line:

    Returns:

    """
    # catch errors where possible
    if "FAILED [" in line:
        logger.critical(line.strip())
    elif any([s in line for s in ["Successfully", "SUCCESS"]]):
        logger.log(35, line.strip())
    elif any([s in line for s in ["WARNING", "SKIPPED"]]):
        logger.warning(line.strip())
    else:
        line = line.strip().translate(translator)
        logger.debug("".join(ch for ch in line if unicodedata.category(ch)[0] != "C"))


def install_base_environment(pip_url):
    for line in execute(["pip", "uninstall", "-y", "py-make"], ignore_error=True):
        process_output(line)
    # install docs
    for install_dir in [join(base_directory, 'docs')]:
        for line in execute(["pip", "install", "-r", "requirements.txt"], cwd=install_dir):
            process_output(line)
    for line in execute(["pip", "install", "-e", ".[dev]", f"--extra-index-url={pip_url}"], cwd=base_directory):
        process_output(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap the development environment")
    parser.add_argument("--index-url", default='https://packages.idmod.org/api/pypi/pypi-production/simple', help="Pip url to install dependencies from")
    parser.add_argument("--verbose", default=False, action='store_true')

    args = parser.parse_args()

    logger.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    file_handler = logging.FileHandler("bootstrap.buildlog")
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    # use colorful logs except the first time
    console_log_level = logging.DEBUG if 'BUILD_DEBUG' in os.environ or args.verbose else logging.INFO
    try:
        import coloredlogs  # noqa: I900

        coloredlogs.install(logger=logger, level=console_log_level, fmt="%(asctime)s [%(levelname)-8.8s]  %(message)s")
        logging.addLevelName(15, 'VERBOSE')
        logging.addLevelName(35, 'SUCCESS')
        logging.addLevelName(50, 'CRITICAL')
    except ImportError:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(console_log_level)
        logger.addHandler(console_handler)

    install_base_environment(args.index_url)
