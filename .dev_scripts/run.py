import argparse
import logging
import os
import subprocess
import sys
from logging import getLogger

# on windows virtual env is not populated through pymake
if sys.platform == "win32" and 'VIRTUAL_ENV' in os.environ:
    sys.path.insert(0, os.environ['VIRTUAL_ENV'] + "\\Lib\\site-packages")
try:
    import coloredlogs  # noqa: E402
except ImportError:
    pass

logging.addLevelName(15, 'VERBOSE')
logging.addLevelName(35, 'SUCCESS')
logging.addLevelName(50, 'CRITICAL')
logger = getLogger(__name__)


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, shell=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def setup_logging(working_dir):
    logger.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-8.8s]  %(message)s")
    file_handler = logging.FileHandler("%s/make.buildlog" % os.path.abspath(working_dir))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    console_log_level = logging.DEBUG if 'BUILD_DEBUG' in os.environ else logging.INFO
    coloredlogs.install(logger=logger, level=console_log_level, fmt="%(asctime)s [%(levelname)-8.8s]  %(message)s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment', nargs='+', help='Environment variables to set ')
    parser.add_argument('-wd', '--working-dir', help='Working Directory')
    parser.add_argument('-rt', '--return-code', default=None, help='Return Code')
    parser.add_argument('-p', '--path', nargs='+', help='Add items to the path')
    parser.add_argument('-ex', help='Add items to the path')

    args = parser.parse_args()
    setup_logging(args.working_dir)
    if args.environment:
        env = dict()
        for e in args.environment:
            parts = e.split("=")
            env[parts[0]] = parts[1]
        logger.info(f"Environment opts: {env}")

    if args.path:
        for p in args.path:
            logger.info(f'Adding {os.path.abspath(p)} to the path')
            sys.path.insert(0, os.path.abspath(p))

    logger.debug(f'Python: {sys.executable}')
    logger.debug('Environment: %s', str(os.environ))

    if args.working_dir:
        logger.info(f'Changing working directory to {os.path.abspath(args.working_dir)}')
        os.chdir(os.path.abspath(args.working_dir))
    logger.info(f'Running {args.ex}')
    try:
        for line in execute(args.ex):
            # catch errors where possible
            if any([s in line for s in
                    ["rollbackFailedOptional:", "extract:yarn:", "Using cache", "copying ", "creating idm",
                     "optional dependency"]]):
                logger.log(15, line.strip())
            elif any([s in line for s in ["ERR!", "ERROR", "FAILED", "Error:", "Failed"]]):
                logger.critical(line.strip())
            elif any([s in line for s in ["Successfully", "SUCCESS", "PASSED"]]):
                logger.log(35, line.strip())
            elif any([s in line for s in ["WARNING", "SKIPPED"]]):
                logger.warning(line.strip())
            else:
                logger.info(line.strip())
        result = 0
    except subprocess.CalledProcessError as e:
        logger.error(f'{e.cmd} did not succeed')
        result = e.returncode
        logger.debug(f'Return Code: {result}')
    sys.exit(result)
