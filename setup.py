import ctypes
from distutils.version import LooseVersion
import json
import os
import pip
import re
import sys
from urlparse import urlparse

from setuptools import setup, find_packages

# from dtk-tools
from simtools.Utilities.General import nostdout
from simtools.Utilities.GitHub.MultiPartFile import GitHubFile
from simtools.Utilities.LocalOS import LocalOS

# to fake out urlparse, setting netloc == 'GITHUB'
GITHUB = 'GITHUB'
GITHUB_URL_PREFIX = 'http://%s' % GITHUB
INSTALL_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'install')

###########################################################################################

def upgrade_pip(my_os):
    """
    Upgrade pip before install other packages
    """
    import subprocess

    if my_os in [LocalOS.MAC, LocalOS.LINUX]:
        subprocess.call("pip install -U pip", shell=True)
    elif my_os in [LocalOS.WINDOWS]:
        subprocess.call("python -m pip install --upgrade pip", shell=True)

def get_requirements_by_os(my_os, requirement_list):
    """
    Update requirements based on OS
    """
    # reqs = OrderedDict([(name, val) for (name, val) in requirements.iteritems() if my_os in val['platform']])
    reqs = []
    for req in requirement_list:
        if my_os in req['platform']:
            # first, OS-specific modifications to requirements

            # OS: Mac or Linux. No wheel needed
            if my_os in [LocalOS.MAC, LocalOS.LINUX]:
                req.pop('wheel', None)

            # OS: Linux. No version for some packages
            if my_os == LocalOS.LINUX:
                if req['package_name'] in ['numpy', 'scipy']:
                    req.pop('version', None)
                    req.pop('test', None)

            # remember this requirement for installation
            reqs.append(req)
    return reqs

def install_linux_pre_requisites():
    """
    Install pre-requisites for Linux
    """
    pass # ck4, will we need this?

def get_installed_packages():
    """
    Check packages in system
    """

    # Flatten the list
    installed_packages = {}
    for package in pip.get_installed_distributions():
        installed_packages[package.project_name] = package.version
    return installed_packages




def download_file(url):
    """
    Download package
    """
    import urllib2

    local_file = get_local_file_path(url)

    req = urllib2.Request(url)
    resp = urllib2.urlopen(req)
    data = resp.read()
    with open(local_file, "wb") as code:
        code.write(data)

    return local_file


def get_local_file_path(url):
    # If it is local file, use it
    if os.path.exists(url):
        return url

    # Compose local file
    file_name = os.path.basename(url)
    local_file = os.path.join(INSTALL_DIRECTORY, file_name)
    return local_file

def build_package_str(my_os, name, val):
    """
    Build package installation string
    """
    package_str = None

    if my_os in [LocalOS.WINDOWS]:
        if val.get('wheel', None):
            package_str = GITHUB_URL_PREFIX + '/' + val['wheel']
        elif val.get('version', None):
            # Win doesn't support >= or <=. Replace with ==
            op = val['test']
            op = re.sub('[><]', '=', op) if not op else op
            package_str = "%s%s%s" % (name, op, val['version'])
        else:
            package_str = name
    elif my_os in [LocalOS.MAC, LocalOS.LINUX]:
        if val.get('test', None) and val.get('version', None):
            package_str = "%s%s%s" % (name, val['test'], val['version'])
        else:
            package_str = name
    return package_str

def install_package(my_os, name, val, upgrade=False):
    """
    Install or upgrade package
    """
    import pip
    package_str = build_package_str(my_os, name, val)
    print('resolved package_str: %s' % package_str)
    # print("urlparse: %s" % urlparse(package_str)[0:1] )

    method, host, path = urlparse(package_str)[0:3]
    # It is an internet file
    if (len(host) > 0 and len(path) > 0) or host == GITHUB:
        local_file = get_local_file_path(package_str)
        print("A")
        if method == 'git+https': # ugly, for scikit-cuda
            local_file = package_str
        else:
            if not os.path.exists(local_file):
                print('host is: %s' % host)
                print('path is: %s' %path)
                # Download file if it does not exist locally
                if host == GITHUB:
                    dependency = GitHubFile(local_file)
                    dependency.pull() # writes to local_file
                else:
                    local_file = download_file(package_str)

        # Install package from local file (just downloaded or existing one)
        print('installing local-file: %s' % local_file)
        if upgrade:
            pip.main(['install', local_file, '--upgrade'])
        else:
            pip.main(['install', local_file])
    # Check if it is local wheel file or tar.gz file
    elif (package_str.endswith('.whl') or package_str.endswith('.tar.gz')) \
            and os.path.exists(get_local_file_path(package_str)):
        print("B")
        # Use local file if it exists
        if upgrade:
            pip.main(['install', get_local_file_path(package_str), '--upgrade'])
        else:
            pip.main(['install', get_local_file_path(package_str)])
    # Just package name w/o version
    else:
        print("C")
        if upgrade:
            pip.main(['install', package_str, '--upgrade'])
        else:
            pip.main(['install', package_str])


def test_package_g(my_os, name, val, installed_packages={}):
    """
    Case: required version > installed version
    """
    version = val.get('version', None)
    test = val.get('test', None)

    if test in ['==', '>=']:
        print("Package %s (%s) already installed with lower version. Upgrading to (%s)..." %  (name, installed_packages[name], version))
        install_package(my_os, name, val, True)
    else:
        # Usually we don't have this case.
        print ("Package %s (%s) already installed. Skipping..." % (name, installed_packages[name]))


def test_package_e(my_os, name, val, installed_packages={}):
    """
    Case: required version == installed version
    """
    test = val.get('test', None)

    if test in ['>=', '<=']:
        print ("Package %s (%s) already installed. Skipping..." % (name, installed_packages[name]))
    elif test in ['==']:
        print ("Package %s (%s) with exact version already installed. Skipping..." % (name, installed_packages[name]))
    else:
        print ("Package %s (%s) installed. Skipping..." % (name, installed_packages[name]))


def test_package_l(my_os, name, val, installed_packages={}):
    """
    Case: required version < installed version
    """
    version = val.get('version', None)

    # Usually we don't have this case.
    print ("Package %s (%s) with higher version installed but require lower version (%s). Installing..." %  (name, installed_packages[name], version))
    install_package(my_os, name, val)


def test_package(my_os, name, val, installed_packages={}):
    """
    Check installation
    """
    version = val.get('version', None)
    if name in installed_packages:
        if not version:
            print ("Package %s (%s) installed. Skipping..." % (name, installed_packages[name]))
            return

        if LooseVersion(version) > LooseVersion(installed_packages[name]):
            test_package_g(my_os, name, val, installed_packages=installed_packages)
        elif LooseVersion(version) == LooseVersion(installed_packages[name]):
            test_package_e(my_os, name, val, installed_packages=installed_packages)
        else:
            test_package_l(my_os, name, val, installed_packages=installed_packages)
    else:
        print ("Package %s not installed. Installing..." % name)
        install_package(my_os, name, val)

def install_packages(my_os, reqs):
    """
    Install required packages
    """
    if my_os == LocalOS.LINUX:
        # Doing the apt-get install pre-requisites
        install_linux_pre_requisites()

    # Get the installed package to not reinstall everything
    installed_packages = get_installed_packages()

    # Go through the requirements
    for req in reqs:
        test_package(my_os, req['package_name'], req, installed_packages=installed_packages)
        # ck4, we would ideally update the installed package list after each installation.

    # Add the develop by default
    sys.argv.append('develop')
    sys.argv.append('--quiet')

    from setuptools import setup, find_packages
    # Suppress the outputs except the errors
    with nostdout(stderr=True):
        setup(name='HistoryMatching',
              version='0.1',
              description='Support for model calibration using the History Matching algorithm',
              url='https://github.com/InstituteforDiseaseModeling/history_matching',
              author='Daniel J. Klein,'
                     'Clark H. Kirkman IV',
              author_email='dklein@idmod.org,'
                           'ckirkman@idmod.org',
              packages=find_packages(),
              package_data={'': ['newlib/kernel.c', 'documentation/history_matching_beginner_documentation.xlsx']},
              entry_points={
                    'console_scripts': ['hm = history_matching.hm:main']
              },
             )

def main():
    # Check OS in case we will need it
    my_os = LocalOS.name
    print ('os: %s' % my_os)

    # Upgrade pip before install other packages
    upgrade_pip(my_os)
    # Get OS-specific requirements
    requirements = json.load(open('requirements.json', 'r'))
    reqs = get_requirements_by_os(my_os=my_os, requirement_list=requirements)

    # Install required packages
    install_packages(my_os, reqs)

    # Success !
    print ("\n=======================================================")
    print ("| history_matching and dependencies installed successfully. |")
    print ("=======================================================")


if __name__ == "__main__":
    # check os first
    if ctypes.sizeof(ctypes.c_voidp) != 8:
        print ("""\nFATAL ERROR: dtk-tools only supports Python 2.7 x64. Please download and install a x86-64 version of python at:
        - Windows: https://www.python.org/downloads/windows/
        - Mac OSX: https://www.python.org/downloads/mac-osx/
        - Linux: https://www.python.org/downloads/source/\n
        Installation is now exiting...""")
        exit()

    main()
