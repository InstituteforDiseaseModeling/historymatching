#!/usr/bin/env python

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import setuptools
import sys

__version__ = '0.0.1'



####################################
#PyBind11 Compilation Stuff
####################################

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        'ckernels',
        ['history_matching/ckernels.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
]

# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag): return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)

####################################
#PyBind11 Compilation Stuff
####################################

setup(name='HistoryMatching',
      version='0.1',
      description='Support for model calibration using the History Matching algorithm',
      author='Daniel J. Klein',
      author_email='dklein@idmod.org',
      url='https://github.com/InstituteforDiseaseModeling/history_matching',
      packages=['history_matching'],
      setup_requires=["pybind11>=2.4"],
      install_requires=[ # Required packages -- install via pip install -e .
        "matplotlib",
        "numpy",
        "pandas",
        "patsy",
        "pycuda",
        "pyDOE",
        "scipy",
        "scikit-cuda",
        "seaborn",
        "statsmodels",
        "pybind11>=2.4"
        ],
        extras_require={ # Required for the Jupyter notebook only -- use pip install -e .[jupyter]
            "jupyter": [
                "jupyter",
                "wand",
                "xlrd",
                "openpyxl",
                ],
      },
      ext_modules=ext_modules,
      cmdclass={'build_ext': BuildExt},
      zip_safe=False
     )

