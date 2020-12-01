#!/usr/bin/env python

# =======
# Imports
# =======

from __future__ import print_function
import os
import sys
import setuptools
import codecs
# from sphinx.setup_command import BuildDoc

# =========
# Read File
# =========

def ReadFile(Filename):
    """
    Reads a file with latin codec.
    """

    with codecs.open(Filename,'r','latin') as File:
        return File.read()

# ================
# Read File to RST
# ================

def ReadFileToRST(Filename):
    """
    Reads atext file and converts it to RST file using pandas.
    """

    try:
        import pypandoc
        rstname = "{}.{}".format(os.path.splitext(Filename)[0],'rst')
        pypandoc.convert(read(Filename),'rst', format='md', outputfile=rstname)
        with open(rstname, 'r') as f:
            rststr = f.read()
        return rststr
    except ImportError:
        return ReadFile(Filename)

# ====
# Main
# ====

def main(argv):

    Directory = os.path.dirname(os.path.realpath(__file__))
    PackageName = "TraceInv"
    PackageNameForDoc = "TraceInv"

    # Version
    version_dummy = {}
    exec(open(os.path.join(Directory,PackageName,'__version__.py'),'r').read(),version_dummy)
    Version = version_dummy['__version__']
    del version_dummy

    # Author
    Author = open(os.path.join(Directory,'AUTHORS.txt'),'r').read().rstrip()

    # Requirements
    Requirements = [i.strip() for i in open(os.path.join(Directory,"requirements.txt"),'r').readlines()]

    # ReadMe
    LongDescription = open(os.path.join(Directory,'README.rst'),'r').read()

    # Build documentation
    # cmdclass = {'build_sphinx': BuildDoc}

    # Setup
    setuptools.setup(
        name = PackageName,
        version = Version,
        author = Author,
        author_email = 'sameli@berkeley.edu',
        description = 'Computes the trace of the inverse of matrix or linear matrix function.',
        long_description = LongDescription,
        long_description_content_type = 'text/x-rst',
        keywords = """matrix-computations matrix-inverse interpolation-techniques 
                cholesky-decomposition randomized-algorithms lanczos-iteration 
                parameter-estimation radial-basis-function polynomial-bases 
                orthogonal-polynomials cross-validation""",
        url = 'https://github.com/ameli/TraceInv',
        download_url = 'https://github.com/ameli/TraceInv/archive/master.zip',
        project_urls = {
            "Documentation": "https://github.com/ameli/TraceInv/blob/master/README.rst",
            "Source": "https://github.com/ameli/TraceInv",
            "Tracker": "https://github.com/ameli/TraceInv/issues",
        },
        platforms = ['Linux','OSX','Windows'],
        packages = setuptools.find_packages(exclude=("tests","examples",)),
        install_requires = Requirements,
        python_requires = '>=2.7',
        setup_requires = ['numpy'],
        tests_require = ['pytest'],
        include_package_data=True,
        # cmdclass=cmdclass,
        # command_options = {
        #     'build_sphinx': {
        #         'project':    ('setup.py',PackageNameForDoc),
        #         'version':    ('setup.py',Version),
        #         'source_dir': ('setup.py','docs')
        #     }
        # },
        extras_require = {
            'extra': [
                'ray'
                ],
            'full': [
                'scikit-sparse',
                'ray'
                ],
            'test': [
                'pytest-cov',
                'codecov'
            ],
            'docs': [
                'sphinx',
                'sphinx-math-dollar',
                'sphinx-toggleprompt',
                'sphinx_rtd_theme',
                'graphviz',
                'sphinx-automodapi'
            ]
        },
        classifiers = [
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Natural Language :: English',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
    )

# ===========
# System Main
# ===========

if __name__ == "__main__":
    main(sys.argv)
