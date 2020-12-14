#!/usr/bin/env python

# =======
# Imports
# =======

from __future__ import print_function
import os
import sys
import codecs
# from sphinx.setup_command import BuildDoc

# Import numpy
try:
    import numpy
except ImportError:
    # Install numpy
    try:
        import pip
        from pip import main
        pip.main(['install','numpy'])
        import numpy
    except:
        raise ImportError('Cannot import numpy.')

# Import setuptools
try:
    import setuptools
    from setuptools.extension import Extension
except ImportError:
    # Install setuptools
    try:
        import pip
        from pip import main
        pip.main(['install','setuptools'])
        import setuptools
        from setuptools.extension import Extension
    except:
        raise ImportError('Cannot import setuptools.')

# Import Cython (to convert pyx to C code)
try:
    from Cython.Build import cythonize
    UseCython = True
except ImportError:
    # Install Cython
    try:
        import pip
        from pip import main
        pip.main(['install','cython'])
        from Cython.Build import cythonize
        UseCython = True
    except:
        print('Cannot import cython. Setup proceeds withput cython.')
        UseCython = False

# Import build_ext module (to build C code)
try:
    from Cython.Distutils import build_ext
except ImportError:
    from distutils.command import build_ext

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
    Reads a text file and converts it to RST file using pandas.
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

# =======================
# Create Cython Extension
# =======================

def CreateCythonExtension(PackageName,SubPackageNames):
    """
    Creates a cython extention for each of the sub-packages that contain
    ``pyx`` files.

    .. note:

        Only include those subpackages in the input list that have cython's
        *.pyx files. If a sub-package is purely python, it should not be included
        in the input list of this function.

    :param SubPackageNames: A list of subpackages.
    :type SubPackageNames: list(string)

    :return: Cythonized extentions object
    :rtype: dict
    """

    Extensions = []

    # Make extension from each of the sub-package names
    for SubPackageName in SubPackageNames:

        Name = PackageName + '.' + SubPackageName + '.*'
        Sources=[os.path.join('.',PackageName,SubPackageName,'*.pyx')]
        IncludeDirs=[os.path.join('.',PackageName,SubPackageName)]

        # Create an extension
        AnExtension = Extension(
            Name,
            sources=Sources,
            include_dirs=IncludeDirs,
            extra_compile_args=['-O3','-march=native','-fopenmp','-fno-stack-protector','-Wall'],
            extra_link_args=['-fopenmp'],
            define_macros=[('NPY_NO_DEPRECATED_API','NPY_1_7_API_VERSION')],
        )

        # Append
        Extensions.append(AnExtension)

    # Add cython signatures for sphinx
    for extension in Extensions:
        extension.cython_directives = {"embedsignature": True}

    # Cythonize
    CythonizedExtensions = cythonize(
        Extensions,
        build_dir="build",
        include_path=[numpy.get_include(),"."],
        compiler_directives={'boundscheck':False,'cdivision':True,'wraparound':False,'nonecheck':False})

    return CythonizedExtensions

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

    # External Modules
    if UseCython:

        # List of cython sub-package that will be built with cython as extension
        SubPackageNames = [
            'ComputeTraceOfInverse',
            'ComputeLogDeterminant',
            '_LinearAlgebra']

        # Cythonize
        ExternalModules = CreateCythonExtension(PackageName,SubPackageNames)
    else:
        # Package will not be built with cython
        ExternalModules = []

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
        packages = setuptools.find_packages(exclude=['tests.*','tests','examples.*','examples']),
        ext_modules = ExternalModules,
        include_dirs=[numpy.get_include()],
        install_requires = Requirements,
        python_requires = '>=2.7',
        setup_requires = [
            'setuptools',
            'cython',
            'pytest-runner'],
        tests_require = ['pytest-cov'],
        include_package_data=True,
        cmdclass = {'build_ext': build_ext},
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
                'sphinx-automodapi',
                'sphinxcontrib-apidoc'
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
