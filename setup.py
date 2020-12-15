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

# =======================
# Check Compiler Has Flag
# =======================

def CheckCompilerHasFlag(compiler,flagname):
    """
    Checks if the C compiler has the given flag. The motivation for this function is that
    
    * In linux, the gcc compiler has the '-fopenmp' flag, which enables compiling with OpenMP.
    * In MacOS, the clang compiler does not recognize '-fopenmp' flag, rather, it should be passed through the
      preprocessor using '-Xpreprocessor -fopenmp -lomp'.

    Thus, we should know in advance which compiler is used to provide correct flags.
    The problem is that in the setup.py script, we cannot figure determine the compiler is gcc or clang. 
    The closet we can get is to call 
        
    .. code-block:: python

        >>> import distutils.ccompiler
        >>> print(distutils.ccompiler.get_default_compiler())

    In linux and mac, the above line yields 'unix', which again we cannot figure which compiler is being used.
    The safest work-around is this function, which compilers a simple c code with a given flagname 
    (such as '-fopenmp') and checks if it compiles. If yes, it is gcc, or not, it is clang.
    """
    import tempfile
    from distutils.errors import CompileError
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except CompileError:
            return False
    return True

# ======================
# Custom Build Extension
# ======================

class CustomBuildExtension(build_ext):
    """
    Customized build_ext that provides correct compile and linker flags to the extensions
    depending on the compiler and the platform.
    """

    # ---------------
    # Build Extension
    # ---------------

    def build_extensions(self):

        # Get compiler type (this is either "unix" (in linux and mac) or "msvc" in windows)
        CompilerType = self.compiler.compiler_type

        # Initialize flags
        ExtraCompileArgs = []
        ExtraLinkArgs = []

        if CompilerType == 'unix':
            
            # This is either linux or mac. We add flags that work both for gcc and mac's clang
            ExtraCompileArgs += ['-O3','-march=native','-fno-stack-protector','-Wall']

            # Check if the compiler accepts '-fopenmp' flag (clang in mac does not, but gcc does)
            HasOpenMPFlag = CheckCompilerHasFlag(self.compiler,'-fopenmp')

            if HasOpenMPFlag:

                # This is gcc. Add '-fopenmp' safely.
                ExtraCompileArgs += ['-fopenmp']
                ExtraLinkArgs += ['-fopenmp']

            else:

                # This is mac's clag. Add '-fopenmp' through preprocessor
                ExtraCompileArgs += ['-Xpreprocessor','-fopenmp']
                ExtraLinkArgs += ['-Xpreprocessor','-fopenmp','-lomp']

        elif CompilerType == 'msvc':

            # This is Microsoft Windows Visual C++ compiler
            ExtraCompileArgs += ['/O2','/Wall','/openmp']
                

        print('NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN')
        print(CompilerType)
        print(HasOpenMPFlag)
        print(ExtraCompileArgs)
        print(ExtraLinkArgs)
        print('NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN')

        # Add the flags to all extensions
        for ext in self.extensions:
            ext.extra_compile_args = ExtraCompileArgs
            ext.extra_link_args = ExtraLinkArgs

        # Cal parent class to build
        build_ext.build_extensions(self)

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

    import distutils.ccompiler
    print('QQQQQQQQQQQQQQQQ')
    print(distutils.ccompiler.get_default_compiler())
    print('QQQQQQQQQQQQQQQQ')


    Extensions = []

    # Make extension from each of the sub-package names
    for SubPackageName in SubPackageNames:

        Name = PackageName + '.' + SubPackageName + '.*'
        Sources  =[os.path.join('.',PackageName,SubPackageName,'*.pyx')]
        IncludeDirs = [os.path.join('.',PackageName,SubPackageName)]
        ExtraCompileArgs = ['-O3','-march=native','-fno-stack-protector','-Wall']
        ExtraLinkArgs = [] 

        if sys.platform == 'darwin':
            ExtraCompileArgs += ['-Xpreprocessor','-fopenmp']
            ExtraLinkArgs += ['-Xpreprocessor','-fopenmp','-lomp']
        else:
            ExtraCompileArgs += ['-fopenmp']
            ExtraLinkArgs += ['-fopenmp']

        # Create an extension
        AnExtension = Extension(
            Name,
            sources=Sources,
            include_dirs=IncludeDirs,
            extra_compile_args=ExtraCompileArgs,
            extra_link_args=ExtraLinkArgs,
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
        # cmdclass = {'build_ext': build_ext},
        cmdclass = {'build_ext': CustomBuildExtension},
        zip_safe=False,
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
