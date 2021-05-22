#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.

# =======
# Imports
# =======

from __future__ import print_function
import os
from os.path import join
import sys
from glob import glob
import subprocess
import codecs
import tempfile
import shutil
from distutils.errors import CompileError, LinkError
import textwrap
import multiprocessing


# ===============
# Install Package
# ===============

def install_package(package):
    """
    Installs packages using pip.

    Example:

    .. code-block:: python

        >>> install_package('numpy>1.11')

    :param package: Name of pakcage with or without its version pin.
    :type package: string
    """

    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# =====================
# Import Setup Packages
# =====================

# Import numpy
try:
    import numpy
except ImportError:
    # Install numpy
    install_package('numpy>1.11')
    import numpy

# Check scipy is installed (needed for build, but not required to be imported)
try:
    import scipy                                                    # noqa F401
except ImportError:
    # Install scipy
    install_package('scipy')

# Import setuptools
try:
    import setuptools
    from setuptools.extension import Extension
except ImportError:
    # Install setuptools
    install_package('setuptools')
    import setuptools
    from setuptools.extension import Extension

# Import Cython (to convert pyx to C code)
try:
    from Cython.Build import cythonize
except ImportError:
    # Install Cython
    install_package('cython')
    from Cython.Build import cythonize

# Import build_ext
try:
    from Cython.Distutils import build_ext
except ImportError:
    from distutils.command import build_ext


# ===========================
# Check environment variables
# ===========================

"""
To compile with cuda, set ``USE_CUDA`` environment variable.

::

    # In Unix
    export USE_CUDA=1

    # In Windows
    $env:USE_CUDA = "1"

    python setup.py install

If you are using ``sudo``, to pass the environment variable, use ``-E`` option:

::

    sudo -E python setup.py install

"""

# If USE_CUDA is set to "1", the package is compiled with cuda lib using nvcc.
use_cuda = False
if 'USE_CUDA' in os.environ and os.environ.get('USE_CUDA') == '1':
    use_cuda = True


# ============
# find in path
# ============

def find_in_path(executable_name, path):
    """
    Recursively searches the executable ``executable_name`` in all of the
    directories in the given path, and returns the full path of the executable
    once its first occurrence is found.. If no executable is found, ``None`` is
    returned. This is used to find CUDA's directories.
    """

    for dir in path.split(os.pathsep):
        executable_path = join(dir, executable_name)
        if os.path.exists(executable_path):
            return os.path.abspath(executable_path)
    return None


# ===========
# locate cuda
# ===========

def locate_cuda():
    """
    Finds the executable ``nvcc`` (or ``nvcc.exe`` if windows). If found,
    creates a dictionary of cuda's executable path, include and lib directories
    and home directory. This is used for GPU.
    """

    if not use_cuda:
        raise EnvironmentError('This function should not be called when '
                               '"USE_CUDA" is not set to "1".')

    # List of environment variables to search for cuda
    environs = ['CUDA_HOME', 'CUDA_ROOT', 'CUDA_PATH']
    cuda_found = False

    # nvcc binary
    nvcc_binary_name = 'nvcc'
    if sys.platform == 'win32':
        nvcc_binary_name = nvcc_binary_name + '.exe'

    # Search in each of the possible environment variables, if they exist
    for env in environs:
        if env in os.environ:

            # Home
            home = os.environ[env]
            if not os.path.exists(home):
                continue

            # nvcc binary
            nvcc = join(home, 'bin', nvcc_binary_name)
            if not os.path.exists(nvcc):
                continue
            else:
                cuda_found = True
                break

    # Brute-force search in all path to find nvcc binary
    if not cuda_found:
        nvcc = find_in_path(nvcc_binary_name, os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The "nvcc" binary could not be located '
                                   'located in $PATH. Either add it to '
                                   'path or set $CUDA_HOME, or $CUDA_ROOT'
                                   'or $CUDA_PATH.')

        home = os.path.dirname(os.path.dirname(nvcc))

    # Include directory
    include = join(home, 'include')
    if not os.path.exists(include):
        raise EnvironmentError("The CUDA's include directory could not be " +
                               "located in %s." % include)

    # Library directory
    lib = join(home, 'lib')
    if not os.path.exists(lib):
        lib64 = join(home, 'lib64')
        if not os.path.exists(lib64):
            raise EnvironmentError("The CUDA's lib directory could not be " +
                                   "located in %s or %s." % (lib, lib64))
        lib = lib64

    # Output dictionary of set of paths
    cuda = {
        'home': home,
        'nvcc': nvcc,
        'include': include,
        'lib': lib
    }

    return cuda


# ================================
# customize unix compiler for nvcc
# ================================

def customize_unix_compiler_for_nvcc(self, cuda):
    """
    Sets compiler to treat 'cpp' and 'cu' file extensions differently. Namely:
    1. A 'cpp' file is treated as usual with the default compiler and the same
       compiler and linker flags as before.
    2. For a 'cu' file, the compiler is switched to 'nvcc' with other compiler
       flags that suites GPU machine.

    This function only should be called for 'unix' compiler (``gcc``, `clang``
    or similar). For windows ``msvc`` compiler, this function does not apply.

    .. note::

        This function should be called when ``USE_CUDA`` is enabled.
    """

    self.src_extensions.append('.cu')

    # Backup default compiler to call them later
    default_compiler_so = self.compiler_so
    super = self._compile

    # =======
    # compile
    # =======

    def _compile(obj, src, ext, cc_args, extra_compile_args, pp_opts):
        """
        Define ``_compile`` method to be called before the original
        ``self.compile`` method. This function modifies the dispatch of the
        compiler depend on the source file extension ('cu', or non 'cu' file),
        then calls the original (backed up) compile function.

        Note: ``extra_compile_args_dict`` is a dictionary with two keys
        ``"nvcc"`` and ``"gcc"``. Respectively, the values of each are lists of
        extra_compile_args for nvcc (to compile .cu files) and other compile
        args to compile other files. This dictionary was created in the
        extra_compile_args when each extension is created (see later in this
        script).
        """

        if os.path.splitext(src)[1] == '.cu':

            # Use nvcc for *.cu files.
            self.set_executable('compiler_so', cuda['nvcc'])

            # Use only a part of extra_postargs dictionary with the key "nvcc"
            _extra_compile_args = extra_compile_args['nvcc']

        else:
            # for any other file extension, use the defaukt compiler. Also, for
            # the extra compile args, use args in "gcc" key of extra_postargs
            _extra_compile_args = extra_compile_args['not_nvcc']

        # Pass back to the default compiler
        super(obj, src, ext, cc_args, _extra_compile_args, pp_opts)

        # Return back the previous default compiler to self.compiler_so
        self.compiler_so = default_compiler_so

    self._compile = _compile


# ===================================
# customize windows compiler for nvcc
# ===================================

def customize_windows_compiler_for_nvcc(self, cuda):
    """
    TODO: This function is not yet fully implemented. There is an issue with
    the self.compile of distutil for windows. The issue is that the ``sources``
    argument in ``compile`` method is NOT a single file, rather is a list of
    all files.

    .. note::

        This function should be called when ``USE_CUDA`` is enabled.
    """

    if not self.initialized:
        self.initialize()

    self.src_extensions.append('.cu')

    # Backup default compiler
    # default_compiler_so = self.compiler_so
    default_cc = self.cc
    super = self.compile

    # =======
    # compile
    # =======

    # def compile(obj, src, ext, cc_args, extra_compile_args_dict, pp_opts):
    def compile(sources,
                output_dir=None, macros=None, include_dirs=None, debug=0,
                extra_preargs=None, extra_postargs=None, depends=None):
        """
        Redefine ``_compile`` method to dispatch relevant compiler for each
        file extension. For ``.cu`` files, the ``nvcc`` compiler will be used.

        Note: ``extra_compile_args_dict`` is a dictionary with two keys
        ``"nvcc"`` and ``"gcc"``. Respectively, the values of each are lists of
        extra_compile_args for nvcc (to compile .cu files) and other compile
        args to compile other files. This dictionary was created in the
        extra_compile_args when each extension is created (see later in this
        script).
        """

        # if os.path.splitext(src)[1] == '.cu':
        if True:

            # Use nvcc for *.cu files.
            # self.set_executable('compiler_so', cuda['nvcc'])
            # self.set_executable('cc', cuda['nvcc'])
            self.cc = cuda['nvcc']

            # Use only a part of extra_postargs dictionary with the key "nvcc"
            extra_postargs = extra_postargs['nvcc']

        else:
            # for any other file extension, use the defaukt compiler. Also, for
            # the extra compile args, use args in "gcc" key of extra_postargs
            extra_postargs = extra_postargs['gcc']

        # Pass back to the default compiler
        # super(obj, src, ext, cc_args, extra_compile_args, pp_opts)
        super(sources, output_dir, macros, include_dirs, debug,
              extra_preargs, extra_postargs, depends)

        # Return back the previous default compiler to self.compiler_so
        # self.compiler_so = default_compiler_so
        self.cc = default_cc

    self.compile = compile


# =======================
# Check Compiler Has Flag
# =======================

def check_compiler_has_flag(compiler, compile_flags, link_flags):
    """
    Checks if the C compiler has a given flag. The motivation for this function
    is that:

    * In Linux, the gcc compiler has ``-fopenmp`` flag, which enables compiling
      with OpenMP.
    * In macOS, the clang compiler does not recognize ``-fopenmp`` flag,
      rather, this flag should be passed through the preprocessor using
      ``-Xpreprocessor -fopenmp``.

    Thus, we should know in advance which compiler is employed to provide the
    correct flags. The problem is that in the setup.py script, we cannot
    determine if the compiler is gcc or clang. The closet we can get is to call

    .. code-block:: python

        >>> import distutils.ccompiler
        >>> print(distutils.ccompiler.get_default_compiler())

    In both Linux and macOS, the above line returns ``unix``, and in windows it
    returns ``msvc`` for Microsoft Visual C++. In the case of Linux and macOS,
    we cannot figure which compiler is being used as both outputs are the same.
    The safest solution so far is this function, which compilers a small c code
    with a given ``flag_name`` and checks if it compiles successfully. In case
    of ``unix``, if it compiles with ``-fopenmp``, it is gcc on Linux,
    otherwise it is likely to be the ``clang`` compiler on macOS.

    :param compiler: The compiler object from build_ext.compiler
    :type compiler: build_ext.compiler

    :param compile_flags: A list of compile flags, such as
        ``['-Xpreprocessor','-fopenmp']``
    :type compile_flags: list(string)

    :param link_flags: A list of linker flags, such as
        ``['-Xpreprocessor','-fopenmp']``
    :type link_flags: list(string)
    """

    if "PYODIDE_PACKAGE_ABI" in os.environ:

        # pyodide doesn't support OpenMP
        return False

    compile_success = True
    current_working_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    filename = 'test.c'
    code = "#include <omp.h>\nint main(int argc, char** argv) { return(0); }"

    # Considerations for Microsoft visual C++ compiler
    if compiler.compiler_type == "msvc":
        link_flags = link_flags + ['/DLL']

    # Write a code in temp directory
    os.chdir(temp_dir)
    with open(filename, 'wt') as file_obj:
        file_obj.write(code)

    try:
        # Try to compile
        objects = compiler.compile([filename], extra_postargs=compile_flags)

        try:
            # Try to link
            compiler.link_shared_lib(
                    objects,
                    "testlib",
                    extra_postargs=link_flags)

        except (LinkError, TypeError):
            # Linker was not successful
            compile_success = False

    except CompileError:
        # Compile was not successful
        compile_success = False

    os.chdir(current_working_dir)
    shutil.rmtree(temp_dir)

    return compile_success


# ======================
# Custom Build Extension
# ======================

class CustomBuildExtension(build_ext):
    """
    Customized ``build_ext`` that provides correct compile and linker flags to
    the extensions depending on the compiler and the operating system platform.

    Default compiler names depending on platform:
        * linux: gcc
        * mac: clang (llvm)
        * windows: msvc (Microsoft Visual C++)

    Compiler flags:
        * gcc   : -O3 -march=native -fno-stack-protector -Wall -fopenmp
        * clang : -O3 -march=native -fno-stack-protector -Wall -Xpreprocessor
                  -fopenmp
        * msvc  : /O2 /Wall /openmp

    Linker flags:
        * gcc   : -fopenmp
        * clang : -Xpreproessor -fopenmp -lomp
        * msvc  : (none)

    Usage:

    This class (CustomBuildExtention) is a child of``build_ext`` class. To use
    this class, add it to the ``cmdclass`` by:

    .. code-block: python

        >>> setup(
        ...     ...
        ...     # cmdclass = {'build_ext' : }                    # default
        ...     cmdclass = {'build_ext' : CustomBuildExtention}  # this class
        ...     ...
        ... )
    """

    # ---------------
    # Build Extension
    # ---------------

    def build_extensions(self):
        """
        Specifies compiler and linker flags depending on the compiler.

        .. warning::

            DO NOT USE '-march=native' flag. By using this flag, the compiler
            optimizes the instructions for the native machine of the build time
            and the executable will not be backward compatible to older CPUs.
            As a result, the package will not be distributable on other
            machines as the installation with the binary wheel crashes on other
            machines with this error:

                'illegal instructions (core dumped).'

            An alternative optimization flag is '-mtune=native', which is
            backward compatible and the package can be installed using wheel
            binary file.
        """

        # Get compiler type. This is "unix" (linux, mac) or "msvc" (windows)
        compiler_type = self.compiler.compiler_type

        # Initialize flags
        extra_compile_args = []
        extra_link_args = []

        if compiler_type == 'msvc':

            # This is Microsoft Windows Visual C++ compiler
            msvc_compile_args = ['/O2', '/Wall', '/openmp']
            msvc_link_args = []
            msvc_has_openmp_flag = check_compiler_has_flag(
                    self.compiler,
                    msvc_compile_args,
                    msvc_link_args)

            if msvc_has_openmp_flag:

                # Add flags
                extra_compile_args += msvc_compile_args
                extra_link_args += msvc_link_args

            else:

                # It does not seem msvc accept -fopenmp flag.
                raise RuntimeError(textwrap.dedent(
                    """
                    OpenMP does not seem to be available on %s compiler.
                    """ % compiler_type))

        else:

            # The compile_type is 'unix'. This is either linux or mac.
            # We add common flags that work both for gcc and mac's clang
            extra_compile_args += ['-O3', '-fno-stack-protector', '-Wall']

            # Assume compiler is gcc (we do not know yet). Check if the
            # compiler accepts '-fopenmp' flag. Note: clang in mac does not
            # accept this flag alone, but gcc does.
            gcc_compile_args = ['-fopenmp']
            gcc_link_args = ['-fopenmp']
            gcc_has_openmp_flag = check_compiler_has_flag(
                    self.compiler,
                    gcc_compile_args,
                    gcc_link_args)

            if gcc_has_openmp_flag:

                # Assuming this is gcc. Add '-fopenmp' safely.
                extra_compile_args += gcc_compile_args
                extra_link_args += gcc_link_args

            else:

                # Assume compiler is clang (we do not know yet). Check if
                # -fopenmp can be passed through preprocessor. This is how
                # clang compiler accepts -fopenmp.
                clang_compile_args = ['-Xpreprocessor', '-fopenmp']
                clang_link_args = ['-Xpreprocessor', '-fopenmp', '-lomp']
                clang_has_openmp_flag = check_compiler_has_flag(
                        self.compiler,
                        clang_compile_args,
                        clang_link_args)

                if clang_has_openmp_flag:

                    # Assuming this is mac's clag. Add '-fopenmp' through
                    # preprocessor
                    extra_compile_args += clang_compile_args
                    extra_link_args += clang_link_args

                else:

                    # It doesn't seem either gcc or clang accept -fopenmp flag.
                    raise RuntimeError(textwrap.dedent(
                        """
                        OpenMP does not seem to be available on %s compiler.
                        """ % compiler_type))

        # Modify compiler flags for cuda
        if use_cuda:

            # Compile flags for nvcc
            if sys.platform == 'win32':
                extra_compile_args_nvcc = ['/Ox']
            else:
                extra_compile_args_nvcc = ['-arch=sm_35', '--ptxas-options=-v',
                                           '-c', '--compiler-options', '-fPIC',
                                           '-O3', '--verbose']

            # Redefine extra_compile_args list to be a dictionary
            extra_compile_args = {
                'not_nvcc': extra_compile_args,
                'nvcc': extra_compile_args_nvcc
            }

        # Add the flags to all extensions
        for ext in self.extensions:
            ext.extra_compile_args = extra_compile_args
            ext.extra_link_args = extra_link_args

        # Parallel compilation (can also be set via build_ext -j or --parallel)
        # Note: parallel build fails in windows since object files are accessed
        # by race condition.
        if sys.platform != 'win32':
            self.parallel = multiprocessing.cpu_count()

        # Modify compiler for cuda
        if use_cuda:
            cuda = locate_cuda()

            if sys.platform == 'win32':
                customize_windows_compiler_for_nvcc(self.compiler, cuda)
            else:
                customize_unix_compiler_for_nvcc(self.compiler, cuda)

        # Remove warning: command line option '-Wstrict-prototypes' is valid
        # for C/ObjC but not for C++
        try:
            if '-Wstrict-prototypes' in self.compiler.compiler_so:
                self.compiler.compiler_so.remove('-Wstrict-prototypes')
        except (AttributeError, ValueError):
            pass

        # Call parent class to build
        build_ext.build_extensions(self)


# =========
# Read File
# =========

def read_file(Filename):
    """
    Reads a file with latin codec.
    """

    with codecs.open(Filename, 'r', 'latin') as file_obj:
        return file_obj.read()


# ================
# Read File to RST
# ================

def read_file_to_rst(filename):
    """
    Reads a markdown text file and converts it to RST file using pandas.
    """

    try:
        import pypandoc
        rstname = "{}.{}".format(os.path.splitext(filename)[0], 'rst')
        pypandoc.convert(
                filename,
                'rst',
                format='markdown',
                outputfile=rstname)

        with open(rstname, 'r') as f:
            rststr = f.read()
        return rststr
    except ImportError:
        return read_file(filename)


# ======================
# does cuda source exist
# ======================

def does_cuda_source_exist(sources):
    """
    Checks files extensions in a list of files in the ``sources``. If a file
    extension ``.cu`` is found, returns ``True``, otherwise returns ``False``.

    :param sources: A list of file names.
    :type sources: list
    """

    has_cuda_source = False
    for source in sources:
        file_extension = os.path.splitext(source)[1]
        if file_extension == '.cu':
            has_cuda_source = True
            break

    return has_cuda_source


# ================
# Create Extension
# ================

def create_extension(
        package_name,
        subpackage_name,
        other_source_dirs=None,
        other_source_files=None,
        other_include_dirs=None):
    """
    Creates an extension for each of the sub-packages that contain
    ``.pyx`` files.

    How to add a new cython sub-package or module:

    In the :func:`main` function, add the name of cython sub-packages or
    modules in the `subpackages_names` list. Note that only include those
    sub-packages in the input list that have cython's *.pyx files. If a
    sub-package is purely python, it should not be included in that list.

    Compile arguments:

        The compiler and linker flags (``extra_compile_args`` and
        ``extra_link_args``) are set to an empty list. We will fill them using
        ``CustomBuildExtension`` class, which depend on the compiler and
        platform, it sets correct flags.

    Parameters:

    :param package_name: Name of the main package
    :type package_name: string

    :param subpackage_name: Name of the subpackage to build its extension.
        In the package_name/subpackage_name directory, all ``pyx``, ``c``,
        ``cpp``, and ``cu`` files (if use_cuda is True) will be added to the
        extension. If there are additional ``c``, ``cpp`` and ``cu`` source
        files in other directories beside the soubpackage directory, use
        ``other_source_dirs`` argument.
    :type subpackage_name: string

    :param other_source_dirs: To add any other source files (only ``c``,
        ``cpp``, and ``cu``, but not ``pyx``) that are outside of the
        subpackage directory, use this argument. The ``other_source_dirs`` is
        a list of directories to include their path to ``include_dir`` and
        to add all of the ``c``, ``cpp``, and ``cu`` files to ``sources``.
        Note that the ``pyx`` files in these other directories will not be
        added. To add a ``pyx`` file, use ``subpackage_name`` argument, which
        creates a separate moule extension for each ``pyx`` file.
    :type other_source_dirs: list(string)

    :param other_source_files: A list of fullpath names of other source files
        (only ``c``, ``cpp``, and ``cu``), that are not in the
        ``subpackage_name`` directory, neither are in the ``other_source_dirs``
        directory.
    :type other_source_files: list(string)

    :param other_include_dirs: A list of fullpath directories of other source
        files, such as other ``*.cpp`` or ``*.cu`` that are not in the
        directories of ``subpackage_name`` and ``other_source_dirs`` arguments.
    :type other_include_dirs: list(string)

    :return: Cythonized extensions object
    :rtype: dict
    """

    # Check directory
    subpackage_dir_name = join(package_name, subpackage_name)
    if not os.path.isdir(subpackage_dir_name):
        raise ValueError('Directory %s does not exists.' % subpackage_dir_name)

    # Wether to create a module for each pyx file or a module for all cpp files
    pyx_sources = join('.', package_name, subpackage_name, '*.pyx')
    if glob(pyx_sources) != []:

        # Creates a directory of modules for each pyx file
        name = package_name + '.' + subpackage_name + '.*'
        sources = [pyx_sources]
    else:
        # Create one so file (not a directory) for all source files (cpp, etc)
        name = package_name + '.' + subpackage_name
        sources = []

    sources += glob(join('.', package_name, subpackage_name, '*.cpp'))

    if use_cuda:
        sources += glob(join('.', package_name, subpackage_name, '*.cu'))

    include_dirs = [join('.', package_name, subpackage_name)]
    extra_compile_args = []  # will be filled by CustomBuildExtension class
    extra_link_args = []     # will be filled by CustomBuildExtension class
    library_dirs = []
    runtime_library_dirs = []
    libraries = []
    language = 'c++'

    # Include any additional source files
    if other_source_files is not None:

        # Check source files exist
        for source_file in other_source_files:
            if not os.path.isfile(source_file):
                raise ValueError('File %s does not exists.' % source_file)

        sources += other_source_files

    # Include any additional include directories
    if other_include_dirs is not None:

        # Check if directories exist
        for include_dir in other_include_dirs:
            if not os.path.isdir(include_dir):
                raise ValueError('Directory %s does not exists.' % include_dir)

        include_dirs += other_include_dirs

    # Glob entire source c, cpp and cufiles in other source directories
    if other_source_dirs is not None:

        for other_source_dir in other_source_dirs:

            # Check directory exists
            other_source_dirname = join(package_name, other_source_dir)
            if not os.path.isdir(other_source_dirname):
                raise ValueError('Directory %s does not exists.'
                                 % other_source_dirname)

            sources += glob(join(other_source_dirname, '*.c'))
            sources += glob(join(other_source_dirname, '*.cpp'))
            include_dirs += join(other_source_dirname)

            if use_cuda:
                sources += \
                    glob(join(other_source_dirname, '*.cu'))

    # Add cuda info
    if use_cuda:
        cuda = locate_cuda()

        # Check if any '*.cu' files exists in the sources
        has_cuda_source = does_cuda_source_exist(sources)

        # Add cuda libraries only if a cuda source exists. This is necessary
        # to run the non-cuda modules on machines without cuda installed.
        if has_cuda_source:
            include_dirs += [cuda['include']]
            library_dirs += [cuda['lib']]
            runtime_library_dirs += [cuda['lib']]
            libraries += ['cudart', 'cublas', 'cusparse']

    # Create an extension
    extension = Extension(
        name,
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs=runtime_library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language=language,
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    )

    return extension


# ====================
# cythonize extensions
# ====================

def cythonize_extensions(extensions):
    """
    Resolving issue with conda-build:

        If the code is build using "conda-build" to be uploaded on anaconda
        cloud, consider setting this environmental variable:

        ::

            export CYTHON_BUILD_IN_SOURCE='true'

        By setting so, this function sets the build directory ``build_dir``
        to ``None``, which then the ``*.c`` files will be written in the source
        code alongside with the source (where ``*.pyx`` are). If this
        environmental variable does not exist, this function sets ``build_dir``
        to ``build``directory , which builds the cython files outside of the
        source code.

        Why this matters?

        Apparently, ``conda-build`` has a bug and emerges if the following two
        conditions are met:

        1. conda builds over multiple variants of the conda recipe (by defining
           jinja variables in the file ``/conda/conda_build_config.yaml``),
           such as defining multiple python versions and then using the jinja
           variable ``{{ python }}`` in ``/conda/meta.yaml``.

        2. Cython builds the ``*.c`` files outside of the source. That is, when
           we set ``build_dir`` to anything but its default in
           ``cythonize(build_dir='some_directory')``. When  ``build_dir`` is
           set to ``None``, cython builds the ``*.c`` files in source. But when
           ``build_dir`` is set to a directory, the ``*.c`` files will be
           written there.

        Now, when the two above are set, ``conda-build`` faces a race condition
        to build multiple versions of the package for variants of python
        versions (if this is the variant variable), and crashes. Either
        conda-build should build only one variant of the ``meta.yaml`` file
        (that is, defining no variant in ``/conda/conda_build.config.yaml``),
        or cython should build in source.

        To resolve this, set ``CYTHON_BUILD_IN_SOURCE`` whenever the package is
        build with build-conda. Also see the github action
        ``./github/workflow/deploy-conda/yaml``

        ::

            env:
                CYTHON_BUILD_IN_SOURCE: 'true'

    Resolving issue with docstring for documentation:

        To build this package only to generate proper cython docstrings for the
        documentation, set the following environment variable:

        ::

            export CYTHON_BUILD_FOR_DOC='true'

        If the documentation is generated by the github actions, set

        ::

            env:
                CYTHON_BUILD_FOR_DOC: 'true'

    .. warning::

        DO NOT USE `linetrace=True` for a production code. Only use linetrace
        to generate the documentation. This is because of serious cython bugs
        caused by linetrace feature, particularly the behavior of ``prange``
        becomes unpredictable since it often halts execution of the program.
    """

    # Add cython signatures for sphinx
    # for extension in extensions:
    #     extension.cython_directives = {"embedsignature": True}

    # If environment var "CYTHON_BUILD_IN_SOURCE" exists, cython builds *.c
    # files in the source code, otherwise in "/build" directory
    cython_build_in_source = os.environ.get('CYTHON_BUILD_IN_SOURCE', None)

    # If this package is build for the documentation, define the environment
    # variable "CYTHON_BUILD_FOR_DOC". By doing so, two things happen:
    # 1. The cython source will be generated in source (not in build directory)
    # 2. The "linetrace" is added to the cython's compiler derivatives.
    cython_build_for_doc = os.environ.get('CYTHON_BUILD_FOR_DOC', None)

    # Build in source or out of source
    if bool(cython_build_in_source) or bool(cython_build_for_doc):
        cython_build_dir = None    # builds *.c in source alongside *.pyx files
    else:
        cython_build_dir = 'build'

    # Compiler derivatives
    compiler_derivatives = {
            'boundscheck': False,
            'cdivision': True,
            'wraparound': False,
            'nonecheck': False,
            'embedsignature': True,
    }

    # Build for doc or not
    if bool(cython_build_for_doc):
        compiler_derivatives['linetrace'] = True

    # Cythonize
    cythonized_extensions = cythonize(
        extensions,
        build_dir=cython_build_dir,
        include_path=[numpy.get_include(), "."],
        language_level="3",
        nthreads=multiprocessing.cpu_count(),
        compiler_directives=compiler_derivatives
    )

    return cythonized_extensions


# ====
# Main
# ====

def main(argv):

    directory = os.path.dirname(os.path.realpath(__file__))
    package_name = "imate"

    # Version
    version_dummy = {}
    version_file = join(directory, package_name, '__version__.py')
    exec(open(version_file, 'r').read(), version_dummy)
    version = version_dummy['__version__']
    del version_dummy

    # Author
    author_file = join(directory, 'AUTHORS.txt')
    author = open(author_file, 'r').read().rstrip()

    # Requirements
    requirements_filename = join(directory, "requirements.txt")
    requirements_file = open(requirements_filename, 'r')
    requirements = [i.strip() for i in requirements_file.readlines()]

    # ReadMe
    readme_file = join(directory, 'README.rst')
    long_description = open(readme_file, 'r').read()

    # Cyhton cpp extentions
    extensions = []
    extensions.append(create_extension(package_name, 'generate_matrix'))
    extensions.append(create_extension(package_name, 'functions'))
    extensions.append(create_extension(package_name, '_linear_algebra'))
    extensions.append(create_extension(package_name, '_c_linear_operator',
                                       other_source_dirs=['_c_basic_algebra']))
    extensions.append(create_extension(package_name,
                                       join('_c_linear_operator', 'tests'),
                                       other_source_dirs=['_c_basic_algebra']))
    extensions.append(create_extension(package_name, '_trace_estimator'))
    extensions.append(create_extension(package_name, '_c_trace_estimator',
                                       other_source_dirs=['_c_linear_operator',
                                                          '_c_basic_algebra']))
    extensions.append(create_extension(package_name, 'traceinv',
                                       other_source_dirs=['functions']))
    extensions.append(create_extension(package_name, 'logdet',
                                       other_source_dirs=['functions']))

    # Cyhton CUDA extentions
    if use_cuda:
        extensions.append(create_extension(package_name, '_cuda_utilities'))
        extensions.append(create_extension(package_name, '_cu_linear_operator',
                                           other_source_dirs=[
                                               '_c_linear_operator',
                                               '_cuda_utilities',
                                               '_cu_basic_algebra',
                                               '_c_basic_algebra']))

        extensions.append(create_extension(package_name, '_cu_trace_estimator',
                                           other_source_dirs=[
                                               '_c_trace_estimator',
                                               '_cu_linear_operator',
                                               '_c_linear_operator',
                                               '_cuda_utilities',
                                               '_cu_basic_algebra',
                                               '_c_basic_algebra']))

        extensions.append(create_extension(package_name,
                                           join(
                                               '_cu_linear_operator', 'tests'),
                                           other_source_dirs=[
                                               '_cu_basic_algebra']))

    # Cythonize
    external_modules = cythonize_extensions(extensions)

    # Description
    description = 'Implicit matrix trace estimator (IMATE). Computes the ' + \
        'trace of function of linear operators and implicit matrices of ' + \
        'possible very large size. Accelerated both on CPU and ' + \
        'CUDA-capable GPU devices.'

    # URLs
    url = 'https://github.com/ameli/imate'
    download_url = url + '/archive/main.zip'
    documentation_url = url + '/blob/main/README.rst'
    tracker_url = url + '/issues'

    # Inputs to setup
    metadata = dict(
        name=package_name,
        version=version,
        author=author,
        author_email='sameli@berkeley.edu',
        description=description,
        long_description=long_description,
        long_description_content_type='text/x-rst',
        keywords="""matrix-computations matrix-inverse interpolation-techniques
                cholesky-decomposition randomized-algorithms lanczos-iteration
                parameter-estimation radial-basis-function polynomial-bases
                orthogonal-polynomials cross-validation""",
        url=url,
        download_url=download_url,
        project_urls={
            "Documentation": documentation_url,
            "Source": url,
            "Tracker": tracker_url,
        },
        platforms=['Linux', 'OSX', 'Windows'],
        packages=setuptools.find_packages(exclude=[
            'tests.*',
            'tests',
            'examples.*',
            'examples']
        ),
        ext_modules=external_modules,
        include_dirs=[numpy.get_include()],
        install_requires=requirements,
        python_requires='>=2.7',
        setup_requires=[
            'setuptools',
            'numpy>1.11',
            'scipy>=1.2.3',
            'cython',
            'pytest-runner'],
        tests_require=[
            'pytest',
            'pytest-cov'],
        include_package_data=True,
        cmdclass={'build_ext': CustomBuildExtension},
        zip_safe=False,    # the package can run out of an .egg file
        extras_require={
            'extra': [
                'scikit-sparse',
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
            ]
        },
        classifiers=[
            'Programming Language :: Cython',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: Implementation :: CPython',
            'Programming Language :: Python :: Implementation :: PyPy',
            'License :: OSI Approved :: MIT License',
            'Operating System :: POSIX :: Linux',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: MacOS',
            'Natural Language :: English',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
    )

    # Setup
    setuptools.setup(**metadata)


# =============
# Script's Main
# =============

if __name__ == "__main__":
    main(sys.argv)
