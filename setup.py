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
import json
import platform
from glob import glob
import subprocess
import codecs
import tempfile
import shutil
from distutils.errors import CompileError, LinkError, DistutilsExecError
import textwrap
import multiprocessing
import re


# ===============
# install package
# ===============

def install_package(package):
    """
    Installs packages using pip.

    Example:

    .. code-block:: python

        >>> install_package('numpy>1.11')

    :param package: Name of package with or without its version pin.
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


# =========================
# get environment variables
# =========================

"""
* To build cython files in source, set ``CYTHON_BUILD_IN_SOURCE`` to ``1``.
* To build for documentation, set ``CYTHON_BUILD_FOR_DOC`` to ``1``.
* To compile with cuda, set ``USE_CUDA`` environment variable.
* To load cuda library at runtime dynamically, set ``CUDA_DYNAMIC_LOADING`` to
  ``1``. When this package wheel is created in *manylinux* environment, the
  cuda's dynamic library files (``libcudart.so``, ``libcublas.so``,
  ``libcusparse.so``) will not bundle to the package, hence the size of the
  wheel remain low. Without the cuda dynamic loading option, the size of the
  wheel package increases to 450MB and cannot be uploaded to PyPi. When the
  cuda dynamic loading is enabled, the end user should install cuda toolkit,
  since the cuda library files are not included in the package anymore, rather,
  they are loaded dynamically at the run time. The downside with the dynamic
  loading is that the user should install the same CUDA major version that the
  package was compiled with.
* To compile for debugging, set ``DEBUG_MODE`` environment variable. This will
  increase the executable size.

::

    # In Unix
    export CYTHON_BUILD_IN_SOURCE=1
    export CYTHON_BUILD_FOR_DOC=1
    export USE_CBLAS=0
    export USE_CUDA=1
    export DEBUG_MODE=1
    export CUDA_DYNAMIC_LOADING=1

    # In Windows
    $env:export CYTHON_BUILD_IN_SOURCE = "1"
    $env:export CYTHON_BUILD_FOR_DOC = "1"
    $env:USE_CBLA = "0"
    $env:USE_CUDA = "1"
    $env:DEBUG_MODE = "1"
    $env:CUDA_DYNAMIC_LOADDING= "1"

    python setup.py install

If you are using ``sudo``, to pass the environment variable, use ``-E`` option:

::

    sudo -E python setup.py install

"""

# If USE_CUDA is set to "1", the package is compiled with cuda lib using nvcc.
use_cuda = False
if 'USE_CUDA' in os.environ and os.environ.get('USE_CUDA') == '1':
    use_cuda = True

# If CUDA_DYNAMIC_LOADING is set to "1", the cuda library is loaded dynamically
cuda_dynamic_loading = False
if 'CUDA_DYNAMIC_LOADING' in os.environ and \
   os.environ.get('CUDA_DYNAMIC_LOADING') == '1':
    cuda_dynamic_loading = True

# If DEBUG_MODE is set to "1", the package is compiled with debug mode.
debug_mode = False
if 'DEBUG_MODE' in os.environ and os.environ.get('DEBUG_MODE') == '1':
    debug_mode = True

# If environment var "CYTHON_BUILD_IN_SOURCE" exists, cython builds *.c files
# in the source code, otherwise in "/build" directory
cython_build_in_source = False
if 'CYTHON_BUILD_IN_SOURCE' in os.environ and \
   os.environ.get('CYTHON_BUILD_IN_SOURCE') == '1':
    cython_build_in_source = True

# If this package is build for the documentation, define the environment
# variable "CYTHON_BUILD_FOR_DOC". By doing so, two things happen:
# 1. The cython source will be generated in source (not in build directory)
# 2. The "linetrace" is added to the cython's compiler derivatives.
cython_build_for_doc = False
if 'CYTHON_BUILD_FOR_DOC' in os.environ and \
   os.environ.get('CYTHON_BUILD_FOR_DOC') == '1':
    cython_build_for_doc = True

# If USE_CBLAS is defined and set to 1, it uses OpenBlas library for dense
# vector and matrix operations. In this case, openblas-dev sld be installed.
use_cblas = False
if 'USE_CBLAS' in os.environ and os.environ.get('USE_CBLAS') == '1':
    use_cblas = True

# If USE_LONG_INT is set to 1, 64-bit integers are used for LongIndexType.
# Otherwise, 32-bit integers are used.
use_long_int = False
if 'USE_LONG_INT' in os.environ and os.environ.get('USE_LONG_INT') == '1':
    use_long_int = True

# If USE_UNSIGNED_LONG_INT is set to 1, unsigned integers are used for the type
# LongIndexType, which doubles the maximum limit of integers. Otherwise, signed
# integers are used.
use_unsigned_long_int = False
if 'USE_UNSIGNED_LONG_INT' in os.environ and \
        os.environ.get('USE_UNSIGNED_LONG_INT') == '1':
    use_unsigned_long_int = True


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
                                   'path or set $CUDA_HOME, or $CUDA_ROOT, '
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

    # For windows, add "x64" or "x86" to the end of lib path
    if sys.platform == "win32":

        # Detect architecture is 64bit or 32bit
        if platform.machine().endswith('64'):
            lib = join(lib, 'x64')
        else:
            lib = join(lib, 'x86')

        if not os.path.exists(lib):
            raise EnvironmentError("The CUDA's lib sub-directory could not " +
                                   "be located in %s." % lib)

    # Get a dictionary of cuda version with keys 'major', 'minor', and 'patch'.
    version = get_cuda_version(home)

    # Output dictionary of set of paths
    cuda = {
        'home': home,
        'nvcc': nvcc,
        'include': include,
        'lib': lib,
        'version': version
    }

    return cuda


# ================
# get cuda version
# ================

def get_cuda_version(cuda_home):
    """
    Gets the version of CUDA library.

    :param cuda_home: The CUDA home paths.
    :type cuda_home: str

    :return: A dictionary with version info containing the keys 'major',
        'minor', and 'patch'.
    :rtype: dict
    """

    version_txt_file = join(cuda_home, 'version.txt')
    version_json_file = join(cuda_home, 'version.json')
    if os.path.isfile(version_txt_file):

        # txt version file is used in CUDA 10 and earlier.
        with open(version_txt_file, 'r') as file:

            # Version_string is like "11.3.1"
            version_string = file.read()

    elif os.path.isfile(version_json_file):

        # json version file is used in CUDA 11 and newer
        with open(version_json_file, 'r') as file:
            info = json.load(file)

            # Version_string is like "11.3.1"
            version_string = info['cuda']['version']

    else:
        # Find cuda version directly by grep-ing include/cuda.h file
        cuda_filename = join(cuda_home, 'include', 'cuda.h')

        # Regex pattern finds a match like "#define CUDA_VERSION 11030"
        regex_pattern = r'^#define CUDA_VERSION.\d+$'
        match = ''

        with open(cuda_filename, 'r') as file:
            for line in file:
                if re.match(regex_pattern, line):
                    match = line
                    break

        if match != '':

            # version_string is like "11030"
            version_string = match.split()[-1]

            # Place a dot to separate major and minor version to parse them
            # later. Here, version_string becomes something like "11.03.0"
            version_string = version_string[:-3] + "." + version_string[-3:]
            version_string = version_string[:-1] + "." + version_string[-1:]

        else:
            error_message_1 = 'Cannot find CUDA "version.txt" or ' + \
                              '"version.json" file in %s. ' % cuda_home
            error_message_2 = 'Cannot find "CUDA_VERSION" in header file %s.' \
                              % cuda_filename
            raise FileNotFoundError(error_message_1 + error_message_2)

    # Convert string to a list of int
    version_string_list = version_string.split(' ')[-1].split('.')
    version_int = [int(v) for v in version_string_list]

    # Output dictionary
    version = {
            'major': None,
            'minor': None,
            'patch': None
    }

    # Fill output dictionary
    if len(version_int) == 0:
        raise ValueError('Cannot detect CUDA major version.')
    else:
        version['major'] = version_int[0]
    if len(version_int) > 1:
        version['minor'] = version_int[1]
    if len(version_int) > 2:
        version['patch'] = version_int[2]

    return version


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
    Sets compiler to treat 'cpp' and 'cu' file extensions differently. Namely:
    1. A 'cpp' file is treated as usual with the default compiler and the same
       compiler and linker flags as before.
    2. For a 'cu' file, the compiler is switched to 'nvcc' with other compiler
       flags that suites GPU machine.

    This function only should be called for 'msvc' compiler.

    .. note::

        This function should be called when ``USE_CUDA`` is enabled.
    """

    self.src_extensions.append('.cu')

    # =======
    # compile
    # =======

    def compile(sources, output_dir=None, macros=None, include_dirs=None,
                debug=0, extra_preargs=None, extra_postargs=None,
                depends=None):
        """
        This method is copied from ``cpython/Lib/distutils/msvccompiler.py``.
        See: github.com/python/cpython/blob/main/Lib/distutils/msvccompiler.py

        This compile method is modified below to allow the ``.cu`` files to be
        compiled with the cuda's nvcc compiler.
        """

        # We altered extra_compile_args (or here, extra_postargs) to be a dict
        # of two keys: 'nvcc' and 'not_nvcc'. Here we extract them.
        extra_postargs_nvcc = extra_postargs['nvcc']
        extra_postargs = extra_postargs['not_nvcc']   # keeping the same name

        if not self.initialized:
            self.initialize()
        compile_info = self._setup_compile(output_dir, macros, include_dirs,
                                           sources, depends, extra_postargs)
        macros, objects, extra_postargs, pp_opts, build = compile_info

        compile_opts = extra_preargs or []
        compile_opts.append('/c')
        if debug:
            compile_opts.extend(self.compile_options_debug)
        else:
            compile_opts.extend(self.compile_options)

        for obj in objects:
            try:
                src, ext = build[obj]
            except KeyError:
                continue
            if debug:
                # pass the full pathname to MSVC in debug mode,
                # this allows the debugger to find the source file
                # without asking the user to browse for it
                src = os.path.abspath(src)

            if ext in self._c_extensions:
                input_opt = "/Tc" + src
            elif ext in self._cpp_extensions:
                input_opt = "/Tp" + src
            elif ext in self._rc_extensions:
                # compile .RC to .RES file
                input_opt = src
                output_opt = "/fo" + obj
                try:
                    self.spawn([self.rc] + pp_opts +
                               [output_opt] + [input_opt])
                except DistutilsExecError as msg:
                    raise CompileError(msg)
                continue
            elif ext in self._mc_extensions:
                # Compile .MC to .RC file to .RES file.
                #   * '-h dir' specifies the directory for the
                #     generated include file
                #   * '-r dir' specifies the target directory of the
                #     generated RC file and the binary message resource
                #     it includes
                #
                # For now (since there are no options to change this),
                # we use the source-directory for the include file and
                # the build directory for the RC file and message
                # resources. This works at least for win32all.
                h_dir = os.path.dirname(src)
                rc_dir = os.path.dirname(obj)
                try:
                    # first compile .MC to .RC and .H file
                    self.spawn([self.mc] +
                               ['-h', h_dir, '-r', rc_dir] + [src])
                    base, _ = os.path.splitext(os.path.basename(src))
                    rc_file = os.path.join(rc_dir, base + '.rc')
                    # then compile .RC to .RES file
                    self.spawn([self.rc] +
                               ["/fo" + obj] + [rc_file])

                except DistutilsExecError as msg:
                    raise CompileError(msg)
                continue
            elif ext in ['.cu']:
                # Adding this elif condition to avoid the else statement below
                pass
            else:
                # how to handle this file?
                raise CompileError("Don't know how to compile %s to %s"
                                   % (src, obj))

            try:
                if ext == '.cu':
                    # Compile with nvcc
                    input_opt = ['-c', src]
                    output_opt = ['-o', obj]

                    # Note: the compile_opts is removed below. All necessary
                    # options for nvcc compiler is in extra_postargs_nvcc
                    self.spawn([cuda['nvcc']] + pp_opts +
                               input_opt + output_opt +
                               extra_postargs_nvcc)
                else:
                    # Compile with msvc
                    output_opt = "/Fo" + obj
                    self.spawn([self.cc] + compile_opts + pp_opts +
                               [input_opt, output_opt] +
                               extra_postargs)

            except DistutilsExecError as msg:
                raise CompileError(msg)

        return objects

    # Replace the previous compile function of distutils.ccompiler with the
    # above modified function. Here, the object ``self`` is ``MVSCCompiler``
    # which is a derived class from ``CCompiler`` in the ``distutils`` package
    # in ``cpython`` package.
    self.compile = compile


# =======================
# check compiler has flag
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
            msvc_compile_args = ['/O2', '/W4', '/openmp']
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
            # We add common flags that work both for gcc and clang
            extra_compile_args += ['-O3', '-fno-stack-protector', '-Wall',
                                   '-Wextra', '-Wundef', '-Wcast-align',
                                   '-Wunreachable-code', '-Wswitch-enum',
                                   '-Wpointer-arith', '-Wcast-align',
                                   '-Wwrite-strings', '-Wsign-compare',
                                   '-pedantic', '-fno-common', '-Wundef']

            # The option '-Wl, ..' will send arguments to the linker. Here,
            # '--strip-all' will remove all symbols from the shared library.
            if not debug_mode:
                extra_compile_args += ['-g0', '-Wl, --strip-all']

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

                    # Assuming this is mac's clang. Add '-fopenmp' through
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
            cuda = locate_cuda()

            # Code generations for various device architectures
            gencodes = ['-gencode', 'arch=compute_35,code=sm_35',
                        '-gencode', 'arch=compute_50,code=sm_50',
                        '-gencode', 'arch=compute_52,code=sm_52',
                        '-gencode', 'arch=compute_60,code=sm_60',
                        '-gencode', 'arch=compute_61,code=sm_61',
                        '-gencode', 'arch=compute_70,code=sm_70',
                        '-gencode', 'arch=compute_75,code=sm_75']

            if cuda['version']['major'] < 11:
                gencodes += \
                        ['-gencode', 'arch=compute_75,code=compute_75']
            else:
                gencodes += \
                        ['-gencode', 'arch=compute_80,code=sm_80',
                         '-gencode', 'arch=compute_86,code=sm_86',
                         '-gencode', 'arch=compute_86,code=compute_86']

            extra_compile_args_nvcc = gencodes + ['--ptxas-options=-v', '-c',
                                                  '--verbose', '--shared',
                                                  '-O3']

            # Adding compiler options
            if sys.platform == 'win32':
                extra_compile_args_nvcc += [
                        '--compiler-options', '-MD',  # Creates shared library
                        '--compiler-options', '-openmp']
            else:
                # There are for linux (macos is not supported in CUDA)
                extra_compile_args_nvcc += [
                        '--compiler-options', '-fPIC',  # only for gcc
                        '--compiler-options', '-fopenmp',
                        '--linker-options', '-lgomp']

            # The option '-Wl, ..' will send arguments to the linker. Here,
            # '--strip-all' will remove all symbols from the shared library.
            if debug_mode:
                extra_compile_args_nvcc += ['-g', '-G']
            else:
                extra_compile_args_nvcc += [
                        '--linker-options', '--strip-all']

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
        # Note: parallel build often fails (especially in windows) since object
        # files are accessed by race condition.
        # if sys.platform != 'win32':
        #     self.parallel = multiprocessing.cpu_count()

        # Modify compiler for cuda
        if use_cuda:
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
# create Extension
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
        creates a separate module extension for each ``pyx`` file.
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
    define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

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

    # Using OpenBlas
    if use_cblas:
        libraries += ['openblas']
        define_macros += [('USE_CBLAS', '1')]

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

            if cuda_dynamic_loading:

                # Link with dl library (dlopen and dlsym). MSVC doesn't need it
                if sys.platform != 'win32':
                    libraries += ['dl']

                # Use source code that uses "extern C" to externally define
                # cuda's API that will be loaded dynamically at the runtime.
                sources += glob(join('.', package_name,
                                     '_cuda_dynamic_loading', '*.cpp'))
                include_dirs += glob(join('.', package_name,
                                          '_cuda_dynamic_loading'))
            else:
                # Do not load cuda dynamically. Rather, bind cuda library so
                # files with the package wheel in manylinux platform.
                libraries += ['cudart', 'cublas', 'cusparse']

            # msvc compiler does not know what to do with the runtime library
            if sys.platform != 'win32':
                runtime_library_dirs += [cuda['lib']]

    # Define macros if they are set as environment variables (see the header
    # file ./imate/_definitions/definitions.h)
    if use_long_int:
        define_macros += [('LONG_INT', '1')]
    if use_unsigned_long_int:
        define_macros += [('UNSIGNED_LONG_INT', '1')]

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
        define_macros=define_macros
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

            export CYTHON_BUILD_IN_SOURCE='1'

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
                CYTHON_BUILD_IN_SOURCE: '1'

    Resolving issue with docstring for documentation:

        To build this package only to generate proper cython docstrings for the
        documentation, set the following environment variable:

        ::

            export CYTHON_BUILD_FOR_DOC='1'

        If the documentation is generated by the github actions, set

        ::

            env:
                CYTHON_BUILD_FOR_DOC: '1'

    .. warning::

        DO NOT USE `linetrace=True` for a production code. Only use linetrace
        to generate the documentation. This is because of serious cython bugs
        caused by linetrace feature, particularly the behavior of ``prange``
        becomes unpredictable since it often halts execution of the program.
    """

    # Add cython signatures for sphinx
    # for extension in extensions:
    #     extension.cython_directives = {"embedsignature": True}

    # Build in source or out of source
    if cython_build_in_source or cython_build_for_doc:
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
    if cython_build_for_doc:
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
# main
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

    extensions.append(create_extension(package_name, 'sample_matrices'))

    extensions.append(create_extension(package_name, 'functions'))

    extensions.append(create_extension(package_name, '_linear_algebra',
                                       other_source_dirs=['_c_basic_algebra']))

    extensions.append(create_extension(package_name, '_c_linear_operator',
                                       other_source_dirs=['_c_basic_algebra']))

    extensions.append(create_extension(package_name,
                                       join('_c_linear_operator', 'tests'),
                                       other_source_dirs=['_c_basic_algebra']))

    extensions.append(create_extension(package_name, '_trace_estimator'))

    extensions.append(create_extension(package_name, '_c_trace_estimator',
                                       other_source_dirs=[
                                           '_c_linear_operator',
                                           '_c_basic_algebra',
                                           '_random_generator',
                                           '_utilities']))

    extensions.append(create_extension(package_name, '_random_generator'))

    extensions.append(create_extension(package_name, 'trace',
                                       other_source_dirs=['functions']))

    extensions.append(create_extension(package_name, 'traceinv',
                                       other_source_dirs=['functions',
                                                          '_c_basic_algebra']))

    extensions.append(create_extension(package_name, 'logdet',
                                       other_source_dirs=['functions']))

    # Cyhton CUDA extensions
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
                                               '_random_generator',
                                               '_cu_linear_operator',
                                               '_c_linear_operator',
                                               '_cuda_utilities',
                                               '_utilities',
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
    description = 'Implicit Matrix Trace Estimator. Computes the trace of ' + \
        'functions of implicit matrices. Accelerated on CPU and ' + \
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
        python_requires='>=3.6',
        setup_requires=[
            'setuptools',
            'wheel',
            'numpy>1.11',
            'scipy>=1.5',
            'cython',
            'pytest-runner'],
        tests_require=[
            'pytest',
            'pytest-cov'],
        include_package_data=True,
        cmdclass={'build_ext': CustomBuildExtension},
        zip_safe=False,  # False: package can be "cimported" by another package
        extras_require={
            ':implementation_name == "cpython"': [
                'matplotlib>=2.0',
                'seaborn'
                ],
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
            'Programming Language :: C++',
            'Programming Language :: Cython',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: Implementation :: CPython',
            'Programming Language :: Python :: Implementation :: PyPy',
            'Environment :: GPU :: NVIDIA CUDA',
            'License :: OSI Approved :: BSD License',
            'Operating System :: POSIX :: Linux',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: MacOS',
            'Natural Language :: English',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    )

    # Setup
    setuptools.setup(**metadata)


# =============
# script's main
# =============

if __name__ == "__main__":
    main(sys.argv)
