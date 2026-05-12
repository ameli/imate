# SPDX-FileCopyrightText: Copyright 2022, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

from .c_get_config cimport is_use_cblas, is_use_mkl, is_use_openmp, \
        is_use_loop_unrolling, is_use_cuda, is_cuda_dynamic_loading, \
        is_debug_mode, is_cython_build_in_source, is_cython_build_for_doc, \
        is_use_long_int, is_use_unsigned_long_int


# ==========
# get config
# ==========

def get_config(key=None):
    """
    Returns the definitions used in the compile-time of the package.

    Parameters
    ----------

    key : str, default=None
        A string with one of the following values:
        
        * ``'use_cblas'``: inquiries if the package is compiled with OpenBLAS
          support.
        * ``'use_mkl'``: inquiries if the package is compiled with MKL support.
        * ``'use_openmp'``: inquiries if the package is compiled with OpenMP
          support.
        * ``'use_loop_unrolling'``: inquiries if the package is compiled with
          loop unrolling (also known as loop unwinding).
        * ``'use_cuda'``: inquiries if the package is compiled with CUDA
          support.
        * ``'cuda_dynamic_loading'``: inquiries if the package is compiled with
          enabling the dynamic loading of CUDA libraries.
        * ``'debug_mode'``: inquiries if the package is compiled with the
          debugging mode enabled.
        * ``'cython_build_in_source'``: inquiries if the Cython source files
          were generated in the source directory during compilation.
        * ``'cython_build_for_doc'``: inquiries if the docstring for Cython
          functions are generated for the purpose of Sphinx documentation.
        * ``'use_long_int'``: inquiries if the 64-bit long integer type is
          used for array indices at compile time.
        * ``'use_unsigned_long_int'``: inquiries if unsigned 64-bit integer
          type is used for array indices at compile time.

        If `None`, the full list of all above configurations is returned. 

    Returns
    -------

    config : dict
        If a ``key`` input argument is given, a boolean value corresponding to
        the status of the key is returned. If ``key`` is set to `None` or no
        ``key`` is given, a dictionary with all the above keys is returned.

    See Also
    --------

    imate.info

    Notes
    -----

    To configure the compile-time definitions, export either of these
    variables and set them to ``0`` or ``1`` as applicable:

        .. tab-set::

            .. tab-item:: UNIX
                :sync: unix

                .. prompt:: bash

                    export USE_CBLAS=1
                    export USE_MKL=0
                    export USE_OPENMP=1
                    export USE_LOOP_UNROLLING=1
                    export USE_CUDA=1
                    export CUDA_DYNAMIC_LOADING=1
                    export DEBUG_MODE=1
                    export CYTHON_BUILD_IN_SOURCE=1
                    export CYTHON_BUILD_FOR_DOC=1
                    export USE_LONG_INT=1
                    export USE_UNSIGNED_LONG_INT=1

            .. tab-item:: Windows (Powershell)
                :sync: win

                .. prompt:: powershell

                    $env:USE_CBLAS = "1"
                    $env:USE_MKL = "0"
                    $env:USE_OPENMP = "1"
                    $env:USE_LOOP_UNROLLING = "1"
                    $env:USE_CUDA = "1"
                    $env:CUDA_DYNAMIC_LOADING = "1"
                    $env:DEBUG_MODE = "1"
                    $env:CYTHON_BUILD_IN_SOURCE = "1"
                    $env:CYTHON_BUILD_FOR_DOC = "1"
                    $env:USE_LONG_INT = "1"
                    $env:USE_UNSIGNED_LONG_INT = "1"

    Examples
    --------

    .. code-block:: python

        >>> from imate import get_config

        >>> # Using a config key
        >>> get_config('use_cuda')
        True

        >>> # Using no key, results in returning all config
        >>> get_config()
        {
            'use_cblas': False,
            'use_mkl': False,
            'use_openmp': True,
            'use_loop_unrolling': True,
            'use_cuda': True,
            'cuda_dynamic_loading': True,
            'debug_mode': False,
            'cython_build_in_source': False,
            'cython_build_for_doc': False,
            'use_long_int': False,
            'use_unsigned_long_int': False,
        }
    """

    if key is None:
        config = {
            'use_cblas': bool(is_use_cblas()),
            'use_mkl': bool(is_use_mkl()),
            'use_openmp': bool(is_use_openmp()),
            'use_loop_unrolling': bool(is_use_loop_unrolling()),
            'use_cuda': bool(is_use_cuda()),
            'cuda_dynamic_loading': bool(is_cuda_dynamic_loading()),
            'debug_mode': bool(is_debug_mode()),
            'cython_build_in_source': bool(is_cython_build_in_source()),
            'cython_build_for_doc': bool(is_cython_build_for_doc()),
            'use_long_int': bool(is_use_long_int()),
            'use_unsigned_long_int': bool(is_use_unsigned_long_int()),
        }
        return config
    elif key == 'use_cblas':
        return bool(is_use_cblas())
    elif key == 'use_mkl':
        return bool(is_use_mkl())
    elif key == 'use_openmp':
        return bool(is_use_openmp())
    elif key == 'use_loop_unrolling':
        return bool(is_use_loop_unrolling())
    elif key == 'use_cuda':
        return bool(is_use_cuda())
    elif key == 'cuda_dynamic_loading':
        return bool(is_cuda_dynamic_loading())
    elif key == 'debug_mode':
        return bool(is_debug_mode())
    elif key == 'cython_build_in_source':
        return bool(is_cython_build_in_source())
    elif key == 'cython_build_for_doc':
        return bool(is_cython_build_for_doc())
    elif key == 'use_long_int':
        return bool(is_use_long_int())
    elif key == 'use_unsigned_long_int':
        return bool(is_use_unsigned_long_int())
    else:
        raise ValueError('Invalid "key".')
