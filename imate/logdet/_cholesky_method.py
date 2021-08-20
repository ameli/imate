# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

import time
import numpy
import scipy
import scipy.linalg
from scipy.sparse import isspmatrix
import multiprocessing
from .._linear_algebra.matrix_utilities import get_data_type_name, get_nnz, \
        get_density

try:
    from sksparse.cholmod import cholesky as sk_cholesky
    suitesparse_installed = True
except ImportError:
    suitesparse_installed = False

from .._linear_algebra import sparse_cholesky
from ..__version__ import __version__

# Test
suitesparse_installed = False


# ===============
# cholesky method
# ===============

def cholesky_method(A, exponent=1.0, cholmod=None):
    """
    Computes log-determinant using Cholesky decomposition.

    This function is essentially a wrapper for the Choleksy function of the
    scipy and scikit-sparse packages and primarily used for testing and
    comparison (benchmarking)  against the randomized methods that are
    implemented in this package.

    :param A: A full-rank matrix.
    :type A: numpy.ndarray

    The log-determinant is computed from the Cholesky decomposition
    :math:`\\mathbf{A} = \\mathbf{L} \\mathbf{L}^{\\intercal}` as

    .. math::

        \\log | \\mathbf{A} | =
        2 \\mathrm{trace}( \\log \\mathrm{diag}(\\mathbf{L})).

    .. note::
        This function uses the `Suite Sparse
        <https://people.engr.tamu.edu/davis/suitesparse.html>`_
        package to compute the Cholesky decomposition. See the
        :ref:`installation <InstallScikitSparse>`.

    The result is exact (no approximation) and should be used as benchmark to
    test other methods.
    """

    # Check input arguments
    check_arguments(A, exponent, cholmod)

    # Determine to use Sparse
    sparse = False
    if scipy.sparse.isspmatrix(A):
        sparse = True

    init_tot_wall_time = time.perf_counter()
    init_cpu_proc_time = time.process_time()

    if exponent == 0:

        # determinant is 1. Logdet is zero
        trace = 0.0

    else:

        # Compute logdet of A without the exponent
        if sparse:

            # Sparse matrix
            if suitesparse_installed and cholmod is not False:
                # Use Suite Sparse
                Factor = sk_cholesky(A)
                trace = Factor.logdet()
            else:
                # Use scipy
                diag_L = sparse_cholesky(
                        A, diagonal_only=True).astype(numpy.complex128)
                logdet_L = numpy.real(numpy.sum(numpy.log(diag_L)))
                trace = 2.0*logdet_L

        else:

            # Dense matrix. Use scipy
            L = scipy.linalg.cholesky(A, lower=True)
            diag_L = numpy.diag(L).astype(numpy.complex128)
            logdet_L = numpy.real(numpy.sum(numpy.log(diag_L)))
            trace = 2.0*logdet_L

        # Taking into account of the exponent
        trace = trace*exponent

    tot_wall_time = time.perf_counter() - init_tot_wall_time
    cpu_proc_time = time.process_time() - init_cpu_proc_time

    # Determine if suitesparse was used
    if suitesparse_installed and cholmod is not False and sparse:
        cholmod_used = True
    else:
        cholmod_used = False

    # Dictionary of output info
    info = {
        'matrix':
        {
            'data_type': get_data_type_name(A),
            'exponent': exponent,
            'size': A.shape[0],
            'sparse': isspmatrix(A),
            'nnz': get_nnz(A),
            'density': get_density(A),
            'num_inquiries': 1
        },
        'device':
        {
            'num_cpu_threads': multiprocessing.cpu_count()
        },
        'time':
        {
            'tot_wall_time': tot_wall_time,
            'alg_wall_time': tot_wall_time,
            'cpu_proc_time': cpu_proc_time,
        },
        'solver':
        {
            'version': __version__,
            'method': 'cholesky',
            'cholmod_used': cholmod_used
        }
    }

    return trace, info


# ===============
# check arguments
# ===============

def check_arguments(A, exponent, cholmod):
    """
    Checks the type and value of the parameters.
    """

    # Check A
    if (not isinstance(A, numpy.ndarray)) and (not scipy.sparse.issparse(A)):
        raise TypeError('Input matrix should be either a "numpy.ndarray" or ' +
                        'a "scipy.sparse" matrix.')
    elif A.shape[0] != A.shape[1]:
        raise ValueError('Input matrix should be a square matrix.')

    # Check exponent
    if exponent is None:
        raise TypeError('"exponent" cannot be None.')
    elif not numpy.isscalar(exponent):
        raise TypeError('"exponent" should be a scalar value.')
    elif isinstance(exponent, complex):
        TypeError('"exponent" cannot be an integer or a float number.')

    # Check cholmod
    if cholmod is not None:
        if not isinstance(cholmod, bool):
            raise TypeError('"cholmod" should be either "None", or boolean.')
        elif cholmod is True and suitesparse_installed is False:
            raise RuntimeError('"cholmod" method is not available. Either ' +
                               'install "scikit-sparse" package, or set ' +
                               '"cholmod" to "False" or "None".')
