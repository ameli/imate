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

try:
    import sksparse
    import sksparse.cholmod.cholesky
    suitesparse_installed = True
except ImportError:
    suitesparse_installed = False

from .._linear_algebra import sparse_cholesky


# ===============
# cholesky method
# ===============

def cholesky_method(A, exponent=1.0):
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
    check_arguments(A, exponent)

    init_wall_time = time.perf_counter()
    init_proc_time = time.process_time()

    if exponent == 0:

        # determinant is 1. Logdet is zero
        trace = 0.0

    else:

        # Compute logdet of A without the exponent
        if scipy.sparse.issparse(A):

            # Sparse matrix
            if suitesparse_installed:
                # Use Suite Sparse
                Factor = sksparse.cholmod.cholesky(A)
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

    wall_time = time.perf_counter() - init_wall_time
    proc_time = time.process_time() - init_proc_time

    # Dictionary of output info
    info = {
        'cpu':
        {
            'wall_time': wall_time,
            'proc_time': proc_time,
        },
        'solver':
        {
            'method': 'cholesky',
        }
    }

    return trace, info


# ===============
# check arguments
# ===============

def check_arguments(A, exponent):
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
