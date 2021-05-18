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

def cholesky_method(A):
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

    if scipy.sparse.issparse(A):

        if suitesparse_installed:
            # Use Suite Sparse
            Factor = sksparse.cholmod.cholesky(A)
            logdet_ = Factor.logdet()
        else:
            # Use scipy
            L_diag = sparse_cholesky(A, diagonal_only=True)
            logdet_ = 2.0 * numpy.sum(numpy.log(L_diag))

    else:

        # Use scipy
        L = scipy.linalg.cholesky(A, lower=True)
        logdet_ = 2.0 * numpy.sum(numpy.log(numpy.diag(L)))

    return logdet_