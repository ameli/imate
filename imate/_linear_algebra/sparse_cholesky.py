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
import scipy.sparse
import scipy.sparse.linalg


# ===============
# Sparse Cholesky
# ===============

def sparse_cholesky(A, diagonal_only=False):
    """
    Computes the Cholesky decomposition of symmetric and positive-definite
    matrix ``A``. This function uses LU decomposition instead of directly
    computing Cholesky decomposition.

    .. note::

        This function does not check if ``A`` is positive-definite. If the
        input matrix is not positive-definite, the Cholesky decomposition does
        not exist and the return value is misleadingly wrong.

    :param A: Symmetric and positive-definite matrix.
    :type A: numpy.ndarray

    :param diagonal_only: If ``True``, returns a column array of the diagonals
        of the Cholesky decomposition. If ``False``, returns the full Cholesky
        matrix as scipy.sparse.csc_matrix.

    :return: Cholesky decomposition of ``A``.
    :rtype: Super.LU
    """

    n = A.shape[0]

    # sparse LU decomposition
    LU = scipy.sparse.linalg.splu(A.tocsc(), diag_pivot_thresh=0,
                                  permc_spec='NATURAL')

    if diagonal_only:

        # Return diagonals only
        return numpy.sqrt(LU.U.diagonal())

    else:

        # return LU.L.dot(sparse.diags(LU.U.diagonal()**0.5))

        # check the matrix A is positive definite.
        if (LU.perm_r == numpy.arange(n)).all() and \
                (LU.U.diagonal() > 0).all():
            return LU.L.dot(scipy.sparse.diags(LU.U.diagonal()**0.5))
        else:
            raise RuntimeError('Matrix is not positive-definite.')
