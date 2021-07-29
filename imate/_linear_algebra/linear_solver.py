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
import scipy.linalg
import scipy.sparse.linalg


# =============
# linear solver
# =============

def linear_solver(A, b, assume_matrix, tol=1e-6):
    """
    Solves the linear system :math:`Ax = b` where :math:`A` can be either
    sparse or dense. It is assumed that :math:`A` is symmetric and positive
    definite.

    :param A: matrix of coefficients, two-dimensional array, can be either
        sparse or dense
    :type A: numpy.ndarray

    :param b: column vector of the right hand side of the linear system,
        one-dimensional array
    :type b: array

    :param tol: Tolerance for the error of solving linear system. This is only
        applicable if ``A`` is sparse.
    :type tol: float

    :return: one-dimensional array of the solution of the linear system
    :rtype: numpy.array
    """

    if assume_matrix == "sym_pos":
        assume_matrix = "pos"

    if scipy.sparse.isspmatrix(A):

        # Use direct method
        # x = scipy.sparse.linalg.spsolve(A,b)
        if assume_matrix == "sym":
            solver = scipy.sparse.linalg.minres
            options = {}
        elif assume_matrix == "pos":
            solver = scipy.sparse.linalg.cg
            options = {'atol': 0}
        elif assume_matrix == "gen":
            solver = scipy.sparse.linalg.gmres
            options = {'atol': 0}
        else:
            raise ValueError('"assume_matrix" is invalid.')

        # Use iterative method
        if b.ndim == 1:
            x = solver(A, b, tol=tol)[0]
        else:
            x = numpy.zeros(b.shape, order='F')
            for i in range(x.shape[1]):
                x[:, i] = solver(A, b[:, i], tol=tol, **options)[0]
    else:
        # Dense matrix
        x = scipy.linalg.solve(A, b, assume_a=assume_matrix)

    return x
