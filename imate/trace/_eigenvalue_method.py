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

import time
import numpy
import numpy.linalg
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg


# =================
# eigenvalue method
# =================

def eigenvalue_method(
        A,
        eigenvalues=None,
        exponent=1.0,
        symmetric=True,
        non_zero_eig_fraction=0.9):
    """
    """

    # Checking input arguments
    check_arguments(A, eigenvalues, exponent, symmetric,
                    non_zero_eig_fraction)

    init_wall_time = time.perf_counter()
    init_proc_time = time.process_time()

    if eigenvalues is None:

        # This is only relevant when A is sparse and we can only find a
        # portion, bit not all, of its eigenvalues.
        if exponent >= 0:
            # when exponent is positive, the largest eigenvalues weight more
            which_eigenvalues = 'LM'
        else:
            # when exponent is negative, the smallest eigenvalues weight more
            which_eigenvalues = 'SM'

        # Compute eigenvalues of A
        eigenvalues = compute_eigenvalues(A, symmetric,
                                          non_zero_eig_fraction,
                                          which_eigenvalues)

    # Compute trace of inverse of matrix
    not_nan = numpy.logical_not(numpy.isnan(eigenvalues))
    trace = numpy.sum(eigenvalues[not_nan]**exponent)

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
            'method': 'eigenvalue',
        }
    }

    return trace, info


# ===============
# check arguments
# ===============

def check_arguments(
        A,
        eigenvalues,
        exponent,
        symmetric,
        non_zero_eig_fraction):
    """
    Checks the type and value of the parameters.
    """

    # Check A
    if (not isinstance(A, numpy.ndarray)) and (not scipy.sparse.issparse(A)):
        raise TypeError('Input matrix should be either a "numpy.ndarray" or ' +
                        'a "scipy.sparse" matrix.')
    elif A.shape[0] != A.shape[1]:
        raise ValueError('Input matrix should be a square matrix.')

    # Check eigenvalues
    if eigenvalues is not None:
        if not isinstance(eigenvalues, numpy.ndarray):
            raise TypeError('"eigenvalues" should be a numpy.ndarray.')
        if eigenvalues.size != A.shape[0]:
            raise ValueError('The length of "eigenvalues" does not match ' +
                             'the size of matrix "A".')

    # Check exponent
    if exponent is None:
        raise TypeError('"exponent" cannot be None.')
    elif not numpy.isscalar(exponent):
        raise TypeError('"exponent" should be a scalar value.')
    elif not isinstance(exponent, int):
        TypeError('"exponent" cannot be an integer.')

    # Check symmetric
    if symmetric is None:
        raise TypeError('"symmetric" cannot be None.')
    elif not numpy.isscalar(symmetric):
        raise TypeError('"symmetric" should be a scalar value.')
    elif not isinstance(symmetric, bool):
        raise TypeError('"symmetric" should be boolean.')

    # Check non_zero_eig_fraction
    if non_zero_eig_fraction is None:
        raise TypeError('"non_zero_eig_fraction" cannot be None.')
    elif not numpy.isscalar(non_zero_eig_fraction):
        raise TypeError('"non_zero_eig_fraction" should be a scalar value.')
    elif not isinstance(non_zero_eig_fraction, float):
        raise TypeError('"non_zero_eig_fraction" should be a float type.')
    elif non_zero_eig_fraction <= 0 or non_zero_eig_fraction >= 1.0:
        raise ValueError('"non_zero_eig_fraction" should be greater then 0.0' +
                         'and smaller than 1.0.')


# ===================
# compute eigenvalues
# ===================

def compute_eigenvalues(
        A,
        symmetric,
        non_zero_eig_fraction,
        which_eigenvalues,
        tol=1e-4):
    """
    """

    if scipy.sparse.isspmatrix(A):

        # Sparse matrix
        n = A.shape[0]
        eigenvalues = numpy.empty(n)
        eigenvalues[:] = numpy.nan

        # find 90% of eigenvalues, assume the rest are very close to zero.
        num_none_zero_eig = int(n*non_zero_eig_fraction)

        if symmetric:
            eigenvalues[:num_none_zero_eig] = \
                scipy.sparse.linalg.eigsh(A, num_none_zero_eig,
                                          which=which_eigenvalues,
                                          return_eigenvectors=False,
                                          tol=tol)
        else:
            eigenvalues[:num_none_zero_eig] = \
                scipy.sparse.linalg.eigs(A, num_none_zero_eig,
                                         which=which_eigenvalues,
                                         return_eigenvectors=False,
                                         tol=tol)
    else:

        # Dense matrix
        if symmetric:
            eigenvalues = scipy.linalg.eigh(A, check_finite=False,
                                            eigvals_only=True)
        else:
            eigenvalues = scipy.linalg.eig(A, check_finite=False)[0]

    return eigenvalues
