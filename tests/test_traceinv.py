#! /usr/bin/env python

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

import sys
import time
import numpy
import scipy.sparse
from imate.sample_matrices import band_matrix, band_matrix_traceinv
from imate import traceinv


# ==============
# relative error
# ==============

def relative_error(estimate, exact):
    """
    Compute the relative error of an estimate, in percent.
    """

    tol = 1e-15
    if numpy.abs(exact) < tol:
        if numpy.abs(estimate - exact) < tol:
            relative_error = 0.0
        else:
            relative_error = numpy.inf
    else:
        relative_error = numpy.abs((estimate - exact) / exact) * 100.0

    return relative_error


# ==============
# traceinv exact
# ==============

def _traceinv_exact(K, B, C, matrix, gram, exponent):
    """
    Finds traceinv directly for the purpose of comparison.
    """

    # Exact solution of traceinv for band matrix
    if B is not None:

        if scipy.sparse.isspmatrix(K):
            K_ = K.toarray()
            B_ = B.toarray()

            if C is not None:
                C_ = C.toarray()
        else:
            K_ = K
            B_ = B

            if C is not None:
                C_ = C

        if exponent == 0:
            if C is not None:
                traceinv_exact = numpy.trace(C_ @ B_)
            else:
                traceinv_exact = numpy.trace(B_)
        else:
            if gram:
                K_ = numpy.matmul(K_.T, K_)

            if exponent > 1:
                K1 = K_.copy()
                for i in range(1, exponent):
                    K_ = numpy.matmul(K_, K1)

            Kinv = numpy.linalg.inv(K_)
            Op = numpy.matmul(Kinv, B_)

            if C is not None:
                Op = Kinv @ C_ @ Op

            traceinv_exact = numpy.trace(Op)

    elif exponent == 1 and not gram:

        # B is identity. Using analytic formula.
        traceinv_exact = band_matrix_traceinv(matrix['a'], matrix['b'],
                                              matrix['size'], True)
    else:
        # B and C are identity. Compute traceinv directly.
        if scipy.sparse.isspmatrix(K):
            K_ = K.toarray()
        else:
            K_ = K

        if exponent == 0:
            traceinv_exact = K_.shape[0]
        else:
            if gram:
                K_ = numpy.matmul(K_.T, K_)

            K_temp = K_.copy()
            for i in range(1, exponent):
                K_ = numpy.matmul(K_, K_temp)

            Kinv = numpy.linalg.inv(K_)
            traceinv_exact = numpy.trace(Kinv)

    return traceinv_exact


# =====================
# test traceinv methods
# =====================

def _test_traceinv_methods(K, B, C, matrix, gram, exponent, assume_matrix):
    """
    Computes the trace of the inverse of matrix ``K`` with multiple method.

    :param K: Invertible matrix.
    :type K: numpy.ndarray
    """

    # Settings
    min_num_samples = 100
    max_num_samples = 200
    lanczos_degree = 30
    error_rtol = 1e-2

    # Use Cholesky method with direct inverse
    if C is None:
        time10 = time.time()
        trace1, _ = traceinv(K, B, method='cholesky', invert_cholesky=False,
                             gram=gram, exponent=exponent, cholmod=None)
        time11 = time.time()
    else:
        trace1 = None
        time10 = numpy.nan
        time11 = numpy.nan

    # Use Cholesky method without direct inverse
    if C is None:
        time20 = time.time()
        trace2, _ = traceinv(K, B, method='cholesky', invert_cholesky=True,
                             gram=gram, exponent=exponent, cholmod=None)
        time21 = time.time()
    else:
        trace2 = None
        time20 = numpy.nan
        time21 = numpy.nan

    # Use eigenvalue method
    if B is None and C is None:
        time30 = time.time()
        trace3, _ = traceinv(K, method='eigenvalue', gram=gram,
                             assume_matrix=assume_matrix, exponent=exponent,
                             non_zero_eig_fraction=0.95)
        time31 = time.time()
    else:
        trace3 = None
        time30 = numpy.nan
        time31 = numpy.nan

    # Use Hutchinson method
    time40 = time.time()
    trace4, _ = traceinv(K, B, C, method='hutchinson',
                         min_num_samples=min_num_samples,
                         max_num_samples=max_num_samples, orthogonalize=True,
                         error_rtol=error_rtol, gram=gram, exponent=exponent,
                         verbose=False)
    time41 = time.time()

    # Use Stochastic Lanczos Quadrature method
    if B is None and C is None:
        time50 = time.time()
        trace5, _ = traceinv(K, method='slq', min_num_samples=min_num_samples,
                             max_num_samples=max_num_samples, orthogonalize=-1,
                             lanczos_degree=lanczos_degree,
                             error_rtol=error_rtol, gram=gram,
                             exponent=exponent, verbose=False)
        time51 = time.time()
    else:
        trace5 = None
        time50 = numpy.nan
        time51 = numpy.nan

    # Elapsed times
    elapsed_time1 = time11 - time10
    elapsed_time2 = time21 - time20
    elapsed_time3 = time31 - time30
    elapsed_time4 = time41 - time40
    elapsed_time5 = time51 - time50

    # Exact value of traceinv computed directly
    traceinv_exact = _traceinv_exact(K, B, C, matrix, gram, exponent)

    # error
    if trace1 is not None:
        error1 = relative_error(trace1, traceinv_exact)
    if trace2 is not None:
        error2 = relative_error(trace2, traceinv_exact)
    if trace3 is not None:
        error3 = relative_error(trace3, traceinv_exact)
    if trace4 is not None:
        error4 = relative_error(trace4, traceinv_exact)
    if trace5 is not None:
        error5 = relative_error(trace5, traceinv_exact)

    # Print results
    print('')
    print('-------------------------------------------------------------')
    print('Method      Options                   traceinv   error   time')
    print('----------  ------------------------  --------  ------  -----')
    if trace1 is not None:
        print('cholesky    without using inverse     %8.3f  %5.2f%%  %5.2f'
              % (trace1, error1, elapsed_time1))
    else:
        print('cholesky    without using inverse          N/A     N/A    N/A')
    if trace2 is not None:
        print('cholesky    using inverse             %8.3f  %5.2f%%  %5.2f'
              % (trace2, error2, elapsed_time2))
    else:
        print('cholesky    using inverse                  N/A     N/A    N/A')
    if trace3 is not None:
        print('eigenvalue  N/A                       %8.3f  %5.2f%%  %5.2f'
              % (trace3, error3, elapsed_time3))
    else:
        print('eigenvalue  N/A                            N/A     N/A    N/A')
    if trace4 is not None:
        print('hutchinson  N/A                       %8.3f  %5.2f%%  %5.2f'
              % (trace4, error4, elapsed_time4))
    else:
        print('hutchinson  N/A                            N/A     N/A    N/A')
    if trace5 is not None:
        print('slq         N/A                       %8.3f  %5.2f%%  %5.2f'
              % (trace5, error5, elapsed_time5))
    else:
        print('slq         N/A                            N/A     N/A    N/A')
    print('-------------------------------------------------------------')
    print('')


# =============
# test traceinv
# =============

def test_traceinv():
    """
    A test for :mod:`imate.traceinv` sub-package.
    """

    matrix_K = {
        'a': 2.0,
        'b': 1.0,
        'size': 50
    }

    matrix_B = {
        'a': 3.0,
        'b': 2.0,
        'size': matrix_K['size']
    }

    matrix_C = {
        'a': 3.0,
        'b': 1.0,
        'size': matrix_K['size']
    }

    exponents = [0, 1, 2]
    dtypes = [r'float32', r'float64']
    grams = [True, False]
    sparses = [True, False]
    B_identities = [False, True]
    C_identities = [False, True]

    for dtype in dtypes:
        for gram in grams:
            for B_identity in B_identities:
                for C_identity in C_identities:

                    if gram:
                        assume_matrix = 'gen'
                    else:
                        assume_matrix = 'sym'

                    # When gram is True:
                    #     1. We generate a 2-band non symmetric matrix K (hence
                    #        we set gram=False in band_matrix).
                    #     2. We compute traceinv of K.T @ K using only K (hence
                    #        we set gram=True in traceinv method).
                    #
                    # When gram is False:
                    #     1. We generate a 3-band symmetric matrix K (hence we
                    #        set gram=True in band_matrix).
                    #     2. We compute traceinv of K using K (hence we set
                    #        gram=False in traceinv method).
                    K = band_matrix(matrix_K['a'], matrix_K['b'],
                                    matrix_K['size'], gram=(not gram),
                                    dtype=dtype)

                    if B_identity:
                        if C_identity:
                            B = None
                            C = None
                        else:
                            # If C is not identity, B should also not be
                            # identity.
                            continue
                    else:
                        B = band_matrix(matrix_B['a'], matrix_B['b'],
                                        matrix_B['size'], gram=True,
                                        dtype=dtype)

                        if C_identity:
                            C = None
                        else:
                            C = band_matrix(matrix_C['a'], matrix_C['b'],
                                            matrix_C['size'], gram=True,
                                            dtype=dtype)

                    for sparse in sparses:
                        if not sparse:
                            K = K.toarray()
                            if B is not None:
                                B = B.toarray()
                            if C is not None:
                                C = C.toarray()

                        for exponent in exponents:
                            print('dtype: %s, ' % (dtype) +
                                  'sparse: %5s, ' % (sparse) +
                                  'gram: %5s, ' % (gram) +
                                  'exponent: %0.4f,\n' % (exponent) +
                                  'assume_matrix: %s, ' % (assume_matrix) +
                                  'B_identity: %5s, ' % (B_identity) +
                                  'C_identity: %5s.' % (C_identity))

                            _test_traceinv_methods(K, B, C, matrix_K, gram,
                                                   exponent, assume_matrix)


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_traceinv())
