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


# =====================
# test traceinv methods
# =====================

def _test_traceinv_methods(K, matrix, gram, exponent, assume_matrix):
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
    time10 = time.time()
    trace1, _ = traceinv(K, method='cholesky', invert_cholesky=False,
                         gram=gram, exponent=exponent, cholmod=None)
    time11 = time.time()

    # Use Cholesky method without direct inverse
    if not scipy.sparse.isspmatrix(K):
        time20 = time.time()
        trace2, _ = traceinv(K, method='cholesky', invert_cholesky=True,
                             gram=gram, exponent=exponent, cholmod=None)
        time21 = time.time()
    else:
        # Do not use Cholesky with inverse method if K is sparse.
        trace2 = None
        time20 = 0
        time21 = 0

    # Use eigenvalue method
    time30 = time.time()
    trace3, _ = traceinv(K, method='eigenvalue', gram=gram,
                         assume_matrix=assume_matrix, exponent=exponent,
                         non_zero_eig_fraction=0.95)
    time31 = time.time()

    # Use Hutchinson method
    time40 = time.time()
    trace4, _ = traceinv(K, method='hutchinson',
                         min_num_samples=min_num_samples,
                         max_num_samples=max_num_samples, orthogonalize=True,
                         error_rtol=error_rtol, gram=gram, exponent=exponent,
                         verbose=False)
    time41 = time.time()

    # Use Stochastic Lanczos Quadrature method
    time50 = time.time()
    trace5, _ = traceinv(K, method='slq', min_num_samples=min_num_samples,
                         max_num_samples=max_num_samples, orthogonalize=-1,
                         lanczos_degree=lanczos_degree, error_rtol=error_rtol,
                         gram=gram, exponent=exponent, verbose=False)
    time51 = time.time()

    # Elapsed times
    elapsed_time1 = time11 - time10
    elapsed_time2 = time21 - time20
    elapsed_time3 = time31 - time30
    elapsed_time4 = time41 - time40
    elapsed_time5 = time51 - time50

    # Exact solution of logdet for band matrix
    if exponent == 1 and not gram:
        traceinv_exact = band_matrix_traceinv(matrix['a'], matrix['b'],
                                              matrix['size'], True)
    else:
        traceinv_exact = trace1

    # error
    error1 = relative_error(trace1, traceinv_exact)
    if trace2 is not None:
        error2 = relative_error(trace2, traceinv_exact)
    error3 = relative_error(trace3, traceinv_exact)
    error4 = relative_error(trace4, traceinv_exact)
    error5 = relative_error(trace5, traceinv_exact)

    # Print results
    print('')
    print('-------------------------------------------------------------')
    print('Method      Options                   traceinv   error   time')
    print('----------  ------------------------  --------  ------  -----')
    print('cholesky    without using inverse     %8.3f  %5.2f%%  %5.2f'
          % (trace1, error1, elapsed_time1))
    if trace2 is not None:
        print('cholesky    using inverse             %8.3f  %5.2f%%  %5.2f'
              % (trace2, error2, elapsed_time2))
    else:
        print('cholesky    using inverse                  N/A     N/A    N/A')
    print('eigenvalue  N/A                       %8.3f  %5.2f%%  %5.2f'
          % (trace3, error3, elapsed_time3))
    print('hutchinson  N/A                       %8.3f  %5.2f%%  %5.2f'
          % (trace4, error4, elapsed_time4))
    print('slq         N/A                       %8.3f  %5.2f%%  %5.2f'
          % (trace5, error5, elapsed_time5))
    print('-------------------------------------------------------------')
    print('')


# =============
# test traceinv
# =============

def test_traceinv():
    """
    A test for :mod:`imate.traceinv` sub-package.
    """

    matrix = {
        'a': 2.0,
        'b': 1.0,
        'size': 50
    }

    exponents = [0, 1, 2]
    dtypes = [r'float32', r'float64']
    grams = [True, False]
    sparses = [True, False]

    for dtype in dtypes:
        for gram in grams:

            if gram:
                assume_matrix = 'gen'
            else:
                assume_matrix = 'sym'

            # When gram is True:
            #     1. We generate a 3-band symmetric matrix K (hence we set
            #        gram=True in band_matrix). Note: this is different than
            #        what we test in test_logdet.py and test_trace.py.
            #        Becase, the golub-kahn method cannot estimate traceinv
            #        of K.T @ K if K is 2-banded. Thus, we input a symmetric
            #        matrix K (itself is gramian).
            #     2. We compute traceinv of K.T @ K using only K (hence we set
            #        gram=True in traceinv method).
            #
            # When gram is False:
            #     1. We generate a 3-band symmetric matrix K (hence we set
            #        gram=True in band_matrix).
            #     2. We compute traceinv of K using K (hence we set
            #        gram=False in traceinv method).
            K = band_matrix(matrix['a'], matrix['b'], matrix['size'],
                            gram=True, dtype=dtype)

            for sparse in sparses:
                if not sparse:
                    K = K.toarray()

                for exponent in exponents:
                    print('dtype: %s, ' % (dtype) +
                          ' sparse: %s, ' % (sparse) +
                          'gram: %s, ' % (gram) +
                          'exponent: %f, ' % (exponent) +
                          'assume_matrix: %s.' % (assume_matrix))

                    _test_traceinv_methods(K, matrix, gram, exponent,
                                           assume_matrix)


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_traceinv())
