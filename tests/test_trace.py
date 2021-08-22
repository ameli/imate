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
from imate.sample_matrices import band_matrix, band_matrix_trace
from imate import trace


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


# ==================
# test trace methods
# ==================

def _test_trace_methods(K, matrix, gram, exponent, assume_matrix):
    """
    Computes the trace of matrix ``K`` with multiple method.

    :param K: Square matrix.
    :type K: numpy.ndarray
    """

    # Settings
    min_num_samples = 100
    max_num_samples = 200
    lanczos_degree = 30
    error_rtol = 1e-2

    # Use exact method
    time10 = time.time()
    trace1, _ = trace(K, method='exact', gram=gram, exponent=exponent)
    time11 = time.time()

    # Use eigenvalue method
    time20 = time.time()
    trace2, _ = trace(K, method='eigenvalue', gram=gram,
                      assume_matrix=assume_matrix, exponent=exponent,
                      non_zero_eig_fraction=0.95)
    time21 = time.time()

    # Use Stochastic Lanczos Quadrature method
    time30 = time.time()
    trace3, _ = trace(K, method='slq', min_num_samples=min_num_samples,
                      max_num_samples=max_num_samples, orthogonalize=-1,
                      lanczos_degree=lanczos_degree, error_rtol=error_rtol,
                      gram=gram, exponent=exponent, verbose=False)
    time31 = time.time()

    # Elapsed times
    elapsed_time1 = time11 - time10
    elapsed_time2 = time21 - time20
    elapsed_time3 = time31 - time30

    # Exact solution of logdet for band matrix
    if exponent == 1:
        trace_exact = band_matrix_trace(matrix['a'], matrix['b'],
                                        matrix['size'], True)
    else:
        trace_exact = trace1

    # error
    error1 = relative_error(trace1, trace_exact)
    error2 = relative_error(trace2, trace_exact)
    error3 = relative_error(trace3, trace_exact)

    # Print results
    print('')
    print('-------------------------------------------------------------')
    print('Method      Options                      trace   error   time')
    print('----------  ------------------------  --------  ------  -----')
    print('exact       N/A                       %8.3f  %5.2f%%  %5.2f'
          % (trace1, error1, elapsed_time1))
    print('eigenvalue  N/A                       %8.3f  %5.2f%%  %5.2f'
          % (trace2, error2, elapsed_time2))
    print('slq         N/A                       %8.3f  %5.2f%%  %5.2f'
          % (trace3, error3, elapsed_time3))
    print('-------------------------------------------------------------')
    print('')


# ==========
# test trace
# ==========

def test_trace():
    """
    A test for :mod:`imate.trace` sub-package.
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
            #     1. We generate a 2-band nonsymmetric matrix K (hence we set
            #        gram=False in band_matrix).
            #     2. We compute trace of K.T @ K using only K (hence we set
            #        gram=True in trace method).
            #
            # When gram is False:
            #     1. We generate a 3-band symmetric matrix K (hence we set
            #        gram=True in band_matrix).
            #     2. We compute trace of K using K (hence we set
            #        gram=False in trace method).
            K = band_matrix(matrix['a'], matrix['b'], matrix['size'],
                            gram=(not gram), dtype=dtype)

            for sparse in sparses:
                if not sparse:
                    K = K.toarray()

                for exponent in exponents:
                    print('dtype: %s, ' % (dtype) +
                          ' sparse: %s, ' % (sparse) +
                          'gram: %s, ' % (gram) +
                          'exponent: %f, ' % (exponent) +
                          'assume_matrix: %s.' % (assume_matrix))

                    _test_trace_methods(K, matrix, gram, exponent,
                                        assume_matrix)


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_trace())
