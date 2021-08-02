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
from imate.sample_matrices import correlation_matrix
from imate import trace


# ==================
# test trace methods
# ==================

def _test_trace_methods(K):
    """
    Computes the trace of matrix ``K`` with multiple method.

    :param K: Square matrix.
    :type K: numpy.ndarray
    """

    # Settings
    num_samples = 30
    lanczos_degree = 30
    exponent = 2

    # Use exact method
    time10 = time.time()
    trace1, _ = trace(K, method='exact', exponent=exponent)
    time11 = time.time()

    # Use eigenvalue method
    time20 = time.time()
    trace2, _ = trace(K, method='eigenvalue', symmetric=True,
                      exponent=exponent)
    time21 = time.time()

    # Use Stochastic Lanczos Quadrature method, with tridiagonalization
    time30 = time.time()
    trace3, _ = trace(K, method='slq', max_num_samples=num_samples,
                      lanczos_degree=lanczos_degree, orthogonalize=-1,
                      exponent=exponent, symmetric=True)
    time31 = time.time()

    # Use Stochastic Lanczos Quadrature method, with bidiagonalization
    time40 = time.time()
    trace4, _ = trace(K, method='slq', max_num_samples=num_samples,
                      lanczos_degree=lanczos_degree, orthogonalize=-1,
                      exponent=exponent, symmetric=False)
    time41 = time.time()

    # Elapsed times
    elapsed_time1 = time11 - time10
    elapsed_time2 = time21 - time20
    elapsed_time3 = time31 - time30
    elapsed_time4 = time41 - time40

    # error
    error1 = 0.0
    error2 = 100.0 * numpy.abs(trace2 - trace1) / trace1
    error3 = 100.0 * numpy.abs(trace3 - trace1) / trace1
    error4 = 100.0 * numpy.abs(trace4 - trace1) / trace1

    # Print results
    print('')
    print('-------------------------------------------------------------')
    print('Method      Options                   traceinv   error   time')
    print('----------  ------------------------  --------  ------  -----')
    print('exact       N/A                       %8.3f  %5.2f%%  %5.2f'
          % (trace1, error1, elapsed_time1))
    print('eigenvalue  N/A                       %8.3f  %5.2f%%  %5.2f'
          % (trace2, error2, elapsed_time2))
    print('slq         with tri-diagonalization  %8.3f  %5.2f%%  %5.2f'
          % (trace3, error3, elapsed_time3))
    print('slq         with bi-diagonalization   %8.3f  %5.2f%%  %5.2f'
          % (trace4, error4, elapsed_time4))
    print('-------------------------------------------------------------')
    print('')


# ==========
# test trace
# ==========

def test_trace():
    """
    A test for :mod:`imate.trace` sub-package.
    """

    dtype = r'float32'

    # Compute trace of K using dense matrix
    print('Using dense matrix')
    K1 = correlation_matrix(size=30, dimension=2, distance_scale=0.05,
                            dtype=dtype, sparse=False)
    _test_trace_methods(K1)

    # # Compute trace of K using sparse matrix
    print('Using sparse matrix')
    K2 = correlation_matrix(size=30, dimension=2, distance_scale=0.05,
                            dtype=dtype, sparse=True, density=2e-1,
                            verbose=False)
    _test_trace_methods(K2)


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_trace())
