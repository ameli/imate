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
from imate.sample_matrices import correlation_matrix
from imate import traceinv


# =====================
# test traceinv methods
# =====================

def _test_traceinv_methods(K):
    """
    Computes the trace of the inverse of matrix ``K`` with multiple method.

    :param K: Invertible matrix.
    :type K: numpy.ndarray
    """

    # Settings
    num_samples = 30
    lanczos_degree = 30
    exponent = 2

    # Use Cholesky method with direct inverse
    time10 = time.time()
    trace1, _ = traceinv(K, method='cholesky', invert_cholesky=False,
                         exponent=exponent)
    time11 = time.time()

    # Use Cholesky method without direct inverse
    if not scipy.sparse.isspmatrix(K):
        time20 = time.time()
        trace2, _ = traceinv(K, method='cholesky', invert_cholesky=True,
                             exponent=exponent)
        time21 = time.time()
    else:
        # Do not use Cholesky with inverse method if K is sparse.
        trace2 = None
        time20 = 0
        time21 = 0

    # Use eigenvalue method
    time30 = time.time()
    trace3, _ = traceinv(K, method='eigenvalue', symmetric=True,
                         exponent=exponent)
    time31 = time.time()

    # Use Hutchinson method
    time40 = time.time()
    trace4, _ = traceinv(K, method='hutchinson', min_num_samples=num_samples,
                         exponent=exponent)
    time41 = time.time()

    # Use Stochastic Lanczos Quadrature method, with tridiagonalization
    time50 = time.time()
    trace5, _ = traceinv(K, method='slq', max_num_samples=num_samples,
                         lanczos_degree=lanczos_degree, orthogonalize=-1,
                         exponent=exponent, symmetric=True)
    time51 = time.time()

    # Use Stochastic Lanczos Quadrature method, with bidiagonalization
    time60 = time.time()
    trace6, _ = traceinv(K, method='slq', max_num_samples=num_samples,
                         lanczos_degree=lanczos_degree, orthogonalize=-1,
                         exponent=exponent, symmetric=False)
    time61 = time.time()

    # Elapsed times
    elapsed_time1 = time11 - time10
    elapsed_time2 = time21 - time20
    elapsed_time3 = time31 - time30
    elapsed_time4 = time41 - time40
    elapsed_time5 = time51 - time50
    elapsed_time6 = time61 - time60

    # error
    error1 = 0.0
    if trace2 is not None:
        error2 = 100.0 * numpy.abs(trace2 - trace1) / trace1
    error3 = 100.0 * numpy.abs(trace3 - trace1) / trace1
    error4 = 100.0 * numpy.abs(trace4 - trace1) / trace1
    error5 = 100.0 * numpy.abs(trace5 - trace1) / trace1
    error6 = 100.0 * numpy.abs(trace6 - trace1) / trace1

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
        print('cholesky    using inverse             N/A          N/A    N/A')
    print('eigenvalue  N/A                       %8.3f  %5.2f%%  %5.2f'
          % (trace3, error3, elapsed_time3))
    print('hutchinson  N/A                       %8.3f  %5.2f%%  %5.2f'
          % (trace4, error4, elapsed_time4))
    print('slq         with tri-diagonalization  %8.3f  %5.2f%%  %5.2f'
          % (trace5, error5, elapsed_time5))
    print('slq         with bi-diagonalization   %8.3f  %5.2f%%  %5.2f'
          % (trace6, error6, elapsed_time6))
    print('-------------------------------------------------------------')
    print('')


# =============
# test traceinv
# =============

def test_traceinv():
    """
    A test for :mod:`imate.traceinv` sub-package.
    """

    dtype = r'float32'

    # Compute trace of inverse of K using dense matrix
    print('Using dense matrix')
    K1 = correlation_matrix(size=30, dimension=2, distance_scale=0.05,
                            dtype=dtype, sparse=False)
    _test_traceinv_methods(K1)

    # # Compute trace of inverse of K using sparse matrix
    print('Using sparse matrix')
    K2 = correlation_matrix(size=30, dimension=2, distance_scale=0.05,
                            dtype=dtype, sparse=True, density=2e-1,
                            verbose=False)
    _test_traceinv_methods(K2)


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_traceinv())
