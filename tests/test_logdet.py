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
from imate import generate_matrix
from imate import logdet


# ===================
# test logdet methods
# ===================

def _test_logdet_methods(K):
    """
    Computes the log-determinant of matrix ``K`` with multiple method.

    :param K: Invertible matrix.
    :type K: numpy.ndarray
    """

    # Use Cholesky method
    time10 = time.time()
    logdet1 = logdet(K, method='cholesky')
    time11 = time.time()

    # Use Stochastic Lanczos Quadrature method, with tri-diagonalization
    time20 = time.time()
    logdet2, _ = logdet(K, method='slq', max_num_samples=50, lanczos_degree=30,
                        symmetric=True)
    time21 = time.time()

    # Use Stochastic Lanczos Quadrature method, with bi-diagonalization
    time30 = time.time()
    logdet3, _ = logdet(K, method='slq', max_num_samples=50, lanczos_degree=30,
                        symmetric=False)
    time31 = time.time()

    # Elapsed times
    elapsed_time1 = time11 - time10
    elapsed_time2 = time21 - time20
    elapsed_time3 = time31 - time30

    # error
    error1 = 0.0
    error2 = 100.0 * numpy.abs(logdet2 - logdet1) / numpy.abs(logdet1)
    error3 = 100.0 * numpy.abs(logdet3 - logdet1) / numpy.abs(logdet1)

    # Print results
    print('')
    print('---------------------------------------------------------')
    print('Method      Options                   logdet  error  time')
    print('----------  ------------------------  ------  -----  ----')
    print('cholesky    N/A                       %6.3f  %4.1f%%  %3.2f'
          % (logdet1, error1, elapsed_time1))
    print('slq         with tri-diagonalization  %6.3f  %4.1f%%  %3.2f'
          % (logdet2, error2, elapsed_time2))
    print('slq         with bi-diagonalization   %6.3f  %4.1f%%  %3.2f'
          % (logdet3, error3, elapsed_time3))
    print('---------------------------------------------------------')
    print('')


# ===========
# test logdet
# ===========

def test_logdet():
    """
    Test for :mod:`logdetInv.logdet` sub-package.
    """

    # Compute trace of inverse of K using dense matrix
    print('Using dense matrix')
    K1 = generate_matrix(size=20, sparse=False)
    K1 = K1 + 0.5*numpy.eye(K1.shape[0])
    _test_logdet_methods(K1)

    # Compute trace of inverse of K using sparse matrix
    print('Using sparse matrix')
    K2 = generate_matrix(size=50, correlation_scale=0.03, density=5e-2,
                         sparse=True)
    K2 = K2 + 0.35*scipy.sparse.eye(K2.shape[0])
    _test_logdet_methods(K2)


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_logdet())
