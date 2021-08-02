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

    # slq method settings
    min_num_samples = 10
    max_num_samples = 100
    lanczos_degree = 50
    error_rtol = 1e-2
    exponent = 2

    # Use direct method
    time00 = time.time()
    if scipy.sparse.issparse(K):
        logdet0 = numpy.real(numpy.log(
            numpy.linalg.det(K.toarray()).astype(numpy.complex128)))
    else:
        logdet0 = numpy.real(numpy.log(
            numpy.linalg.det(K).astype(numpy.complex128)))
    logdet0 *= exponent
    time01 = time.time()

    # Use eigenvalue method
    time10 = time.time()
    logdet1, _ = logdet(K, method='eigenvalue', symmetric=True,
                        non_zero_eig_fraction=0.8, exponent=exponent)
    time11 = time.time()

    # Use Cholesky method
    time20 = time.time()
    logdet2, _ = logdet(K, method='cholesky', exponent=exponent)
    time21 = time.time()

    # Use Stochastic Lanczos Quadrature method, with tri-diagonalization
    time30 = time.time()
    logdet3, _ = logdet(K, method='slq', min_num_samples=min_num_samples,
                        max_num_samples=max_num_samples,
                        lanczos_degree=lanczos_degree, error_rtol=error_rtol,
                        exponent=exponent, symmetric=True)
    time31 = time.time()

    # Use Stochastic Lanczos Quadrature method, with bi-diagonalization
    time40 = time.time()
    logdet4, _ = logdet(K, method='slq', min_num_samples=min_num_samples,
                        max_num_samples=max_num_samples,
                        lanczos_degree=lanczos_degree, error_rtol=error_rtol,
                        exponent=exponent, symmetric=True)
    time41 = time.time()

    # Elapsed times
    elapsed_time0 = time01 - time00
    elapsed_time1 = time11 - time10
    elapsed_time2 = time21 - time20
    elapsed_time3 = time31 - time30
    elapsed_time4 = time41 - time40

    # error
    error0 = 0.0
    error1 = 100.0 * numpy.abs(logdet1 - logdet0) / numpy.abs(logdet0)
    error2 = 100.0 * numpy.abs(logdet2 - logdet0) / numpy.abs(logdet0)
    error3 = 100.0 * numpy.abs(logdet3 - logdet0) / numpy.abs(logdet0)
    error4 = 100.0 * numpy.abs(logdet4 - logdet0) / numpy.abs(logdet0)

    # Print results
    print('')
    print('------------------------------------------------------------')
    print('Method      Options                     logdet  error   time')
    print('----------  ------------------------  --------  -----  -----')
    print('direct      N/A                       %+8.3f  %4.1f%%  %5.2f'
          % (logdet0, error0, elapsed_time0))
    print('eigenvalue  N/A                       %+8.3f  %4.1f%%  %5.2f'
          % (logdet1, error1, elapsed_time1))
    print('cholesky    N/A                       %+8.3f  %4.1f%%  %5.2f'
          % (logdet2, error2, elapsed_time2))
    print('slq         with tri-diagonalization  %+8.3f  %4.1f%%  %5.2f'
          % (logdet3, error3, elapsed_time3))
    print('slq         with bi-diagonalization   %+8.3f  %4.1f%%  %5.2f'
          % (logdet4, error4, elapsed_time4))
    print('------------------------------------------------------------')
    print('')


# ===========
# test logdet
# ===========

def test_logdet():
    """
    A test for :mod:`logdetInv.logdet` sub-package.
    """

    dtype = r'float32'

    # Compute trace of inverse of K using dense matrix
    print('Using dense matrix')
    K1 = correlation_matrix(size=20, dimension=2, dtype=dtype, sparse=False)
    K1 = K1 + 0.5*numpy.eye(K1.shape[0])
    _test_logdet_methods(K1)

    # Compute trace of inverse of K using sparse matrix
    print('Using sparse matrix')
    K2 = correlation_matrix(size=50, dimension=2, distance_scale=0.03,
                            density=8e-3, dtype=dtype, sparse=True)
    K2 = K2 + 0.35*scipy.sparse.eye(K2.shape[0])
    _test_logdet_methods(K2)


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_logdet())
