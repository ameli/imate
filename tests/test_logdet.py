#! /usr/bin/env python

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

def test_logdet_methods(K):
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
    logdet2 = logdet(K, method='SLQ', num_iterations=50, lanczos_degree=30,
                     symmetric=True)
    time21 = time.time()

    # Use Stochastic Lanczos Quadrature method, with bi-diagonalization
    time30 = time.time()
    logdet3 = logdet(K, method='SLQ', num_iterations=50, lanczos_degree=30,
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
    print('Method      Options                   LogDet  error  time')
    print('----------  ------------------------  ------  -----  ----')
    print('Cholesky    N/A                       %0.3f  %0.2f%%  %0.2f'
          % (logdet1, error1, elapsed_time1))
    print('SLQ         with tri-diagonalization  %0.3f  %0.2f%%  %0.2f'
          % (logdet2, error2, elapsed_time2))
    print('SLQ         with bi-diagonalization   %0.3f  %0.2f%%  %0.2f'
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
    test_logdet_methods(K1)

    # Compute trace of inverse of K using sparse matrix
    print('Using sparse matrix')
    K2 = generate_matrix(size=20, sparse=True)
    # K2 = generate_matrix(size=50, correlation_scale=0.03, sparse=True)
    K2 = K2 + 0.35*scipy.sparse.eye(K2.shape[0])
    test_logdet_methods(K2)


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_logdet())
