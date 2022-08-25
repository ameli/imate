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
from imate.sample_matrices import toeplitz, toeplitz_schatten
from imate import schatten


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
# test schatten methods
# =====================

def _test_schatten_methods(K, matrix, gram, p, assume_matrix):
    """
    Computes the log-determinant of matrix ``K`` with multiple method.

    :param K: Invertible matrix.
    :type K: numpy.ndarray
    """

    # slq method settings
    min_num_samples = 100
    max_num_samples = 200
    lanczos_degree = 30
    error_rtol = 1e-2

    # Use direct method
    # Exact solution of schatten for band matrix
    schatten0 = toeplitz_schatten(matrix['a'], matrix['b'],
                                  size=matrix['size'], p=p)
    time00 = 0.0
    time01 = 0.0

    # Use eigenvalue method
    time10 = time.time()
    schatten1 = schatten(K, gram=gram, p=p, method='eigenvalue',
                         assume_matrix=assume_matrix,
                         non_zero_eig_fraction=0.95)
    # Test
    print(schatten1)
    time11 = time.time()

    # Use Cholesky method
    if p <= 0:
        time20 = time.time()
        schatten2 = schatten(K, gram=gram, p=p, method='cholesky',
                             cholmod=None)
        time21 = time.time()
    else:
        schatten2 = None

    # Use Stochastic Lanczos Quadrature method
    time30 = time.time()
    schatten3 = schatten(K, gram=gram, p=p, method='slq',
                         min_num_samples=min_num_samples,
                         max_num_samples=max_num_samples, orthogonalize=-1,
                         lanczos_degree=lanczos_degree, error_rtol=error_rtol,
                         verbose=False)
    time31 = time.time()

    # Elapsed times
    elapsed_time0 = time01 - time00
    elapsed_time1 = time11 - time10
    if schatten2 is not None:
        elapsed_time2 = time21 - time20
    else:
        elapsed_time2 = None
    elapsed_time3 = time31 - time30

    # error
    error0 = relative_error(schatten0, schatten0)
    error1 = relative_error(schatten1, schatten0)
    if schatten2 is not None:
        error2 = relative_error(schatten2, schatten0)
    else:
        error2 = None
    error3 = relative_error(schatten3, schatten0)

    # Print results
    print('')
    print('-------------------------------------------------------------')
    print('Method      Options                   schatten   error   time')
    print('----------  ------------------------  --------  ------  -----')
    print('direct      N/A                       %+8.3f  %5.2f%%  %5.2f'
          % (schatten0, error0, elapsed_time0))
    print('eigenvalue  N/A                       %+8.3f  %5.2f%%  %5.2f'
          % (schatten1, error1, elapsed_time1))
    if schatten2 is not None:
        print('cholesky    N/A                       %+8.3f  %5.2f%%  %5.2f'
              % (schatten2, error2, elapsed_time2))
    else:
        print('cholesky    without using inverse          N/A     N/A    N/A')
    print('slq         N/A                       %+8.3f  %5.2f%%  %5.2f'
          % (schatten3, error3, elapsed_time3))
    print('-------------------------------------------------------------')
    print('')


# =============
# test schatten
# =============

def test_schatten():
    """
    A test for :mod:`schatten` sub-package.
    """

    matrix = {
        'a': 2.0,
        'b': 1.0,
        'size': 50
    }

    exponents = [0, 1, 2]
    dtypes = [r'float32', r'float64']
    sparses = [True, False]
    gram = False
    assume_matrix = 'gen'

    for dtype in dtypes:

        K = toeplitz(matrix['a'], matrix['b'], matrix['size'], gram=gram,
                     dtype=dtype)

        for sparse in sparses:
            if not sparse:
                K = K.toarray()

            for p in exponents:
                print('dtype: %s, ' % (dtype) +
                      'sparse: %5s, ' % (sparse) +
                      'gram: %5s, ' % (gram) +
                      'exponent: %0.4f,\n' % (p) +
                      'assume_matrix: %s.' % (assume_matrix))

                _test_schatten_methods(K, matrix, gram, p, assume_matrix)


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_schatten())
