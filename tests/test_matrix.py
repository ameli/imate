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
import numpy
import scipy.sparse
from imate import Matrix, get_config

__all__ = ['test_matrix']


# ==================
# benchmark solution
# ==================

def _benchmark_solution(A, vector, transpose=False):
    """
    Multiplies ``A`` by ``vector`` using numpy.

    :param A: An array of size ``(n, m)``.
    :type A: numpy.ndarray, or any scipy.sparse array

    :param vector: The input vector to multiply with, with the size of ``m``.
    :type vector: numpy.array

    :param transpose: If `True`, the transpose of the input matrix is used.
    :type transpose: boolean

    :return: The product ``A * vector``. The length of this array is ``n``.
    :rtype: numpy.array
    """

    if transpose:
        A_ = A.T
    else:
        A_ = A

    # Matrix-vector multiplication
    vector_typed = vector.astype(A.dtype)
    benchmark = A_.dot(vector_typed)

    return benchmark


# ===========
# check error
# ===========

def _check_error(error, n, dtype, transpose, symmetric):
    """
    Suppose a, b, and x and the elements of A, B, and x. Let e be the precision
    (epsilon) floating point error of a number. Thus, the arithmetic error of
    A*x can be found, roughly, by expanding (A+e)*(x+e), and ignoring all
    higher-order terms of the powers of e. The first order term involving e^1
    is:

        (A + x) * e.

    Assuming A and x are normally distributed as N(0, 1), on average, the
    above error term is 2 * e.
    """

    # tolerance should be adjusted by dtype, since the dot product results
    # between Aop and numpy's dot product can only agree up to dtype precision.
    if dtype == 'float16':
        precision = 0.000977
    elif dtype == 'float32':
        precision = 1.1920929e-7
    elif dtype == 'float64':
        precision = 2.220446049250313e-16
    elif dtype == 'float128':
        # Numpy seems to not perform log-double precision matvec on
        # long-double data, rather, uses double precision. As such, we use
        # tolerance for double precision (not long-double)
        precision = 1.084202172485504434e-19   # for log-double

    # Arithmetic standard-deviation error of A*x (see above documentation)
    sigma = 2.0 * precision

    # Print success
    success = True
    z_score = 3.29  # 99.9% confidence interval for A[i, j] and x[j] ~ N(0, 1)
    matvec_tolerance = 2.0 * sigma * z_score * n
    if any(error > matvec_tolerance):
        print('ERROR: product and benchmark mismatch.')
        print('       tolerance: %e' % matvec_tolerance)
        print('       dtype: %s' % dtype)
        print('       transpose: %s' % transpose)
        print('       symmetric: %s' % symmetric)
        success = False

        for i in range(error.size):
            if error[i] > matvec_tolerance:
                print('dtype: %s, i: %04d, error: %4e' % (dtype, i, error[i]))

        print('')

    return success


# ========
# test dot
# ========

def _test_dot(A, transpose=False, symmetric=False, gpu=False):
    """
    """

    # Define linear operator object
    Aop = Matrix(A, symmetric=symmetric)
    Aop.initialize(gpu=gpu)

    # Matrix shape
    num_rows, num_columns = A.shape

    # Input random vectors
    if transpose:
        vector = numpy.random.randn(num_rows)
    else:
        vector = numpy.random.randn(num_columns)
    vector = vector.astype(numpy.float32)

    # The nogil environment is arbitrary
    if transpose:
        product = Aop.transpose_dot(vector)
    else:
        product = Aop.dot(vector)

    # Benchmark product
    benchmark = _benchmark_solution(A, vector, transpose=transpose)

    # Error
    error = numpy.abs(product - benchmark)

    # Compare error with tolerance
    success = _check_error(error, product.size, A.dtype, transpose, symmetric)

    return success


# ====
# test
# ====

def _test(A, symmetric=False, gpu=False):
    """
    """

    A_matrices = {
        # 'float16': A.astype(numpy.float16),
        # 'float32': A.astype(numpy.float32),
        'float64': A.astype(numpy.float64),
        # 'float128': A.astype(numpy.float128)
    }

    successes = []

    for dtype in A_matrices.keys():

        # Currently, 16-bit data type is not implemented
        if (dtype == 'float16'):
            continue

        # CUDA does not support 128-bit data type
        if (gpu is True) and (dtype == 'float128'):
            continue

        # OpenBLAS does not support 128-but data type
        if get_config()['use_cblas'] and (dtype == 'float128'):
            continue

        A_ = A_matrices[dtype]

        # Test both dot() and transpose_dot() functions
        for transpose in [False, True]:

            success = _test_dot(A_, transpose=transpose, symmetric=symmetric,
                                gpu=gpu)
            successes.append(success)

    return successes


# ===========
# test matrix
# ===========

def test_matrix():
    """
    A test for :mod:`imate.Matrix` class.
    """

    successes = []

    # n and m are the shapes of matrices A and B
    n, m = 100, 50

    # Density for sparse matrices
    density = 0.3

    # Dense matrix
    A = numpy.random.randn(n, m)
    A_sym = numpy.random.randn(n, n)
    A_sym = 0.5 * (A_sym + A_sym.T)

    A_c = numpy.ascontiguousarray(A)
    A_f = numpy.asfortranarray(A)
    A_sym_c = numpy.ascontiguousarray(A_sym)
    A_sym_f = numpy.asfortranarray(A_sym)

    successes += _test(A_c, symmetric=False)
    successes += _test(A_f, symmetric=False)
    successes += _test(A_sym_c, symmetric=True)
    successes += _test(A_sym_f, symmetric=True)

    # Sparse CSR
    A_csr = scipy.sparse.random(n, m, density=density, format='csr')
    A_csr_sym = scipy.sparse.random(n, n, density=density, format='csr')
    A_csr_sym = 0.5 * (A_csr_sym + A_csr_sym.T)

    successes += _test(A_csr, symmetric=False)
    successes += _test(A_csr_sym, symmetric=True)

    # Sparse CSC
    A_csc = scipy.sparse.random(n, m, density=density, format='csc')
    A_csc_sym = scipy.sparse.random(n, n, density=density, format='csc')
    A_csc_sym = 0.5 * (A_csc_sym + A_csc_sym.T)

    successes += _test(A_csc, symmetric=False)
    successes += _test(A_csc_sym, symmetric=True)

    # Sparse LIL
    A_lil = scipy.sparse.random(n, m, density=density, format='lil')
    A_lil_sym = scipy.sparse.random(n, n, density=density, format='lil')
    A_lil_sym = 0.5 * (A_lil_sym + A_lil_sym.T)

    successes += _test(A_lil, symmetric=False)
    successes += _test(A_lil_sym, symmetric=True)

    successes = numpy.array(successes, dtype=bool)
    if successes.all():
        print('OK')
    else:
        print(successes)
        raise RuntimeError('Test failed.')


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_matrix())
