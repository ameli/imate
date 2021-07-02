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
from ..py_c_matrix cimport pycMatrix

__all__ = ['test_c_matrix']


# ==================
# benchmark solution
# ==================

def _benchmark_solution(A, vector):
    """
    Multiplies ``A`` by ``vector`` using numpy.

    :param A: An array of size ``(n, m)``.
    :type A: numpy.ndarray, or any scipy.sparse array

    :param vector: The input vector to multiply with, with the size of ``m``.
    :type vector: numpy.array

    :return: The product ``A * vector``. The length of this array is ``n``.
    :rtype: numpy.array
    """

    # Matrix-vector multiplication
    benchmark = A.dot(numpy.asarray(vector))

    return benchmark


# ========
# test dot
# ========

cdef int _test_dot(A) except *:
    """
    """

    # Define linear operator object
    cdef pycMatrix Aop = pycMatrix(A)

    # Matrix shape
    num_rows, num_columns = A.shape

    # Input random vectors
    vector = numpy.random.randn(num_columns)

    # Output product
    product = numpy.empty((num_rows), dtype=float)

    # The nogil environment is arbitrary
    Aop.dot(vector, product)

    # Benchmark product
    benchmark = _benchmark_solution(A, numpy.asarray(vector))

    # Error
    error = numpy.abs(numpy.asarray(product) - benchmark)

    # Print success
    cdef int success = 1
    cdef double tolerance = 1e-14
    if any(error > tolerance):
        print('ERROR: product and benchmark mismatch.')
        success = 0

        print('error:')
        for i in range(error.size):
            print('error: %e' % error[i])

            print(error[i])

    return success


# ==================
# test transpose dot
# ==================

cdef int _test_transpose_dot(A) except *:
    """
    """

    # Define linear operator object
    cdef pycMatrix Aop = pycMatrix(A)

    # Matrix shape
    num_rows, num_columns = A.shape

    # Input random vectors
    vector = numpy.random.randn(num_rows)

    # Output product
    product = numpy.empty((num_columns), dtype=float)

    # The nogil environment is arbitrary
    Aop.transpose_dot(vector, product)

    # Benchmark product
    benchmark = _benchmark_solution(A.T, numpy.asarray(vector))

    # Error
    error = numpy.abs(numpy.asarray(product) - benchmark)

    # Print success
    cdef int success = 1
    cdef double tolerance = 1e-14
    if any(error > tolerance):
        print('ERROR: product and benchmark mismatch.')
        success = 0

        print('error:')
        for i in range(error.size):
            print('error: %e' % error[i])

    return success


# ====
# test
# ====

cpdef int _test(A) except *:
    """
    """

    # Dot test for pycMatrix.dot()
    success1 = _test_dot(A)

    # Transose dot est for pycMatrix.transpose_dot()
    success2 = _test_transpose_dot(A)

    if success1 and success2:
        return 1
    else:
        return 0


# =============
# test c matrix
# =============

def test_c_matrix():
    """
    A test for :mod:`imate.linear_operator.constant_matrix` module.
    """

    success = []

    # n and m are the shapes of matrices A and B
    n, m = 10, 7

    # Parameter t
    t = 2.2

    # Dense matrix
    success.append(_test(numpy.random.randn(n, m)))

    # Sparse CSR
    success.append(_test(scipy.sparse.random(n, m, density=0.2, format='csr')))

    # Sparse CSC
    success.append(_test(scipy.sparse.random(n, m, density=0.2, format='csc')))

    # Sparse LIL
    success.append(_test(scipy.sparse.random(n, m, density=0.2, format='lil')))

    success = numpy.array(success, dtype=bool)
    if success.all():
        print('OK')
    else:
        print(success)
