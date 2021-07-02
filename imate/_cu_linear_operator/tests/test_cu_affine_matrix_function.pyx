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
from ..py_cu_affine_matrix_function cimport pycuAffineMatrixFunction

__all__ = ['test_cu_affine_matrix_function']


# ==================
# benchmark solution
# ==================

def _benchmark_solution(A, B, t, vector):
    """
    Multiplies ``A + t[i] * B`` by ``vector`` using numpy.

    :param A: An array of size ``(n, m)``.
    :type A: numpy.ndarray, or any scipy.sparse array

    :param B: An array of size ``(n, m)``.
    :type B: numpy.ndarray, or any scipy.sparse array

    :param t: The parameter of the affine matrix function
    :type t: DataType

    :param vector: The input vector to multiply with, with the size of ``m``.
    :type vector: numpy.array

    :return: The product ``(A + t * B) * vector``. The length of this array is
        ``n``.
    :rtype: numpy.array
    """

    n, m = A.shape

    if B is None:
        if scipy.sparse.issparse(A):
            K = A + t * scipy.sparse.eye(n, m, dtype=float)
        else:
            K = A + t * numpy.eye(n, m, dtype=float)
    else:
        K = A + t * B

    # Matrix-vector multiplication
    benchmark = K.dot(numpy.asarray(vector))

    return benchmark


# ========
# test dot
# ========

cdef int _test_dot(A, B, t) except *:
    """
    """

    # Define linear operator object
    cdef pycuAffineMatrixFunction Aop = pycuAffineMatrixFunction(A, B)

    # Set parameter
    cdef double parameters
    if t is not None:
        parameters = t
        Aop.set_parameters(parameters)

    # Matrix shape
    num_rows, num_columns = A.shape

    # Input random vectors
    vector = numpy.random.randn(num_columns)

    # Output product
    product = numpy.empty((num_rows), dtype=float)

    # The nogil environment is arbitrary
    Aop.dot(vector, product)

    # Benchmark product
    benchmark = _benchmark_solution(A, B, t, numpy.asarray(vector))

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


# ==================
# test transpose dot
# ==================

cdef int _test_transpose_dot(A, B, t) except *:
    """
    """

    # Define linear operator object
    cdef pycuAffineMatrixFunction Aop = pycuAffineMatrixFunction(A, B)

    # Set parameter
    cdef double parameters
    if t is not None:
        parameters = t
        Aop.set_parameters(parameters)

    # Matrix shape
    num_rows, num_columns = A.shape

    # Input random vectors
    vector = numpy.random.randn(num_rows)

    # Output product
    product = numpy.empty((num_columns), dtype=float)

    # The nogil environment is arbitrary
    Aop.transpose_dot(vector, product)

    # Benchmark product
    if B is None:
        benchmark = _benchmark_solution(A.T, B, t, numpy.asarray(vector))
    else:
        benchmark = _benchmark_solution(A.T, B.T, t, numpy.asarray(vector))

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

cpdef int _test(A, B, t) except *:
    """
    """

    # Dot test for AffineMatrixFunction.dot()
    success1 = _test_dot(A, B, t)

    # Transpose dot test for AffineMatrixFunction.transpose_dot()
    success2 = _test_transpose_dot(A, B, t)

    if success1 and success2:
        return 1
    else:
        return 0


# =============================
# test c affine matrix function
# =============================

def test_cu_affine_matrix_function():
    """
    A test for :mod:`imate.linear_operator.affine_matrix_function` module.
    """

    success = []

    # n and m are the shapes of matrices A and B
    n, m = 10, 7

    # Parameter t
    t = 2.2

    # Both A and B are random
    success.append(_test(
            numpy.random.randn(n, m),
            numpy.random.randn(n, m),
            t))

    # No B and t
    success.append(_test(
            numpy.random.randn(n, m),
            None,
            t))

    # Sparse CSR
    success.append(_test(
            scipy.sparse.random(n, m, density=0.2, format='csr'),
            scipy.sparse.random(n, m, density=0.2, format='csr'),
            t))

    # Sparse CSC
    success.append(_test(
            scipy.sparse.random(n, m, density=0.2, format='csc'),
            scipy.sparse.random(n, m, density=0.2, format='csc'),
            t))

    # Sparse LIL
    success.append(_test(
            scipy.sparse.random(n, m, density=0.2, format='lil'),
            scipy.sparse.random(n, m, density=0.2, format='lil'),
            t))

    success = numpy.array(success, dtype=bool)
    if success.all():
        print('OK')
    else:
        print(success)
