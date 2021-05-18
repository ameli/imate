# =======
# Imports
# =======

import sys
import numpy
import scipy.sparse
from ..._linear_algebra.types cimport data_type
from ..affine_matrix_function cimport AffineMatrixFunction

__all__ = ['test_affine_matrix_function']


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
    :type t: data_type

    :param vector: The input vector to multiply with, with the size of ``m``.
    :type vector: numpy.array

    :return: The product ``(A + t * B) * vector``. The length of this array is
        ``n``.
    :rtype: numpy.array
    """

    n, m = A.shape

    if (t is None) or (B is None):
        K = A
    elif numpy.isscalar(B):
        if B == 0:
            K = A
        elif B == 1:
            if scipy.sparse.issparse(B):
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

cdef int _test_dot(A, B, t):
    """
    """

    # Define linear operator object
    cdef AffineMatrixFunction Aop = AffineMatrixFunction(A, B)

    # Set parameter
    cdef data_type parameters
    if t is not None:
        parameters = t
        Aop.set_parameters(&parameters)

    # Matrix shape
    num_rows, num_columns = A.shape

    # Input random vectors
    cdef double[:] vector = numpy.random.randn(num_columns)

    # Output product
    cdef double[:] product = numpy.empty((num_rows), dtype=float)

    # The nogil environment is arbitrary
    with nogil:
        Aop.dot(&vector[0], &product[0])

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

cdef int _test_transpose_dot(A, B, t):
    """
    """

    # Define linear operator object
    cdef AffineMatrixFunction Aop = AffineMatrixFunction(A, B)

    # Set parameter
    cdef data_type parameters
    if t is not None:
        parameters = t
        Aop.set_parameters(&parameters)

    # Matrix shape
    num_rows, num_columns = A.shape

    # Input random vectors
    cdef double[:] vector = numpy.random.randn(num_columns)

    # Output product
    cdef double[:] product = numpy.empty((num_rows), dtype=float)

    # The nogil environment is arbitrary
    with nogil:
        Aop.dot(&vector[0], &product[0])

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


# ====
# test
# ====

cpdef int _test(A, B, t) except *:
    """
    """

    # Test for AffineMatrixFunction.dot()
    success1 = _test_dot(A, B, t)

    # Test for AffineMatrixFunction.transpose_dot()
    success2 = _test_transpose_dot(A, B, t)

    if success1 and success2:
        return 1
    else:
        return 0


# ===========================
# test affine matrix function
# ===========================

def test_affine_matrix_function():
    """
    Test for :mod:`imate.linear_operator.affine_matrix_function` module.
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
            None))

    # B as zero matrix
    success.append(_test(
            numpy.random.randn(n, m),
            0,
            None))

    # B as identity matrix
    success.append(_test(
            numpy.random.randn(n, m),
            1,
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
