# =======
# Imports
# =======

import sys
import numpy
import scipy.sparse
from ..._linear_algebra.types cimport data_type
from ..constant_matrix cimport ConstantMatrix

__all__ = ['test_constant_matrix']


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

cdef int _test_dot(A):
    """
    """

    # Define linear operator object
    cdef ConstantMatrix Aop = ConstantMatrix(A)

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

    return success


# ==================
# test transpose dot
# ==================

cdef int _test_transpose_dot(A):
    """
    """

    # Define linear operator object
    cdef ConstantMatrix Aop = ConstantMatrix(A)

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

    return success


# ====
# test
# ====

cpdef int _test(A) except *:
    """
    """

    # Test for ConstantMatrix.dot()
    success1 = _test_dot(A)

    # Test for ConstantMatrix.transpose_dot()
    success2 = _test_transpose_dot(A)

    if success1 and success2:
        return 1
    else:
        return 0


# ====================
# test constant matrix
# ====================

def test_constant_matrix():
    """
    Test for :mod:`imate.linear_operator.constant_matrix` module.
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
