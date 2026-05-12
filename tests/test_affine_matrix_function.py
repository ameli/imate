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
from imate import AffineMatrixFunction, get_config

__all__ = ['test_affine_matrix_function']


# ==================
# benchmark solution
# ==================

def _benchmark_solution(A, B, t, vector, transpose=False):
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

    :param transpose: If `True`, the transpose of the input matrix is used.
    :type transpose: boolean

    :return: The product ``(A + t * B) * vector``. The length of this array is
        ``n``.
    :rtype: numpy.array
    """

    n, m = A.shape

    if B is None:
        if scipy.sparse.issparse(A):
            K = A + t * scipy.sparse.eye(n, m, dtype=A.dtype)
        else:
            K = A + t * numpy.eye(n, m, dtype=A.dtype)
    else:
        K = A + t * B

    if transpose:
        K_ = K.T
    else:
        K_ = K

    # Matrix-vector multiplication
    vector_typed = vector.astype(A.dtype)
    benchmark = K_.dot(vector_typed)

    return benchmark


# ===========
# check error
# ===========

def _check_error(error, n, t, dtype, transpose, symmetric, factor=1.0):
    """
    Suppose a, b, and x and the elements of A, B, and x. Let e be the floating
    point precision (epsilon) of a number. Thus, the arithmetic error of
    (a+tb)*x can be found, roughly, by expanding (a+e + (t+e)*(b+e))*(x+e),
    and ignoring all higher order terms of the powers of e. The first
    order term involving e^1 is:

        (A + tB + x + tx + bx) * e.

    Assuming A , B, and x, are normally distributed as N(0, 1), on average,
    the above error term is (3 + 2t) * e.
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

    # Arithmetic standard-deviation error of (A+tB)*x (see above documentation)
    sigma = (3.0 + 2.0 * t) * precision

    # Print success
    success = True
    z_score = 3.29  # 99.9% confidence interval for A[i, j] and x[j] ~ N(0, 1)
    matvec_tolerance = 2.0 * z_score * sigma * n * factor
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

def _test_dot(
        A,
        B,
        t,
        transpose=False,
        symmetric=False,
        gpu=False,
        factor=1.0):
    """
    """

    # Define linear operator object
    Aop = AffineMatrixFunction(A, B, A_is_symmetric=symmetric,
                               B_is_symmetric=symmetric)
    Aop.initialize(gpu=gpu)

    # Set parameter
    if t is not None:
        parameters = t
        Aop.set_parameters(parameters)

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
    benchmark = _benchmark_solution(A, B, t, vector, transpose=transpose)

    # Error
    error = numpy.abs(product - benchmark)

    # Compare error with tolerance
    success = _check_error(error, product.size, t, A.dtype, transpose,
                           symmetric, factor=factor)

    return success


# ====
# test
# ====

def _test(A, B, t, symmetric=False, gpu=False, factor=1.0):
    """
    """

    A_matrices = {
        'float16': A.astype(numpy.float16),
        'float32': A.astype(numpy.float32),
        'float64': A.astype(numpy.float64),
        'float128': A.astype(numpy.float128)
    }

    if B is not None:
        B_matrices = {
            'float16': B.astype(numpy.float16),
            'float32': B.astype(numpy.float32),
            'float64': B.astype(numpy.float64),
            'float128': B.astype(numpy.float128)
        }

    # Make 128-bit data truly contain 19 decimals by filling decimals 16 to 19
    precision = 1.084202172485504434e-19   # epsilon for log-double
    A_matrices['float128'] = A_matrices['float128'] * \
        (1.0 + (1.0e+0 + 1.0e+1 + 1.0e+2 + 1.0e+3) * precision)

    if B is not None:
        B_matrices['float128'] = B_matrices['float128'] * \
            (1.0 + (1.0e+0 + 1.0e+1 + 1.0e+2 + 1.0e+3) * precision)

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
        if B is not None:
            B_ = B_matrices[dtype]
        else:
            B_ = None

        # Test both dot() and transpose_dot() functions
        for transpose in [False, True]:

            success = _test_dot(A_, B_, t, transpose=transpose,
                                symmetric=symmetric, gpu=gpu, factor=factor)
            successes.append(success)

    return successes


# ===========================
# test affine matrix function
# ===========================

def test_affine_matrix_function():
    """
    A test for :mod:`imate.AffineMatrixFunction` class.

    Notes:

    Usually, the matrix-vector product between imate's AffineMatrixFunction
    and numpy/scipy matches up to machine precision. However, under the
    following conditions, the error of the two packages (imate and numpy/scipy)
    slightly goes beyond machine precision, such as twice or often up to 20
    times the machine precision. All these cases happen for float128 data. Here
    are these conditions:

    1. float128 data, A is lil sparse, B is lil sparse.
       In this case, imate tries to convert lil to csr, and I guess here is
       where it looses some precision.

    2. float128 data, A is any matrix (dense, csr, csc, lil), and B is identity
       of the same type (dense, csr, csc, lil). In this case, imate inquires
       that B is identity (see is_identity_matrix() function). If yes, it
       does not perform matrix-vector products on B, rather, does vector
       addition, which reduces the error. But, numpy/scipy does not have such
       feature, and performs the full matrix-vector product. Such discrepancy
       of floating point arithmetic error is observable on float128 data.

    The mismatch of errors in above above cases are very normal. To suppress
    printing error, for these special cases, we increase the error tolerance
    by a factor of 20.
    """

    successes = []

    # n and m are the shapes of matrices A and B
    n, m = 100, 50

    # Density for sparse matrices
    density = 0.3

    # Parameter t
    t = 2.2

    # Increase error tolerance factor for specific cases where we except there
    # is a higher discrepancy between imate and numpy/scipy in matrix
    # multiplication. This usually applies to float 128, when B is identity,
    # or when A and B are lil sparse matrices.
    factor = 20.0

    # Dense matrix
    A = numpy.random.randn(n, m)
    A_sym = numpy.random.randn(n, n)
    A_sym = 0.5 * (A_sym + A_sym.T)

    B = numpy.random.randn(n, m)
    B_sym = numpy.random.randn(n, n)
    B_sym = 0.5 * (B_sym + B_sym.T)

    # Identity matrix
    I = numpy.eye(n, n)                                            # noqa: E741
    I_c = numpy.ascontiguousarray(I)
    I_f = numpy.asfortranarray(I)

    A_c = numpy.ascontiguousarray(A)
    A_f = numpy.asfortranarray(A)
    A_sym_c = numpy.ascontiguousarray(A_sym)
    A_sym_f = numpy.asfortranarray(A_sym)

    B_c = numpy.ascontiguousarray(B)
    B_f = numpy.asfortranarray(B)
    B_sym_c = numpy.ascontiguousarray(B_sym)
    B_sym_f = numpy.asfortranarray(B_sym)

    # Using B as None
    successes += _test(A_c, None, t, symmetric=False)
    successes += _test(A_f, None, t, symmetric=False)
    successes += _test(A_sym_c, None, t, symmetric=True)
    successes += _test(A_sym_f, None, t, symmetric=True)

    # Using B as identity matrix and a square matrix A
    successes += _test(A_sym_c, I_c, t, symmetric=False, factor=factor)
    successes += _test(A_sym_f, I_f, t, symmetric=False, factor=factor)
    successes += _test(A_sym_c, I_c, t, symmetric=True, factor=factor)
    successes += _test(A_sym_f, I_f, t, symmetric=True, factor=factor)

    # Using B as non-None
    successes += _test(A_c, B_c, t, symmetric=False)
    successes += _test(A_f, B_f, t, symmetric=False)
    successes += _test(A_sym_c, B_sym_c, t, symmetric=True)
    successes += _test(A_sym_f, B_sym_f, t, symmetric=True)

    # # Sparse CSR
    A_csr = scipy.sparse.random(n, m, density=density, format='csr')
    A_csr_sym = scipy.sparse.random(n, n, density=density, format='csr')

    B_csr = scipy.sparse.random(n, m, density=density, format='csr')
    B_csr_sym = scipy.sparse.random(n, n, density=density, format='csr')
    I_csr = scipy.sparse.eye(n, n, format='csr')

    A_csr_sym = 0.5 * (A_csr_sym + A_csr_sym.T)
    B_csr_sym = 0.5 * (B_csr_sym + B_csr_sym.T)

    successes += _test(A_csr, None, t, symmetric=False)
    successes += _test(A_csr_sym, None, t, symmetric=True)
    successes += _test(A_csr_sym, I_csr, t, symmetric=False, factor=factor)
    successes += _test(A_csr_sym, I_csr, t, symmetric=True, factor=factor)
    successes += _test(A_csr, B_csr, t, symmetric=False)
    successes += _test(A_csr_sym, B_csr_sym, t, symmetric=True)

    # Sparse CSC
    A_csc = scipy.sparse.random(n, m, density=density, format='csc')
    A_csc_sym = scipy.sparse.random(n, n, density=density, format='csc')
    A_csc_sym = 0.5 * (A_csc_sym + A_csc_sym.T)

    B_csc = scipy.sparse.random(n, m, density=density, format='csc')
    B_csc_sym = scipy.sparse.random(n, n, density=density, format='csc')
    B_csc_sym = 0.5 * (B_csc_sym + B_csc_sym.T)
    I_csc = scipy.sparse.eye(n, n, format='csc')

    successes += _test(A_csc, None, t, symmetric=False)
    successes += _test(A_csc_sym, None, t, symmetric=True)
    successes += _test(A_csc_sym, I_csc, t, symmetric=False, factor=factor)
    successes += _test(A_csc_sym, I_csc, t, symmetric=True, factor=factor)
    successes += _test(A_csc, B_csc, t, symmetric=False)
    successes += _test(A_csc_sym, B_csc_sym, t, symmetric=True)

    # Sparse LIL
    A_lil = scipy.sparse.random(n, m, density=density, format='lil')
    A_lil_sym = scipy.sparse.random(n, n, density=density, format='lil')
    A_lil_sym = 0.5 * (A_lil_sym + A_lil_sym.T).tolil()

    B_lil = scipy.sparse.random(n, m, density=density, format='lil')
    B_lil_sym = scipy.sparse.random(n, n, density=density, format='lil')
    B_lil_sym = 0.5 * (B_lil_sym + B_lil_sym.T).tolil()
    I_lil = scipy.sparse.eye(n, n, format='lil')

    successes += _test(A_lil, None, t, symmetric=False)
    successes += _test(A_lil_sym, None, t, symmetric=True)
    successes += _test(A_lil_sym, I_lil, t, symmetric=False, factor=factor)
    successes += _test(A_lil_sym, I_lil, t, symmetric=True, factor=factor)
    successes += _test(A_lil, B_lil, t, symmetric=False, factor=factor)
    successes += _test(A_lil_sym, B_lil_sym, t, symmetric=True, factor=factor)

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
    sys.exit(test_affine_matrix_function())
